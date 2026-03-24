import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson
import typer

app = typer.Typer()


def fetch_project_info(data):
    workspace, project_id = data
    project = workspace.project(project_id)
    versions = project.versions()
    return project, versions


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    return float(stripped)


def _best_row(
    rows: list[dict[str, float]], metric: str, maximize: bool = True
) -> dict[str, float] | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[metric]) if maximize else min(
        candidates, key=lambda row: row[metric]
    )


def _resolve_dataset_split(dataset_path: Path, split: str) -> tuple[Path, Path]:
    split_dir_map = {
        "train": "train",
        "valid": "valid",
        "val": "valid",
        "test": "test",
    }
    split_dir_name = split_dir_map.get(split.lower())
    if split_dir_name is None:
        valid_splits = ", ".join(sorted(split_dir_map))
        raise typer.BadParameter(f"Unsupported split '{split}'. Expected one of: {valid_splits}")

    split_dir = dataset_path / split_dir_name
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise typer.BadParameter(f"Missing annotation file for split '{split}': {annotation_path}")
    return split_dir, annotation_path


def _load_coco_annotations(
    annotation_path: Path,
) -> tuple[
    dict[int, dict],
    dict[int, list[dict]],
    list[dict],
    dict[int, int],
    dict[int, int],
]:
    dataset = orjson.loads(annotation_path.read_bytes())
    images = sorted(dataset.get("images", []), key=lambda image: image["id"])
    annotations = dataset.get("annotations", [])
    categories = sorted(dataset.get("categories", []), key=lambda category: category["id"])

    images_by_id = {image["id"]: image for image in images}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    label_to_category_id = {
        label_index: category["id"] for label_index, category in enumerate(categories)
    }
    category_id_to_label = {
        category["id"]: label_index for label_index, category in enumerate(categories)
    }
    return (
        images_by_id,
        annotations_by_image,
        categories,
        label_to_category_id,
        category_id_to_label,
    )


def _mask_iou(mask_a, mask_b) -> float:
    import numpy as np

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    intersection = width * height
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _decode_segmentation(segmentation, height: int, width: int):
    import numpy as np
    from pycocotools import mask as coco_mask

    if not segmentation:
        return np.zeros((height, width), dtype=bool)

    if isinstance(segmentation, dict):
        encoded = dict(segmentation)
        counts = encoded.get("counts")
        if isinstance(counts, str):
            encoded["counts"] = counts.encode("utf-8")
        decoded = coco_mask.decode(encoded)
    else:
        encoded = coco_mask.frPyObjects(segmentation, height, width)
        decoded = coco_mask.decode(encoded)

    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def _encode_binary_mask(mask):
    import numpy as np
    from pycocotools import mask as coco_mask

    encoded = coco_mask.encode(np.asfortranarray(mask.astype("uint8")))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        encoded["counts"] = counts.decode("utf-8")
    return encoded


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _percent_label(value: float) -> str:
    return f"{value * 100:.1f}%"


def _score_image(row: dict) -> tuple[float, int, int, str]:
    return (row["f1"], -row["false_negatives"], -row["false_positives"], row["file_name"])


def _write_csv(output_path: Path, rows: list[dict]):
    if not rows:
        output_path.write_text("")
        return

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_overlay_image(
    image_path: Path,
    output_path: Path,
    gt_items: list[dict],
    pred_items: list[dict],
    summary_text: str,
):
    import numpy as np
    from PIL import Image, ImageDraw

    colors = {
        "gt": (46, 204, 113),
        "pred": (231, 76, 60),
    }
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    def blend_mask(mask, color, alpha=0.28):
        if mask is None:
            return
        mask_indices = mask.astype(bool)
        if not mask_indices.any():
            return
        image_array[mask_indices] = (
            image_array[mask_indices] * (1 - alpha) + np.array(color) * alpha
        ).astype("uint8")

    for item in gt_items:
        blend_mask(item.get("mask"), colors["gt"])
    for item in pred_items:
        blend_mask(item.get("mask"), colors["pred"])

    annotated = Image.fromarray(image_array)
    draw = ImageDraw.Draw(annotated)

    for item in gt_items:
        box = item["bbox"]
        draw.rectangle(box, outline=colors["gt"], width=3)
        draw.text((box[0] + 4, max(4, box[1] - 16)), f"GT {item['class_name']}", fill=colors["gt"])

    for item in pred_items:
        box = item["bbox"]
        label = f"P {item['class_name']} {item['score']:.2f}"
        draw.rectangle(box, outline=colors["pred"], width=3)
        draw.text((box[0] + 4, box[1] + 4), label, fill=colors["pred"])

    draw.rectangle((0, 0, annotated.width, 24), fill=(0, 0, 0))
    draw.text((8, 5), summary_text, fill=(255, 255, 255))
    annotated.save(output_path, quality=92)


def _run_coco_eval(coco_gt, results: list[dict], iou_type: str) -> dict[str, object]:
    from contextlib import redirect_stdout
    import io

    import numpy as np
    from pycocotools.cocoeval import COCOeval

    if not results:
        return {
            "stats": None,
            "per_class_ap50": {},
            "per_class_ap": {},
        }

    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    capture = io.StringIO()
    with redirect_stdout(capture):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    iou_thresholds = evaluator.params.iouThrs
    ap50_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))

    per_class_ap = {}
    per_class_ap50 = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        class_precision = precision[:, :, category_index, 0, -1]
        class_precision = class_precision[class_precision > -1]
        category_id = int(category_id)
        per_class_ap[category_id] = float(class_precision.mean()) if class_precision.size else 0.0

        class_precision_ap50 = precision[ap50_index, :, category_index, 0, -1]
        class_precision_ap50 = class_precision_ap50[class_precision_ap50 > -1]
        per_class_ap50[category_id] = (
            float(class_precision_ap50.mean()) if class_precision_ap50.size else 0.0
        )

    return {
        "stats": [float(value) for value in evaluator.stats.tolist()],
        "per_class_ap50": per_class_ap50,
        "per_class_ap": per_class_ap,
    }


def _write_evaluation_plots(
    output_dir: Path,
    per_class_rows: list[dict],
    per_image_rows: list[dict],
    prediction_rows: list[dict],
    coco_metrics: dict[str, dict[str, object]],
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    class_names = [row["class_name"] for row in per_class_rows]
    perf_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Per-class quality",
            "Per-class counts",
            "Prediction confidence",
            "Per-image F1",
        ),
    )
    for metric in ("precision", "recall", "f1", "mean_matched_iou"):
        perf_fig.add_trace(
            go.Bar(
                x=class_names,
                y=[row[metric] for row in per_class_rows],
                name=metric,
            ),
            row=1,
            col=1,
        )
    for metric in ("true_positives", "false_positives", "false_negatives"):
        perf_fig.add_trace(
            go.Bar(
                x=class_names,
                y=[row[metric] for row in per_class_rows],
                name=metric,
            ),
            row=1,
            col=2,
        )

    matched_scores = [row["score"] for row in prediction_rows if row["match_status"] == "tp"]
    unmatched_scores = [row["score"] for row in prediction_rows if row["match_status"] == "fp"]
    perf_fig.add_trace(
        go.Histogram(x=matched_scores, name="true positives", opacity=0.75),
        row=2,
        col=1,
    )
    perf_fig.add_trace(
        go.Histogram(x=unmatched_scores, name="false positives", opacity=0.75),
        row=2,
        col=1,
    )
    perf_fig.add_trace(
        go.Histogram(
            x=[row["f1"] for row in per_image_rows],
            name="per-image F1",
            nbinsx=20,
        ),
        row=2,
        col=2,
    )

    perf_fig.update_layout(
        barmode="group",
        template="plotly_white",
        hovermode="x unified",
        height=900,
        width=1400,
    )
    perf_fig.write_html(plots_dir / "performance_overview.html", include_plotlyjs="cdn")

    iou_scatter = go.Figure()
    if prediction_rows:
        iou_scatter.add_trace(
            go.Scatter(
                x=[row["score"] for row in prediction_rows],
                y=[row["best_iou"] for row in prediction_rows],
                mode="markers",
                marker={
                    "size": 9,
                    "color": [
                        "#2ecc71" if row["match_status"] == "tp" else "#e74c3c"
                        for row in prediction_rows
                    ],
                },
                text=[
                    f"{row['file_name']}<br>{row['class_name']}<br>{row['match_status']}"
                    for row in prediction_rows
                ],
                hovertemplate="%{text}<br>score=%{x:.3f}<br>IoU=%{y:.3f}<extra></extra>",
                name="predictions",
            )
        )
    iou_scatter.update_layout(
        title="Prediction score vs best IoU",
        xaxis_title="confidence",
        yaxis_title="best IoU",
        template="plotly_white",
        height=650,
        width=1100,
    )
    iou_scatter.write_html(plots_dir / "prediction_scatter.html", include_plotlyjs="cdn")

    if any(metrics["stats"] for metrics in coco_metrics.values()):
        coco_fig = make_subplots(
            rows=1,
            cols=len([name for name, metrics in coco_metrics.items() if metrics["stats"] is not None]),
            subplot_titles=[
                f"{name.upper()} COCO AP"
                for name, metrics in coco_metrics.items()
                if metrics["stats"] is not None
            ],
        )
        col_index = 1
        for metric_name, metrics in coco_metrics.items():
            if metrics["stats"] is None:
                continue
            coco_fig.add_trace(
                go.Bar(
                    x=class_names,
                    y=[
                        metrics["per_class_ap"].get(row["category_id"], 0.0)
                        for row in per_class_rows
                    ],
                    name=f"{metric_name} AP",
                ),
                row=1,
                col=col_index,
            )
            coco_fig.add_trace(
                go.Bar(
                    x=class_names,
                    y=[
                        metrics["per_class_ap50"].get(row["category_id"], 0.0)
                        for row in per_class_rows
                    ],
                    name=f"{metric_name} AP50",
                ),
                row=1,
                col=col_index,
            )
            col_index += 1

        coco_fig.update_layout(
            barmode="group",
            template="plotly_white",
            hovermode="x unified",
            height=550,
            width=max(900, 600 * (col_index - 1)),
        )
        coco_fig.write_html(plots_dir / "coco_metrics.html", include_plotlyjs="cdn")


@app.command()
def metrics(
    metrics_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to write generated plots to. Defaults to a sibling folder.",
    ),
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    if output_dir is None:
        output_dir = metrics_path.parent / f"{metrics_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, float]] = []
    val_rows: list[dict[str, float]] = []
    lr_rows: list[dict[str, float]] = []

    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for raw_row in reader:
            row = {name: _parse_float(raw_row.get(name)) for name in fieldnames}
            step = row.get("step")
            epoch = row.get("epoch")
            if step is None or epoch is None:
                continue

            row["step"] = int(step)
            row["epoch"] = int(epoch)

            if row.get("train/loss") is not None:
                train_rows.append(row)
            if any(row.get(metric) is not None for metric in ("val/mAP_50", "val/loss")):
                val_rows.append(row)
            if row.get("train/lr") is not None:
                lr_rows.append(row)

    if not any((train_rows, val_rows, lr_rows)):
        typer.echo(f"No plottable metrics found in {metrics_path}")
        raise typer.Exit(code=1)

    def add_series(
        fig,
        row_index: int,
        col_index: int,
        rows: list[dict[str, float]],
        metrics_to_plot: list[str],
    ) -> bool:
        plotted = False
        for metric in metrics_to_plot:
            points = [
                (row["step"], row[metric])
                for row in rows
                if row.get(metric) is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points, strict=False)
            fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="lines+markers",
                    name=metric,
                    legendgroup=metric,
                ),
                row=row_index,
                col=col_index,
            )
            plotted = True
        return plotted

    summary_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training losses",
            "Validation overview",
            "Validation quality",
            "Learning rate",
        ),
    )
    training_plotted = add_series(
        summary_fig,
        1,
        1,
        train_rows,
        ["train/loss", "train/loss_ce", "train/loss_bbox", "train/loss_giou"],
    )
    validation_overview_plotted = add_series(
        summary_fig,
        1,
        2,
        val_rows,
        ["val/loss", "val/mAP_50", "val/mAP_50_95", "val/F1"],
    )
    validation_quality_plotted = add_series(
        summary_fig,
        2,
        1,
        val_rows,
        [
            "val/precision",
            "val/recall",
            "val/mAR",
            "val/ema_mAP_50",
            "val/ema_mAP_50_95",
        ],
    )
    lr_plotted = add_series(
        summary_fig,
        2,
        2,
        lr_rows,
        ["train/lr", "train/lr_max", "train/lr_min"],
    )
    for index, plotted in enumerate(
        (
            training_plotted,
            validation_overview_plotted,
            validation_quality_plotted,
            lr_plotted,
        ),
        start=1,
    ):
        row_index = 1 if index <= 2 else 2
        col_index = 1 if index in (1, 3) else 2
        summary_fig.update_xaxes(title_text="step", row=row_index, col=col_index)
        if not plotted:
            summary_fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{index} domain",
                yref=f"y{index} domain",
                text="No data",
                showarrow=False,
            )

    summary_fig.update_layout(
        height=900,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
    )
    summary_path = output_dir / "training_summary.html"
    summary_fig.write_html(summary_path, include_plotlyjs="cdn")

    per_class_metrics = sorted(
        name for name in fieldnames if name.startswith("val/AP/")
    )
    per_class_path = None
    if per_class_metrics and val_rows:
        per_class_fig = go.Figure()
        for metric in per_class_metrics:
            points = [
                (row["step"], row[metric])
                for row in val_rows
                if row.get(metric) is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points, strict=False)
            per_class_fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="lines+markers",
                    name=metric,
                )
            )

        per_class_fig.update_layout(
            title="Per-class AP",
            xaxis_title="step",
            yaxis_title="value",
            height=600,
            width=1400,
            template="plotly_white",
            hovermode="x unified",
        )
        per_class_path = output_dir / "per_class_ap.html"
        per_class_fig.write_html(per_class_path, include_plotlyjs="cdn")

    best_map50 = _best_row(val_rows, "val/mAP_50")
    best_ema_map50 = _best_row(val_rows, "val/ema_mAP_50")
    best_f1 = _best_row(val_rows, "val/F1")
    lowest_val_loss = _best_row(val_rows, "val/loss", maximize=False)
    final_train = max(train_rows, key=lambda row: row["step"]) if train_rows else None

    typer.echo(f"Analyzed metrics from {metrics_path}")
    typer.echo(f"Plots written to {output_dir}")
    typer.echo(f"  - {summary_path.name}")
    if per_class_path is not None:
        typer.echo(f"  - {per_class_path.name}")

    typer.echo("\nValidation highlights:")
    if best_map50 is not None:
        typer.echo(
            f"  - best val/mAP_50: {best_map50['val/mAP_50']:.4f} "
            f"(epoch {best_map50['epoch']}, step {best_map50['step']})"
        )
    if best_ema_map50 is not None:
        typer.echo(
            f"  - best val/ema_mAP_50: {best_ema_map50['val/ema_mAP_50']:.4f} "
            f"(epoch {best_ema_map50['epoch']}, step {best_ema_map50['step']})"
        )
    if best_f1 is not None:
        typer.echo(
            f"  - best val/F1: {best_f1['val/F1']:.4f} "
            f"(epoch {best_f1['epoch']}, step {best_f1['step']})"
        )
    if lowest_val_loss is not None:
        typer.echo(
            f"  - lowest val/loss: {lowest_val_loss['val/loss']:.4f} "
            f"(epoch {lowest_val_loss['epoch']}, step {lowest_val_loss['step']})"
        )
    if final_train is not None and final_train.get("train/loss") is not None:
        typer.echo(
            f"  - final train/loss: {final_train['train/loss']:.4f} "
            f"(epoch {final_train['epoch']}, step {final_train['step']})"
        )


@app.command()
def evaluate(
    checkpoint_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    dataset_path: Path = typer.Argument(..., exists=True, file_okay=False),
    split: str = typer.Option("test", help="Dataset split to evaluate: train, valid, or test."),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to write overlays, plots, and summaries to.",
    ),
    model_type: str = typer.Option(
        "rfdetr-seg-nano",
        help="Model architecture to instantiate for the checkpoint.",
    ),
    resolution: int = typer.Option(372, help="Inference resolution."),
    threshold: float = typer.Option(0.25, help="Prediction confidence threshold."),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for TP/FP matching."),
    max_overlay_images: int = typer.Option(
        100,
        help="Maximum number of overlay images to save. Worst examples are saved first.",
    ),
    limit: int | None = typer.Option(
        None,
        help="Optional cap on the number of images to evaluate for debugging.",
    ),
):
    import json

    import torch
    from pycocotools.coco import COCO
    from rfdetr import (
        RFDETRBase,
        RFDETRLarge,
        RFDETRMedium,
        RFDETRNano,
        RFDETRSeg2XLarge,
        RFDETRSegLarge,
        RFDETRSegMedium,
        RFDETRSegNano,
        RFDETRSegPreview,
        RFDETRSegSmall,
        RFDETRSegXLarge,
        RFDETRSmall,
    )

    model_registry = {
        "rfdetr-base": RFDETRBase,
        "rfdetr-nano": RFDETRNano,
        "rfdetr-small": RFDETRSmall,
        "rfdetr-medium": RFDETRMedium,
        "rfdetr-large": RFDETRLarge,
        "rfdetr-seg-preview": RFDETRSegPreview,
        "rfdetr-seg-nano": RFDETRSegNano,
        "rfdetr-seg-small": RFDETRSegSmall,
        "rfdetr-seg-medium": RFDETRSegMedium,
        "rfdetr-seg-large": RFDETRSegLarge,
        "rfdetr-seg-xlarge": RFDETRSegXLarge,
        "rfdetr-seg-2xlarge": RFDETRSeg2XLarge,
    }
    model_class = model_registry.get(model_type)
    if model_class is None:
        valid_models = ", ".join(sorted(model_registry))
        raise typer.BadParameter(f"Unsupported model type '{model_type}'. Expected one of: {valid_models}")

    split_dir, annotation_path = _resolve_dataset_split(dataset_path, split)
    (
        images_by_id,
        annotations_by_image,
        categories,
        label_to_category_id,
        _category_id_to_label,
    ) = _load_coco_annotations(annotation_path)

    if output_dir is None:
        output_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_{split_dir.name}_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_head_size = int(checkpoint["model"]["class_embed.bias"].shape[0])
    dataset_label_count = len(categories)
    if checkpoint_head_size > dataset_label_count:
        typer.echo(
            "Checkpoint has more class slots than this dataset; "
            f"restricting evaluation to the first {dataset_label_count} label slots."
        )

    model = model_class(pretrain_weights=str(checkpoint_path), resolution=resolution)
    category_names = {
        category["id"]: category["name"] for category in categories
    }

    image_records = [images_by_id[image_id] for image_id in sorted(images_by_id)]
    if limit is not None:
        image_records = image_records[:limit]

    bbox_results: list[dict] = []
    segm_results: list[dict] = []
    per_image_rows: list[dict] = []
    prediction_rows: list[dict] = []
    overlay_payloads: list[dict] = []
    per_class_counts = {
        category["id"]: {"tp": 0, "fp": 0, "fn": 0, "matched_iou_sum": 0.0, "matched_iou_count": 0}
        for category in categories
    }

    typer.echo(
        f"Evaluating {len(image_records)} images from {split_dir} using {checkpoint_path.name}"
    )

    with typer.progressbar(image_records, label="Running inference") as progress:
        for image_record in progress:
            image_id = image_record["id"]
            image_path = split_dir / image_record["file_name"]
            gt_annotations = annotations_by_image.get(image_id, [])
            detections = model.predict(str(image_path), threshold=threshold)

            pred_items = []
            for index in range(len(detections)):
                class_id = int(detections.class_id[index])
                if class_id >= dataset_label_count:
                    continue

                mapped_category_id = label_to_category_id[class_id]
                bbox_xyxy = [float(value) for value in detections.xyxy[index].tolist()]
                score = float(detections.confidence[index])
                mask = detections.mask[index].astype(bool) if detections.mask is not None else None
                pred_items.append(
                    {
                        "prediction_index": index,
                        "category_id": mapped_category_id,
                        "class_name": category_names[mapped_category_id],
                        "bbox": bbox_xyxy,
                        "score": score,
                        "mask": mask,
                    }
                )
                bbox_results.append(
                    {
                        "image_id": image_id,
                        "category_id": mapped_category_id,
                        "bbox": _xyxy_to_xywh(bbox_xyxy),
                        "score": score,
                    }
                )
                if mask is not None:
                    segm_results.append(
                        {
                            "image_id": image_id,
                            "category_id": mapped_category_id,
                            "segmentation": _encode_binary_mask(mask),
                            "score": score,
                        }
                    )

            gt_items = []
            image_height = int(image_record["height"])
            image_width = int(image_record["width"])
            for annotation in gt_annotations:
                gt_items.append(
                    {
                        "annotation_id": annotation["id"],
                        "category_id": annotation["category_id"],
                        "class_name": category_names[annotation["category_id"]],
                        "bbox": _xywh_to_xyxy(annotation["bbox"]),
                        "mask": _decode_segmentation(
                            annotation.get("segmentation"),
                            image_height,
                            image_width,
                        ),
                    }
                )

            gt_matched = set()
            image_matched_ious: list[float] = []
            true_positives = 0
            false_positives = 0

            for pred_item in sorted(pred_items, key=lambda item: item["score"], reverse=True):
                best_match_index = None
                best_iou = 0.0
                for gt_index, gt_item in enumerate(gt_items):
                    if gt_index in gt_matched or gt_item["category_id"] != pred_item["category_id"]:
                        continue

                    if pred_item["mask"] is not None and gt_item["mask"] is not None:
                        overlap = _mask_iou(pred_item["mask"], gt_item["mask"])
                    else:
                        overlap = _bbox_iou(pred_item["bbox"], gt_item["bbox"])

                    if overlap > best_iou:
                        best_iou = overlap
                        best_match_index = gt_index

                if best_match_index is not None and best_iou >= iou_threshold:
                    gt_matched.add(best_match_index)
                    true_positives += 1
                    image_matched_ious.append(best_iou)
                    per_class_counts[pred_item["category_id"]]["tp"] += 1
                    per_class_counts[pred_item["category_id"]]["matched_iou_sum"] += best_iou
                    per_class_counts[pred_item["category_id"]]["matched_iou_count"] += 1
                    match_status = "tp"
                else:
                    false_positives += 1
                    per_class_counts[pred_item["category_id"]]["fp"] += 1
                    match_status = "fp"

                prediction_rows.append(
                    {
                        "image_id": image_id,
                        "file_name": image_record["file_name"],
                        "class_name": pred_item["class_name"],
                        "category_id": pred_item["category_id"],
                        "score": pred_item["score"],
                        "best_iou": best_iou,
                        "match_status": match_status,
                    }
                )

            false_negatives = 0
            for gt_index, gt_item in enumerate(gt_items):
                if gt_index in gt_matched:
                    continue
                false_negatives += 1
                per_class_counts[gt_item["category_id"]]["fn"] += 1

            precision = _safe_divide(true_positives, true_positives + false_positives)
            recall = _safe_divide(true_positives, true_positives + false_negatives)
            f1 = _safe_divide(2 * precision * recall, precision + recall)
            mean_matched_iou = _safe_divide(sum(image_matched_ious), len(image_matched_ious))
            per_image_row = {
                "image_id": image_id,
                "file_name": image_record["file_name"],
                "ground_truth_count": len(gt_items),
                "prediction_count": len(pred_items),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": mean_matched_iou,
            }
            per_image_rows.append(per_image_row)
            overlay_payloads.append(
                {
                    "image_path": image_path,
                    "file_name": image_record["file_name"],
                    "gt_items": gt_items,
                    "pred_items": pred_items,
                    **per_image_row,
                }
            )

    coco_gt = COCO(str(annotation_path))
    coco_metrics = {
        "bbox": _run_coco_eval(coco_gt, bbox_results, "bbox"),
        "segm": _run_coco_eval(coco_gt, segm_results, "segm"),
    }

    per_class_rows = []
    for category in categories:
        counts = per_class_counts[category["id"]]
        precision = _safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        mean_matched_iou = _safe_divide(
            counts["matched_iou_sum"],
            counts["matched_iou_count"],
        )
        per_class_rows.append(
            {
                "category_id": category["id"],
                "class_name": category["name"],
                "true_positives": counts["tp"],
                "false_positives": counts["fp"],
                "false_negatives": counts["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": mean_matched_iou,
            }
        )

    per_image_rows.sort(key=_score_image)
    overlay_payloads.sort(key=_score_image)
    overlays_to_render = overlay_payloads[:max_overlay_images]
    for item in overlays_to_render:
        summary = (
            f"F1 {item['f1']:.2f} | P {item['precision']:.2f} | R {item['recall']:.2f} | "
            f"TP {item['true_positives']} FP {item['false_positives']} FN {item['false_negatives']}"
        )
        output_path = overlays_dir / f"{Path(item['file_name']).stem}_overlay.jpg"
        _render_overlay_image(
            item["image_path"],
            output_path,
            item["gt_items"],
            item["pred_items"],
            summary,
        )

    _write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    _write_csv(output_dir / "prediction_metrics.csv", prediction_rows)
    _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "split": split_dir.name,
        "image_count": len(per_image_rows),
        "threshold": threshold,
        "iou_threshold": iou_threshold,
        "per_class": per_class_rows,
        "coco_metrics": coco_metrics,
        "overall": {
            "precision": _safe_divide(
                sum(row["true_positives"] for row in per_image_rows),
                sum(row["true_positives"] + row["false_positives"] for row in per_image_rows),
            ),
            "recall": _safe_divide(
                sum(row["true_positives"] for row in per_image_rows),
                sum(row["true_positives"] + row["false_negatives"] for row in per_image_rows),
            ),
            "mean_f1": _safe_divide(sum(row["f1"] for row in per_image_rows), len(per_image_rows)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _write_evaluation_plots(
        output_dir=output_dir,
        per_class_rows=per_class_rows,
        per_image_rows=per_image_rows,
        prediction_rows=prediction_rows,
        coco_metrics=coco_metrics,
    )

    typer.echo(f"Evaluation complete. Artifacts written to {output_dir}")
    typer.echo(f"  - overlays: {overlays_dir}")
    typer.echo(f"  - plots: {output_dir / 'plots'}")
    typer.echo(f"  - per_image_metrics.csv")
    typer.echo(f"  - prediction_metrics.csv")
    typer.echo(f"  - per_class_metrics.csv")
    typer.echo(f"  - summary.json")


@app.command()
def analyze(dataset_path: Path):
    split_reports = []

    def percent(count: int, total: int) -> str:
        if total == 0:
            return "0.0%"
        return f"{(count / total) * 100:.1f}%"

    def resolve_annotation_files(path: Path) -> list[Path]:
        if path.is_file():
            return [path]

        direct_annotation = path / "_annotations.coco.json"
        if direct_annotation.exists():
            return [direct_annotation]

        split_names = ("train", "valid", "test")
        split_files = [
            path / split_name / "_annotations.coco.json"
            for split_name in split_names
            if (path / split_name / "_annotations.coco.json").exists()
        ]
        if split_files:
            return split_files

        return sorted(path.rglob("_annotations.coco.json"))

    def print_report(annotation_path: Path):
        dataset = orjson.loads(annotation_path.read_bytes())
        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])
        categories = dataset.get("categories", [])

        category_names = {category["id"]: category["name"] for category in categories}
        used_category_ids = {annotation["category_id"] for annotation in annotations}
        report_category_ids = sorted(
            category["id"]
            for category in categories
            if category["id"] in used_category_ids
            or category.get("supercategory") != "none"
        )

        annotation_counts = Counter(
            annotation["category_id"] for annotation in annotations
        )
        image_class_sets: dict[int, set[int]] = defaultdict(set)
        for annotation in annotations:
            image_class_sets[annotation["image_id"]].add(annotation["category_id"])

        image_ids = {image["id"] for image in images}
        images_with_annotations = len(image_class_sets)
        unlabeled_images = len(image_ids) - images_with_annotations

        image_presence_counts = Counter()
        combination_counts = Counter()

        for image in images:
            class_ids = frozenset(image_class_sets.get(image["id"], set()))
            if not class_ids:
                continue

            for class_id in class_ids:
                image_presence_counts[class_id] += 1

            combination_counts[class_ids] += 1

        label = annotation_path.parent.name
        typer.echo(f"\n[{label}] {annotation_path}")
        typer.echo(
            f"Images: {len(images)} | Annotated images: {images_with_annotations} | "
            f"Unlabeled images: {unlabeled_images} | Annotations: {len(annotations)}"
        )

        typer.echo("\nAnnotation counts by class:")
        for category_id in sorted(
            report_category_ids,
            key=lambda cid: (
                -annotation_counts[cid],
                category_names.get(cid, str(cid)),
            ),
        ):
            typer.echo(
                f"  - {category_names.get(category_id, str(category_id))}: "
                f"{annotation_counts[category_id]} ({percent(annotation_counts[category_id], len(annotations))})"
            )

        typer.echo("\nImages containing each class:")
        for category_id in sorted(
            report_category_ids,
            key=lambda cid: (
                -image_presence_counts[cid],
                category_names.get(cid, str(cid)),
            ),
        ):
            typer.echo(
                f"  - {category_names.get(category_id, str(category_id))}: "
                f"{image_presence_counts[category_id]} images "
                f"({percent(image_presence_counts[category_id], len(images))})"
            )

        typer.echo("\nImage class combinations:")
        for class_ids, count in combination_counts.most_common():
            class_names = sorted(
                category_names.get(class_id, str(class_id)) for class_id in class_ids
            )
            typer.echo(
                f"  - {', '.join(class_names)}: {count} images ({percent(count, len(images))})"
            )

        split_reports.append(
            {
                "label": label,
                "annotation_path": annotation_path,
                "category_names": category_names,
                "report_category_ids": report_category_ids,
                "annotation_counts": annotation_counts,
                "annotation_total": len(annotations),
            }
        )

    def print_split_comparison():
        if len(split_reports) < 2:
            return

        typer.echo("\nSplit class weight comparison:")

        all_category_ids = sorted(
            {
                category_id
                for report in split_reports
                for category_id in report["report_category_ids"]
            }
        )

        for category_id in all_category_ids:
            class_name = next(
                (
                    report["category_names"].get(category_id)
                    for report in split_reports
                    if category_id in report["category_names"]
                ),
                str(category_id),
            )
            weights = []
            for report in split_reports:
                total = report["annotation_total"]
                weight = (
                    report["annotation_counts"][category_id] / total if total else 0.0
                )
                weights.append((report["label"], weight))

            min_label, min_weight = min(weights, key=lambda item: item[1])
            max_label, max_weight = max(weights, key=lambda item: item[1])
            spread = max_weight - min_weight
            status = "OK" if spread <= 0.05 else "WARNING"

            typer.echo(
                f"  - {class_name}: "
                + ", ".join(f"{label}={weight * 100:.1f}%" for label, weight in weights)
                + f" | spread={spread * 100:.1f}pp | {status}"
            )
            if status == "WARNING":
                typer.echo(
                    f"    largest difference between {min_label} and {max_label}"
                )

    annotation_files = resolve_annotation_files(dataset_path)
    if not annotation_files:
        typer.echo(f"No COCO annotation files found under: {dataset_path}")
        raise typer.Exit(code=1)

    typer.echo(
        f"Analyzing {len(annotation_files)} annotation file(s) from {dataset_path}"
    )
    for annotation_path in annotation_files:
        print_report(annotation_path)
    print_split_comparison()


@app.command()
def wizard():
    import questionary
    import roboflow

    # fix torch multiprocessing sharing strategy
    import torch.multiprocessing as mp
    from questionary import Choice

    mp.set_sharing_strategy("file_system")

    roboflow.login()

    rf = roboflow.Roboflow()

    # List all projects for your workspace
    workspace = rf.workspace()
    projects = []
    with ThreadPoolExecutor() as executor:
        for project, versions in executor.map(
            fetch_project_info,
            [(workspace, project.split("/")[-1]) for project in workspace.projects()],
        ):
            projects.append((project, versions))

    projects.sort(key=lambda p: p[0].updated, reverse=True)

    project, versions = questionary.select(
        "Select your project",
        choices=[
            Choice(title=project.id, value=(project, versions))
            for project, versions in projects
        ],
    ).ask()

    version = questionary.select(
        "Select the dataset version",
        choices=[
            Choice(title=version.id.split("/")[-1], value=version)
            for version in versions
        ],
    ).ask()

    batch_size, grad_accum_steps = questionary.select(
        "Select the GPU VRAM",
        choices=[
            Choice(title="RTX  4080 (16GB)", value=(16, 1)),
            Choice(title="RTX  4090 (24GB)", value=(32, 1)),
            Choice(title="RTX  5090 (32GB)", value=(48, 1)),
            Choice(title="RTX A6000 (48GB)", value=(68, 1)),
        ],
    ).ask()

    dataset_path = f"./datasets/{version.id}"
    version.download(model_format="coco", location=dataset_path)

    from rfdetr import RFDETRSegNano

    model = RFDETRSegNano()

    model.train(
        dataset_dir=dataset_path,
        epochs=100,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=372,
        early_stopping=True,
        early_stopping_patience=3,
        progress_bar=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=False,
        num_queries=50,
        num_select=20,
        output_dir="runs",
    )


if __name__ == "__main__":
    app()
