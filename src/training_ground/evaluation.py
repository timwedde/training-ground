import csv
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from urllib.parse import quote

import numpy as np
import typer
from faster_coco_eval import COCO

from .coco import encode_binary_mask, load_coco_annotations
from .coco_eval import run_coco_eval
from .evaluation_plots import write_evaluation_plots
from .geometry import xyxy_to_xywh
from .matching import (
    build_ground_truth,
    build_label_to_category_id,
    image_relative_path,
    match_predictions,
)
from .overlays import render_fn_overlay, render_fp_overlay, render_overlay
from .predictors import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    Predictor,
    YOLO26PredictorAdapter,
    build_pred_items,
    create_model,
)
from .settings_sweep import CONFIDENCE_FLOOR, SETTINGS_REPORT, write_settings_report
from .tiling import DEFAULT_MERGE_IOU
from .training_backends import (
    BOX_TASK,
    RFDETR_BACKEND,
    SEGMENTATION_TASK,
    YOLO26_BACKEND,
    Backend,
    Task,
)

COCO_CONFIDENCE_FLOOR = CONFIDENCE_FLOOR


def resolve_dataset_split(dataset_path: Path, split: str) -> tuple[Path, Path]:
    split_dir_map = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    split_dir_name = split_dir_map.get(split.lower())
    if split_dir_name is None:
        valid_splits = ", ".join(sorted(split_dir_map))
        raise typer.BadParameter(
            f"Unsupported split '{split}'. Expected one of: {valid_splits}"
        )

    split_dir = dataset_path / split_dir_name
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise typer.BadParameter(
            f"Missing annotation file for split '{split}': {annotation_path}"
        )
    return split_dir, annotation_path


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def score_image(row: dict) -> tuple:
    return (
        row["f1"],
        -row["false_negatives"],
        -row["false_positives"],
        row["file_name"],
    )


def write_csv(output_path: Path, rows: list[dict]):
    if not rows:
        output_path.write_text("")
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(
    dataset_path: Path,
    checkpoint_path: Path,
    split: str,
    threshold: float | None,
    iou_threshold: float,
    model: Predictor | None = None,
    backend: Backend = RFDETR_BACKEND,
    output_dir: Path | None = None,
    model_size: str | None = None,
    task: Task = SEGMENTATION_TASK,
    settings_sweep: bool = True,
    tiled: bool = False,
) -> Path:
    if tiled and (backend != YOLO26_BACKEND or task != BOX_TASK):
        raise typer.BadParameter("Tiling is available only for YOLO26 box detection.")

    split_dir, annotation_path = resolve_dataset_split(dataset_path, split)
    images_by_id, annotations_by_image, categories = load_coco_annotations(
        annotation_path
    )

    if output_dir is None:
        output_dir = (
            checkpoint_path.parent
            / f"{checkpoint_path.stem}_{split_dir.name}_evaluation"
        )
        try:
            output_dir.mkdir(parents=True)
        except FileExistsError:
            output_dir = Path(
                mkdtemp(prefix=f"{output_dir.name}-", dir=output_dir.parent)
            )
    else:
        output_dir.mkdir(parents=True)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Create directories for false positive and false negative overlays per class
    false_positives_dir = output_dir / "false_positives"
    false_negatives_dir = output_dir / "false_negatives"
    false_positives_dir.mkdir(parents=True, exist_ok=True)
    false_negatives_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = create_model(
            checkpoint_path,
            backend,
            model_size=model_size,
            task=task,
            tiled=tiled,
        )

    camera_profile = getattr(model, "camera_profile", False)
    operating_threshold = threshold
    camera_thresholds = None
    tiled_model = (
        model if isinstance(model, YOLO26PredictorAdapter) and model.tiled else None
    )
    if tiled_model is not None:
        report = write_settings_report(
            output_dir, dataset_path, model, task, iou_threshold, settings_sweep
        )
        recommended = report["sweep"].get("recommended")
        merge_iou = recommended["nms_iou"] if recommended else DEFAULT_MERGE_IOU
        if operating_threshold is None and recommended:
            operating_threshold = recommended["confidence"]
    if operating_threshold is None:
        if camera_profile:
            from .camera_inference import CLASS_THRESHOLDS

            camera_thresholds = CLASS_THRESHOLDS
        else:
            operating_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    inference_threshold = (
        COCO_CONFIDENCE_FLOOR
        if operating_threshold is None
        else min(COCO_CONFIDENCE_FLOOR, operating_threshold)
    )

    category_names = {category["id"]: category["name"] for category in categories}
    label_to_category_id = build_label_to_category_id(model, categories)

    image_records = [images_by_id[image_id] for image_id in sorted(images_by_id)]

    bbox_results: list[dict] = []
    segm_results: list[dict] = []
    per_image_rows: list[dict] = []
    prediction_rows: list[dict] = []
    per_class_counts = {
        category["id"]: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_iou_sum": 0.0,
            "matched_iou_count": 0,
        }
        for category in categories
    }

    typer.echo(f"Evaluating {len(image_records)} images from {split_dir}")

    with typer.progressbar(image_records, label="Running inference") as progress:
        for image_record in progress:
            image_id = image_record["id"]
            relative_path = image_relative_path(image_record)
            image_path = split_dir / relative_path
            gt_annotations = annotations_by_image.get(image_id, [])
            # Lower the inference cutoff once, then apply the two report thresholds
            # independently to the same detections.
            tile_metrics = {}
            if tiled_model is None:
                detections = model.predict(
                    str(image_path), threshold=inference_threshold
                )
            else:
                start = perf_counter()
                tiles = tiled_model.predict_tiles(str(image_path), inference_threshold)
                detections = tiled_model.merge_tiles(tiles, merge_iou)
                tile_metrics = {
                    "tile_count": len(tiles),
                    "inference_ms": (perf_counter() - start) * 1000,
                }
            scores = detections.confidence
            cutoff = operating_threshold
            if camera_thresholds is not None:
                cutoff = np.asarray(camera_thresholds, dtype=scores.dtype)[
                    detections.class_id.astype(int, copy=False)
                ]
            at_operating_point = scores >= cutoff if camera_profile else scores > cutoff
            all_items = build_pred_items(
                detections, category_names, label_to_category_id
            )
            pred_items = []
            for item, for_coco, for_operating_metrics in zip(
                all_items,
                scores > COCO_CONFIDENCE_FLOOR,
                at_operating_point,
                strict=True,
            ):
                if for_operating_metrics:
                    pred_items.append(item)
                if not for_coco:
                    continue
                bbox_results.append(
                    {
                        "image_id": image_id,
                        "category_id": item["category_id"],
                        "bbox": xyxy_to_xywh(item["bbox"]),
                        "score": item["score"],
                    }
                )
                if task == SEGMENTATION_TASK and item["mask"] is not None:
                    segm_results.append(
                        {
                            "image_id": image_id,
                            "category_id": item["category_id"],
                            "segmentation": encode_binary_mask(item["mask"]),
                            "score": item["score"],
                        }
                    )

            gt_items = build_ground_truth(
                image_record, gt_annotations, category_names, task
            )
            matches, gt_matched = match_predictions(
                pred_items, gt_items, task, iou_threshold
            )
            image_matched_ious = []
            true_positives = false_positives = 0
            fp_items_per_class = defaultdict(list)

            for pred_item, best_match_index, best_iou in matches:
                counts = per_class_counts[pred_item["category_id"]]
                if best_match_index is not None:
                    true_positives += 1
                    image_matched_ious.append(best_iou)
                    counts["tp"] += 1
                    counts["matched_iou_sum"] += best_iou
                    counts["matched_iou_count"] += 1
                    match_status = "tp"
                else:
                    false_positives += 1
                    counts["fp"] += 1
                    match_status = "fp"
                    # Track this FP item for the class
                    fp_items_per_class[pred_item["category_id"]].append(pred_item)

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
            fn_items_per_class = defaultdict(list)
            for gt_index, gt_item in enumerate(gt_items):
                if gt_index not in gt_matched:
                    false_negatives += 1
                    per_class_counts[gt_item["category_id"]]["fn"] += 1
                    # Track this FN item for the class
                    fn_items_per_class[gt_item["category_id"]].append(gt_item)

            precision, recall, f1 = compute_metrics(
                true_positives, false_positives, false_negatives
            )
            mean_matched_iou = safe_divide(
                sum(image_matched_ious), len(image_matched_ious)
            )
            per_image_rows.append(
                {
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
                    **tile_metrics,
                }
            )
            output_path = overlays_dir / relative_path.with_name(
                f"{relative_path.name}_overlay.jpg"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_overlay(
                image_path,
                output_path,
                gt_items,
                pred_items,
                f"F1 {f1:.2f} | P {precision:.2f} | R {recall:.2f} | "
                f"TP {true_positives} FP {false_positives} FN {false_negatives}",
            )

            for folder, groups, suffix, render in (
                (false_positives_dir, fp_items_per_class, "fp", render_fp_overlay),
                (false_negatives_dir, fn_items_per_class, "fn", render_fn_overlay),
            ):
                for category_id, items in groups.items():
                    class_name = category_names[category_id]
                    class_folder = quote(class_name, safe="")
                    if class_folder in {"", ".", ".."}:
                        class_folder = f"class_{category_id}_{class_folder}"
                    target = (
                        folder
                        / class_folder
                        / relative_path.with_name(f"{relative_path.name}_{suffix}.jpg")
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    render(image_path, target, items, class_name)

    coco_gt = COCO(str(annotation_path))
    coco_metrics = {"bbox": run_coco_eval(coco_gt, bbox_results, "bbox")}
    if task == SEGMENTATION_TASK:
        coco_metrics["segm"] = run_coco_eval(coco_gt, segm_results, "segm")

    per_class_rows = []
    for category in categories:
        counts = per_class_counts[category["id"]]
        precision, recall, f1 = compute_metrics(
            counts["tp"], counts["fp"], counts["fn"]
        )
        mean_matched_iou = safe_divide(
            counts["matched_iou_sum"], counts["matched_iou_count"]
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

    per_image_rows.sort(key=score_image)

    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "prediction_metrics.csv", prediction_rows)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    total_tp = sum(row["true_positives"] for row in per_image_rows)
    total_fp = sum(row["false_positives"] for row in per_image_rows)
    total_fn = sum(row["false_negatives"] for row in per_image_rows)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "backend": backend,
        "task": task,
        "split": split_dir.name,
        "image_count": len(per_image_rows),
        "threshold": operating_threshold,
        "camera_profile": camera_profile,
        "coco_confidence_floor": COCO_CONFIDENCE_FLOOR,
        "iou_threshold": iou_threshold,
        "per_class": per_class_rows,
        "coco_metrics": coco_metrics,
        "settings_report": SETTINGS_REPORT,
        "overall": {
            "precision": safe_divide(total_tp, total_tp + total_fp),
            "recall": safe_divide(total_tp, total_tp + total_fn),
            "mean_f1": safe_divide(
                sum(row["f1"] for row in per_image_rows), len(per_image_rows)
            ),
        },
    }
    if camera_thresholds is not None:
        summary["class_thresholds"] = {
            name: camera_thresholds[class_id]
            for class_id, name in model.class_id_to_name.items()
        }
    if tiled_model is not None:
        summary["tiling"] = {
            **report["model"]["tiling"],
            "merge_iou": merge_iou,
            "mean_tiles_per_image": safe_divide(
                sum(row["tile_count"] for row in per_image_rows), len(per_image_rows)
            ),
            "mean_inference_ms_per_image": safe_divide(
                sum(row["inference_ms"] for row in per_image_rows), len(per_image_rows)
            ),
        }
    write_evaluation_plots(
        output_dir=output_dir,
        per_class_rows=per_class_rows,
        per_image_rows=per_image_rows,
        prediction_rows=prediction_rows,
        coco_metrics=coco_metrics,
        coco_confidence_floor=COCO_CONFIDENCE_FLOOR,
    )
    if tiled_model is None:
        write_settings_report(
            output_dir, dataset_path, model, task, iou_threshold, settings_sweep
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return output_dir
