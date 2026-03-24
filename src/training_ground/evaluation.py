import csv
import json
from pathlib import Path

import faster_coco_eval

faster_coco_eval.init_as_pycocotools()
import numpy as np
import torch
import typer
from PIL import Image, ImageDraw
from pycocotools.coco import COCO  # noqa: E402

from .coco import decode_segmentation, encode_binary_mask, load_coco_annotations
from .coco_eval import run_coco_eval
from .evaluation_plots import write_evaluation_plots
from .geometry import bbox_iou, mask_iou, xywh_to_xyxy, xyxy_to_xywh

OVERLAY_ALPHA = 0.28
OVERLAY_QUALITY = 92


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


def render_overlay(
    image_path: Path,
    output_path: Path,
    gt_items: list,
    pred_items: list,
    summary_text: str,
):
    colors = {"gt": (46, 204, 113), "pred": (231, 76, 60)}
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    def blend_mask(mask, color, alpha=OVERLAY_ALPHA):
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
        draw.text(
            (box[0] + 4, max(4, box[1] - 16)),
            f"GT {item['class_name']}",
            fill=colors["gt"],
        )

    for item in pred_items:
        box = item["bbox"]
        label = f"P {item['class_name']} {item['score']:.2f}"
        draw.rectangle(box, outline=colors["pred"], width=3)
        draw.text((box[0] + 4, box[1] + 4), label, fill=colors["pred"])

    draw.rectangle((0, 0, annotated.width, 24), fill=(0, 0, 0))
    draw.text((8, 5), summary_text, fill=(255, 255, 255))
    annotated.save(output_path, quality=OVERLAY_QUALITY)


def render_fp_overlay(
    image_path: Path,
    output_path: Path,
    fp_items: list,
    class_name: str,
):
    """Render overlay showing only false positive predictions for a specific class."""
    color = (231, 76, 60)  # Red for false positives
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    def blend_mask(mask, alpha=OVERLAY_ALPHA):
        if mask is None:
            return
        mask_indices = mask.astype(bool)
        if not mask_indices.any():
            return
        image_array[mask_indices] = (
            image_array[mask_indices] * (1 - alpha) + np.array(color) * alpha
        ).astype("uint8")

    for item in fp_items:
        blend_mask(item.get("mask"))

    annotated = Image.fromarray(image_array)
    draw = ImageDraw.Draw(annotated)

    for item in fp_items:
        box = item["bbox"]
        label = f"FP {item['class_name']} {item['score']:.2f}"
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 4), label, fill=color)

    summary = f"False Positives for '{class_name}' - Count: {len(fp_items)}"
    draw.rectangle((0, 0, annotated.width, 24), fill=(0, 0, 0))
    draw.text((8, 5), summary, fill=(255, 255, 255))
    annotated.save(output_path, quality=OVERLAY_QUALITY)


def render_fn_overlay(
    image_path: Path,
    output_path: Path,
    fn_items: list,
    class_name: str,
):
    """Render overlay showing only false negative ground truth for a specific class."""
    color = (46, 204, 113)  # Green for false negatives (missed GT)
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    def blend_mask(mask, alpha=OVERLAY_ALPHA):
        if mask is None:
            return
        mask_indices = mask.astype(bool)
        if not mask_indices.any():
            return
        image_array[mask_indices] = (
            image_array[mask_indices] * (1 - alpha) + np.array(color) * alpha
        ).astype("uint8")

    for item in fn_items:
        blend_mask(item.get("mask"))

    annotated = Image.fromarray(image_array)
    draw = ImageDraw.Draw(annotated)

    for item in fn_items:
        box = item["bbox"]
        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0] + 4, max(4, box[1] - 16)),
            f"FN {item['class_name']}",
            fill=color,
        )

    summary = f"False Negatives for '{class_name}' - Count: {len(fn_items)}"
    draw.rectangle((0, 0, annotated.width, 24), fill=(0, 0, 0))
    draw.text((8, 5), summary, fill=(255, 255, 255))
    annotated.save(output_path, quality=OVERLAY_QUALITY)


def write_csv(output_path: Path, rows: list[dict]):
    if not rows:
        output_path.write_text("")
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(
    checkpoint_path: Path,
    dataset_path: Path,
    split: str,
    output_dir: Path | None,
    model_type: str,
    resolution: int,
    threshold: float,
    iou_threshold: float,
    max_overlay_images: int,
    limit: int | None,
) -> Path:
    from rfdetr.detr import (
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
        raise typer.BadParameter(
            f"Unsupported model type '{model_type}'. Expected one of: {valid_models}"
        )

    split_dir, annotation_path = resolve_dataset_split(dataset_path, split)
    images_by_id, annotations_by_image, categories, label_to_category_id, _ = (
        load_coco_annotations(annotation_path)
    )

    if output_dir is None:
        output_dir = (
            checkpoint_path.parent
            / f"{checkpoint_path.stem}_{split_dir.name}_evaluation"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Create directories for false positive and false negative overlays per class
    false_positives_dir = output_dir / "false_positives"
    false_negatives_dir = output_dir / "false_negatives"
    false_positives_dir.mkdir(parents=True, exist_ok=True)
    false_negatives_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_head_size = int(checkpoint["model"]["class_embed.bias"].shape[0])
    dataset_label_count = len(categories)
    if checkpoint_head_size > dataset_label_count:
        typer.echo(
            f"Checkpoint has more class slots than dataset; restricting to first {dataset_label_count}."
        )

    model = model_class(pretrain_weights=str(checkpoint_path), resolution=resolution)
    category_names = {category["id"]: category["name"] for category in categories}

    image_records = [images_by_id[image_id] for image_id in sorted(images_by_id)]
    if limit is not None:
        image_records = image_records[:limit]

    bbox_results: list[dict] = []
    segm_results: list[dict] = []
    per_image_rows: list[dict] = []
    prediction_rows: list[dict] = []
    overlay_payloads: list[dict] = []
    per_class_counts = {
        category["id"]: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_iou_sum": 0.0,
            "matched_iou_count": 0,
            "fp_payloads": [],  # Track false positive images per class
            "fn_payloads": [],  # Track false negative images per class
        }
        for category in categories
    }

    typer.echo(f"Evaluating {len(image_records)} images from {split_dir}")

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
                bbox_xyxy = [float(v) for v in detections.xyxy[index].tolist()]
                score = float(detections.confidence[index])
                mask = (
                    detections.mask[index].astype(bool)
                    if detections.mask is not None
                    else None
                )
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
                        "bbox": xyxy_to_xywh(bbox_xyxy),
                        "score": score,
                    }
                )
                if mask is not None:
                    segm_results.append(
                        {
                            "image_id": image_id,
                            "category_id": mapped_category_id,
                            "segmentation": encode_binary_mask(mask),
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
                        "bbox": xywh_to_xyxy(annotation["bbox"]),
                        "mask": decode_segmentation(
                            annotation.get("segmentation"), image_height, image_width
                        ),
                    }
                )

            gt_matched: set[int] = set()
            image_matched_ious: list[float] = []
            true_positives = 0
            false_positives = 0

            # Track FP items per class for this image
            fp_items_per_class: dict[int, list] = {
                cat_id: [] for cat_id in per_class_counts
            }

            for pred_item in sorted(
                pred_items, key=lambda item: item["score"], reverse=True
            ):
                best_match_index = None
                best_iou = 0.0
                for gt_index, gt_item in enumerate(gt_items):
                    if (
                        gt_index in gt_matched
                        or gt_item["category_id"] != pred_item["category_id"]
                    ):
                        continue

                    if pred_item["mask"] is not None and gt_item["mask"] is not None:
                        overlap = mask_iou(pred_item["mask"], gt_item["mask"])
                    else:
                        overlap = bbox_iou(pred_item["bbox"], gt_item["bbox"])

                    if overlap > best_iou:
                        best_iou = overlap
                        best_match_index = gt_index

                counts = per_class_counts[pred_item["category_id"]]
                if best_match_index is not None and best_iou >= iou_threshold:
                    gt_matched.add(best_match_index)
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
            # Track FN items per class for this image
            fn_items_per_class: dict[int, list] = {
                cat_id: [] for cat_id in per_class_counts
            }
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
                }
            )
            overlay_payloads.append(
                {
                    "image_path": image_path,
                    "file_name": image_record["file_name"],
                    "gt_items": gt_items,
                    "pred_items": pred_items,
                    **per_image_rows[-1],
                }
            )

            # Track FP and FN payloads per class for generating class-specific overlays
            for category_id, fp_items in fp_items_per_class.items():
                if fp_items:
                    per_class_counts[category_id]["fp_payloads"].append(
                        {
                            "image_path": image_path,
                            "file_name": image_record["file_name"],
                            "fp_items": fp_items,
                        }
                    )
            for category_id, fn_items in fn_items_per_class.items():
                if fn_items:
                    per_class_counts[category_id]["fn_payloads"].append(
                        {
                            "image_path": image_path,
                            "file_name": image_record["file_name"],
                            "fn_items": fn_items,
                        }
                    )

    coco_gt = COCO(str(annotation_path))
    coco_metrics = {
        "bbox": run_coco_eval(coco_gt, bbox_results, "bbox"),
        "segm": run_coco_eval(coco_gt, segm_results, "segm"),
    }

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
    overlay_payloads.sort(key=score_image)
    for item in overlay_payloads[:max_overlay_images]:
        summary = (
            f"F1 {item['f1']:.2f} | P {item['precision']:.2f} | R {item['recall']:.2f} | "
            f"TP {item['true_positives']} FP {item['false_positives']} FN {item['false_negatives']}"
        )
        render_overlay(
            item["image_path"],
            overlays_dir / f"{Path(item['file_name']).stem}_overlay.jpg",
            item["gt_items"],
            item["pred_items"],
            summary,
        )

    # Generate false positive overlays per class
    typer.echo(f"Generating false positive overlays per class...")
    for category in categories:
        category_id = category["id"]
        class_name = category["name"]
        counts = per_class_counts[category_id]

        if not counts["fp_payloads"]:
            continue

        # Create class-specific subdirectory
        class_fp_dir = false_positives_dir / class_name.replace(" ", "_")
        class_fp_dir.mkdir(parents=True, exist_ok=True)

        # Save up to max_overlay_images per class, sorted by number of FPs (descending)
        fp_payloads_sorted = sorted(
            counts["fp_payloads"], key=lambda x: len(x["fp_items"]), reverse=True
        )[:max_overlay_images]

        for payload in fp_payloads_sorted:
            output_filename = f"{Path(payload['file_name']).stem}_fp.jpg"
            render_fp_overlay(
                payload["image_path"],
                class_fp_dir / output_filename,
                payload["fp_items"],
                class_name,
            )

    # Generate false negative overlays per class
    typer.echo(f"Generating false negative overlays per class...")
    for category in categories:
        category_id = category["id"]
        class_name = category["name"]
        counts = per_class_counts[category_id]

        if not counts["fn_payloads"]:
            continue

        # Create class-specific subdirectory
        class_fn_dir = false_negatives_dir / class_name.replace(" ", "_")
        class_fn_dir.mkdir(parents=True, exist_ok=True)

        # Save up to max_overlay_images per class, sorted by number of FNs (descending)
        fn_payloads_sorted = sorted(
            counts["fn_payloads"], key=lambda x: len(x["fn_items"]), reverse=True
        )[:max_overlay_images]

        for payload in fn_payloads_sorted:
            output_filename = f"{Path(payload['file_name']).stem}_fn.jpg"
            render_fn_overlay(
                payload["image_path"],
                class_fn_dir / output_filename,
                payload["fn_items"],
                class_name,
            )

    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "prediction_metrics.csv", prediction_rows)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    total_tp = sum(row["true_positives"] for row in per_image_rows)
    total_fp = sum(row["false_positives"] for row in per_image_rows)
    total_fn = sum(row["false_negatives"] for row in per_image_rows)
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
            "precision": safe_divide(total_tp, total_tp + total_fp),
            "recall": safe_divide(total_tp, total_tp + total_fn),
            "mean_f1": safe_divide(
                sum(row["f1"] for row in per_image_rows), len(per_image_rows)
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_evaluation_plots(
        output_dir=output_dir,
        per_class_rows=per_class_rows,
        per_image_rows=per_image_rows,
        prediction_rows=prediction_rows,
        coco_metrics=coco_metrics,
    )

    return output_dir
