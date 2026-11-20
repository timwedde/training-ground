from collections import defaultdict
from pathlib import Path

from .coco import decode_segmentation
from .geometry import bbox_iou, mask_iou, xywh_to_xyxy
from .predictors import Predictor
from .training_backends import SEGMENTATION_TASK, Task


def build_label_to_category_id(
    model: Predictor,
    categories: list[dict],
) -> dict[int, int]:
    category_name_to_id = {}
    for category in categories:
        name = category["name"]
        if name in category_name_to_id:
            raise ValueError(f"COCO class name {name!r} is ambiguous.")
        category_name_to_id[name] = category["id"]
    model_names = model.class_id_to_name
    if len(set(model_names.values())) != len(model_names):
        raise ValueError("Model class names are ambiguous.")
    return {
        class_id: category_name_to_id[name]
        for class_id, name in model_names.items()
        if name in category_name_to_id
    }


def build_ground_truth(image_record, annotations, category_names, task: Task):
    height, width = int(image_record["height"]), int(image_record["width"])
    return [
        {
            "category_id": item["category_id"],
            "class_name": category_names[item["category_id"]],
            "bbox": xywh_to_xyxy(item["bbox"]),
            "mask": (
                decode_segmentation(item.get("segmentation"), height, width)
                if task == SEGMENTATION_TASK
                else None
            ),
        }
        for item in annotations
    ]


def match_predictions(predictions, ground_truth, task: Task, iou_threshold: float):
    ground_truth_by_class = defaultdict(list)
    for index, item in enumerate(ground_truth):
        ground_truth_by_class[item["category_id"]].append((index, item))

    matched = set()
    results = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        best_index = None
        best_iou = 0.0
        for index, item in ground_truth_by_class[prediction["category_id"]]:
            if index in matched:
                continue
            if (
                task == SEGMENTATION_TASK
                and prediction["mask"] is not None
                and item["mask"] is not None
            ):
                overlap = mask_iou(prediction["mask"], item["mask"])
            else:
                overlap = bbox_iou(prediction["bbox"], item["bbox"])
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None and best_iou >= iou_threshold:
            matched.add(best_index)
        else:
            best_index = None
        results.append((prediction, best_index, best_iou))
    return results, matched


def image_relative_path(image_record: dict) -> Path:
    relative_path = Path(image_record["file_name"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Invalid COCO image path: {relative_path}")
    return relative_path
