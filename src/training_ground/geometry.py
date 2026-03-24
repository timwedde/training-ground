import numpy as np


def mask_iou(mask_a, mask_b) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
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


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]
