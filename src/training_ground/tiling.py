import cv2
import numpy as np

from .training_backends import YOLO26_IMAGE_SIZE

TILE_BATCH_SIZE = 8
TILE_PADDING_VALUE = 114
DEFAULT_MERGE_IOU = 0.5


def tile_windows(height: int, width: int) -> list[tuple[int, int, int, int]]:
    # Use the same grid as the Ultralytics training dataset.
    from ultralytics.data.utils import tile_windows as training_windows

    return training_windows((height, width), YOLO26_IMAGE_SIZE)


def pad_to_tile(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.copyMakeBorder(
        image,
        0,
        max(0, YOLO26_IMAGE_SIZE - height),
        0,
        max(0, YOLO26_IMAGE_SIZE - width),
        cv2.BORDER_CONSTANT,
        value=(TILE_PADDING_VALUE,) * 3,
    )


def restore_boxes(boxes: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = window
    restored = boxes.astype(np.float32, copy=True)
    restored[:, :4] = np.clip(
        restored[:, :4], 0, [x2 - x1, y2 - y1, x2 - x1, y2 - y1]
    ) + [x1, y1, x1, y1]
    valid = (restored[:, 2:4] > restored[:, :2]).all(axis=1)
    return restored[valid]


def merge_tile_boxes(tiles: list[np.ndarray], iou_threshold: float) -> np.ndarray:
    """Greedy NMS between tiles, with score ties resolved in tile order."""
    boxes = np.concatenate(tiles)
    tile_ids = np.repeat(np.arange(len(tiles)), [len(tile) for tile in tiles])
    areas = np.prod(boxes[:, 2:4] - boxes[:, :2], axis=1)
    keep = []
    for class_id in np.unique(boxes[:, 5]):
        indices = np.flatnonzero(boxes[:, 5] == class_id)
        pending = indices[np.argsort(-boxes[indices, 4], kind="stable")]
        while pending.size:
            best, pending = pending[0], pending[1:]
            keep.append(best)
            intersection = np.prod(
                np.maximum(
                    np.minimum(boxes[best, 2:4], boxes[pending, 2:4])
                    - np.maximum(boxes[best, :2], boxes[pending, :2]),
                    0,
                ),
                axis=1,
            )
            union = areas[best] + areas[pending] - intersection
            suppress = (tile_ids[pending] != tile_ids[best]) & (
                intersection > iou_threshold * union
            )
            pending = pending[~suppress]

    keep = np.asarray(sorted(keep), dtype=np.intp)
    return boxes[keep[np.argsort(-boxes[keep, 4], kind="stable")]]
