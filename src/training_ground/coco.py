from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson
from faster_coco_eval import mask as coco_mask


def load_coco_annotations(
    annotation_path: Path,
) -> tuple[
    dict[int, dict],
    dict[int, list[dict]],
    list[dict],
]:
    dataset = orjson.loads(annotation_path.read_bytes())
    images = sorted(dataset.get("images", []), key=lambda image: image["id"])
    annotations = dataset.get("annotations", [])
    categories = sorted(
        dataset.get("categories", []), key=lambda category: category["id"]
    )

    images_by_id = {image["id"]: image for image in images}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    return images_by_id, annotations_by_image, categories


def decode_segmentation(segmentation, height: int, width: int):
    if not segmentation:
        return np.zeros((height, width), dtype=bool)

    if isinstance(segmentation, dict):
        encoded = dict(segmentation)
        counts = encoded.get("counts")
        if isinstance(counts, str):
            encoded["counts"] = counts.encode("utf-8")
        elif isinstance(counts, list):
            encoded = coco_mask.frPyObjects(encoded, height, width)
        decoded = coco_mask.decode(encoded)
    else:
        encoded = coco_mask.frPyObjects(segmentation, height, width)
        decoded = coco_mask.decode(encoded)

    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def encode_binary_mask(mask):
    encoded = coco_mask.encode(np.asfortranarray(mask.astype("uint8")))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        encoded["counts"] = counts.decode("utf-8")
    return encoded
