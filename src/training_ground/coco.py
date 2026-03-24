from collections import defaultdict
from pathlib import Path

# Initialize faster-coco-eval as drop-in replacement for pycocotools
import faster_coco_eval
import numpy as np
import orjson

faster_coco_eval.init_as_pycocotools()
from pycocotools import mask as coco_mask  # noqa: E402


def load_coco_annotations(
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
    categories = sorted(
        dataset.get("categories", []), key=lambda category: category["id"]
    )

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


def decode_segmentation(segmentation, height: int, width: int):
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


def encode_binary_mask(mask):
    encoded = coco_mask.encode(np.asfortranarray(mask.astype("uint8")))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        encoded["counts"] = counts.decode("utf-8")
    return encoded
