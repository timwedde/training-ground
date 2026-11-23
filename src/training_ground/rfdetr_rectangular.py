import json
from pathlib import Path

DEFAULT_SHORT_SIDE = 360


def rectangular_input(
    dataset_path: Path, short_side: int, block_size: int
) -> tuple[int, tuple[int, int]]:
    from rfdetr.datasets.coco import _COCO_MAX_SIZE, RandomResize

    short_side = ((short_side + block_size - 1) // block_size) * block_size
    shapes = set()
    for split in ("train", "valid", "test"):
        path = dataset_path / split / "_annotations.coco.json"
        images = json.loads(path.read_bytes())["images"]
        if not images:
            raise ValueError(f"No images in the {split} split.")
        for image in images:
            height, width = RandomResize._get_size(
                image["height"], image["width"], short_side, _COCO_MAX_SIZE
            )
            shapes.add(
                tuple(
                    ((side + block_size - 1) // block_size) * block_size
                    for side in (height, width)
                )
            )
    if len(shapes) != 1:
        sizes = ", ".join(f"{w}×{h}" for h, w in sorted(shapes))
        raise ValueError(
            "Rectangular RF-DETR export requires one padded input shape across "
            f"train, validation, and test. Found {sizes}."
        )
    return short_side, shapes.pop()


def predict_rectangular(predictor, image_path, threshold, shape, short_side, means):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from rfdetr.datasets.coco import _COCO_MAX_SIZE, RandomResize
    from torchvision.transforms.functional import pil_to_tensor

    with Image.open(image_path) as image:
        width, height = image.size
        resized, _ = RandomResize([short_side], max_size=_COCO_MAX_SIZE)(
            image.convert("RGB")
        )
        rw, rh = resized.size
        pixels = pil_to_tensor(resized).float().div_(255)
    input_h, input_w = shape
    if rw > input_w or rh > input_h:
        raise ValueError(
            f"{image_path} resizes to {rw}×{rh}, outside the trained "
            f"rectangular input {input_w}×{input_h}."
        )
    # The training collator pads on the right and bottom with normalized zeros.
    canvas = torch.tensor(means, dtype=pixels.dtype)[:, None, None]
    canvas = canvas.expand(-1, input_h, input_w).clone()
    canvas[:, :rh, :rw] = pixels
    detections = predictor.predict(
        canvas, shape=shape, threshold=threshold, include_source_image=False
    )
    boxes = detections.xyxy
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, rw) * (width / rw)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, rh) * (height / rh)
    detections = detections[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    if detections.mask is not None:
        masks = np.empty((len(detections), height, width), dtype=bool)
        for i, mask in enumerate(detections.mask):
            masks[i] = cv2.resize(
                mask[:rh, :rw].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST_EXACT,
            ).astype(bool)
        detections.mask = masks
    return detections
