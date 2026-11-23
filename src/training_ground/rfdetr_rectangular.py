import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

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


def _rectangular_probe_batch(
    micro_batch_size,
    resolution,
    device,
    num_classes,
    segmentation_head=False,
    max_targets_per_image=1,
    num_channels=3,
    *,
    shape,
):
    import torch
    from rfdetr.utilities.tensors import NestedTensor

    height, width = shape
    samples = NestedTensor(
        torch.randn(micro_batch_size, num_channels, height, width, device=device),
        torch.zeros(micro_batch_size, height, width, dtype=torch.bool, device=device),
    )
    count = max(1, max_targets_per_image)
    size = torch.tensor([height, width], device=device)
    targets = []
    for index in range(micro_batch_size):
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device).expand(count, 4),
            "labels": torch.full((count,), int(num_classes > 1), device=device),
            "image_id": torch.tensor(index, device=device),
            "orig_size": size,
            "size": size,
            "iscrowd": torch.zeros(count, dtype=torch.int64, device=device),
            "area": torch.full((count,), 0.04, device=device),
        }
        if segmentation_head:
            target["masks"] = torch.zeros(count, height, width, dtype=torch.bool, device=device)
        targets.append(target)
    return samples, targets


def probe_rectangular_batch_size(model, shape, **train_options) -> tuple[int, int]:
    from rfdetr.detr import _prepare_run_config
    from rfdetr.training import auto_batch

    # RF-DETR 1.12.1 only builds square probe tensors. Replace the batch factory
    # during sizing so RF-DETR still owns the loss, optimizer, and OOM search.
    probe_batch = partial(_rectangular_probe_batch, shape=shape)
    weights = model.model.model
    original_device = next(weights.parameters()).device
    try:
        with patch.object(auto_batch, "_make_synthetic_batch", probe_batch):
            config, _, _ = _prepare_run_config(model, batch_size="auto", **train_options)
    finally:
        # train() builds a new model. Release the probe's copy before it starts.
        weights.to(original_device)
    return config.batch_size, config.grad_accum_steps


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
