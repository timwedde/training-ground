import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

from .coco import load_coco_annotations

DEFAULT_SHORT_SIDE = 360
VRAM_BUDGET_FRACTION = 0.90


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
    target_counts,
):
    import torch
    from rfdetr.utilities.tensors import NestedTensor

    height, width = shape
    samples = NestedTensor(
        torch.randn(micro_batch_size, num_channels, height, width, device=device),
        torch.zeros(micro_batch_size, height, width, dtype=torch.bool, device=device),
    )
    size = torch.tensor([height, width], device=device)
    targets = []
    for index, count in enumerate(target_counts):
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


def _resolve_auto_batch_with_vram_budget(
    resolve, shape, object_counts, model_context, model_config, train_config, devices
):
    import torch
    from rfdetr.training import auto_batch

    device = model_context.device
    model = model_context.model
    free, _ = torch.cuda.mem_get_info(device)
    available = free + torch.cuda.memory_reserved(device)
    budget = int(available * VRAM_BUDGET_FRACTION)
    gradient_bytes = sum(p.nbytes for p in model.parameters() if p.requires_grad)
    # The probe's shadow optimizer already holds one copy of every trainable
    # parameter. Count that copy toward EMA and reserve the rest separately.
    ema_bytes = 0
    if train_config.use_ema:
        ema_bytes = sum(p.nbytes for p in model.parameters() if not p.requires_grad)
        ema_bytes += sum(b.nbytes for b in model.buffers())
    peaks = {}
    probe_step = auto_batch._probe_step

    def fits_budget(model, criterion, micro_batch_size, *args, **kwargs):
        def measure(counts):
            # Exercise the mask loss even when all sampled images are empty.
            if not any(counts):
                counts = [*counts[:-1], 1]
            factory = partial(_rectangular_probe_batch, shape=shape, target_counts=counts)
            torch.cuda.reset_peak_memory_stats(device)
            with patch.object(auto_batch, "_make_synthetic_batch", factory):
                if not probe_step(model, criterion, micro_batch_size, *args, **kwargs):
                    return None
            return torch.cuda.max_memory_allocated(device) + ema_bytes + gradient_bytes

        # Sample evenly spaced quantiles to make the representative mix repeatable.
        typical = [
            object_counts[(2 * i + 1) * len(object_counts) // (2 * micro_batch_size)]
            for i in range(micro_batch_size)
        ]
        peak = measure(typical)
        if peak is None or (micro_batch_size > 1 and peak > budget):
            return False
        dense = object_counts[-micro_batch_size:]
        dense_peak = peak if dense == typical else measure(dense)
        if dense_peak is None or (micro_batch_size > 1 and dense_peak > budget):
            return False
        peaks[micro_batch_size] = (peak, dense_peak)
        return True

    with patch.object(auto_batch, "_probe_step", fits_budget):
        result = resolve(
            model_context, model_config, train_config,
            safety_margin=1.0, max_micro_batch=min(128, len(object_counts)), devices=devices,
        )
    peak, dense_peak = peaks[result.safe_micro_batch]
    auto_batch.logger.info(
        "[auto-batch] rectangular batch %s: representative peak %.2f GiB, dense peak %.2f GiB, "
        "budget %.2f GiB (%.0f%% of %.2f GiB available)",
        result.safe_micro_batch,
        peak / 2**30,
        dense_peak / 2**30,
        budget / 2**30,
        VRAM_BUDGET_FRACTION * 100,
        available / 2**30,
    )
    return result


def probe_rectangular_batch_size(
    model, shape, *, dataset_dir: str, **train_options
) -> tuple[int, int]:
    from rfdetr.detr import _prepare_run_config
    from rfdetr.training import auto_batch

    images, annotations, _ = load_coco_annotations(
        Path(dataset_dir) / "train" / "_annotations.coco.json"
    )
    object_counts = sorted(
        sum(obj.get("iscrowd", 0) == 0 for obj in annotations.get(image_id, ()))
        for image_id in images
    )

    # RF-DETR 1.12.1 only builds square probe tensors. Replace the batch factory
    # during sizing and budget its training-step peaks using the native search.
    resolve = partial(
        _resolve_auto_batch_with_vram_budget, auto_batch.resolve_auto_batch_config, shape, object_counts
    )
    weights = model.model.model
    original_device = next(weights.parameters()).device
    try:
        with patch.object(auto_batch, "resolve_auto_batch_config", resolve):
            config, _, _ = _prepare_run_config(
                model,
                dataset_dir=dataset_dir,
                batch_size="auto",
                auto_batch_max_targets_per_image=max(1, object_counts[-1]),
                auto_batch_ema_headroom=1.0,
                **train_options,
            )
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
