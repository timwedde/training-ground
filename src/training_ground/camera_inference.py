"""RF-DETR Medium STM inference settings from fw-jetson/camera.

Reference: src/model_specs.rs, src/gpu_pipeline.cu and src/inference.rs.
Keep this profile explicit: unrelated training datasets retain RF-DETR defaults.
"""

import numpy as np
import torch
import torch.nn.functional as F
import supervision as sv
from PIL import Image

IMAGE_SIZE = 432
MASK_SIZE = 216
NUM_SELECT = 20
CLASS_THRESHOLDS = (1.0, 0.9, 0.55, 0.6, 1.0)


def matches_camera_model(model) -> bool:
    config = getattr(model, "model_config", None)
    names = list(getattr(model, "class_names", []) or [])
    return (
        type(model).__name__ == "RFDETRSegMedium"
        and len(names) == 4
        and names[1:] == ["pole", "stick", "tree"]
        and getattr(config, "resolution", None) == IMAGE_SIZE
        and getattr(config, "mask_downsample_ratio", None) == 2
        and getattr(config, "num_select", None) == NUM_SELECT
    )


def _resize_weights(source: int, target: int) -> np.ndarray:
    # Match CUDA's triangular downsampling filter. For small input images,
    # use ordinary bilinear support to avoid empty sampling windows.
    scale = np.float32(source / target)
    support = max(scale, 1.0)
    centers = (np.arange(target, dtype=np.float32) + 0.5) * scale - 0.5
    weights = np.maximum(
        0.0,
        1.0
        - np.abs(
            (np.arange(source, dtype=np.float32)[None] - centers[:, None]) / support
        ),
    )
    return weights / weights.sum(axis=1, keepdims=True)


def prepare_image(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    scale = min(np.float32(IMAGE_SIZE / width), np.float32(IMAGE_SIZE / height))
    rw, rh = int(width * scale), int(height * scale)
    resized = np.einsum(
        "yh,hwc,xw->yxc",
        _resize_weights(height, rh),
        pixels,
        _resize_weights(width, rw),
        optimize=True,
    )
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    x, y = (IMAGE_SIZE - rw) // 2, (IMAGE_SIZE - rh) // 2
    canvas[y : y + rh, x : x + rw] = resized / 255.0
    canvas = (canvas - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(canvas.transpose(2, 0, 1).copy()).unsqueeze(0)


def postprocess(boxes, logits, masks, width: int, height: int, threshold=None):
    if logits.shape[-1] != 5 or masks.shape[-2:] != (MASK_SIZE, MASK_SIZE):
        raise ValueError(
            "Camera STM profile requires five class slots and 216x216 mask logits."
        )
    # Exclude garbage/background BEFORE top-k, just like select_queries_kernel.
    enabled = logits[0, :, 1:4].float().flatten()
    order = torch.argsort(enabled, descending=True, stable=True)[:NUM_SELECT]
    scores = enabled[order].sigmoid()
    labels, queries = order % 3 + 1, order // 3
    thresholds = scores.new_tensor(CLASS_THRESHOLDS)[labels]
    if threshold is not None:
        thresholds = torch.full_like(thresholds, threshold)
    keep = scores >= thresholds
    scores, labels, queries = scores[keep], labels[keep], queries[keep]
    selected_boxes = boxes[0, queries].float()
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    padding = selected_boxes.new_tensor(
        [(IMAGE_SIZE - width * scale) / 2, (IMAGE_SIZE - height * scale) / 2]
    )
    centers = (selected_boxes[:, :2] * IMAGE_SIZE - padding) / scale
    sizes = selected_boxes[:, 2:] * IMAGE_SIZE / scale
    # Camera clamps the origin, retaining the regressed width and height.
    origins = (centers - sizes / 2).clamp(min=0)
    origins = torch.minimum(origins, origins.new_tensor([width, height]))
    xyxy = torch.cat((origins, origins + sizes.clamp(min=0)), dim=1)

    if len(queries):
        # Match the camera's width-based centered crop, half-pixel coordinates,
        # border clamping, and strict mask logit > 0 threshold.
        crop_height = int(np.floor(height * MASK_SIZE / width + 0.5))
        crop_y = (MASK_SIZE - crop_height) / 2
        ys = (
            crop_y
            + (torch.arange(height, device=masks.device) + 0.5) * crop_height / height
        )
        xs = (torch.arange(width, device=masks.device) + 0.5) * MASK_SIZE / width
        gy, gx = torch.meshgrid(
            ys * 2 / MASK_SIZE - 1, xs * 2 / MASK_SIZE - 1, indexing="ij"
        )
        grid = torch.stack((gx, gy), dim=-1)[None].expand(len(queries), -1, -1, -1)
        resized = F.grid_sample(
            masks[0, queries, None].float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        binary_masks = (resized[:, 0] > 0).cpu().numpy()
    else:
        binary_masks = np.zeros((0, height, width), dtype=bool)
    return sv.Detections(
        xyxy=xyxy.cpu().numpy(),
        confidence=scores.cpu().numpy(),
        class_id=labels.cpu().numpy(),
        mask=binary_masks,
    )


@torch.inference_mode()
def predict(model, image_path: str, threshold: float | None):
    with Image.open(image_path) as image:
        width, height = image.size
        inputs = prepare_image(image).to(
            model.model.device, dtype=model._optimized_dtype
        )
    outputs = model.model.inference_model(inputs)
    if isinstance(outputs, tuple):
        boxes, logits, masks = outputs
    else:
        boxes, logits, masks = (
            outputs[k] for k in ("pred_boxes", "pred_logits", "pred_masks")
        )
    return postprocess(boxes, logits, masks, width, height, threshold)
