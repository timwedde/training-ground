from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Backend = Literal["rfdetr", "yolo26"]

RFDETR_BACKEND: Backend = "rfdetr"
YOLO26_BACKEND: Backend = "yolo26"
RFDETR_MODEL_FAMILY = "RF-DETR"
RFDETR_SEG_MODEL_LABEL = "RF-DETR Seg"
RFDETR_SEG_MODELS = {
    "nano": ("RF-DETR Seg Nano", "RFDETRSegNano"),
    "small": ("RF-DETR Seg Small", "RFDETRSegSmall"),
    "medium": ("RF-DETR Seg Medium", "RFDETRSegMedium"),
    "large": ("RF-DETR Seg Large", "RFDETRSegLarge"),
    "xlarge": ("RF-DETR Seg XLarge", "RFDETRSegXLarge"),
    "2xlarge": ("RF-DETR Seg 2XLarge", "RFDETRSeg2XLarge"),
}
YOLO26_MODEL_FAMILY = "YOLO26"

YOLO26_SEG_MODELS = {
    "nano": "yolo26n-seg.pt",
    "small": "yolo26s-seg.pt",
    "medium": "yolo26m-seg.pt",
    "large": "yolo26l-seg.pt",
    "xlarge": "yolo26x-seg.pt",
}

YOLO26_SIZE_LABELS = {
    "nano": "Nano",
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
    "xlarge": "XLarge",
}
RFDETR_SIZE_LABELS = {
    "nano": "Nano",
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
    "xlarge": "XLarge",
    "2xlarge": "2XLarge",
}


@dataclass
class TrainingArtifacts:
    backend: Backend
    model_name: str
    model_size: str | None
    runs_dir: Path
    primary_checkpoint_path: Path
    secondary_checkpoint_path: Path | None
    metrics_path: Path
    eval_dir: Path
    onnx_path: Path


def normalize_backend(value: str) -> Backend:
    normalized = value.strip().lower()
    if normalized not in {RFDETR_BACKEND, YOLO26_BACKEND}:
        raise ValueError(f"Unsupported backend: {value}")
    return cast(Backend, normalized)


def yolo26_run_name(model_size: str) -> str:
    return f"yolo26-{model_size}"
