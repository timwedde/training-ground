from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Backend = Literal["rfdetr", "yolo26"]

RFDETR_BACKEND: Backend = "rfdetr"
YOLO26_BACKEND: Backend = "yolo26"
RFDETR_SEG_NANO_NAME = "RF-DETR Seg Nano"
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

