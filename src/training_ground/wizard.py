import json
import math
import os
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import onnx
import onnx.helper
import questionary
import typer
import yaml
from questionary import Choice

from .coco import decode_segmentation
from .evaluation import create_rfdetr_predictor, run_evaluation
from .metrics_plotting import plot_training_metrics
from .training_backends import (
    RFDETR_BACKEND,
    RFDETR_SEG_MODEL_LABEL,
    RFDETR_SEG_MODELS,
    RFDETR_SIZE_LABELS,
    YOLO26_BACKEND,
    YOLO26_SEG_MODELS,
    YOLO26_SIZE_LABELS,
    TrainingArtifacts,
    yolo26_run_name,
)
from .upload import (
    artifact_files_for_training,
    build_training_metadata,
    slugify_dataset_name,
    upload_artifact_bundle,
    write_upload_metadata,
)

DEFAULT_TRAINING_RESOLUTION = 372
YOLO26_IMAGE_SIZE = 640
BACK_CHOICE = "__back__"
RFDETR_GPU_CHOICES = [
    Choice(title="RTX  4080 (16GB)", value=(16, 1)),
    Choice(title="RTX  4090 (24GB)", value=(32, 1)),
    Choice(title="RTX  5090 (32GB)", value=(48, 1)),
    Choice(title="RTX A6000 (48GB)", value=(64, 1)),
    Choice(title="RTX  A100 (80GB)", value=(94, 1)),
]
YOLO26_GPU_CHOICES = [
    Choice(title="RTX  4080 (16GB)", value=(16, 1)),
    Choice(title="RTX  4090 (24GB)", value=(32, 1)),
    Choice(title="RTX  5090 (32GB)", value=(48, 1)),
    Choice(title="RTX A6000 (48GB)", value=(64, 1)),
    Choice(title="RTX  A100 (80GB)", value=(94, 1)),
]


def _float32_to_bfloat16(fval: float, truncate: bool = False) -> int:
    ival = int.from_bytes(struct.pack("<f", fval), "little")
    if truncate:
        return ival >> 16
    if math.isnan(fval):
        return 0x7FC0
    rounded = ((ival >> 16) & 1) + 0x7FFF
    return (ival + rounded) >> 16


if not hasattr(onnx.helper, "float32_to_bfloat16"):
    onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16


def fetch_project_info(data):
    workspace, project_id = data
    project = workspace.project(project_id)
    return project, project.versions()


def _patch_rfdetr_training_pretrain_loader():
    from rfdetr.models.weights import load_pretrain_weights
    from rfdetr.training.module_model import RFDETRModelModule

    if getattr(RFDETRModelModule._load_pretrain_weights, "__name__", "") == (
        "_load_pretrain_weights_with_pe_interpolation"
    ):
        return

    def _load_pretrain_weights_with_pe_interpolation(self) -> None:
        load_pretrain_weights(self.model, self.model_config)

    RFDETRModelModule._load_pretrain_weights = (
        _load_pretrain_weights_with_pe_interpolation
    )


def _copy_or_link_image(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    try:
        target_path.symlink_to(source_path.resolve())
    except OSError:
        shutil.copy2(source_path, target_path)


def _normalize_coco_annotations(annotation_path: Path, output_path: Path) -> list[str]:
    import cv2
    import numpy as np

    with annotation_path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)

    images_by_id = {image["id"]: image for image in dataset.get("images", [])}
    categories = sorted(
        dataset.get("categories", []), key=lambda category: category["id"]
    )
    category_id_map = {
        category["id"]: index + 1 for index, category in enumerate(categories)
    }

    normalized_annotations = []
    for annotation in dataset.get("annotations", []):
        normalized = dict(annotation)
        normalized["category_id"] = category_id_map[annotation["category_id"]]
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, dict):
            image = images_by_id[annotation["image_id"]]
            decoded = decode_segmentation(
                segmentation,
                int(image["height"]),
                int(image["width"]),
            )
            contours, _ = cv2.findContours(
                decoded.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            polygons = []
            for contour in contours:
                contour = contour.reshape(-1, 2)
                if len(contour) >= 3:
                    polygons.append(contour.astype(float).reshape(-1).tolist())
            normalized["segmentation"] = polygons
        elif isinstance(segmentation, list):
            normalized["segmentation"] = [
                segment for segment in segmentation if len(segment) >= 6
            ]
        else:
            normalized["segmentation"] = []
        normalized_annotations.append(normalized)

    normalized_dataset = dict(dataset)
    normalized_dataset["categories"] = [
        {"id": index + 1, **{k: v for k, v in category.items() if k != "id"}}
        for index, category in enumerate(categories)
    ]
    normalized_dataset["annotations"] = normalized_annotations

    output_path.write_text(json.dumps(normalized_dataset), encoding="utf-8")
    return [str(category["name"]) for category in categories]


def prepare_yolo26_dataset(dataset_path: Path) -> Path:
    from ultralytics.data.converter import convert_coco

    annotations_dir = dataset_path / "_yolo26_annotations"
    yolo_root = dataset_path / "yolo26"
    if annotations_dir.exists():
        shutil.rmtree(annotations_dir)
    if yolo_root.exists():
        shutil.rmtree(yolo_root)

    annotations_dir.mkdir(parents=True, exist_ok=True)

    class_names: list[str] | None = None
    split_aliases = {"train": "train", "valid": "valid", "test": "test"}
    for split_name, output_name in split_aliases.items():
        annotation_path = dataset_path / split_name / "_annotations.coco.json"
        names = _normalize_coco_annotations(
            annotation_path, annotations_dir / f"{output_name}.json"
        )
        class_names = class_names or names

    # Roboflow already provides COCO annotations. We normalize them once and then
    # use Ultralytics' COCO converter to produce YOLO26-compatible segmentation labels.
    convert_coco(
        labels_dir=str(annotations_dir),
        save_dir=str(yolo_root),
        use_segments=True,
        cls91to80=False,
    )

    for split_name in split_aliases:
        source_split_dir = dataset_path / split_name
        target_split_dir = yolo_root / "images" / split_name
        target_split_dir.mkdir(parents=True, exist_ok=True)
        for image_path in source_split_dir.iterdir():
            if image_path.name == "_annotations.coco.json" or not image_path.is_file():
                continue
            _copy_or_link_image(image_path, target_split_dir / image_path.name)

    data_yaml_path = yolo_root / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(yolo_root.resolve()),
                "train": "images/train",
                "val": "images/valid",
                "test": "images/test",
                "names": class_names or [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml_path


def train_rfdetr_seg_nano(
    dataset_path: Path,
    batch_size: int,
    grad_accum_steps: int,
    model_size: str = "nano",
) -> TrainingArtifacts:
    import rfdetr.detr as rfdetr_detr
    import torch.multiprocessing as mp
    from onnxsim import simplify

    mp.set_sharing_strategy("file_system")
    _patch_rfdetr_training_pretrain_loader()

    model_name, constructor_name = RFDETR_SEG_MODELS[model_size]
    model = getattr(rfdetr_detr, constructor_name)()
    model.train(
        dataset_dir=str(dataset_path),
        epochs=100,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=DEFAULT_TRAINING_RESOLUTION,
        early_stopping=True,
        early_stopping_patience=3,
        progress_bar=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=False,
        num_queries=50,
        num_select=20,
        output_dir="runs",
    )

    runs_dir = Path("runs")
    typer.echo("Exporting model to ONNX...")
    model.export(output_dir=str(runs_dir))
    onnx_path = runs_dir / "inference_model.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx_model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model_simp, onnx_path)

    metrics_path = runs_dir / "metrics.csv"
    typer.echo("Generating training metrics plots...")
    plot_training_metrics(metrics_path)

    checkpoint_ema = runs_dir / "checkpoint_best_ema.pth"
    checkpoint_regular = runs_dir / "checkpoint_best_regular.pth"
    eval_dir = runs_dir / f"{checkpoint_ema.stem}_test_evaluation"
    typer.echo("Running evaluation on best EMA checkpoint...")
    run_evaluation(
        checkpoint_path=checkpoint_ema,
        dataset_path=dataset_path,
        split="test",
        threshold=0.5,
        iou_threshold=0.5,
        model=create_rfdetr_predictor(checkpoint_ema, model=model),
        model_size=model_size,
        backend=RFDETR_BACKEND,
        output_dir=eval_dir,
    )

    return TrainingArtifacts(
        backend=RFDETR_BACKEND,
        model_name=model_name,
        model_size=model_size,
        runs_dir=runs_dir,
        primary_checkpoint_path=checkpoint_ema,
        secondary_checkpoint_path=checkpoint_regular,
        metrics_path=metrics_path,
        eval_dir=eval_dir,
        onnx_path=onnx_path,
    )


def train_yolo26_seg(
    dataset_path: Path,
    batch_size: int,
    model_size: str,
) -> TrainingArtifacts:
    from ultralytics import YOLO

    data_yaml_path = prepare_yolo26_dataset(dataset_path)
    selected_model_name = YOLO26_SEG_MODELS[model_size]
    run_name = yolo26_run_name(model_size)
    model = YOLO(selected_model_name)
    model.train(
        data=str(data_yaml_path),
        epochs=200,
        batch=batch_size,
        imgsz=YOLO26_IMAGE_SIZE,
        patience=3,
        workers=8,
        project="runs",
        name=run_name,
        exist_ok=True,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("YOLO26 training completed but Ultralytics did not expose a save_dir")

    runs_dir = Path(save_dir)
    best_path = runs_dir / "weights" / "best.pt"
    last_path = runs_dir / "weights" / "last.pt"
    metrics_path = runs_dir / "results.csv"
    if not best_path.exists():
        raise RuntimeError(f"YOLO26 training did not produce expected best checkpoint: {best_path}")
    if not last_path.exists():
        raise RuntimeError(f"YOLO26 training did not produce expected last checkpoint: {last_path}")
    if not metrics_path.exists():
        raise RuntimeError(f"YOLO26 training did not produce expected metrics file: {metrics_path}")

    typer.echo("Exporting YOLO26 model to ONNX...")
    export_model = YOLO(str(best_path))
    exported_path = Path(
        export_model.export(format="onnx", imgsz=YOLO26_IMAGE_SIZE, simplify=True)
    )
    onnx_path = runs_dir / "model.onnx"
    if exported_path.resolve() != onnx_path.resolve():
        shutil.copy2(exported_path, onnx_path)

    typer.echo("Generating training metrics plots...")
    plot_training_metrics(metrics_path)

    eval_dir = runs_dir / f"{best_path.stem}_test_evaluation"
    typer.echo("Running evaluation on best checkpoint...")
    run_evaluation(
        checkpoint_path=best_path,
        dataset_path=dataset_path,
        split="test",
        threshold=0.5,
        iou_threshold=0.5,
        backend=YOLO26_BACKEND,
        output_dir=eval_dir,
    )

    return TrainingArtifacts(
        backend=YOLO26_BACKEND,
        model_name=selected_model_name,
        model_size=model_size,
        runs_dir=runs_dir,
        primary_checkpoint_path=best_path,
        secondary_checkpoint_path=last_path,
        metrics_path=metrics_path,
        eval_dir=eval_dir,
        onnx_path=onnx_path,
    )


def run_wizard():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import roboflow

    roboflow.login()
    rf = roboflow.Roboflow()

    typer.echo("Fetching projects...")
    workspace = rf.workspace()
    projects = []
    with ThreadPoolExecutor() as executor:
        for project, versions in executor.map(
            fetch_project_info,
            [(workspace, project.split("/")[-1]) for project in workspace.projects()],
        ):
            projects.append((project, versions))

    projects.sort(key=lambda p: p[0].updated, reverse=True)

    project_index = 0
    version_index = 0
    project = None
    versions = None
    version = None
    backend = None
    model_size = None
    batch_size = None
    grad_accum_steps = 1

    step = "project"
    while True:
        if step == "project":
            choices = [
                Choice(title=p.id, value=index) for index, (p, _) in enumerate(projects)
            ]
            selected = questionary.select(
                "Select project",
                choices=choices,
                default=project_index,
                instruction="(Enter to continue)",
            ).ask()
            project_index = int(selected)
            project, versions = projects[project_index]
            step = "version"
            continue

        if step == "version":
            assert versions is not None
            version_choices = [
                Choice(title=v.id.split("/")[-1], value=index)
                for index, v in enumerate(versions)
            ]
            version_choices.append(Choice(title="Back", value=BACK_CHOICE))
            selected = questionary.select(
                "Select dataset version",
                choices=version_choices,
                default=version_index,
                instruction="(Select Back to return)",
            ).ask()
            if selected == BACK_CHOICE:
                step = "project"
                continue
            version_index = int(selected)
            version = versions[version_index]
            step = "backend"
            continue

        if step == "backend":
            selected = questionary.select(
                "Select model backend",
                choices=[
                    Choice(title="YOLO26", value=YOLO26_BACKEND),
                    Choice(title=RFDETR_SEG_MODEL_LABEL, value=RFDETR_BACKEND),
                    Choice(title="Back", value=BACK_CHOICE),
                ],
                default=backend or YOLO26_BACKEND,
                instruction="(Select Back to return)",
            ).ask()
            if selected == BACK_CHOICE:
                step = "version"
                continue
            backend = selected
            step = "model_size"
            continue

        if step == "model_size":
            if backend == YOLO26_BACKEND:
                selected = questionary.select(
                    "Select YOLO26 model size",
                    choices=[
                        Choice(title=YOLO26_SIZE_LABELS[size], value=size)
                        for size in ("nano", "small", "medium", "large", "xlarge")
                    ]
                    + [Choice(title="Back", value=BACK_CHOICE)],
                    default=model_size or "nano",
                    instruction="(Select Back to return)",
                ).ask()
            else:
                selected = questionary.select(
                    "Select RF-DETR model size",
                    choices=[
                        Choice(title=RFDETR_SIZE_LABELS[size], value=size)
                        for size in (
                            "nano",
                            "small",
                            "medium",
                            "large",
                            "xlarge",
                            "2xlarge",
                        )
                    ]
                    + [Choice(title="Back", value=BACK_CHOICE)],
                    default=model_size or "nano",
                    instruction="(Select Back to return)",
                ).ask()
            if selected == BACK_CHOICE:
                step = "backend"
                continue
            model_size = selected
            step = "gpu"
            continue

        if step == "gpu":
            gpu_choices = (
                YOLO26_GPU_CHOICES if backend == YOLO26_BACKEND else RFDETR_GPU_CHOICES
            )
            batch_size_selection = questionary.select(
                "Select GPU VRAM",
                choices=gpu_choices
                + [
                    Choice(title="Custom", value="custom"),
                    Choice(title="Back", value=BACK_CHOICE),
                ],
                instruction="(Select Back to return)",
            ).ask()
            if batch_size_selection == BACK_CHOICE:
                step = "model_size"
                continue

            if batch_size_selection == "custom":
                custom_value = questionary.text(
                    "Enter custom batch size",
                    validate=lambda value: value.isdigit() and int(value) > 0,
                    instruction="(Ctrl+C to abort, Enter to continue)",
                ).ask()
                batch_size = int(custom_value)
                grad_accum_steps = 1
            else:
                batch_size, grad_accum_steps = batch_size_selection
            break

    assert project is not None
    assert version is not None
    assert backend is not None
    assert model_size is not None
    assert batch_size is not None

    dataset_path = Path("datasets") / version.id
    version.download(model_format="coco", location=str(dataset_path))
    dataset_name = project.id.split("/")[-1]

    if backend == RFDETR_BACKEND:
        artifacts = train_rfdetr_seg_nano(
            dataset_path=dataset_path,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            model_size=model_size,
        )
    else:
        artifacts = train_yolo26_seg(
            dataset_path=dataset_path,
            batch_size=batch_size,
            model_size=model_size or "nano",
        )

    metadata = build_training_metadata(dataset_name, artifacts)
    write_upload_metadata(artifacts.runs_dir, dataset_name, metadata)

    typer.echo("Preparing to upload training run to GCS...")
    import asyncio

    run_id = asyncio.run(
        upload_artifact_bundle(
            runs_dir=artifacts.runs_dir,
            dataset_name=dataset_name,
            artifact_files=artifact_files_for_training(artifacts),
            metadata=metadata,
        )
    )
    typer.echo(
        f"Training run {slugify_dataset_name(dataset_name)}-{run_id} complete and uploaded successfully"
    )
