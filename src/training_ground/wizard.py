import json
import math
import os
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import onnx
import onnx.helper
import questionary
import typer
import yaml
from questionary import Choice

from .evaluation import RFDETRPredictorAdapter, load_rfdetr_checkpoint, run_evaluation
from .metrics_plotting import plot_training_metrics
from .training_backends import (
    BOX_TASK,
    RFDETR_BACKEND,
    RFDETR_BOX_MODEL_LABEL,
    RFDETR_SEG_MODEL_LABEL,
    RFDETR_SIZE_LABELS,
    SEGMENTATION_TASK,
    YOLO26_BACKEND,
    YOLO26_SIZE_LABELS,
    Task,
    TrainingArtifacts,
    rfdetr_models_for_task,
    yolo26_models_for_task,
    yolo26_run_name,
)
from .upload import (
    artifact_files_for_training,
    build_training_metadata,
    slugify_dataset_name,
    upload_artifact_bundle,
    write_upload_metadata,
)

RFDETR_SEG_NANO_RESOLUTION = 372
DEFAULT_BACKEND = RFDETR_BACKEND
DEFAULT_RFDETR_MODEL_SIZE = "medium"
YOLO26_IMAGE_SIZE = 640
BACK_CHOICE = "__back__"
RFDETR_GPU_CHOICES = [
    Choice(title="RTX  4080 (16GB)", value=(16, 1)),
    Choice(title="RTX  4090 (24GB)", value=(32, 1)),
    Choice(title="RTX  5090 (32GB)", value=(48, 1)),
    Choice(title="RTX A6000 (48GB)", value=(64, 1)),
    Choice(title="RTX  A100 (80GB)", value=(94, 1)),
]
RFDETR_MASK_DOWNSAMPLE_RATIO_CHOICES = [
    Choice(
        title="4 (default, quarter-resolution masks)",
        value=4,
    ),
    Choice(
        title="2 (high quality, half-resolution masks; ~4x mask-head pixels)",
        value=2,
    ),
    Choice(
        title="1 (full-resolution masks; ~16x mask-head pixels)",
        value=1,
    ),
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


def _copy_or_link_image(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    try:
        target_path.symlink_to(source_path.resolve())
    except OSError:
        shutil.copy2(source_path, target_path)


def _normalize_coco_annotations(
    dataset: dict, output_path: Path, class_ids: dict[str, int]
) -> None:
    category_names = {
        category["id"]: category["name"] for category in dataset["categories"]
    }
    if len(set(category_names.values())) != len(dataset["categories"]):
        raise ValueError(f"Duplicate COCO categories in {output_path.stem} split.")

    normalized_annotations = []
    for annotation in dataset.get("annotations", []):
        normalized = dict(annotation)
        name = category_names.get(annotation["category_id"])
        if name not in class_ids:
            raise ValueError(
                f"{output_path.stem} annotation {annotation['id']} uses class "
                f"{name or annotation['category_id']!r}, which is missing from the training classes."
            )
        normalized["category_id"] = class_ids[name]
        normalized_annotations.append(normalized)

    normalized_dataset = dict(dataset)
    normalized_dataset["categories"] = [
        {"id": class_id, "name": name} for name, class_id in class_ids.items()
    ]
    normalized_dataset["annotations"] = normalized_annotations

    output_path.write_text(json.dumps(normalized_dataset), encoding="utf-8")


def prepare_yolo26_dataset(dataset_path: Path, yolo_root: Path, task: Task) -> Path:
    from ultralytics.data.converter import convert_coco

    if yolo_root.exists():
        raise FileExistsError(yolo_root)
    splits = ("train", "valid", "test")
    train_annotations = json.loads(
        (dataset_path / "train" / "_annotations.coco.json").read_bytes()
    )
    class_names = [
        category["name"]
        for category in sorted(
            train_annotations["categories"], key=lambda item: item["id"]
        )
    ]
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Training COCO categories must have unique class names.")
    class_ids = {name: index for index, name in enumerate(class_names, start=1)}

    images_by_split: dict[str, list[Path]] = {}
    with TemporaryDirectory() as temp_dir:
        annotations_dir = Path(temp_dir)
        for split in splits:
            dataset = (
                train_annotations
                if split == "train"
                else json.loads(
                    (dataset_path / split / "_annotations.coco.json").read_bytes()
                )
            )
            _normalize_coco_annotations(
                dataset, annotations_dir / f"{split}.json", class_ids
            )
            image_paths = [Path(image["file_name"]) for image in dataset["images"]]
            for path in image_paths:
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Invalid COCO image path: {path}")
                if not (dataset_path / split / path).is_file():
                    raise FileNotFoundError(dataset_path / split / path)
            images_by_split[split] = image_paths

        convert_coco(
            labels_dir=str(annotations_dir),
            save_dir=str(yolo_root),
            use_segments=task == SEGMENTATION_TASK,
            cls91to80=False,
        )

    for split_name, image_paths in images_by_split.items():
        source_split_dir = dataset_path / split_name
        target_split_dir = yolo_root / "images" / split_name
        target_split_dir.mkdir(parents=True, exist_ok=True)
        for path in image_paths:
            _copy_or_link_image(source_split_dir / path, target_split_dir / path)

    data_yaml_path = yolo_root / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(yolo_root.resolve()),
                "train": "images/train",
                "val": "images/valid",
                "test": "images/test",
                "names": class_names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml_path


def create_run_directory(name: str) -> Path:
    runs_root = Path("runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(mkdtemp(prefix=f"{name}-", dir=runs_root))
    typer.echo(f"Training artifacts directory: {runs_dir}")
    return runs_dir


def train_rfdetr(
    dataset_path: Path,
    batch_size: int,
    grad_accum_steps: int,
    task: Task,
    model_size: str = "nano",
    mask_downsample_ratio: int = 4,
    limit_queries: bool = False,
) -> TrainingArtifacts:
    import rfdetr.detr as rfdetr_detr
    import torch.multiprocessing as mp
    from onnxsim import simplify

    mp.set_sharing_strategy("file_system")
    runs_dir = create_run_directory(f"rfdetr-{task}-{model_size}")
    model_name, constructor_name = rfdetr_models_for_task(task)[model_size]
    model_kwargs = {"num_queries": 50, "num_select": 20} if limit_queries else {}
    if task == SEGMENTATION_TASK:
        model_kwargs["mask_downsample_ratio"] = mask_downsample_ratio
    model = getattr(rfdetr_detr, constructor_name)(**model_kwargs)
    aug_config: dict[str, object] = {}
    num_workers = 2
    prefetch_factor = 1
    persistent_workers = False
    pin_memory = False
    typer.echo(
        "RF-DETR augmentation disabled: "
        f"aug_config={aug_config}, "
        "multi_scale=False, "
        "expanded_scales=False, "
        "scale_jitter=False, "
        'augmentation_backend="torchvision"'
    )
    typer.echo(
        "RF-DETR dataloader: "
        f"workers={num_workers}, "
        f"prefetch_factor={prefetch_factor}, "
        f"persistent_workers={persistent_workers}, "
        f"pin_memory={pin_memory}"
    )
    if task == SEGMENTATION_TASK:
        typer.echo(
            "RF-DETR mask output: "
            f"downsample_ratio={mask_downsample_ratio}, "
            f"linear_resolution=1/{mask_downsample_ratio} of training input"
        )
    train_kwargs = {}
    if model_size == "nano" and task == SEGMENTATION_TASK:
        train_kwargs["resolution"] = RFDETR_SEG_NANO_RESOLUTION

    model.train(
        dataset_dir=str(dataset_path),
        epochs=100,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        early_stopping=True,
        early_stopping_patience=3,
        progress_bar="tqdm",
        compute_val_loss=True,
        tensorboard=False,
        aug_config=aug_config,
        multi_scale=False,
        expanded_scales=False,
        scale_jitter=False,
        augmentation_backend="torchvision",
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        output_dir=str(runs_dir),
        **train_kwargs,
    )

    del model
    checkpoint_ema = runs_dir / "checkpoint_best_ema.pth"
    checkpoint_regular = runs_dir / "checkpoint_best_regular.pth"
    best_model = load_rfdetr_checkpoint(
        checkpoint_ema, model_size=model_size, task=task
    )
    typer.echo("Exporting model to ONNX...")
    onnx_path = best_model.export(
        output_dir=str(runs_dir), output_name="inference_model", verbose=False
    )
    onnx_model = onnx.load(onnx_path)
    onnx_model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model_simp, onnx_path)

    metrics_path = runs_dir / "metrics.csv"
    typer.echo("Generating training metrics plots...")
    plot_training_metrics(metrics_path)

    eval_dir = runs_dir / f"{checkpoint_ema.stem}_test_evaluation"
    typer.echo("Running evaluation on best EMA checkpoint...")
    run_evaluation(
        checkpoint_path=checkpoint_ema,
        dataset_path=dataset_path,
        split="test",
        threshold=None,
        iou_threshold=0.5,
        model=RFDETRPredictorAdapter(best_model),
        model_size=model_size,
        backend=RFDETR_BACKEND,
        task=task,
        output_dir=eval_dir,
    )

    return TrainingArtifacts(
        backend=RFDETR_BACKEND,
        task=task,
        model_name=model_name,
        model_size=model_size,
        runs_dir=runs_dir,
        primary_checkpoint_path=checkpoint_ema,
        secondary_checkpoint_path=checkpoint_regular,
        metrics_path=metrics_path,
        eval_dir=eval_dir,
        onnx_path=onnx_path,
        mask_downsample_ratio=(
            mask_downsample_ratio if task == SEGMENTATION_TASK else None
        ),
    )


def train_yolo26(
    dataset_path: Path,
    batch_size: int,
    task: Task,
    model_size: str,
) -> TrainingArtifacts:
    from ultralytics import YOLO

    runs_dir = create_run_directory(yolo26_run_name(task, model_size))
    data_yaml_path = prepare_yolo26_dataset(dataset_path, runs_dir / "dataset", task)
    selected_model_name = yolo26_models_for_task(task)[model_size]
    model = YOLO(selected_model_name)
    model.train(
        data=str(data_yaml_path),
        epochs=200,
        batch=batch_size,
        imgsz=YOLO26_IMAGE_SIZE,
        patience=10,
        workers=8,
        project=str(runs_dir.parent),
        name=runs_dir.name,
        # The directory was reserved for this invocation before conversion.
        exist_ok=True,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError(
            "YOLO26 training completed but Ultralytics did not expose a save_dir"
        )

    runs_dir = Path(save_dir).resolve()
    best_path = runs_dir / "weights" / "best.pt"
    last_path = runs_dir / "weights" / "last.pt"
    metrics_path = runs_dir / "results.csv"
    if not best_path.exists():
        raise RuntimeError(
            f"YOLO26 training did not produce expected best checkpoint: {best_path}"
        )
    if not last_path.exists():
        raise RuntimeError(
            f"YOLO26 training did not produce expected last checkpoint: {last_path}"
        )
    if not metrics_path.exists():
        raise RuntimeError(
            f"YOLO26 training did not produce expected metrics file: {metrics_path}"
        )

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
        threshold=None,
        iou_threshold=0.5,
        backend=YOLO26_BACKEND,
        task=task,
        output_dir=eval_dir,
    )

    return TrainingArtifacts(
        backend=YOLO26_BACKEND,
        task=task,
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
    task = SEGMENTATION_TASK
    backend = None
    model_size = None
    mask_downsample_ratio = 4
    limit_queries = False
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
            step = "task"
            continue

        if step == "task":
            selected = questionary.select(
                "Select annotation type",
                choices=[
                    Choice(title="Boxes", value=BOX_TASK),
                    Choice(title="Segmentation", value=SEGMENTATION_TASK),
                    Choice(title="Back", value=BACK_CHOICE),
                ],
                default=task,
                instruction="(Select Back to return)",
            ).ask()
            if selected == BACK_CHOICE:
                step = "version"
                continue
            task = selected
            model_size = None
            step = "backend"
            continue

        if step == "backend":
            selected = questionary.select(
                "Select model backend",
                choices=[
                    Choice(title="YOLO26", value=YOLO26_BACKEND),
                    Choice(
                        title=(
                            RFDETR_BOX_MODEL_LABEL
                            if task == BOX_TASK
                            else RFDETR_SEG_MODEL_LABEL
                        ),
                        value=RFDETR_BACKEND,
                    ),
                    Choice(title="Back", value=BACK_CHOICE),
                ],
                default=backend or DEFAULT_BACKEND,
                instruction="(Select Back to return)",
            ).ask()
            if selected == BACK_CHOICE:
                step = "task"
                continue
            if selected != backend:
                model_size = None
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
                rfdetr_models = rfdetr_models_for_task(task)
                selected = questionary.select(
                    "Select RF-DETR model size",
                    choices=[
                        Choice(title=RFDETR_SIZE_LABELS[size], value=size)
                        for size in rfdetr_models
                    ]
                    + [Choice(title="Back", value=BACK_CHOICE)],
                    default=model_size or DEFAULT_RFDETR_MODEL_SIZE,
                    instruction="(Select Back to return)",
                ).ask()
            if selected == BACK_CHOICE:
                step = "backend"
                continue
            model_size = selected
            if backend == YOLO26_BACKEND:
                step = "gpu"
            else:
                step = "mask_ratio" if task == SEGMENTATION_TASK else "queries"
            continue

        if step == "mask_ratio":
            selected = questionary.select(
                "Select RF-DETR mask downsample ratio",
                choices=RFDETR_MASK_DOWNSAMPLE_RATIO_CHOICES
                + [Choice(title="Back", value=BACK_CHOICE)],
                default=mask_downsample_ratio,
                instruction="(Lower ratios improve mask resolution and use more VRAM)",
            ).ask()
            if selected == BACK_CHOICE:
                step = "model_size"
                continue
            mask_downsample_ratio = int(selected)
            step = "queries"
            continue

        if step == "queries":
            selected = questionary.select(
                "Select RF-DETR query and output limits",
                choices=[
                    Choice(title="Model defaults", value=False),
                    Choice(title="50 queries / 20 outputs per image", value=True),
                    Choice(title="Back", value=BACK_CHOICE),
                ],
                default=limit_queries,
                instruction="(Select Back to return)",
            ).ask()
            if selected == BACK_CHOICE:
                step = "mask_ratio" if task == SEGMENTATION_TASK else "model_size"
                continue
            limit_queries = selected is True
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
                step = "queries" if backend == RFDETR_BACKEND else "model_size"
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
        artifacts = train_rfdetr(
            dataset_path=dataset_path,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            task=task,
            model_size=model_size,
            mask_downsample_ratio=mask_downsample_ratio,
            limit_queries=limit_queries,
        )
    else:
        artifacts = train_yolo26(
            dataset_path=dataset_path,
            batch_size=batch_size,
            task=task,
            model_size=model_size or "nano",
        )

    metadata = build_training_metadata(dataset_name, artifacts)
    write_upload_metadata(artifacts.runs_dir, dataset_name, metadata)
    typer.echo(f"Checkpoint: {artifacts.primary_checkpoint_path}")
    typer.echo(f"ONNX model: {artifacts.onnx_path}")
    typer.echo(f"Evaluation report: {artifacts.eval_dir}")

    typer.echo("Preparing to upload training run to GCS...")
    import asyncio

    run_id = asyncio.run(
        upload_artifact_bundle(
            dataset_name=dataset_name,
            artifact_files=artifact_files_for_training(artifacts),
            metadata=metadata,
        )
    )
    typer.echo(
        f"Training run {slugify_dataset_name(dataset_name)}-{run_id} complete and uploaded successfully"
    )
