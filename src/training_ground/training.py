import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import cv2
import typer
import yaml

from .evaluation import run_evaluation
from .metrics_plotting import plot_training_metrics
from .predictors import RFDETRPredictorAdapter, load_rfdetr_checkpoint
from .tiling import pad_to_tile
from .training_backends import (
    BOX_TASK,
    RFDETR_BACKEND,
    SEGMENTATION_TASK,
    YOLO26_BACKEND,
    YOLO26_IMAGE_SIZE,
    Task,
    TrainingArtifacts,
    rfdetr_models_for_task,
    yolo26_models_for_task,
    yolo26_run_name,
)

RFDETR_SEG_NANO_RESOLUTION = 372


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


def prepare_yolo26_dataset(
    dataset_path: Path, yolo_root: Path, task: Task, *, tiled: bool = False
) -> Path:
    from ultralytics.data.converter import convert_coco
    from ultralytics.utils.patches import imread

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

    images_by_split: dict[str, list[tuple[Path, Path]]] = {}
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
            image_paths = []
            for image in dataset["images"]:
                path = Path(image["file_name"])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Invalid COCO image path: {path}")
                if not (dataset_path / split / path).is_file():
                    raise FileNotFoundError(dataset_path / split / path)
                prepared_path = path
                if tiled and min(image["width"], image["height"]) < YOLO26_IMAGE_SIZE:
                    # Padding before YOLO loads the image prevents its loader from upscaling small crops.
                    prepared_path = path.with_name(f"{path.name}.png")
                    image["file_name"] = prepared_path.as_posix()
                    image["width"] = max(image["width"], YOLO26_IMAGE_SIZE)
                    image["height"] = max(image["height"], YOLO26_IMAGE_SIZE)
                image_paths.append((path, prepared_path))
            images_by_split[split] = image_paths
            _normalize_coco_annotations(
                dataset, annotations_dir / f"{split}.json", class_ids
            )

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
        for path, prepared_path in image_paths:
            source = source_split_dir / path
            target = target_split_dir / prepared_path
            if path == prepared_path:
                _copy_or_link_image(source, target)
            else:
                image = imread(str(source))
                if image is None:
                    raise ValueError(f"Could not read image: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), pad_to_tile(image)):
                    raise OSError(f"Could not write padded image: {target}")

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
    task: Task,
    model_size: str = "nano",
    mask_downsample_ratio: int = 4,
    limit_queries: bool = False,
    *,
    batch_size: int | None = None,
) -> TrainingArtifacts:
    import onnx
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
    if batch_size is not None:
        train_kwargs["batch_size"] = batch_size
    elif model.model.device.type == "cuda":
        train_kwargs["batch_size"] = "auto"
    if model_size == "nano" and task == SEGMENTATION_TASK:
        train_kwargs["resolution"] = RFDETR_SEG_NANO_RESOLUTION

    model.train(
        dataset_dir=str(dataset_path),
        epochs=100,
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_use_ema=True,
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
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated.")
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
    task: Task,
    model_size: str,
    *,
    batch_size: int | None = None,
    tiled: bool = False,
) -> TrainingArtifacts:
    from ultralytics import YOLO

    if tiled and task != BOX_TASK:
        raise typer.BadParameter("Tiling is available only for YOLO26 box detection.")
    run_name = yolo26_run_name(task, model_size)
    runs_dir = create_run_directory(f"{run_name}-tiled" if tiled else run_name)
    data_yaml_path = prepare_yolo26_dataset(
        dataset_path, runs_dir / "dataset", task, tiled=tiled
    )
    selected_model_name = yolo26_models_for_task(task)[model_size]
    model = YOLO(selected_model_name)
    workers = 8
    if tiled:
        available_cpus = os.process_cpu_count() or 1
        # Reserve CPU capacity for the trainer and limit the number of prefetched batches.
        workers = min(16, max(0, available_cpus - 2))
        typer.echo(
            f"YOLO26 tiled training requests {workers} loader workers ({available_cpus} available CPUs)."
        )
    tile_options = (
        {
            "tile": True,
            "nms": False,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "scale": 0.0,
            "translate": 0.0,
            "multi_scale": 0.0,
        }
        if tiled
        else {}
    )
    model.train(
        data=str(data_yaml_path),
        epochs=200,
        batch=0.95 if batch_size is None else batch_size,
        imgsz=YOLO26_IMAGE_SIZE,
        patience=10,
        workers=workers,
        project=str(runs_dir.parent),
        name=runs_dir.name,
        # The directory was reserved for this invocation before conversion.
        exist_ok=True,
        **tile_options,
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
        export_model.export(
            format="onnx",
            imgsz=YOLO26_IMAGE_SIZE,
            simplify=True,
            nms=False if tiled else None,
        )
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
        tiled=tiled,
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
