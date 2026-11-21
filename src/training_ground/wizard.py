import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import questionary
import typer
from questionary import Choice

from .training import train_rfdetr, train_yolo26
from .training_backends import (
    BOX_TASK,
    RFDETR_BACKEND,
    RFDETR_BOX_MODEL_LABEL,
    RFDETR_SEG_MODEL_LABEL,
    RFDETR_SIZE_LABELS,
    SEGMENTATION_TASK,
    YOLO26_BACKEND,
    YOLO26_IMAGE_SIZE,
    YOLO26_SIZE_LABELS,
    rfdetr_models_for_task,
)
from .upload import (
    artifact_files_for_training,
    build_training_metadata,
    slugify_dataset_name,
    upload_artifact_bundle,
    write_upload_metadata,
)

DEFAULT_BACKEND = RFDETR_BACKEND
DEFAULT_RFDETR_MODEL_SIZE = "medium"
BACK_CHOICE = "__back__"
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


def _select(message, choices, default, *, back=True, instruction=None):
    if back:
        choices = [*choices, Choice(title="Back", value=BACK_CHOICE)]
    selected = questionary.select(
        message, choices=choices, default=default, instruction=instruction
    ).ask()
    if selected is None:
        raise typer.Abort()
    return selected


def run_wizard(
    batch_size: int | None = None,
    dataset_path: Path | None = None,
    *,
    image_size: int | None = None,
    rect: bool = False,
):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    local_dataset = dataset_path is not None
    if not local_dataset:
        import roboflow

        roboflow.login()
        rf = roboflow.Roboflow()

        typer.echo("Fetching projects...")
        workspace = rf.workspace()

        def fetch_project_info(project_id):
            project = workspace.project(project_id.split("/")[-1])
            return project, project.versions()

        with ThreadPoolExecutor() as executor:
            projects = list(executor.map(fetch_project_info, workspace.projects()))
        if not projects:
            raise typer.BadParameter("No projects found in your Roboflow workspace.")

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
    tiled = False
    rectangular_batches = rect

    step = "task" if local_dataset else "project"
    while True:
        if step == "project":
            choices = [
                Choice(title=p.id, value=index) for index, (p, _) in enumerate(projects)
            ]
            selected = _select(
                "Select project",
                choices=choices,
                default=project_index,
                back=False,
            )
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
            selected = _select(
                "Select dataset version",
                choices=version_choices,
                default=min(version_index, len(versions) - 1)
                if versions
                else BACK_CHOICE,
            )
            if selected == BACK_CHOICE:
                step = "project"
                continue
            version_index = int(selected)
            version = versions[version_index]
            step = "task"
            continue

        if step == "task":
            selected = _select(
                "Select annotation type",
                choices=[
                    Choice(title="Boxes", value=BOX_TASK),
                    Choice(title="Segmentation", value=SEGMENTATION_TASK),
                ],
                default=task,
                back=not local_dataset,
            )
            if selected == BACK_CHOICE:
                step = "version"
                continue
            task = selected
            model_size = None
            step = "backend"
            continue

        if step == "backend":
            selected = _select(
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
                ],
                default=backend or DEFAULT_BACKEND,
            )
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
                title = "Select YOLO26 model size"
                sizes = YOLO26_SIZE_LABELS
                default_size = "nano"
            else:
                title = "Select RF-DETR model size"
                sizes = {
                    size: RFDETR_SIZE_LABELS[size]
                    for size in rfdetr_models_for_task(task)
                }
                default_size = DEFAULT_RFDETR_MODEL_SIZE
            selected = _select(
                title,
                choices=[
                    Choice(title=label, value=size) for size, label in sizes.items()
                ],
                default=model_size or default_size,
            )
            if selected == BACK_CHOICE:
                step = "backend"
                continue
            model_size = selected
            if backend == YOLO26_BACKEND:
                if task == SEGMENTATION_TASK:
                    step = "image_size" if rect else "batch_shape"
                else:
                    step = "tiling"
                continue
            step = "mask_ratio" if task == SEGMENTATION_TASK else "queries"
            continue

        if step == "batch_shape":
            selected = _select(
                "Select YOLO26 segmentation training batch shape",
                choices=[
                    Choice(title="Square (default)", value=False),
                    Choice(title="Rectangular (less padding, no mosaic)", value=True),
                ],
                default=rectangular_batches,
            )
            if selected == BACK_CHOICE:
                step = "model_size"
                continue
            rectangular_batches = selected is True
            step = "image_size"
            continue

        if step == "image_size":
            if image_size is not None:
                break
            choices = [
                Choice(
                    title=f"{YOLO26_IMAGE_SIZE} px (default)", value=YOLO26_IMAGE_SIZE
                ),
            ]
            if not rectangular_batches:
                choices.append(Choice(title="448 px (for 432 × 432 images)", value=448))
            choices.append(Choice(title="Custom size", value="custom"))
            selected = _select(
                "Select YOLO26 segmentation "
                + ("long side" if rectangular_batches else "input size"),
                choices=choices,
                default=YOLO26_IMAGE_SIZE,
            )
            if selected == BACK_CHOICE:
                step = "model_size" if rect else "batch_shape"
                continue
            if selected == "custom":
                selected = questionary.text(
                    "Size in pixels (rounded up to a multiple of 32)",
                    validate=lambda value: (
                        (value.isdecimal() and int(value) > 0)
                        or "Enter a positive whole number."
                    ),
                ).ask()
                if selected is None:
                    raise typer.Abort()
            image_size = int(selected)
            break

        if step == "tiling":
            selected = _select(
                "Select YOLO26 image mode",
                choices=[
                    Choice(title="Full image", value=False),
                    Choice(
                        title="Native-resolution tiles (640 px, 20% overlap)",
                        value=True,
                    ),
                ],
                default=tiled,
            )
            if selected == BACK_CHOICE:
                step = "model_size"
                continue
            tiled = selected is True
            break

        if step == "mask_ratio":
            selected = _select(
                "Select RF-DETR mask downsample ratio",
                choices=RFDETR_MASK_DOWNSAMPLE_RATIO_CHOICES,
                default=mask_downsample_ratio,
                instruction="(Lower ratios improve mask resolution and use more VRAM)",
            )
            if selected == BACK_CHOICE:
                step = "model_size"
                continue
            mask_downsample_ratio = int(selected)
            step = "queries"
            continue

        if step == "queries":
            selected = _select(
                "Select RF-DETR query and output limits",
                choices=[
                    Choice(title="Model defaults", value=False),
                    Choice(title="50 queries / 20 outputs per image", value=True),
                ],
                default=limit_queries,
            )
            if selected == BACK_CHOICE:
                step = "mask_ratio" if task == SEGMENTATION_TASK else "model_size"
                continue
            limit_queries = selected is True
            break

    assert backend is not None
    assert model_size is not None

    if image_size is not None and (
        backend != YOLO26_BACKEND or task != SEGMENTATION_TASK
    ):
        raise typer.BadParameter("--image-size applies only to YOLO26 segmentation.")
    if rect and (backend != YOLO26_BACKEND or task != SEGMENTATION_TASK):
        raise typer.BadParameter("--rect applies only to YOLO26 segmentation.")

    if dataset_path is None:
        assert project is not None and version is not None
        dataset_path = Path("datasets") / version.id
        version.download(model_format="coco", location=str(dataset_path))
        dataset_name = project.id.split("/")[-1]
    else:
        dataset_path = dataset_path.resolve()
        dataset_name = dataset_path.name

    if backend == RFDETR_BACKEND:
        artifacts = train_rfdetr(
            dataset_path=dataset_path,
            batch_size=batch_size,
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
            model_size=model_size,
            tiled=tiled,
            image_size=image_size,
            rect=rectangular_batches,
        )

    metadata = build_training_metadata(dataset_name, artifacts)
    write_upload_metadata(artifacts.runs_dir, dataset_name, metadata)
    typer.echo(f"Checkpoint: {artifacts.primary_checkpoint_path}")
    typer.echo(f"ONNX model: {artifacts.onnx_path}")
    typer.echo(f"Evaluation report: {artifacts.eval_dir}")

    typer.echo("Preparing to upload training run to GCS...")
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
