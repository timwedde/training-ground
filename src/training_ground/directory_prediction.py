import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

import numpy as np
import typer
from PIL import Image

from .geometry import xyxy_to_xywh
from .overlays import render_prediction_overlay
from .predictors import build_pred_items, create_model
from .training_backends import RFDETR_BACKEND, SEGMENTATION_TASK, Backend, Task
from .upload import METADATA_FILENAMES

DEFAULT_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".avif", ".webp"}


def iter_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_SUFFIXES
    )


def _resolve_roboflow_project(
    workspace, checkpoint_path: Path, project_name: str | None
):
    if project_name:
        return workspace.project(project_name.split("/")[-1])

    for parent in (checkpoint_path.parent, checkpoint_path.parent.parent):
        for filename in METADATA_FILENAMES:
            path = parent / filename
            if not path.is_file():
                continue
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except OSError, ValueError:
                typer.echo(f"Could not read run metadata: {path}", err=True)
                continue
            name = metadata.get("dataset_name") if isinstance(metadata, dict) else None
            if isinstance(name, str) and name.strip():
                typer.echo(
                    f"Auto-resolved Roboflow project from metadata: {name.strip()}"
                )
                return workspace.project(name.strip().split("/")[-1])

    import questionary
    from questionary import Choice

    typer.echo("Fetching Roboflow projects...")
    project_ids = [project.split("/")[-1] for project in workspace.projects()]

    def fetch_project(project_id):
        try:
            return workspace.project(project_id)
        except Exception as exc:  # noqa: BLE001 - an unavailable project should not hide the rest
            typer.echo(
                f"Could not load Roboflow project {project_id} ({type(exc).__name__}).",
                err=True,
            )
            return None

    with ThreadPoolExecutor() as executor:
        projects = [
            project
            for project in executor.map(fetch_project, project_ids)
            if project is not None
        ]
    if not projects:
        raise typer.BadParameter("No projects found in your Roboflow workspace.")

    projects.sort(key=lambda project: getattr(project, "updated", 0), reverse=True)
    project = questionary.select(
        "Select Roboflow project for upload",
        choices=[Choice(title=project.id, value=project) for project in projects],
        instruction="(Enter to select)",
    ).ask()
    if project is None:
        raise typer.Abort()
    return project


def _mask_polygons(mask):
    if mask is None:
        return []

    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        contour.astype(float).ravel().tolist()
        for contour in contours
        if len(contour) >= 3
    ]


def generate_coco_json(
    image_path: Path, pred_items: list[dict], width: int, height: int
) -> dict:
    categories = {}
    annotations = []
    for annotation_id, item in enumerate(pred_items, start=1):
        category_id = item["category_id"]
        categories[category_id] = item["class_name"]
        bbox = xyxy_to_xywh(item["bbox"])
        annotations.append(
            {
                "id": annotation_id,
                "image_id": 1,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": _mask_polygons(item["mask"]),
            }
        )

    return {
        "images": [
            {"id": 1, "file_name": image_path.name, "width": width, "height": height}
        ],
        "categories": [
            {"id": category_id, "name": name, "supercategory": "none"}
            for category_id, name in categories.items()
        ],
        "annotations": annotations,
    }


def upload_single_image(
    project: Any, image_path: Path, pred_items: list[dict]
) -> tuple[str, bool, str | None]:
    try:
        if pred_items:
            with Image.open(image_path) as image:
                width, height = image.size

            with TemporaryDirectory() as workspace:
                annotation_path = Path(workspace) / "_annotations.coco.json"
                annotation_path.write_text(
                    json.dumps(
                        generate_coco_json(image_path, pred_items, width, height)
                    ),
                    encoding="utf-8",
                )
                results = project.upload(
                    image_path=str(image_path),
                    annotation_path=str(annotation_path),
                    is_prediction=True,
                )
        else:
            results = project.upload(image_path=str(image_path), is_prediction=True)
        if (
            len(results) != 1
            or not results[0]["image"]
            or (pred_items and results[0]["annotation"] is None)
        ):
            return str(image_path), False, "Roboflow did not confirm the upload."
    except Exception as exc:  # noqa: BLE001 - report worker failures without exposing credential-bearing URLs
        return str(image_path), False, type(exc).__name__

    return str(image_path), True, None


def run_prediction_directory(
    input_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    threshold: float | None,
    backend: Backend = RFDETR_BACKEND,
    model_size: str | None = None,
    task: Task = SEGMENTATION_TASK,
    upload: bool = False,
    project_name: str | None = None,
) -> dict:
    if not input_dir.exists() or not input_dir.is_dir():
        raise typer.BadParameter(f"Input directory does not exist: {input_dir}")

    input_root = input_dir.resolve()
    output_root = output_dir.resolve()
    if output_root == input_root:
        raise typer.BadParameter(
            "Output directory must differ from the input directory."
        )
    image_paths = iter_image_paths(input_dir)
    if output_root.is_relative_to(input_root):
        image_paths = [
            path
            for path in image_paths
            if not path.resolve().is_relative_to(output_root)
        ]
    if not image_paths:
        raise typer.BadParameter(
            "No supported images found. Expected JPEG, PNG, AVIF, or WEBP files."
        )

    project = None
    if upload:
        import roboflow

        roboflow.login()
        rf = roboflow.Roboflow()
        project = _resolve_roboflow_project(
            rf.workspace(), checkpoint_path, project_name
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(checkpoint_path, backend, model_size=model_size, task=task)

    category_names = model.class_id_to_name
    label_to_category_id = {index: index for index in category_names}

    fallback_count = 0
    file_count = 0
    pending = set()
    upload_workers = 8
    upload_executor = ThreadPoolExecutor(max_workers=upload_workers) if upload else None
    uploaded = upload_failures = 0

    def record_uploads(completed):
        nonlocal uploaded, upload_failures
        for future in completed:
            image_name, success, error = future.result()
            if success:
                uploaded += 1
            else:
                upload_failures += 1
                typer.echo(f"Failed to upload {image_name}: {error}", err=True)

    try:
        typer.echo(f"Running inference on {len(image_paths)} images from {input_dir}")
        with typer.progressbar(image_paths, label="Running inference") as progress:
            for image_path in progress:
                relative_path = image_path.relative_to(input_dir)
                detections = model.predict(str(image_path), threshold=threshold)
                pred_items = build_pred_items(
                    detections, category_names, label_to_category_id
                )
                items_by_class: dict[int, list[dict]] = {}
                for item in pred_items:
                    items_by_class.setdefault(item["category_id"], []).append(item)
                groups = []
                for class_id, items in sorted(items_by_class.items()):
                    folder = quote(items[0]["class_name"], safe="")
                    if folder in {"", ".", "..", "no_detections"}:
                        folder = f"class_{class_id}_{folder}"
                    groups.append((folder, items))
                if not groups:
                    groups = [("no_detections", [])]

                for folder, items in groups:
                    requested_output_path = output_dir / folder / relative_path
                    requested_output_path.parent.mkdir(parents=True, exist_ok=True)
                    saved_path = render_prediction_overlay(
                        image_path=image_path,
                        output_path=requested_output_path,
                        pred_items=items,
                        summary_text=(
                            f"Predictions: {len(items)} | "
                            + (
                                f"threshold {threshold:.2f}"
                                if threshold is not None
                                else "model confidence defaults"
                            )
                        ),
                    )
                    if saved_path != requested_output_path:
                        fallback_count += 1
                    file_count += 1

                if upload_executor is not None and project is not None:
                    if len(pending) == upload_workers:
                        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                        record_uploads(completed)
                    pending.add(
                        upload_executor.submit(
                            upload_single_image, project, image_path, pred_items
                        )
                    )
        record_uploads(wait(pending).done)
    finally:
        if upload_executor is not None:
            upload_executor.shutdown(cancel_futures=True)

    if upload:
        typer.echo(f"Uploaded {uploaded} of {len(image_paths)} images to Roboflow.")

    return {
        "image_count": len(image_paths),
        "file_count": file_count,
        "fallback_count": fallback_count,
        "upload_failure_count": upload_failures,
    }
