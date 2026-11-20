import csv
import json
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Protocol
from urllib.parse import quote

import numpy as np
import typer
from faster_coco_eval import COCO
from PIL import Image, ImageDraw

from .coco import decode_segmentation, encode_binary_mask, load_coco_annotations
from .coco_eval import run_coco_eval
from .evaluation_plots import write_evaluation_plots
from .geometry import bbox_iou, mask_iou, xywh_to_xyxy, xyxy_to_xywh
from .training_backends import (
    RFDETR_BACKEND,
    SEGMENTATION_TASK,
    YOLO26_BACKEND,
    Backend,
    Task,
    rfdetr_models_for_task,
)

OVERLAY_ALPHA = 0.28
OVERLAY_QUALITY = 92
SUMMARY_BAR_HEIGHT = 24
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
COCO_CONFIDENCE_FLOOR = 0.001
DEFAULT_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".avif", ".webp"}
DEFAULT_PRED_COLOR = (231, 76, 60)
CLASS_PRED_COLORS = {
    "trees": (46, 204, 113),
    "poles": (155, 89, 182),
    "sticks": (231, 76, 60),
}


class Predictor(Protocol):
    class_id_to_name: dict[int, str]

    def predict(self, image_path: str, threshold: float | None): ...


class YOLO26PredictorAdapter:
    def __init__(self, checkpoint_path: Path):
        from ultralytics import YOLO

        typer.echo(f"Loading YOLO26 model from checkpoint: {checkpoint_path}")
        self._model = YOLO(str(checkpoint_path))
        names = getattr(self._model, "names", None) or getattr(
            self._model.model, "names", {}
        )
        if isinstance(names, dict):
            self.class_id_to_name = {
                int(index): str(name) for index, name in names.items()
            }
        else:
            self.class_id_to_name = dict(enumerate(map(str, names)))

    def predict(self, image_path: str, threshold: float | None):
        results = self._model.predict(
            source=image_path,
            conf=DEFAULT_CONFIDENCE_THRESHOLD if threshold is None else threshold,
            verbose=False,
            imgsz=640,
            retina_masks=True,
        )
        result = results[0]
        boxes = result.boxes
        masks = result.masks

        xyxy = (
            boxes.xyxy.detach().cpu().numpy()
            if boxes is not None and boxes.xyxy is not None
            else np.zeros((0, 4), dtype=np.float32)
        )
        confidence = (
            boxes.conf.detach().cpu().numpy()
            if boxes is not None and boxes.conf is not None
            else np.zeros((0,), dtype=np.float32)
        )
        class_id = (
            boxes.cls.detach().cpu().numpy().astype(np.int32)
            if boxes is not None and boxes.cls is not None
            else np.zeros((0,), dtype=np.int32)
        )
        mask_data = (
            masks.data.detach().cpu().numpy().astype(bool, copy=False)
            if masks is not None and masks.data is not None
            else None
        )

        return DetectionBatch(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=mask_data,
        )


class RFDETRPredictorAdapter:
    def __init__(self, model):
        from .camera_inference import matches_camera_model

        self.camera_profile = matches_camera_model(model)
        if self.camera_profile:
            typer.echo(
                "Camera STM settings: input=432, masks=216, top-k=20, "
                "pole=0.90, stick=0.55, tree=0.60; centered letterbox."
            )
        self._predictor = model.inference(compile=False)
        if self.camera_profile:
            self.class_id_to_name = {
                slot: model.class_names[slot] for slot in (1, 2, 3)
            }
        else:
            context = self._predictor._prediction_context()
            self.class_id_to_name = context.class_id_to_name

    def predict(self, image_path: str, threshold: float | None):
        if self.camera_profile:
            from .camera_inference import predict

            return predict(self._predictor, image_path, threshold)
        from rfdetr import ClassThresholds

        confidence = DEFAULT_CONFIDENCE_THRESHOLD if threshold is None else threshold
        return self._predictor.predict(
            image_path,
            # A uniform class threshold excludes RF-DETR's unmapped no-object slot.
            threshold=ClassThresholds({}, default=confidence),
            include_source_image=False,
        )


class DetectionBatch:
    def __init__(self, xyxy, confidence, class_id, mask):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.mask = mask

    def __len__(self) -> int:
        return len(self.class_id)


def load_rfdetr_checkpoint(
    checkpoint_path: Path,
    model_size: str | None = None,
    task: Task = SEGMENTATION_TASK,
):
    import rfdetr
    import rfdetr.detr as rfdetr_detr
    from rfdetr.config import ModelConfig

    typer.echo(f"Loading RF-DETR model from checkpoint: {checkpoint_path}")
    # Preserve checkpoint architecture (including mask resolution), but use
    # this machine's default device rather than the saved training device.
    device = ModelConfig.model_fields["device"].default
    try:
        model = rfdetr.from_checkpoint(checkpoint_path.resolve(), device=device)
    except ValueError as exc:
        if "Could not infer model class from checkpoint" not in str(exc):
            raise

        legacy_model_size = model_size or "nano"
        typer.echo(
            "Checkpoint does not contain self-describing RF-DETR model metadata; "
            f"falling back to --model-size={legacy_model_size}."
        )
        _, constructor_name = rfdetr_models_for_task(task)[legacy_model_size]
        constructor = getattr(rfdetr_detr, constructor_name)
        model = constructor(
            pretrain_weights=str(checkpoint_path.resolve()), device=device
        )

    model_config = getattr(model, "model_config", None)
    if model_config is not None:
        typer.echo(
            "RF-DETR checkpoint architecture: "
            f"model={type(model).__name__}, "
            f"resolution={getattr(model_config, 'resolution', 'unknown')}, "
            "mask_downsample_ratio="
            f"{getattr(model_config, 'mask_downsample_ratio', 'unknown')}, "
            f"num_queries={getattr(model_config, 'num_queries', 'unknown')}, "
            f"num_select={getattr(model_config, 'num_select', 'unknown')}"
        )
    return model


def create_yolo26_predictor(checkpoint_path: Path) -> Predictor:
    return YOLO26PredictorAdapter(checkpoint_path)


def create_model(
    checkpoint_path: Path,
    backend: Backend,
    model_size: str | None = None,
    task: Task = SEGMENTATION_TASK,
) -> Predictor:
    if backend == YOLO26_BACKEND:
        return create_yolo26_predictor(checkpoint_path)
    if backend == RFDETR_BACKEND:
        return RFDETRPredictorAdapter(
            load_rfdetr_checkpoint(checkpoint_path, model_size=model_size, task=task)
        )
    raise ValueError(f"Unsupported backend: {backend}")


def resolve_dataset_split(dataset_path: Path, split: str) -> tuple[Path, Path]:
    split_dir_map = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    split_dir_name = split_dir_map.get(split.lower())
    if split_dir_name is None:
        valid_splits = ", ".join(sorted(split_dir_map))
        raise typer.BadParameter(
            f"Unsupported split '{split}'. Expected one of: {valid_splits}"
        )

    split_dir = dataset_path / split_dir_name
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise typer.BadParameter(
            f"Missing annotation file for split '{split}': {annotation_path}"
        )
    return split_dir, annotation_path


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def score_image(row: dict) -> tuple:
    return (
        row["f1"],
        -row["false_negatives"],
        -row["false_positives"],
        row["file_name"],
    )


def draw_box_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    fill: tuple[int, int, int],
):
    left, top, right, bottom = draw.textbbox(position, text)
    padding = 2
    background = (
        left - padding,
        top - padding,
        right + padding,
        bottom + padding,
    )
    draw.rectangle(background, fill=fill)
    draw.text(position, text, fill=(255, 255, 255))


def label_y_inside_box(box: list[float]) -> float:
    return max(SUMMARY_BAR_HEIGHT + 4, box[1] + 4)


def label_y_above_box(box: list[float]) -> float:
    return max(SUMMARY_BAR_HEIGHT + 4, box[1] - 16)


def pred_color_for_class(class_name: str) -> tuple[int, int, int]:
    return CLASS_PRED_COLORS.get(class_name.lower(), DEFAULT_PRED_COLOR)


def render_overlay(
    image_path: Path,
    output_path: Path,
    gt_items: list,
    pred_items: list,
    summary_text: str,
):
    colors = {"gt": (46, 204, 113), "pred": DEFAULT_PRED_COLOR}
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    def blend_mask(mask, color, alpha=OVERLAY_ALPHA):
        if mask is None:
            return
        mask_indices = mask.astype(bool)
        if not mask_indices.any():
            return
        image_array[mask_indices] = (
            image_array[mask_indices] * (1 - alpha) + np.array(color) * alpha
        ).astype("uint8")

    for item in gt_items:
        blend_mask(item.get("mask"), colors["gt"])
    for item in pred_items:
        blend_mask(item.get("mask"), pred_color_for_class(item["class_name"]))

    annotated = Image.fromarray(image_array)
    draw = ImageDraw.Draw(annotated)

    for item in gt_items:
        box = item["bbox"]
        draw.rectangle(box, outline=colors["gt"], width=3)
        draw_box_label(
            draw,
            (box[0] + 4, label_y_above_box(box)),
            f"GT {item['class_name']}",
            colors["gt"],
        )

    for item in pred_items:
        box = item["bbox"]
        label = f"{item['class_name']} {item['score']:.2f}"
        pred_color = pred_color_for_class(item["class_name"])
        draw.rectangle(box, outline=pred_color, width=3)
        draw_box_label(draw, (box[0] + 4, label_y_inside_box(box)), label, pred_color)

    draw.rectangle((0, 0, annotated.width, SUMMARY_BAR_HEIGHT), fill=(0, 0, 0))
    draw.text((8, 5), summary_text, fill=(255, 255, 255))
    annotated.save(output_path, quality=OVERLAY_QUALITY)


def render_prediction_overlay(
    image_path: Path,
    output_path: Path,
    pred_items: list,
    summary_text: str,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    def blend_mask(mask, color, alpha=OVERLAY_ALPHA):
        if mask is None:
            return
        mask_indices = mask.astype(bool)
        if not mask_indices.any():
            return
        image_array[mask_indices] = (
            image_array[mask_indices] * (1 - alpha) + np.array(color) * alpha
        ).astype("uint8")

    for item in pred_items:
        blend_mask(item.get("mask"), pred_color_for_class(item["class_name"]))

    annotated = Image.fromarray(image_array)
    draw = ImageDraw.Draw(annotated)

    for item in pred_items:
        box = item["bbox"]
        label = f"{item['class_name']} {item['score']:.2f}"
        pred_color = pred_color_for_class(item["class_name"])
        draw.rectangle(box, outline=pred_color, width=3)
        draw_box_label(draw, (box[0] + 4, label_y_inside_box(box)), label, pred_color)

    draw.rectangle((0, 0, annotated.width, SUMMARY_BAR_HEIGHT), fill=(0, 0, 0))
    draw.text((8, 5), summary_text, fill=(255, 255, 255))

    try:
        annotated.save(output_path, quality=OVERLAY_QUALITY)
        return output_path
    except OSError:
        fallback_path = output_path.with_suffix(".png")
        annotated.save(fallback_path)
        return fallback_path


def render_fp_overlay(
    image_path: Path,
    output_path: Path,
    fp_items: list,
    class_name: str,
):
    """Render overlay showing only false positive predictions for a specific class."""
    color = (231, 76, 60)  # Red for false positives
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    def blend_mask(mask, alpha=OVERLAY_ALPHA):
        if mask is None:
            return
        mask_indices = mask.astype(bool)
        if not mask_indices.any():
            return
        image_array[mask_indices] = (
            image_array[mask_indices] * (1 - alpha) + np.array(color) * alpha
        ).astype("uint8")

    for item in fp_items:
        blend_mask(item.get("mask"))

    annotated = Image.fromarray(image_array)
    draw = ImageDraw.Draw(annotated)

    for item in fp_items:
        box = item["bbox"]
        label = f"FP {item['class_name']} {item['score']:.2f}"
        draw.rectangle(box, outline=color, width=3)
        draw_box_label(draw, (box[0] + 4, label_y_inside_box(box)), label, color)

    summary = f"False Positives for '{class_name}' - Count: {len(fp_items)}"
    draw.rectangle((0, 0, annotated.width, SUMMARY_BAR_HEIGHT), fill=(0, 0, 0))
    draw.text((8, 5), summary, fill=(255, 255, 255))
    annotated.save(output_path, quality=OVERLAY_QUALITY)


def render_fn_overlay(
    image_path: Path,
    output_path: Path,
    fn_items: list,
    class_name: str,
):
    """Render overlay showing only false negative ground truth for a specific class."""
    color = (46, 204, 113)  # Green for false negatives (missed GT)
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).copy()

    def blend_mask(mask, alpha=OVERLAY_ALPHA):
        if mask is None:
            return
        mask_indices = mask.astype(bool)
        if not mask_indices.any():
            return
        image_array[mask_indices] = (
            image_array[mask_indices] * (1 - alpha) + np.array(color) * alpha
        ).astype("uint8")

    for item in fn_items:
        blend_mask(item.get("mask"))

    annotated = Image.fromarray(image_array)
    draw = ImageDraw.Draw(annotated)

    for item in fn_items:
        box = item["bbox"]
        draw.rectangle(box, outline=color, width=3)
        draw_box_label(
            draw,
            (box[0] + 4, label_y_above_box(box)),
            f"FN {item['class_name']}",
            color,
        )

    summary = f"False Negatives for '{class_name}' - Count: {len(fn_items)}"
    draw.rectangle((0, 0, annotated.width, SUMMARY_BAR_HEIGHT), fill=(0, 0, 0))
    draw.text((8, 5), summary, fill=(255, 255, 255))
    annotated.save(output_path, quality=OVERLAY_QUALITY)


def write_csv(output_path: Path, rows: list[dict]):
    if not rows:
        output_path.write_text("")
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def iter_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_SUFFIXES
    )


def build_pred_items(
    detections: Any,
    category_names: dict[int, str],
    label_to_category_id: dict[int, int],
) -> list[dict]:
    pred_items = []
    for index in range(len(detections)):
        class_id = int(detections.class_id[index])
        if class_id not in label_to_category_id:
            raise ValueError(
                f"Prediction class ID {class_id} has no matching category."
            )
        mapped_category_id = label_to_category_id[class_id]
        bbox_xyxy = [float(v) for v in detections.xyxy[index].tolist()]
        score = float(detections.confidence[index])
        mask = (
            detections.mask[index].astype(bool, copy=False)
            if detections.mask is not None
            else None
        )
        pred_items.append(
            {
                "prediction_index": index,
                "category_id": mapped_category_id,
                "class_name": category_names.get(
                    mapped_category_id, f"class_{class_id}"
                ),
                "bbox": bbox_xyxy,
                "score": score,
                "mask": mask,
            }
        )
    return pred_items


def build_label_to_category_id(
    model: Predictor,
    categories: list[dict],
) -> dict[int, int]:
    category_name_to_id = {}
    for category in categories:
        name = category["name"]
        if name in category_name_to_id:
            raise ValueError(f"COCO class name {name!r} is ambiguous.")
        category_name_to_id[name] = category["id"]
    model_names = model.class_id_to_name
    if len(set(model_names.values())) != len(model_names):
        raise ValueError("Model class names are ambiguous.")
    return {
        class_id: category_name_to_id[name]
        for class_id, name in model_names.items()
        if name in category_name_to_id
    }


def resolve_roboflow_project(checkpoint_path: Path, project_name: str | None = None) -> str:
    if project_name:
        return project_name.split("/")[-1]

    # Try parent directory for metadata
    for parent in (checkpoint_path.parent, checkpoint_path.parent.parent):
        for name in ("metadata.json", "upload_metadata.json"):
            meta_path = parent / name
            if meta_path.exists():
                try:
                    import json
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    stored_name = meta.get("dataset_name")
                    if stored_name and stored_name.strip():
                        project_name = stored_name.strip()
                        typer.echo(f"Auto-resolved Roboflow project from metadata: {project_name}")
                        break
                except Exception:
                    pass
        if project_name:
            break

    if not project_name:
        import questionary
        import roboflow
        from questionary import Choice

        roboflow.login()
        rf = roboflow.Roboflow()
        workspace = rf.workspace()

        typer.echo("Fetching Roboflow projects...")
        project_list = workspace.projects()
        if not project_list:
            raise typer.BadParameter("No projects found in your Roboflow workspace.")

        def fetch_project(project_id: str):
            try:
                return workspace.project(project_id)
            except Exception:
                return None

        project_ids = [proj.split("/")[-1] for proj in project_list]
        projects = []
        with ThreadPoolExecutor() as executor:
            for proj in executor.map(fetch_project, project_ids):
                if proj is not None:
                    projects.append(proj)

        projects.sort(key=lambda p: getattr(p, "updated", 0), reverse=True)

        choices = [
            Choice(title=p.id, value=p.id.split("/")[-1])
            for p in projects
        ]

        project_name = questionary.select(
            "Select Roboflow project for upload",
            choices=choices,
            instruction="(Enter to select)",
        ).ask()

        if not project_name:
            raise typer.Abort()

    return project_name.split("/")[-1]


def generate_coco_json(
    image_path: Path,
    pred_items: list[dict],
    width: int,
    height: int,
) -> dict:
    categories = []
    category_set = set()
    annotations = []
    annotation_id = 1

    for item in pred_items:
        cat_id = item["category_id"]
        class_name = item["class_name"]

        if cat_id not in category_set:
            categories.append({
                "id": cat_id,
                "name": class_name,
                "supercategory": "none"
            })
            category_set.add(cat_id)

        bbox = item["bbox"]  # [x1, y1, x2, y2]
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        h = y_max - y_min

        anno = {
            "id": annotation_id,
            "image_id": 1,
            "category_id": cat_id,
            "bbox": [x_min, y_min, w, h],
            "area": w * h,
            "iscrowd": 0,
            "segmentation": []
        }

        mask = item["mask"]
        if mask is not None:
            import cv2
            import numpy as np
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            polygons = []
            for contour in contours:
                contour = contour.reshape(-1, 2)
                if len(contour) >= 3:
                    polygons.append(contour.astype(float).reshape(-1).tolist())
            anno["segmentation"] = polygons

        annotations.append(anno)
        annotation_id += 1

    coco_data = {
        "images": [
            {
                "id": 1,
                "file_name": image_path.name,
                "width": width,
                "height": height
            }
        ],
        "categories": categories,
        "annotations": annotations
    }
    return coco_data


def upload_single_image(
    project: Any,
    image_path: Path,
    pred_items: list[dict],
) -> tuple[str, bool, str | None]:
    """
    Helper function to upload a single image to Roboflow.
    Runs inside a background thread.
    Returns (image_name, success, error_message)
    """
    try:
        if pred_items:
            import tempfile

            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size

            coco_data = generate_coco_json(image_path, pred_items, width, height)

            with tempfile.TemporaryDirectory() as temp_dir:
                anno_path = Path(temp_dir) / "_annotations.coco.json"
                anno_path.write_text(json.dumps(coco_data), encoding="utf-8")

                results = project.upload(
                    image_path=str(image_path),
                    annotation_path=str(anno_path),
                    is_prediction=True,
                )
        else:
            results = project.upload(
                image_path=str(image_path),
                is_prediction=True,
            )
        if (
            len(results) != 1
            or not results[0]["image"]
            or (pred_items and results[0]["annotation"] is None)
        ):
            return str(image_path), False, "Roboflow did not confirm the upload."
        return str(image_path), True, None
    except Exception as exc:
        # The SDK may embed a request URL with credentials in the exception.
        return str(image_path), False, type(exc).__name__



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
        raise typer.BadParameter("Output directory must differ from the input directory.")
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
        resolved_project_name = resolve_roboflow_project(checkpoint_path, project_name)
        typer.echo(f"Connecting to Roboflow project: {resolved_project_name}")
        workspace = rf.workspace()
        project = workspace.project(resolved_project_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model(checkpoint_path, backend, model_size=model_size, task=task)

    category_names = model.class_id_to_name
    label_to_category_id = {index: index for index in category_names}

    fallback_count = 0
    saved_paths: list[Path] = []
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
                    saved_paths.append(saved_path)

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
        "file_count": len(saved_paths),
        "saved_paths": saved_paths,
        "output_dir": output_dir,
        "fallback_count": fallback_count,
        "upload_failure_count": upload_failures,
    }


def run_evaluation(
    dataset_path: Path,
    checkpoint_path: Path,
    split: str,
    threshold: float | None,
    iou_threshold: float,
    model: Predictor | None = None,
    backend: Backend = RFDETR_BACKEND,
    output_dir: Path | None = None,
    model_size: str | None = None,
    task: Task = SEGMENTATION_TASK,
) -> Path:
    split_dir, annotation_path = resolve_dataset_split(dataset_path, split)
    images_by_id, annotations_by_image, categories = load_coco_annotations(
        annotation_path
    )

    if output_dir is None:
        output_dir = (
            checkpoint_path.parent
            / f"{checkpoint_path.stem}_{split_dir.name}_evaluation"
        )
        try:
            output_dir.mkdir(parents=True)
        except FileExistsError:
            output_dir = Path(
                mkdtemp(prefix=f"{output_dir.name}-", dir=output_dir.parent)
            )
    else:
        output_dir.mkdir(parents=True)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Create directories for false positive and false negative overlays per class
    false_positives_dir = output_dir / "false_positives"
    false_negatives_dir = output_dir / "false_negatives"
    false_positives_dir.mkdir(parents=True, exist_ok=True)
    false_negatives_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = create_model(
            checkpoint_path,
            backend,
            model_size=model_size,
            task=task,
        )

    camera_profile = getattr(model, "camera_profile", False)
    operating_threshold = threshold
    camera_thresholds = None
    if operating_threshold is None:
        if camera_profile:
            from .camera_inference import CLASS_THRESHOLDS

            camera_thresholds = CLASS_THRESHOLDS
        else:
            operating_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    inference_threshold = (
        COCO_CONFIDENCE_FLOOR
        if operating_threshold is None
        else min(COCO_CONFIDENCE_FLOOR, operating_threshold)
    )

    category_names = {category["id"]: category["name"] for category in categories}
    label_to_category_id = build_label_to_category_id(model, categories)

    image_records = [images_by_id[image_id] for image_id in sorted(images_by_id)]

    bbox_results: list[dict] = []
    segm_results: list[dict] = []
    per_image_rows: list[dict] = []
    prediction_rows: list[dict] = []
    per_class_counts = {
        category["id"]: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_iou_sum": 0.0,
            "matched_iou_count": 0,
        }
        for category in categories
    }

    typer.echo(f"Evaluating {len(image_records)} images from {split_dir}")

    with typer.progressbar(image_records, label="Running inference") as progress:
        for image_record in progress:
            image_id = image_record["id"]
            relative_path = Path(image_record["file_name"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Invalid COCO image path: {relative_path}")
            image_path = split_dir / relative_path
            gt_annotations = annotations_by_image.get(image_id, [])
            # Lower the inference cutoff once, then apply the two report thresholds
            # independently to the same detections.
            detections = model.predict(str(image_path), threshold=inference_threshold)
            scores = detections.confidence
            cutoff = operating_threshold
            if camera_thresholds is not None:
                cutoff = np.asarray(camera_thresholds, dtype=scores.dtype)[
                    detections.class_id.astype(int, copy=False)
                ]
            at_operating_point = scores >= cutoff if camera_profile else scores > cutoff
            all_items = build_pred_items(
                detections, category_names, label_to_category_id
            )
            pred_items = []
            for item, for_coco, for_operating_metrics in zip(
                all_items,
                scores > COCO_CONFIDENCE_FLOOR,
                at_operating_point,
                strict=True,
            ):
                if for_operating_metrics:
                    pred_items.append(item)
                if not for_coco:
                    continue
                bbox_results.append(
                    {
                        "image_id": image_id,
                        "category_id": item["category_id"],
                        "bbox": xyxy_to_xywh(item["bbox"]),
                        "score": item["score"],
                    }
                )
                if task == SEGMENTATION_TASK and item["mask"] is not None:
                    segm_results.append(
                        {
                            "image_id": image_id,
                            "category_id": item["category_id"],
                            "segmentation": encode_binary_mask(item["mask"]),
                            "score": item["score"],
                        }
                    )

            gt_items = []
            image_height = int(image_record["height"])
            image_width = int(image_record["width"])
            for annotation in gt_annotations:
                gt_items.append(
                    {
                        "annotation_id": annotation["id"],
                        "category_id": annotation["category_id"],
                        "class_name": category_names[annotation["category_id"]],
                        "bbox": xywh_to_xyxy(annotation["bbox"]),
                        "mask": (
                            decode_segmentation(
                                annotation.get("segmentation"),
                                image_height,
                                image_width,
                            )
                            if task == SEGMENTATION_TASK
                            else None
                        ),
                    }
                )

            gt_matched: set[int] = set()
            image_matched_ious: list[float] = []
            true_positives = 0
            false_positives = 0

            fp_items_per_class = defaultdict(list)

            for pred_item in sorted(
                pred_items, key=lambda item: item["score"], reverse=True
            ):
                best_match_index = None
                best_iou = 0.0
                for gt_index, gt_item in enumerate(gt_items):
                    if (
                        gt_index in gt_matched
                        or gt_item["category_id"] != pred_item["category_id"]
                    ):
                        continue

                    if (
                        task == SEGMENTATION_TASK
                        and pred_item["mask"] is not None
                        and gt_item["mask"] is not None
                    ):
                        overlap = mask_iou(pred_item["mask"], gt_item["mask"])
                    else:
                        overlap = bbox_iou(pred_item["bbox"], gt_item["bbox"])

                    if overlap > best_iou:
                        best_iou = overlap
                        best_match_index = gt_index

                counts = per_class_counts[pred_item["category_id"]]
                if best_match_index is not None and best_iou >= iou_threshold:
                    gt_matched.add(best_match_index)
                    true_positives += 1
                    image_matched_ious.append(best_iou)
                    counts["tp"] += 1
                    counts["matched_iou_sum"] += best_iou
                    counts["matched_iou_count"] += 1
                    match_status = "tp"
                else:
                    false_positives += 1
                    counts["fp"] += 1
                    match_status = "fp"
                    # Track this FP item for the class
                    fp_items_per_class[pred_item["category_id"]].append(pred_item)

                prediction_rows.append(
                    {
                        "image_id": image_id,
                        "file_name": image_record["file_name"],
                        "class_name": pred_item["class_name"],
                        "category_id": pred_item["category_id"],
                        "score": pred_item["score"],
                        "best_iou": best_iou,
                        "match_status": match_status,
                    }
                )

            false_negatives = 0
            fn_items_per_class = defaultdict(list)
            for gt_index, gt_item in enumerate(gt_items):
                if gt_index not in gt_matched:
                    false_negatives += 1
                    per_class_counts[gt_item["category_id"]]["fn"] += 1
                    # Track this FN item for the class
                    fn_items_per_class[gt_item["category_id"]].append(gt_item)

            precision, recall, f1 = compute_metrics(
                true_positives, false_positives, false_negatives
            )
            mean_matched_iou = safe_divide(
                sum(image_matched_ious), len(image_matched_ious)
            )
            per_image_rows.append(
                {
                    "image_id": image_id,
                    "file_name": image_record["file_name"],
                    "ground_truth_count": len(gt_items),
                    "prediction_count": len(pred_items),
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_matched_iou": mean_matched_iou,
                }
            )
            output_path = overlays_dir / relative_path.with_name(
                f"{relative_path.name}_overlay.jpg"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_overlay(
                image_path,
                output_path,
                gt_items,
                pred_items,
                f"F1 {f1:.2f} | P {precision:.2f} | R {recall:.2f} | "
                f"TP {true_positives} FP {false_positives} FN {false_negatives}",
            )

            for folder, groups, suffix, render in (
                (false_positives_dir, fp_items_per_class, "fp", render_fp_overlay),
                (false_negatives_dir, fn_items_per_class, "fn", render_fn_overlay),
            ):
                for category_id, items in groups.items():
                    class_name = category_names[category_id]
                    class_folder = quote(class_name, safe="")
                    if class_folder in {"", ".", ".."}:
                        class_folder = f"class_{category_id}_{class_folder}"
                    target = folder / class_folder / relative_path.with_name(
                        f"{relative_path.name}_{suffix}.jpg"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    render(image_path, target, items, class_name)

    coco_gt = COCO(str(annotation_path))
    coco_metrics = {"bbox": run_coco_eval(coco_gt, bbox_results, "bbox")}
    if task == SEGMENTATION_TASK:
        coco_metrics["segm"] = run_coco_eval(coco_gt, segm_results, "segm")

    per_class_rows = []
    for category in categories:
        counts = per_class_counts[category["id"]]
        precision, recall, f1 = compute_metrics(
            counts["tp"], counts["fp"], counts["fn"]
        )
        mean_matched_iou = safe_divide(
            counts["matched_iou_sum"], counts["matched_iou_count"]
        )
        per_class_rows.append(
            {
                "category_id": category["id"],
                "class_name": category["name"],
                "true_positives": counts["tp"],
                "false_positives": counts["fp"],
                "false_negatives": counts["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": mean_matched_iou,
            }
        )

    per_image_rows.sort(key=score_image)

    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "prediction_metrics.csv", prediction_rows)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    total_tp = sum(row["true_positives"] for row in per_image_rows)
    total_fp = sum(row["false_positives"] for row in per_image_rows)
    total_fn = sum(row["false_negatives"] for row in per_image_rows)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "backend": backend,
        "task": task,
        "split": split_dir.name,
        "image_count": len(per_image_rows),
        "threshold": operating_threshold,
        "camera_profile": camera_profile,
        "coco_confidence_floor": COCO_CONFIDENCE_FLOOR,
        "iou_threshold": iou_threshold,
        "per_class": per_class_rows,
        "coco_metrics": coco_metrics,
        "overall": {
            "precision": safe_divide(total_tp, total_tp + total_fp),
            "recall": safe_divide(total_tp, total_tp + total_fn),
            "mean_f1": safe_divide(
                sum(row["f1"] for row in per_image_rows), len(per_image_rows)
            ),
        },
    }
    if camera_thresholds is not None:
        summary["class_thresholds"] = {
            name: camera_thresholds[class_id]
            for class_id, name in model.class_id_to_name.items()
        }
    write_evaluation_plots(
        output_dir=output_dir,
        per_class_rows=per_class_rows,
        per_image_rows=per_image_rows,
        prediction_rows=prediction_rows,
        coco_metrics=coco_metrics,
        coco_confidence_floor=COCO_CONFIDENCE_FLOOR,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return output_dir
