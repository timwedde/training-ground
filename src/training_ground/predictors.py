from itertools import batched
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import typer

from .tiling import (
    DEFAULT_MERGE_IOU,
    TILE_BATCH_SIZE,
    TILE_PADDING_VALUE,
    merge_tile_boxes,
    pad_to_tile,
    restore_boxes,
    tile_windows,
)
from .training_backends import (
    BOX_TASK,
    RFDETR_BACKEND,
    SEGMENTATION_TASK,
    YOLO26_BACKEND,
    YOLO26_IMAGE_SIZE,
    Backend,
    Task,
    rfdetr_models_for_task,
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class Predictor(Protocol):
    class_id_to_name: dict[int, str]

    def model_settings(self) -> dict: ...

    def predict(self, image_path: str, threshold: float | None): ...


class YOLO26PredictorAdapter:
    def __init__(self, checkpoint_path: Path, *, tiled: bool = False):
        from ultralytics import YOLO

        typer.echo(f"Loading YOLO26 model from checkpoint: {checkpoint_path}")
        self._model = YOLO(str(checkpoint_path))
        self.tiled = tiled
        if tiled and self._model.task != "detect":
            raise typer.BadParameter("Tiling requires a YOLO26 detection checkpoint.")
        self.checkpoint_name = checkpoint_path.name
        names = getattr(self._model, "names", None) or getattr(
            self._model.model, "names", {}
        )
        if isinstance(names, dict):
            self.class_id_to_name = {
                int(index): str(name) for index, name in names.items()
            }
        else:
            self.class_id_to_name = dict(enumerate(map(str, names)))

    def model_settings(self) -> dict:
        from ultralytics.utils import DEFAULT_CFG, PREDICT_IOU

        layers = getattr(self._model.model, "model", None)
        head = layers[-1] if layers is not None else None
        predictor = self._model.predictor
        inference_model = predictor.model if predictor is not None else None
        end_to_end = (
            inference_model.end2end
            if inference_model is not None
            else head is not None and getattr(head, "cv2", None) is None
        )
        settings = {
            "variant": (
                self._model.model.yaml.get("yaml_file", self.checkpoint_name)
                if head is not None
                else self.checkpoint_name
            ),
            "image_size": [YOLO26_IMAGE_SIZE, YOLO26_IMAGE_SIZE],
            "max_detections": (
                int(predictor.args.max_det)
                if predictor is not None
                else DEFAULT_CFG.max_det
            ),
            "nms_iou": None if end_to_end else PREDICT_IOU,
            "can_sweep_nms": head is not None and not end_to_end,
        }
        training_size = getattr(self._model.model, "args", {}).get("imgsz")
        if isinstance(training_size, int):
            settings["training_image_size"] = [training_size, training_size]
        elif isinstance(training_size, (list, tuple)) and len(training_size) == 2:
            settings["training_image_size"] = list(training_size)
        if self._model.task == "segment" and getattr(head, "proto", None) is not None:
            # Prototype resolution for the nominal square input. Returned masks
            # are resized to the source image by the predictor.
            upsample = head.proto.upsample.stride
            feature_size = YOLO26_IMAGE_SIZE // int(head.stride[0])
            settings["mask_prototype_size"] = [
                feature_size * int(stride) for stride in upsample
            ]
        if self.tiled:
            settings.update(nms_iou=DEFAULT_MERGE_IOU, can_sweep_nms=True)
            settings["tiling"] = {
                "tile_size": YOLO26_IMAGE_SIZE,
                "stride": round(YOLO26_IMAGE_SIZE * 0.8),
                "grid": "row_major_snap_to_edge",
                "padding": {"edges": ["right", "bottom"], "value": TILE_PADDING_VALUE},
                "input_layout": "NCHW",
                "input_color": "RGB",
                "input_scale": 1 / 255,
                "output_columns": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
                "merge": "class_aware_cross_tile_nms",
                "max_detections_scope": "per_tile",
                "default_confidence": DEFAULT_CONFIDENCE_THRESHOLD,
            }
        return settings

    def predict(
        self, image_path: str, threshold: float | None, *, iou: float | None = None
    ):
        confidence_cutoff = (
            DEFAULT_CONFIDENCE_THRESHOLD if threshold is None else threshold
        )
        if self.tiled:
            return self.merge_tiles(
                self.predict_tiles(image_path, confidence_cutoff), iou
            )
        results = self._model.predict(
            source=image_path,
            conf=confidence_cutoff,
            verbose=False,
            imgsz=YOLO26_IMAGE_SIZE,
            retina_masks=True,
            iou=iou,
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

    def predict_tiles(self, image_path: str, threshold: float) -> list[np.ndarray]:
        from ultralytics.utils.patches import imread

        image = imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        detections = []
        for windows in batched(tile_windows(*image.shape[:2]), TILE_BATCH_SIZE):
            crops = [pad_to_tile(image[y1:y2, x1:x2]) for x1, y1, x2, y2 in windows]
            results = self._model.predict(
                source=crops,
                imgsz=YOLO26_IMAGE_SIZE,
                conf=threshold,
                rect=False,
                tile=False,
                nms=False,
                verbose=False,
            )
            detections.extend(
                restore_boxes(result.boxes.data.detach().cpu().numpy(), window)
                for result, window in zip(results, windows, strict=True)
            )
        return detections

    @staticmethod
    def merge_tiles(tiles: list[np.ndarray], iou: float | None) -> DetectionBatch:
        boxes = merge_tile_boxes(tiles, DEFAULT_MERGE_IOU if iou is None else iou)
        return DetectionBatch(
            boxes[:, :4], boxes[:, 4], boxes[:, 5].astype(np.int32), None
        )


class RFDETRPredictorAdapter:
    def __init__(self, model):
        from .camera_inference import matches_camera_model

        self.camera_profile = matches_camera_model(model)
        self.model_config = model.model_config
        self.variant = type(model).__name__
        self.num_class_slots = int(model.model.model.class_embed.out_features)
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

    def model_settings(self) -> dict:
        config = self.model_config
        ratio = config.mask_downsample_ratio if config.segmentation_head else None
        return {
            "variant": self.variant,
            "image_size": [config.resolution, config.resolution],
            "mask_size": [config.resolution // ratio] * 2 if ratio else None,
            "mask_downsample_ratio": ratio,
            "num_queries": config.num_queries,
            "num_select": config.num_select,
            "num_class_slots": self.num_class_slots,
            "nms_iou": None,
        }

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


def create_model(
    checkpoint_path: Path,
    backend: Backend,
    model_size: str | None = None,
    task: Task = SEGMENTATION_TASK,
    *,
    tiled: bool = False,
) -> Predictor:
    if tiled and (backend != YOLO26_BACKEND or task != BOX_TASK):
        raise typer.BadParameter("Tiling is available only for YOLO26 box detection.")
    if backend == YOLO26_BACKEND:
        return YOLO26PredictorAdapter(checkpoint_path, tiled=tiled)
    if backend == RFDETR_BACKEND:
        return RFDETRPredictorAdapter(
            load_rfdetr_checkpoint(checkpoint_path, model_size=model_size, task=task)
        )
    raise ValueError(f"Unsupported backend: {backend}")


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
