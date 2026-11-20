import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import typer

from .coco import load_coco_annotations
from .matching import (
    build_ground_truth,
    build_label_to_category_id,
    image_relative_path,
    match_predictions,
)
from .predictors import Predictor, YOLO26PredictorAdapter, build_pred_items
from .training_backends import Task

SETTINGS_REPORT = "recommended_settings.json"
CONFIDENCE_FLOOR = 0.001
CONFIDENCE_CANDIDATES = (
    CONFIDENCE_FLOOR,
    *(percent / 100 for percent in range(1, 100)),
)


class ConfidenceSweep:
    """Accumulate score-ordered matches without retaining images or masks."""

    def __init__(self):
        self.support = Counter()
        self.detections = defaultdict(list)

    def add_image(self, ground_truth: list[dict], matches: list[tuple]):
        self.support.update(item["category_id"] for item in ground_truth)
        for prediction, matched_index, _iou in matches:
            self.detections[prediction["category_id"]].append(
                (prediction["score"], matched_index is not None)
            )

    def best_global(self, *, inclusive: bool):
        return _best_confidence(
            [row for rows in self.detections.values() for row in rows],
            self.support.total(),
            inclusive=inclusive,
        )

    def best_by_class(
        self, mapping: dict[int, int], class_names: dict[int, str], *, inclusive: bool
    ):
        return [
            {
                "class_id": class_id,
                "name": class_names[class_id],
                "ground_truth_count": self.support[category_id],
                "best": _best_confidence(
                    self.detections[category_id],
                    self.support[category_id],
                    inclusive=inclusive,
                ),
            }
            for class_id, category_id in sorted(mapping.items())
        ]


def _best_confidence(rows: list[tuple[float, bool]], support: int, *, inclusive: bool):
    if not support or not rows:
        return None

    # Each higher cutoff retains a prefix of the score-ordered matches.
    ranked = sorted(rows, key=lambda row: row[0], reverse=True)
    scores = np.asarray([score for score, _ in ranked], dtype=np.float32)
    thresholds = np.asarray(CONFIDENCE_CANDIDATES, dtype=scores.dtype)
    counts = np.searchsorted(
        -scores, -thresholds, side="right" if inclusive else "left"
    )
    prefix_tp = np.cumsum([0, *(is_tp for _, is_tp in ranked)])
    best = None
    for confidence, prediction_count, tp in zip(
        CONFIDENCE_CANDIDATES, counts, prefix_tp[counts], strict=True
    ):
        if tp == 0:
            continue
        tp, prediction_count = int(tp), int(prediction_count)
        precision = tp / prediction_count
        f1 = 2 * tp / (prediction_count + support)
        rank = (f1, precision, confidence)
        if best is None or rank > best[0]:
            best = (
                rank,
                {
                    "confidence": confidence,
                    "precision": precision,
                    "recall": tp / support,
                    "f1": f1,
                    "true_positives": tp,
                    "false_positives": prediction_count - tp,
                    "false_negatives": support - tp,
                },
            )
    return best[1] if best else None


def write_settings_report(
    output_dir: Path,
    dataset_path: Path,
    model: Predictor,
    task: Task,
    iou_threshold: float,
    sweep: bool,
) -> dict:
    architecture = model.model_settings()
    can_sweep_nms = architecture.pop("can_sweep_nms", False)
    classes = [
        {"class_id": class_id, "name": name}
        for class_id, name in sorted(model.class_id_to_name.items())
    ]
    architecture.update(
        num_classes=len(classes),
        classes=classes,
        camera_profile=getattr(model, "camera_profile", False),
    )
    report = {"model": architecture}
    annotation_path = dataset_path / "valid" / "_annotations.coco.json"
    if not sweep:
        report["sweep"] = {"status": "disabled"}
    elif not annotation_path.is_file():
        report["sweep"] = {"status": "validation_split_missing"}
    else:
        report["sweep"] = _sweep_validation(
            annotation_path,
            model,
            task,
            iou_threshold,
            architecture.get("nms_iou"),
            can_sweep_nms,
        )
        best = report["sweep"]["recommended"]
        if best is not None:
            nms = (
                f", NMS IoU {best['nms_iou']:g}" if best["nms_iou"] is not None else ""
            )
            typer.echo(
                f"Validation recommendation: confidence {best['confidence']:g}"
                f"{nms}, F1 {best['f1']:.3f}"
            )
    (output_dir / SETTINGS_REPORT).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _sweep_validation(
    annotation_path: Path,
    model: Predictor,
    task: Task,
    iou_threshold: float,
    default_nms_iou: float | None,
    can_sweep_nms: bool,
) -> dict:
    image_records, annotations, categories = load_coco_annotations(annotation_path)
    category_names = {category["id"]: category["name"] for category in categories}
    mapping = build_label_to_category_id(model, categories)
    next_id = max(category_names, default=-1) + 1
    for class_id, name in sorted(model.class_id_to_name.items()):
        if class_id not in mapping:
            # A class absent from validation can still introduce false positives.
            mapping[class_id] = next_id
            category_names[next_id] = name
            next_id += 1

    nms_candidates = (
        sorted({0.3, 0.5, 0.7, default_nms_iou})
        if can_sweep_nms and default_nms_iou is not None
        else [default_nms_iou]
    )
    scores = {iou: ConfidenceSweep() for iou in nms_candidates}
    inclusive = getattr(model, "camera_profile", False)
    yolo = model if isinstance(model, YOLO26PredictorAdapter) else None

    with typer.progressbar(
        image_records.values(), label="Sweeping settings on validation"
    ) as images:
        for image in images:
            image_path = annotation_path.parent / image_relative_path(image)
            ground_truth = build_ground_truth(
                image, annotations.get(image["id"], []), category_names, task
            )
            tiles = (
                yolo.predict_tiles(str(image_path), CONFIDENCE_FLOOR)
                if yolo is not None and yolo.tiled
                else None
            )
            for iou, accumulator in scores.items():
                if tiles is not None:
                    detections = YOLO26PredictorAdapter.merge_tiles(tiles, iou)
                elif yolo is not None and can_sweep_nms:
                    detections = yolo.predict(
                        str(image_path), threshold=CONFIDENCE_FLOOR, iou=iou
                    )
                else:
                    detections = model.predict(
                        str(image_path), threshold=CONFIDENCE_FLOOR
                    )
                predictions = build_pred_items(detections, category_names, mapping)
                matches, _ = match_predictions(
                    predictions, ground_truth, task, iou_threshold
                )
                accumulator.add_image(ground_truth, matches)
                del detections, predictions, matches

    candidates = []
    for iou, accumulator in scores.items():
        best = accumulator.best_global(inclusive=inclusive)
        if best is not None:
            candidates.append({**best, "nms_iou": iou})

    recommended = max(
        candidates,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["nms_iou"] == default_nms_iou,
            row["confidence"],
        ),
        default=None,
    )
    by_class = []
    if recommended is not None:
        accumulator = scores[recommended["nms_iou"]]
        by_class = accumulator.best_by_class(
            mapping, model.class_id_to_name, inclusive=inclusive
        )
    return {
        "status": "complete" if recommended else "no_true_positives",
        "split": "valid",
        "image_count": len(image_records),
        "objective": "micro F1",
        "matching_iou": iou_threshold,
        "confidence_floor": CONFIDENCE_FLOOR,
        "confidence_grid_step": 0.01,
        "nms_iou_candidates": nms_candidates,
        "recommended": recommended,
        "per_class": by_class,
    }
