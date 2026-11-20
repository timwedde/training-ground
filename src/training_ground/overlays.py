from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

OVERLAY_ALPHA = 0.28
OVERLAY_QUALITY = 92
SUMMARY_BAR_HEIGHT = 24
GROUND_TRUTH_COLOR = (46, 204, 113)
PREDICTION_COLOR = (231, 76, 60)
CLASS_COLORS = {
    "trees": GROUND_TRUTH_COLOR,
    "poles": (155, 89, 182),
    "sticks": PREDICTION_COLOR,
}


class _Annotation(NamedTuple):
    box: list[float]
    mask: np.ndarray | None
    color: tuple[int, int, int]
    label: str
    label_offset: int


def _ground_truth_items(items: list[dict], prefix: str) -> list[_Annotation]:
    return [
        _Annotation(
            item["bbox"],
            item.get("mask"),
            GROUND_TRUTH_COLOR,
            f"{prefix} {item['class_name']}",
            -16,
        )
        for item in items
    ]


def _prediction_items(
    items: list[dict], *, false_positive: bool = False
) -> list[_Annotation]:
    annotations = []
    for item in items:
        name = item["class_name"]
        prefix = "FP " if false_positive else ""
        color = (
            PREDICTION_COLOR
            if false_positive
            else CLASS_COLORS.get(name.lower(), PREDICTION_COLOR)
        )
        annotations.append(
            _Annotation(
                item["bbox"],
                item.get("mask"),
                color,
                f"{prefix}{name} {item['score']:.2f}",
                4,
            )
        )
    return annotations


def _render(
    image_path: Path, annotations: list[_Annotation], summary: str
) -> Image.Image:
    with Image.open(image_path) as image:
        pixels = np.array(image.convert("RGB"))
    for annotation in annotations:
        if annotation.mask is not None:
            mask = np.asarray(annotation.mask, dtype=bool)
            pixels[mask] = (
                pixels[mask] * (1 - OVERLAY_ALPHA)
                + np.array(annotation.color) * OVERLAY_ALPHA
            ).astype("uint8")

    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for box, _, color, label, label_offset in annotations:
        draw.rectangle(box, outline=color, width=3)
        position = (box[0] + 4, max(SUMMARY_BAR_HEIGHT + 4, box[1] + label_offset))
        left, top, right, bottom = draw.textbbox(position, label)
        draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=color)
        draw.text(position, label, fill="white")

    draw.rectangle((0, 0, image.width, SUMMARY_BAR_HEIGHT), fill="black")
    draw.text((8, 5), summary, fill="white")
    return image


def render_overlay(
    image_path: Path,
    output_path: Path,
    gt_items: list[dict],
    pred_items: list[dict],
    summary_text: str,
) -> None:
    annotations = _ground_truth_items(gt_items, "GT") + _prediction_items(pred_items)
    _render(image_path, annotations, summary_text).save(
        output_path, quality=OVERLAY_QUALITY
    )


def render_prediction_overlay(
    image_path: Path, output_path: Path, pred_items: list[dict], summary_text: str
) -> Path:
    image = _render(image_path, _prediction_items(pred_items), summary_text)
    try:
        image.save(output_path, quality=OVERLAY_QUALITY)
        return output_path
    except OSError:
        fallback_path = output_path.with_suffix(".png")
        image.save(fallback_path)
        return fallback_path


def render_fp_overlay(
    image_path: Path, output_path: Path, items: list[dict], class_name: str
) -> None:
    summary = f"False Positives for '{class_name}' - Count: {len(items)}"
    _render(image_path, _prediction_items(items, false_positive=True), summary).save(
        output_path, quality=OVERLAY_QUALITY
    )


def render_fn_overlay(
    image_path: Path, output_path: Path, items: list[dict], class_name: str
) -> None:
    summary = f"False Negatives for '{class_name}' - Count: {len(items)}"
    _render(image_path, _ground_truth_items(items, "FN"), summary).save(
        output_path, quality=OVERLAY_QUALITY
    )
