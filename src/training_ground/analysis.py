from collections import Counter, defaultdict
from pathlib import Path

import orjson
import typer


def percent(count: int, total: int) -> str:
    return f"{(count / total) * 100:.1f}%" if total else "0.0%"


def resolve_annotation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    direct = path / "_annotations.coco.json"
    if direct.exists():
        return [direct]

    return sorted(path.rglob("_annotations.coco.json"))


def analyze_dataset(dataset_path: Path):
    annotation_files = resolve_annotation_files(dataset_path)
    if not annotation_files:
        typer.echo(f"No COCO annotation files found under: {dataset_path}")
        raise typer.Exit(code=1)

    typer.echo(
        f"Analyzing {len(annotation_files)} annotation file(s) from {dataset_path}"
    )
    split_reports = []

    for annotation_path in annotation_files:
        report = _analyze_split(annotation_path)
        split_reports.append(report)

    if len(split_reports) >= 2:
        _compare_splits(split_reports)


def _analyze_split(annotation_path: Path) -> dict:
    dataset = orjson.loads(annotation_path.read_bytes())
    images = dataset.get("images", [])
    annotations = dataset.get("annotations", [])
    categories = dataset.get("categories", [])

    category_names = {category["id"]: category["name"] for category in categories}
    used_category_ids = {annotation["category_id"] for annotation in annotations}
    report_category_ids = sorted(
        category["id"]
        for category in categories
        if category["id"] in used_category_ids
        or category.get("supercategory") != "none"
    )

    annotation_counts = Counter(annotation["category_id"] for annotation in annotations)
    image_class_sets: dict[int, set[int]] = defaultdict(set)
    for annotation in annotations:
        image_class_sets[annotation["image_id"]].add(annotation["category_id"])

    image_ids = {image["id"] for image in images}
    images_with_annotations = len(image_class_sets)
    unlabeled_images = len(image_ids) - images_with_annotations

    image_presence_counts = Counter()
    combination_counts = Counter()

    for image in images:
        class_ids = frozenset(image_class_sets.get(image["id"], set()))
        if not class_ids:
            continue
        for class_id in class_ids:
            image_presence_counts[class_id] += 1
        combination_counts[class_ids] += 1

    label = annotation_path.parent.name
    typer.echo(f"\n[{label}] {annotation_path}")
    typer.echo(
        f"Images: {len(images)} | Annotated images: {images_with_annotations} | "
        f"Unlabeled images: {unlabeled_images} | Annotations: {len(annotations)}"
    )

    typer.echo("\nAnnotation counts by class:")
    for category_id in sorted(
        report_category_ids,
        key=lambda cid: (-annotation_counts[cid], category_names.get(cid, str(cid))),
    ):
        typer.echo(
            f"  - {category_names.get(category_id, str(category_id))}: "
            f"{annotation_counts[category_id]} ({percent(annotation_counts[category_id], len(annotations))})"
        )

    typer.echo("\nImages containing each class:")
    for category_id in sorted(
        report_category_ids,
        key=lambda cid: (
            -image_presence_counts[cid],
            category_names.get(cid, str(cid)),
        ),
    ):
        typer.echo(
            f"  - {category_names.get(category_id, str(category_id))}: "
            f"{image_presence_counts[category_id]} images ({percent(image_presence_counts[category_id], len(images))})"
        )

    typer.echo("\nImage class combinations:")
    for class_ids, count in combination_counts.most_common():
        class_names = sorted(
            category_names.get(class_id, str(class_id)) for class_id in class_ids
        )
        typer.echo(
            f"  - {', '.join(class_names)}: {count} images ({percent(count, len(images))})"
        )

    return {
        "label": label,
        "annotation_path": annotation_path,
        "category_names": category_names,
        "report_category_ids": report_category_ids,
        "annotation_counts": annotation_counts,
        "annotation_total": len(annotations),
    }


def _compare_splits(split_reports: list[dict]):
    typer.echo("\nSplit class weight comparison:")

    all_category_ids = sorted(
        {
            category_id
            for report in split_reports
            for category_id in report["report_category_ids"]
        }
    )

    for category_id in all_category_ids:
        class_name = next(
            (
                report["category_names"].get(category_id)
                for report in split_reports
                if category_id in report["category_names"]
            ),
            str(category_id),
        )
        weights = []
        for report in split_reports:
            total = report["annotation_total"]
            weight = report["annotation_counts"][category_id] / total if total else 0.0
            weights.append((report["label"], weight))

        min_label, min_weight = min(weights, key=lambda item: item[1])
        max_label, max_weight = max(weights, key=lambda item: item[1])
        spread = max_weight - min_weight
        status = "OK" if spread <= 0.05 else "WARNING"

        typer.echo(
            f"  - {class_name}: "
            + ", ".join(f"{label}={weight * 100:.1f}%" for label, weight in weights)
            + f" | spread={spread * 100:.1f}pp | {status}"
        )
        if status == "WARNING":
            typer.echo(f"    largest difference between {min_label} and {max_label}")
