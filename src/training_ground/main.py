from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson
import typer

app = typer.Typer()


def fetch_project_info(data):
    workspace, project_id = data
    project = workspace.project(project_id)
    versions = project.versions()
    return project, versions


@app.command()
def analyze(dataset_path: Path):
    split_reports = []

    def percent(count: int, total: int) -> str:
        if total == 0:
            return "0.0%"
        return f"{(count / total) * 100:.1f}%"

    def resolve_annotation_files(path: Path) -> list[Path]:
        if path.is_file():
            return [path]

        direct_annotation = path / "_annotations.coco.json"
        if direct_annotation.exists():
            return [direct_annotation]

        split_names = ("train", "valid", "test")
        split_files = [
            path / split_name / "_annotations.coco.json"
            for split_name in split_names
            if (path / split_name / "_annotations.coco.json").exists()
        ]
        if split_files:
            return split_files

        return sorted(path.rglob("_annotations.coco.json"))

    def print_report(annotation_path: Path):
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

        annotation_counts = Counter(
            annotation["category_id"] for annotation in annotations
        )
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
            key=lambda cid: (
                -annotation_counts[cid],
                category_names.get(cid, str(cid)),
            ),
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
                f"{image_presence_counts[category_id]} images "
                f"({percent(image_presence_counts[category_id], len(images))})"
            )

        typer.echo("\nImage class combinations:")
        for class_ids, count in combination_counts.most_common():
            class_names = sorted(
                category_names.get(class_id, str(class_id)) for class_id in class_ids
            )
            typer.echo(
                f"  - {', '.join(class_names)}: {count} images ({percent(count, len(images))})"
            )

        split_reports.append(
            {
                "label": label,
                "annotation_path": annotation_path,
                "category_names": category_names,
                "report_category_ids": report_category_ids,
                "annotation_counts": annotation_counts,
                "annotation_total": len(annotations),
            }
        )

    def print_split_comparison():
        if len(split_reports) < 2:
            return

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
                weight = (
                    report["annotation_counts"][category_id] / total if total else 0.0
                )
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
                typer.echo(
                    f"    largest difference between {min_label} and {max_label}"
                )

    annotation_files = resolve_annotation_files(dataset_path)
    if not annotation_files:
        typer.echo(f"No COCO annotation files found under: {dataset_path}")
        raise typer.Exit(code=1)

    typer.echo(
        f"Analyzing {len(annotation_files)} annotation file(s) from {dataset_path}"
    )
    for annotation_path in annotation_files:
        print_report(annotation_path)
    print_split_comparison()


@app.command()
def wizard():
    import questionary
    import roboflow

    # fix torch multiprocessing sharing strategy
    import torch.multiprocessing as mp
    from questionary import Choice

    mp.set_sharing_strategy("file_system")

    roboflow.login()

    rf = roboflow.Roboflow()

    # List all projects for your workspace
    workspace = rf.workspace()
    projects = []
    with ThreadPoolExecutor() as executor:
        for project, versions in executor.map(
            fetch_project_info,
            [(workspace, project.split("/")[-1]) for project in workspace.projects()],
        ):
            projects.append((project, versions))

    projects.sort(key=lambda p: p[0].updated, reverse=True)

    project, versions = questionary.select(
        "Select your project",
        choices=[
            Choice(title=project.id, value=(project, versions))
            for project, versions in projects
        ],
    ).ask()

    version = questionary.select(
        "Select the dataset version",
        choices=[
            Choice(title=version.id.split("/")[-1], value=version)
            for version in versions
        ],
    ).ask()

    batch_size, grad_accum_steps = questionary.select(
        "Select the GPU VRAM",
        choices=[
            Choice(title="RTX  4080 (16GB)", value=(16, 1)),
            Choice(title="RTX  4090 (24GB)", value=(32, 1)),
            Choice(title="RTX  5090 (32GB)", value=(48, 1)),
            Choice(title="RTX A6000 (48GB)", value=(68, 1)),
        ],
    ).ask()

    dataset_path = f"./datasets/{version.id}"
    version.download(model_format="coco", location=dataset_path)

    from rfdetr import RFDETRSegNano

    model = RFDETRSegNano()

    model.train(
        dataset_dir=dataset_path,
        epochs=100,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=372,
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


if __name__ == "__main__":
    app()
