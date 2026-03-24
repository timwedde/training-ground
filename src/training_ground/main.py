import csv
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


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    return float(stripped)


def _best_row(
    rows: list[dict[str, float]], metric: str, maximize: bool = True
) -> dict[str, float] | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[metric]) if maximize else min(
        candidates, key=lambda row: row[metric]
    )


@app.command()
def metrics(
    metrics_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to write generated plots to. Defaults to a sibling folder.",
    ),
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    if output_dir is None:
        output_dir = metrics_path.parent / f"{metrics_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, float]] = []
    val_rows: list[dict[str, float]] = []
    lr_rows: list[dict[str, float]] = []

    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for raw_row in reader:
            row = {name: _parse_float(raw_row.get(name)) for name in fieldnames}
            step = row.get("step")
            epoch = row.get("epoch")
            if step is None or epoch is None:
                continue

            row["step"] = int(step)
            row["epoch"] = int(epoch)

            if row.get("train/loss") is not None:
                train_rows.append(row)
            if any(row.get(metric) is not None for metric in ("val/mAP_50", "val/loss")):
                val_rows.append(row)
            if row.get("train/lr") is not None:
                lr_rows.append(row)

    if not any((train_rows, val_rows, lr_rows)):
        typer.echo(f"No plottable metrics found in {metrics_path}")
        raise typer.Exit(code=1)

    def add_series(
        fig,
        row_index: int,
        col_index: int,
        rows: list[dict[str, float]],
        metrics_to_plot: list[str],
    ) -> bool:
        plotted = False
        for metric in metrics_to_plot:
            points = [
                (row["step"], row[metric])
                for row in rows
                if row.get(metric) is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points, strict=False)
            fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="lines+markers",
                    name=metric,
                    legendgroup=metric,
                ),
                row=row_index,
                col=col_index,
            )
            plotted = True
        return plotted

    summary_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training losses",
            "Validation overview",
            "Validation quality",
            "Learning rate",
        ),
    )
    training_plotted = add_series(
        summary_fig,
        1,
        1,
        train_rows,
        ["train/loss", "train/loss_ce", "train/loss_bbox", "train/loss_giou"],
    )
    validation_overview_plotted = add_series(
        summary_fig,
        1,
        2,
        val_rows,
        ["val/loss", "val/mAP_50", "val/mAP_50_95", "val/F1"],
    )
    validation_quality_plotted = add_series(
        summary_fig,
        2,
        1,
        val_rows,
        [
            "val/precision",
            "val/recall",
            "val/mAR",
            "val/ema_mAP_50",
            "val/ema_mAP_50_95",
        ],
    )
    lr_plotted = add_series(
        summary_fig,
        2,
        2,
        lr_rows,
        ["train/lr", "train/lr_max", "train/lr_min"],
    )
    for index, plotted in enumerate(
        (
            training_plotted,
            validation_overview_plotted,
            validation_quality_plotted,
            lr_plotted,
        ),
        start=1,
    ):
        row_index = 1 if index <= 2 else 2
        col_index = 1 if index in (1, 3) else 2
        summary_fig.update_xaxes(title_text="step", row=row_index, col=col_index)
        if not plotted:
            summary_fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{index} domain",
                yref=f"y{index} domain",
                text="No data",
                showarrow=False,
            )

    summary_fig.update_layout(
        height=900,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
    )
    summary_path = output_dir / "training_summary.html"
    summary_fig.write_html(summary_path, include_plotlyjs="cdn")

    per_class_metrics = sorted(
        name for name in fieldnames if name.startswith("val/AP/")
    )
    per_class_path = None
    if per_class_metrics and val_rows:
        per_class_fig = go.Figure()
        for metric in per_class_metrics:
            points = [
                (row["step"], row[metric])
                for row in val_rows
                if row.get(metric) is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points, strict=False)
            per_class_fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="lines+markers",
                    name=metric,
                )
            )

        per_class_fig.update_layout(
            title="Per-class AP",
            xaxis_title="step",
            yaxis_title="value",
            height=600,
            width=1400,
            template="plotly_white",
            hovermode="x unified",
        )
        per_class_path = output_dir / "per_class_ap.html"
        per_class_fig.write_html(per_class_path, include_plotlyjs="cdn")

    best_map50 = _best_row(val_rows, "val/mAP_50")
    best_ema_map50 = _best_row(val_rows, "val/ema_mAP_50")
    best_f1 = _best_row(val_rows, "val/F1")
    lowest_val_loss = _best_row(val_rows, "val/loss", maximize=False)
    final_train = max(train_rows, key=lambda row: row["step"]) if train_rows else None

    typer.echo(f"Analyzed metrics from {metrics_path}")
    typer.echo(f"Plots written to {output_dir}")
    typer.echo(f"  - {summary_path.name}")
    if per_class_path is not None:
        typer.echo(f"  - {per_class_path.name}")

    typer.echo("\nValidation highlights:")
    if best_map50 is not None:
        typer.echo(
            f"  - best val/mAP_50: {best_map50['val/mAP_50']:.4f} "
            f"(epoch {best_map50['epoch']}, step {best_map50['step']})"
        )
    if best_ema_map50 is not None:
        typer.echo(
            f"  - best val/ema_mAP_50: {best_ema_map50['val/ema_mAP_50']:.4f} "
            f"(epoch {best_ema_map50['epoch']}, step {best_ema_map50['step']})"
        )
    if best_f1 is not None:
        typer.echo(
            f"  - best val/F1: {best_f1['val/F1']:.4f} "
            f"(epoch {best_f1['epoch']}, step {best_f1['step']})"
        )
    if lowest_val_loss is not None:
        typer.echo(
            f"  - lowest val/loss: {lowest_val_loss['val/loss']:.4f} "
            f"(epoch {lowest_val_loss['epoch']}, step {lowest_val_loss['step']})"
        )
    if final_train is not None and final_train.get("train/loss") is not None:
        typer.echo(
            f"  - final train/loss: {final_train['train/loss']:.4f} "
            f"(epoch {final_train['epoch']}, step {final_train['step']})"
        )


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
