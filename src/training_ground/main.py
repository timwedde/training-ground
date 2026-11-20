import asyncio
from pathlib import Path
from typing import Annotated

import typer

from .analysis import analyze_dataset
from .directory_prediction import run_prediction_directory
from .evaluation import run_evaluation
from .metrics_plotting import plot_training_metrics
from .settings_sweep import SETTINGS_REPORT
from .training_backends import (
    RFDETR_BACKEND,
    SEGMENTATION_TASK,
    YOLO26_BACKEND,
    normalize_backend,
    normalize_task,
)
from .upload import (
    artifact_files_for_training,
    build_training_metadata,
    resolve_dataset_name,
    resolve_training_artifacts,
    slugify_dataset_name,
    upload_artifact_bundle,
)
from .wizard import run_wizard

app = typer.Typer()


@app.command()
def wizard(
    batch_size: Annotated[
        int | None,
        typer.Option(min=1, help="Override automatic training batch sizing."),
    ] = None,
):
    """
    Complete a full training run.
    """
    run_wizard(batch_size=batch_size)


@app.command()
def metrics(
    metrics_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
):
    """
    Analyze and plot training metrics from a CSV file.
    """
    typer.echo(f"Analyzed metrics from {metrics_path}")

    plots_dir, per_class_path = plot_training_metrics(metrics_path)

    typer.echo(f"Plots written to {plots_dir}")
    typer.echo("  - training_summary.html")
    if per_class_path:
        typer.echo(f"  - {per_class_path.name}")


@app.command()
def analyze(dataset_path: Path):
    """
    Analyze a dataset and print summary statistics.
    """
    analyze_dataset(dataset_path)


@app.command()
def evaluate(
    dataset_path: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    checkpoint_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    split: Annotated[
        str, typer.Option(help="Dataset split: train, valid, or test.")
    ] = "test",
    threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Confidence for precision, recall, F1, and overlays. Defaults to STM class thresholds or 0.5. COCO AP uses a separate 0.001 floor.",
        ),
    ] = None,
    iou_threshold: Annotated[
        float, typer.Option(help="IoU threshold for TP/FP matching.")
    ] = 0.5,
    backend: Annotated[
        str, typer.Option(help="Model backend: yolo26 or rfdetr.")
    ] = YOLO26_BACKEND,
    model_size: Annotated[
        str | None,
        typer.Option(
            help="Model size fallback for RF-DETR checkpoints without embedded model metadata."
        ),
    ] = None,
    task: Annotated[
        str, typer.Option(help="Annotation type: boxes or segmentation.")
    ] = SEGMENTATION_TASK,
    settings_sweep: Annotated[
        bool,
        typer.Option(help="Tune confidence and supported NMS settings on validation."),
    ] = True,
):
    """
    Evaluate a model checkpoint on a dataset split.
    """
    output_dir = run_evaluation(
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        split=split,
        threshold=threshold,
        iou_threshold=iou_threshold,
        backend=normalize_backend(backend),
        model_size=model_size,
        task=normalize_task(task),
        settings_sweep=settings_sweep,
    )

    typer.echo(f"Evaluation complete. Artifacts written to {output_dir}")
    typer.echo("  - overlays")
    typer.echo("  - plots")
    typer.echo("  - per_image_metrics.csv")
    typer.echo("  - prediction_metrics.csv")
    typer.echo("  - per_class_metrics.csv")
    typer.echo("  - summary.json")
    typer.echo(f"  - {SETTINGS_REPORT}")


@app.command("predict-dir")
def predict_dir(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    checkpoint_path: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Model checkpoint path.")
    ] = Path("best.pt"),
    threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Override confidence thresholds. Defaults to STM class thresholds or 0.5.",
        ),
    ] = None,
    backend: Annotated[
        str, typer.Option(help="Model backend: yolo26 or rfdetr.")
    ] = RFDETR_BACKEND,
    model_size: Annotated[
        str | None,
        typer.Option(
            help="Model size fallback for RF-DETR checkpoints without embedded model metadata."
        ),
    ] = None,
    task: Annotated[
        str, typer.Option(help="Model task: boxes or segmentation.")
    ] = SEGMENTATION_TASK,
    upload: Annotated[
        bool, typer.Option("--upload", help="Upload processed samples to Roboflow.")
    ] = False,
    project: Annotated[
        str | None, typer.Option(help="Roboflow project ID for upload.")
    ] = None,
):
    """
    Run predictions on every image in a directory tree.
    """
    result = run_prediction_directory(
        input_dir=input_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        threshold=threshold,
        backend=normalize_backend(backend),
        model_size=model_size,
        task=normalize_task(task),
        upload=upload,
        project_name=project,
    )

    typer.echo(
        f"Prediction complete. Processed {result['image_count']} images; "
        f"wrote {result['file_count']} files to {output_dir}"
    )
    if result["fallback_count"]:
        typer.echo(
            f"{result['fallback_count']} files were saved as .png because the original format was not writable"
        )
    if result["upload_failure_count"]:
        raise typer.Exit(code=1)


@app.command()
def upload(
    runs_dir: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, help="Directory for the training run."
        ),
    ],
    dataset_name: Annotated[
        str | None,
        typer.Option(
            help="Dataset name for the upload. Defaults to run metadata or the run directory name."
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(
            help="Training backend: yolo26 or rfdetr. Stored metadata takes precedence."
        ),
    ] = YOLO26_BACKEND,
    task: Annotated[
        str,
        typer.Option(
            help="Training task: boxes or segmentation. Stored metadata takes precedence."
        ),
    ] = SEGMENTATION_TASK,
):
    """
    Upload a training run to GCP.
    """
    artifacts, stored_metadata = resolve_training_artifacts(
        runs_dir, normalize_backend(backend), normalize_task(task)
    )
    for path in (
        artifacts.primary_checkpoint_path,
        artifacts.metrics_path,
        artifacts.eval_dir,
        artifacts.onnx_path,
    ):
        if not path.exists():
            typer.echo(f"Required artifact not found: {path}", err=True)
            raise typer.Exit(code=1)
    if (
        artifacts.secondary_checkpoint_path
        and not artifacts.secondary_checkpoint_path.exists()
    ):
        typer.echo(
            f"Required artifact not found: {artifacts.secondary_checkpoint_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        resolved_dataset_name = resolve_dataset_name(runs_dir, dataset_name)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    metadata = build_training_metadata(resolved_dataset_name, artifacts)
    if stored_metadata:
        metadata.update(
            {
                key: value
                for key, value in stored_metadata.items()
                if metadata.get(key) is None and value is not None
            }
        )

    run_id = asyncio.run(
        upload_artifact_bundle(
            dataset_name=resolved_dataset_name,
            artifact_files=artifact_files_for_training(artifacts),
            metadata=metadata,
        )
    )
    typer.echo(
        f"Upload complete: run ID {slugify_dataset_name(resolved_dataset_name)}-{run_id}"
    )


if __name__ == "__main__":
    app()
