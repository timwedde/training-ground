import asyncio
from pathlib import Path

import typer

from .analysis import analyze_dataset
from .evaluation import run_evaluation, run_prediction_directory
from .metrics_plotting import plot_training_metrics
from .training_backends import RFDETR_BACKEND, YOLO26_BACKEND, normalize_backend
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
def wizard():
    """
    Complete a full training run.
    """
    run_wizard()


@app.command()
def metrics(
    metrics_path: Path = typer.Argument(
        "runs/metrics.csv", exists=True, dir_okay=False
    ),
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
    dataset_path: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint_path: Path = typer.Argument(
        "runs/yolo26-nano/weights/best.pt", exists=True, dir_okay=False
    ),
    split: str = typer.Option("test", help="Dataset split: train, valid, or test."),
    threshold: float = typer.Option(0.5, help="Prediction confidence threshold."),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for TP/FP matching."),
    backend: str = typer.Option(
        YOLO26_BACKEND,
        "--backend",
        help="Model backend: yolo26 or rfdetr.",
    ),
    model_size: str | None = typer.Option(
        None,
        "--model-size",
        help="Model size fallback for legacy RF-DETR checkpoints without embedded model metadata.",
    ),
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
    )

    typer.echo(f"Evaluation complete. Artifacts written to {output_dir}")
    typer.echo("  - overlays")
    typer.echo("  - plots")
    typer.echo("  - per_image_metrics.csv")
    typer.echo("  - prediction_metrics.csv")
    typer.echo("  - per_class_metrics.csv")
    typer.echo("  - summary.json")


@app.command("predict-dir")
def predict_dir(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Argument(..., file_okay=False),
    checkpoint_path: Path = typer.Option(
        "best.pt",
        exists=True,
        dir_okay=False,
        help="Model checkpoint path.",
    ),
    threshold: float = typer.Option(0.5, help="Prediction confidence threshold."),
    backend: str = typer.Option(
        RFDETR_BACKEND,
        "--backend",
        help="Model backend: yolo26 or rfdetr.",
    ),
    model_size: str | None = typer.Option(
        None,
        "--model-size",
        help="Model size fallback for legacy RF-DETR checkpoints without embedded model metadata.",
    ),
    upload: bool = typer.Option(
        False,
        "--upload",
        help="Upload processed samples to Roboflow.",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Roboflow project ID for upload.",
    ),
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
        upload=upload,
        project_name=project,
    )

    typer.echo(
        f"Prediction complete. Wrote {result['image_count']} files to {output_dir}"
    )
    if result["fallback_count"]:
        typer.echo(
            f"{result['fallback_count']} files were saved as .png because the original format was not writable"
        )


@app.command()
def upload(
    runs_dir: Path = typer.Argument(
        "runs", exists=True, file_okay=False, help="Training runs directory."
    ),
    dataset_name: str | None = typer.Option(
        None,
        "--dataset-name",
        help="Dataset name to scope the uploaded run. Defaults to stored upload metadata or the runs directory name.",
    ),
    backend: str = typer.Option(
        YOLO26_BACKEND,
        "--backend",
        help="Training backend: yolo26 or rfdetr.",
    ),
):
    """
    Upload a training run to GCP.
    """
    artifacts, stored_metadata = resolve_training_artifacts(
        runs_dir, normalize_backend(backend)
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
            runs_dir=runs_dir,
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
