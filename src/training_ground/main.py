import asyncio
from pathlib import Path

import typer

from .analysis import analyze_dataset
from .evaluation import run_evaluation
from .metrics_plotting import plot_training_metrics
from .upload import upload_training_run
from .wizard import run_wizard

app = typer.Typer()


@app.command()
def metrics(
    metrics_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Directory for generated plots."
    ),
):
    typer.echo(f"Analyzed metrics from {metrics_path}")

    try:
        plots_dir, per_class_path = plot_training_metrics(metrics_path, output_dir)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"Plots written to {plots_dir}")
    typer.echo("  - training_summary.html")
    if per_class_path:
        typer.echo(f"  - {per_class_path.name}")

    typer.echo("\nValidation highlights:")


@app.command()
def evaluate(
    checkpoint_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    dataset_path: Path = typer.Argument(..., exists=True, file_okay=False),
    split: str = typer.Option("test", help="Dataset split: train, valid, or test."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory."
    ),
    model_type: str = typer.Option("rfdetr-seg-nano", help="Model architecture."),
    resolution: int = typer.Option(372, help="Inference resolution."),
    threshold: float = typer.Option(0.25, help="Prediction confidence threshold."),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for TP/FP matching."),
    max_overlay_images: int = typer.Option(
        100, help="Max overlay images to save (worst first)."
    ),
    limit: int | None = typer.Option(
        None, help="Cap images to evaluate for debugging."
    ),
):
    run_evaluation(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        split=split,
        output_dir=output_dir,
        model_type=model_type,
        resolution=resolution,
        threshold=threshold,
        iou_threshold=iou_threshold,
        max_overlay_images=max_overlay_images,
        limit=limit,
    )

    typer.echo("Evaluation complete. Artifacts written to {output_dir}")
    typer.echo("  - overlays")
    typer.echo("  - plots")
    typer.echo("  - per_image_metrics.csv")
    typer.echo("  - prediction_metrics.csv")
    typer.echo("  - per_class_metrics.csv")
    typer.echo("  - summary.json")


@app.command()
def analyze(dataset_path: Path):
    analyze_dataset(dataset_path)


@app.command()
def wizard():
    run_wizard()


@app.command()
def upload(
    runs_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Training runs directory."
    ),
):
    checkpoint_ema = runs_dir / "checkpoint_best_ema.pth"
    checkpoint_regular = runs_dir / "checkpoint_best_regular.pth"
    metrics_path = runs_dir / "metrics.csv"
    eval_dir = runs_dir / "checkpoint_best_ema_test_evaluation"
    onnx_path = runs_dir / "inference_model.onnx"

    if not checkpoint_ema.exists():
        typer.echo(f"EMA checkpoint not found: {checkpoint_ema}", err=True)
        raise typer.Exit(code=1)
    if not checkpoint_regular.exists():
        typer.echo(f"Regular checkpoint not found: {checkpoint_regular}", err=True)
        raise typer.Exit(code=1)
    if not metrics_path.exists():
        typer.echo(f"Metrics file not found: {metrics_path}", err=True)
        raise typer.Exit(code=1)
    if not eval_dir.exists():
        typer.echo(f"Evaluation directory not found: {eval_dir}", err=True)
        raise typer.Exit(code=1)
    if not onnx_path.exists():
        typer.echo(f"ONNX model not found: {onnx_path}", err=True)
        raise typer.Exit(code=1)

    run_id = asyncio.run(
        upload_training_run(
            runs_dir=runs_dir,
            checkpoint_ema_path=checkpoint_ema,
            checkpoint_regular_path=checkpoint_regular,
            metrics_path=metrics_path,
            eval_dir=eval_dir,
            onnx_path=onnx_path,
        )
    )
    typer.echo(f"Upload complete: run ID {run_id}")


if __name__ == "__main__":
    app()
