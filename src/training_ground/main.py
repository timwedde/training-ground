import asyncio
from pathlib import Path

import typer

from .analysis import analyze_dataset
from .evaluation import run_evaluation, run_prediction_directory
from .metrics_plotting import plot_training_metrics
from .upload import upload_training_run
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
        "runs/checkpoint_best_ema.pth", exists=True, dir_okay=False
    ),
    split: str = typer.Option("test", help="Dataset split: train, valid, or test."),
    threshold: float = typer.Option(0.5, help="Prediction confidence threshold."),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for TP/FP matching."),
):
    """
    Evaluate a model checkpoint on a dataset split.
    """
    run_evaluation(
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        split=split,
        threshold=threshold,
        iou_threshold=iou_threshold,
    )

    typer.echo("Evaluation complete. Artifacts written to {output_dir}")
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
        "runs/checkpoint_best_ema.pth",
        exists=True,
        dir_okay=False,
        help="Model checkpoint path.",
    ),
    threshold: float = typer.Option(0.5, help="Prediction confidence threshold."),
):
    """
    Run predictions on every image in a directory tree.
    """
    result = run_prediction_directory(
        input_dir=input_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        threshold=threshold,
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
):
    """
    Upload a training run to GCP.
    """
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
