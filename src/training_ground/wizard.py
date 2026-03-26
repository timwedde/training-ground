from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import questionary
import typer
from questionary import Choice

from .evaluation import run_evaluation
from .metrics_plotting import plot_training_metrics
from .upload import upload_training_run


def fetch_project_info(data):
    workspace, project_id = data
    project = workspace.project(project_id)
    return project, project.versions()


def run_wizard():
    import roboflow
    import torch.multiprocessing as mp

    mp.set_sharing_strategy("file_system")

    roboflow.login()
    rf = roboflow.Roboflow()

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
        choices=[Choice(title=p.id, value=(p, v)) for p, v in projects],
    ).ask()

    version = questionary.select(
        "Select the dataset version",
        choices=[Choice(title=v.id.split("/")[-1], value=v) for v in versions],
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

    # Import here to avoid slow startup when just using other CLI commands
    from rfdetr.detr import RFDETRSegNano

    model = RFDETRSegNano(resolution=372)
    try:
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
    except Exception as e:
        typer.echo(f"\n❌ Training failed: {e}", err=True)
        raise

    # Training completed successfully - proceed with metrics, evaluation, and upload
    typer.echo("\n✅ Training completed successfully!")

    runs_dir = Path("runs")

    # Export model to ONNX (required)
    typer.echo("\n📦 Exporting model to ONNX format...")
    model.export(simplify=True)
    onnx_path = runs_dir / "model.onnx"
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export did not produce expected file: {onnx_path}")
    typer.echo("  ✓ ONNX export complete")

    # Generate metrics plots
    typer.echo("\n📊 Generating training metrics plots...")
    metrics_path = runs_dir / "metrics.csv"
    try:
        plot_training_metrics(metrics_path, runs_dir / "metrics_plots")
        typer.echo("  ✓ Metrics plots generated")
    except Exception as e:
        typer.echo(f"  ⚠️  Warning: Failed to generate metrics plots: {e}")

    # Run evaluation on best EMA checkpoint
    typer.echo("\n🔍 Running evaluation on best EMA checkpoint...")
    checkpoint_ema = runs_dir / "checkpoint_best_ema.pth"
    try:
        run_evaluation(
            checkpoint_path=checkpoint_ema,
            dataset_path=Path(dataset_path),
            split="test",
            output_dir=None,
            model_type="rfdetr-seg-nano",
            resolution=372,
            threshold=0.25,
            iou_threshold=0.5,
            max_overlay_images=100,
            limit=None,
        )
        typer.echo("  ✓ Evaluation complete")
        eval_dir = runs_dir / f"{checkpoint_ema.stem}_test_evaluation"
    except Exception as e:
        typer.echo(f"  ⚠️  Warning: Evaluation failed: {e}")
        eval_dir = None

    # Upload to GCS if evaluation succeeded
    if eval_dir and eval_dir.exists():
        typer.echo("\n☁️  Preparing to upload training run to GCS...")
        try:
            import asyncio

            checkpoint_regular = runs_dir / "checkpoint_best_regular.pth"

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
            typer.echo(
                f"\n🎉 Training run {run_id} complete and uploaded successfully!"
            )
        except Exception as e:
            typer.echo(f"\n❌ Upload failed: {e}", err=True)
            typer.echo(
                "   Your training artifacts are still available locally in the 'runs/' directory."
            )
    else:
        typer.echo("\n⚠️  Skipping upload - evaluation did not produce output.")
        typer.echo(
            "   Your training artifacts are available locally in the 'runs/' directory."
        )
