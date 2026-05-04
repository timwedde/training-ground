import math
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import onnx
import onnx.helper
import questionary
import typer
from questionary import Choice


# patch ONNX export
def _float32_to_bfloat16(fval: float, truncate: bool = False) -> int:
    """
    Converts a float32 value to a bfloat16 (as int).
    Restores compatibility for onnx-graphsurgeon.
    """
    # ival is the integer representation of the float32 bits
    ival = int.from_bytes(struct.pack("<f", fval), "little")

    if truncate:
        return ival >> 16

    # NaN requires at least 1 significand bit set
    if math.isnan(fval):
        return 0x7FC0  # sign=0, exp=all-ones, sig=0b1000000

    # Drop bottom 16-bits and round remaining bits using round-to-nearest-even
    rounded = ((ival >> 16) & 1) + 0x7FFF
    return (ival + rounded) >> 16


# Check if the function is missing and inject it
if not hasattr(onnx.helper, "float32_to_bfloat16"):
    onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16

from .evaluation import run_evaluation
from .metrics_plotting import plot_training_metrics
from .upload import (
    slugify_dataset_name,
    upload_training_run,
    write_upload_metadata,
)

DEFAULT_TRAINING_RESOLUTION = 372


def fetch_project_info(data):
    workspace, project_id = data
    project = workspace.project(project_id)
    return project, project.versions()


def _patch_rfdetr_training_pretrain_loader():
    """
    Patch RF-DETR's training module to support custom input resolutions again.

    Why this exists:
    - `model.train(resolution=...)` is still a supported RF-DETR API.
    - In the currently installed `rfdetr`, the training-specific pretrained
      weight loader does not interpolate backbone positional embeddings before
      `load_state_dict(...)`.
    - That breaks training as soon as we request a non-default resolution such
      as 372, because the pretrained checkpoint was saved at 312 and its
      positional embedding tensor no longer matches the resized model.

    What this patch does:
    - Replaces `RFDETRModelModule._load_pretrain_weights` with the shared
      `rfdetr.models.weights.load_pretrain_weights(...)` helper.
    - That helper already performs the required positional-embedding
      interpolation, so custom-resolution training works again without changing
      RF-DETR's public training API.

    Scope:
    - This is a narrow runtime patch applied only by this training wizard.
    - It leaves inference/export code paths untouched.
    """
    from rfdetr.models.weights import load_pretrain_weights
    from rfdetr.training.module_model import RFDETRModelModule

    if getattr(RFDETRModelModule._load_pretrain_weights, "__name__", "") == (
        "_load_pretrain_weights_with_pe_interpolation"
    ):
        return

    def _load_pretrain_weights_with_pe_interpolation(self) -> None:
        load_pretrain_weights(self.model, self.model_config)

    RFDETRModelModule._load_pretrain_weights = _load_pretrain_weights_with_pe_interpolation


def run_wizard():
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import roboflow

    roboflow.login()
    rf = roboflow.Roboflow()

    typer.echo("Fetching projects...")
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
        "Select project",
        choices=[Choice(title=p.id, value=(p, v)) for p, v in projects],
    ).ask()

    version = questionary.select(
        "Select dataset version",
        choices=[Choice(title=v.id.split("/")[-1], value=v) for v in versions],
    ).ask()

    batch_size_selection = questionary.select(
        "Select GPU VRAM",
        choices=[
            Choice(title="RTX  4080 (16GB)", value=(16, 1)),
            Choice(title="RTX  4090 (24GB)", value=(32, 1)),
            Choice(title="RTX  5090 (32GB)", value=(48, 1)),
            Choice(title="RTX A6000 (48GB)", value=(64, 1)),
            Choice(title="RTX  A100 (80GB)", value=(94, 1)),
            Choice(title="Custom", value="custom"),
        ],
    ).ask()

    if batch_size_selection == "custom":
        batch_size = int(
            questionary.text(
                "Enter custom batch size",
                validate=lambda value: value.isdigit() and int(value) > 0,
            ).ask()
        )
        grad_accum_steps = 1
    else:
        batch_size, grad_accum_steps = batch_size_selection

    dataset_path = f"./datasets/{version.id}"
    version.download(model_format="coco", location=dataset_path)
    dataset_name = project.id.split("/")[-1]

    import torch.multiprocessing as mp

    mp.set_sharing_strategy("file_system")
    from rfdetr.detr import RFDETRSegNano

    # RF-DETR 1.6.x regressed custom-resolution training: train(resolution=...)
    # mutates the model config, but the training-time pretrained-weight loader
    # still tries to load 312-resolution positional embeddings directly into the
    # resized model. We patch that loader so it interpolates positional
    # embeddings before loading, which restores the previously working
    # custom-resolution flow for this wizard.
    _patch_rfdetr_training_pretrain_loader()

    model = RFDETRSegNano()
    model.train(
        dataset_dir=dataset_path,
        epochs=100,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        resolution=DEFAULT_TRAINING_RESOLUTION,
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

    runs_dir = Path("runs")
    write_upload_metadata(runs_dir, dataset_name)

    typer.echo("Exporting model to ONNX...")
    model.export(output_dir=str(runs_dir))
    onnx_path = runs_dir / "inference_model.onnx"
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export did not produce expected file: {onnx_path}")

    from onnxsim import simplify

    typer.echo("Simplifying ONNX model...")
    onnx_model = onnx.load(onnx_path)

    onnx_model_simp, check = simplify(onnx_model)

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(onnx_model_simp, onnx_path)

    typer.echo("ONNX export complete")

    typer.echo("Generating training metrics plots...")
    metrics_path = runs_dir / "metrics.csv"
    plot_training_metrics(metrics_path)

    typer.echo("Running evaluation on best EMA checkpoint...")
    checkpoint_ema = runs_dir / "checkpoint_best_ema.pth"
    run_evaluation(
        checkpoint_path=checkpoint_ema,
        dataset_path=Path(dataset_path),
        split="test",
        threshold=0.5,
        iou_threshold=0.5,
        model=model,
    )
    eval_dir = runs_dir / f"{checkpoint_ema.stem}_test_evaluation"

    typer.echo("Preparing to upload training run to GCS...")
    import asyncio

    run_id = asyncio.run(
        upload_training_run(
            runs_dir=runs_dir,
            dataset_name=dataset_name,
            checkpoint_ema_path=checkpoint_ema,
            checkpoint_regular_path=runs_dir / "checkpoint_best_regular.pth",
            metrics_path=metrics_path,
            eval_dir=eval_dir,
            onnx_path=onnx_path,
        )
    )
    typer.echo(
        f"Training run {slugify_dataset_name(dataset_name)}-{run_id} complete and uploaded successfully"
    )
