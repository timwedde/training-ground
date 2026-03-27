import csv
from pathlib import Path

import plotly.graph_objects as go
import typer
from plotly.subplots import make_subplots


def best_row(rows: list[dict], metric: str, maximize: bool = True) -> dict | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    return (
        max(candidates, key=lambda row: row[metric])
        if maximize
        else min(candidates, key=lambda row: row[metric])
    )


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    return float(stripped) if stripped else None


def plot_training_metrics(metrics_path: Path) -> tuple[Path, Path | None]:
    output_dir = metrics_path.parent / f"{metrics_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    lr_rows: list[dict] = []

    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        for raw_row in reader:
            row = {name: parse_float(raw_row.get(name)) for name in fieldnames}
            step = row.get("step")
            epoch = row.get("epoch")
            if step is None or epoch is None:
                continue

            row["step"] = int(step)
            row["epoch"] = int(epoch)

            if row.get("train/loss") is not None:
                train_rows.append(row)
            if any(row.get(m) is not None for m in ("val/mAP_50", "val/loss")):
                val_rows.append(row)
            if row.get("train/lr") is not None:
                lr_rows.append(row)

    if not any((train_rows, val_rows, lr_rows)):
        raise ValueError(f"No plottable metrics found in {metrics_path}")

    _write_training_summary(output_dir, train_rows, val_rows, lr_rows)
    per_class_path = (
        _write_per_class_ap(output_dir, val_rows, fieldnames) if val_rows else None
    )

    _print_highlights(val_rows, train_rows)

    return output_dir, per_class_path


def _write_training_summary(
    output_dir: Path, train_rows: list[dict], val_rows: list[dict], lr_rows: list[dict]
):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training losses",
            "Validation overview",
            "Validation quality",
            "Learning rate",
        ),
    )

    def add_series(fig, row_i, col_i, rows, metrics):
        plotted = False
        for metric in metrics:
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
                row=row_i,
                col=col_i,
            )
            plotted = True
        return plotted

    training_plotted = add_series(
        fig,
        1,
        1,
        train_rows,
        ["train/loss", "train/loss_ce", "train/loss_bbox", "train/loss_giou"],
    )
    add_series(
        fig, 1, 2, val_rows, ["val/loss", "val/mAP_50", "val/mAP_50_95", "val/F1"]
    )
    add_series(
        fig,
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
    add_series(fig, 2, 2, lr_rows, ["train/lr", "train/lr_max", "train/lr_min"])

    fig.update_xaxes(title_text="step", row=1, col=1)
    if not training_plotted:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
            text="No data",
            showarrow=False,
        )

    fig.update_layout(
        height=900,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(output_dir / "training_summary.html", include_plotlyjs="cdn")


def _write_per_class_ap(
    output_dir: Path, val_rows: list[dict], fieldnames: list[str] | tuple[str, ...]
) -> Path | None:
    per_class_metrics = sorted(
        name for name in fieldnames if name.startswith("val/AP/")
    )
    if not per_class_metrics:
        return None

    fig = go.Figure()
    for metric in per_class_metrics:
        points = [
            (row["step"], row[metric])
            for row in val_rows
            if row.get(metric) is not None
        ]
        if not points:
            continue
        xs, ys = zip(*points, strict=False)
        fig.add_trace(
            go.Scatter(x=list(xs), y=list(ys), mode="lines+markers", name=metric)
        )

    fig.update_layout(
        title="Per-class AP",
        xaxis_title="step",
        yaxis_title="value",
        height=600,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
    )
    path = output_dir / "per_class_ap.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def _print_highlights(val_rows: list[dict], train_rows: list[dict]):
    best_map50 = best_row(val_rows, "val/mAP_50")
    best_ema_map50 = best_row(val_rows, "val/ema_mAP_50")
    best_f1 = best_row(val_rows, "val/F1")
    lowest_val_loss = best_row(val_rows, "val/loss", maximize=False)
    final_train = max(train_rows, key=lambda row: row["step"]) if train_rows else None

    if best_map50:
        typer.echo(
            f"  - best val/mAP_50: {best_map50['val/mAP_50']:.4f} (epoch {best_map50['epoch']}, step {best_map50['step']})"
        )
    if best_ema_map50:
        typer.echo(
            f"  - best val/ema_mAP_50: {best_ema_map50['val/ema_mAP_50']:.4f} (epoch {best_ema_map50['epoch']}, step {best_ema_map50['step']})"
        )
    if best_f1:
        typer.echo(
            f"  - best val/F1: {best_f1['val/F1']:.4f} (epoch {best_f1['epoch']}, step {best_f1['step']})"
        )
    if lowest_val_loss:
        typer.echo(
            f"  - lowest val/loss: {lowest_val_loss['val/loss']:.4f} (epoch {lowest_val_loss['epoch']}, step {lowest_val_loss['step']})"
        )
    if final_train and final_train.get("train/loss") is not None:
        typer.echo(
            f"  - final train/loss: {final_train['train/loss']:.4f} (epoch {final_train['epoch']}, step {final_train['step']})"
        )
