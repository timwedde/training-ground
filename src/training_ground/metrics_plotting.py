import csv
from pathlib import Path
from typing import Iterable

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


def _read_rows(metrics_path: Path) -> tuple[list[dict], list[str]]:
    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{name: parse_float(raw.get(name)) for name in fieldnames} for raw in reader]
    return rows, fieldnames


def plot_training_metrics(metrics_path: Path) -> tuple[Path, Path | None]:
    output_dir = metrics_path.parent / f"{metrics_path.stem}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = _read_rows(metrics_path)
    if not rows:
        raise ValueError(f"No rows found in {metrics_path}")

    if "step" in fieldnames or "train/loss" in fieldnames:
        return _plot_rfdetr_metrics(output_dir, rows, fieldnames, metrics_path)
    return _plot_yolo26_metrics(output_dir, rows, fieldnames, metrics_path)


def _plot_rfdetr_metrics(
    output_dir: Path,
    rows: list[dict],
    fieldnames: list[str],
    metrics_path: Path,
) -> tuple[Path, Path | None]:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    lr_rows: list[dict] = []

    for row in rows:
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
        raise ValueError(f"No plottable RF-DETR metrics found in {metrics_path}")

    _write_summary_plot(
        output_dir=output_dir,
        subplot_titles=(
            "Training losses",
            "Validation overview",
            "Validation quality",
            "Learning rate",
        ),
        series_groups=(
            (1, 1, train_rows, ["train/loss", "train/loss_ce", "train/loss_bbox", "train/loss_giou"]),
            (1, 2, val_rows, ["val/loss", "val/mAP_50", "val/mAP_50_95", "val/F1"]),
            (2, 1, val_rows, ["val/precision", "val/recall", "val/mAR", "val/ema_mAP_50", "val/ema_mAP_50_95"]),
            (2, 2, lr_rows, ["train/lr", "train/lr_max", "train/lr_min"]),
        ),
    )
    per_class_path = (
        _write_per_class_ap(output_dir, val_rows, fieldnames, prefix="val/AP/")
        if val_rows
        else None
    )
    _print_rfdetr_highlights(val_rows, train_rows)
    return output_dir, per_class_path


def _plot_yolo26_metrics(
    output_dir: Path,
    rows: list[dict],
    fieldnames: list[str],
    metrics_path: Path,
) -> tuple[Path, Path | None]:
    normalized_rows: list[dict] = []
    for index, row in enumerate(rows):
        epoch = row.get("epoch")
        if epoch is None:
            continue
        normalized = dict(row)
        normalized["epoch"] = int(epoch)
        normalized["step"] = int(epoch)
        normalized["row_index"] = index
        normalized_rows.append(normalized)

    if not normalized_rows:
        raise ValueError(f"No plottable YOLO26 metrics found in {metrics_path}")

    _write_summary_plot(
        output_dir=output_dir,
        subplot_titles=(
            "Training losses",
            "Detection metrics",
            "Segmentation metrics",
            "Learning rate",
        ),
        series_groups=(
            (1, 1, normalized_rows, ["train/box_loss", "train/seg_loss", "train/cls_loss"]),
            (1, 2, normalized_rows, ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]),
            (2, 1, normalized_rows, ["metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"]),
            (2, 2, normalized_rows, ["lr/pg0", "lr/pg1", "lr/pg2"]),
        ),
    )

    per_class_path = None
    _print_yolo26_highlights(normalized_rows)
    return output_dir, per_class_path


def _write_summary_plot(
    output_dir: Path,
    subplot_titles: tuple[str, str, str, str],
    series_groups: tuple[tuple[int, int, list[dict], list[str]], ...],
) -> None:
    fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)

    for row_i, col_i, rows, metrics in series_groups:
        plotted = _add_series(fig, row_i, col_i, rows, metrics)
        if not plotted:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{'' if row_i == col_i == 1 else (row_i - 1) * 2 + col_i} domain",
                yref=f"y{'' if row_i == col_i == 1 else (row_i - 1) * 2 + col_i} domain",
                text="No data",
                showarrow=False,
            )

    fig.update_xaxes(title_text="step")
    fig.update_layout(
        height=900,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(output_dir / "training_summary.html", include_plotlyjs="cdn")


def _add_series(
    fig: go.Figure,
    row_i: int,
    col_i: int,
    rows: list[dict],
    metrics: Iterable[str],
) -> bool:
    plotted = False
    for metric in metrics:
        points = [(row["step"], row[metric]) for row in rows if row.get(metric) is not None]
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


def _write_per_class_ap(
    output_dir: Path,
    val_rows: list[dict],
    fieldnames: list[str] | tuple[str, ...],
    prefix: str,
) -> Path | None:
    per_class_metrics = sorted(name for name in fieldnames if name.startswith(prefix))
    if not per_class_metrics:
        return None

    fig = go.Figure()
    for metric in per_class_metrics:
        points = [(row["step"], row[metric]) for row in val_rows if row.get(metric) is not None]
        if not points:
            continue
        xs, ys = zip(*points, strict=False)
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines+markers", name=metric))

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


def _print_rfdetr_highlights(val_rows: list[dict], train_rows: list[dict]) -> None:
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


def _print_yolo26_highlights(rows: list[dict]) -> None:
    highlights = (
        ("metrics/mAP50-95(M)", True),
        ("metrics/mAP50(M)", True),
        ("metrics/precision(M)", True),
        ("metrics/recall(M)", True),
        ("train/seg_loss", False),
    )
    for metric, maximize in highlights:
        row = best_row(rows, metric, maximize=maximize)
        if row is None:
            continue
        qualifier = "best" if maximize else "lowest"
        typer.echo(
            f"  - {qualifier} {metric}: {row[metric]:.4f} (epoch {row['epoch']}, step {row['step']})"
        )
