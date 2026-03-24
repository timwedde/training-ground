from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def write_evaluation_plots(
    output_dir: Path,
    per_class_rows: list[dict],
    per_image_rows: list[dict],
    prediction_rows: list[dict],
    coco_metrics: dict[str, dict[str, object]],
):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _write_performance_overview(
        plots_dir, per_class_rows, per_image_rows, prediction_rows
    )
    _write_prediction_scatter(plots_dir, prediction_rows)
    if any(metrics["stats"] for metrics in coco_metrics.values()):
        _write_coco_metrics(plots_dir, per_class_rows, coco_metrics)


def _write_performance_overview(
    plots_dir: Path,
    per_class_rows: list[dict],
    per_image_rows: list[dict],
    prediction_rows: list[dict],
):
    class_names = [row["class_name"] for row in per_class_rows]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Per-class quality",
            "Per-class counts",
            "Prediction confidence",
            "Per-image F1",
        ),
    )
    for metric in ("precision", "recall", "f1", "mean_matched_iou"):
        fig.add_trace(
            go.Bar(
                x=class_names,
                y=[row[metric] for row in per_class_rows],
                name=metric,
            ),
            row=1,
            col=1,
        )
    for metric in ("true_positives", "false_positives", "false_negatives"):
        fig.add_trace(
            go.Bar(
                x=class_names,
                y=[row[metric] for row in per_class_rows],
                name=metric,
            ),
            row=1,
            col=2,
        )

    matched_scores = [
        row["score"] for row in prediction_rows if row["match_status"] == "tp"
    ]
    unmatched_scores = [
        row["score"] for row in prediction_rows if row["match_status"] == "fp"
    ]
    fig.add_trace(
        go.Histogram(x=matched_scores, name="true positives", opacity=0.75),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=unmatched_scores, name="false positives", opacity=0.75),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=[row["f1"] for row in per_image_rows],
            name="per-image F1",
            nbinsx=20,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        hovermode="x unified",
        height=900,
        width=1400,
    )
    fig.write_html(plots_dir / "performance_overview.html", include_plotlyjs="cdn")


def _write_prediction_scatter(plots_dir: Path, prediction_rows: list[dict]):
    fig = go.Figure()
    if prediction_rows:
        fig.add_trace(
            go.Scatter(
                x=[row["score"] for row in prediction_rows],
                y=[row["best_iou"] for row in prediction_rows],
                mode="markers",
                marker={
                    "size": 9,
                    "color": [
                        "#2ecc71" if row["match_status"] == "tp" else "#e74c3c"
                        for row in prediction_rows
                    ],
                },
                text=[
                    f"{row['file_name']}<br>{row['class_name']}<br>{row['match_status']}"
                    for row in prediction_rows
                ],
                hovertemplate="%{text}<br>score=%{x:.3f}<br>IoU=%{y:.3f}<extra></extra>",
                name="predictions",
            )
        )
    fig.update_layout(
        title="Prediction score vs best IoU",
        xaxis_title="confidence",
        yaxis_title="best IoU",
        template="plotly_white",
        height=650,
        width=1100,
    )
    fig.write_html(plots_dir / "prediction_scatter.html", include_plotlyjs="cdn")


def _write_coco_metrics(
    plots_dir: Path,
    per_class_rows: list[dict],
    coco_metrics: dict[str, dict[str, object]],
):
    valid_metrics = [
        (name, metrics)
        for name, metrics in coco_metrics.items()
        if metrics["stats"] is not None
    ]
    fig = make_subplots(
        rows=1,
        cols=len(valid_metrics),
        subplot_titles=[f"{name.upper()} COCO AP" for name, metrics in valid_metrics],
    )

    class_names = [row["class_name"] for row in per_class_rows]
    for col_index, (metric_name, metrics) in enumerate(valid_metrics, start=1):
        fig.add_trace(
            go.Bar(
                x=class_names,
                y=[
                    metrics["per_class_ap"].get(row["category_id"], 0.0)
                    for row in per_class_rows
                ],
                name=f"{metric_name} AP",
            ),
            row=1,
            col=col_index,
        )
        fig.add_trace(
            go.Bar(
                x=class_names,
                y=[
                    metrics["per_class_ap50"].get(row["category_id"], 0.0)
                    for row in per_class_rows
                ],
                name=f"{metric_name} AP50",
            ),
            row=1,
            col=col_index,
        )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        hovermode="x unified",
        height=550,
        width=max(900, 600 * len(valid_metrics)),
    )
    fig.write_html(plots_dir / "coco_metrics.html", include_plotlyjs="cdn")
