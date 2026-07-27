"""
plots/plot_metrics.py
======================
Publication-quality "metric vs epoch" line plots (300 dpi, PNG/PDF/SVG)
for every scalar metric tracked by the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

logger = logging.getLogger("plots.metrics")


def _get_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.family": "serif",
        }
    )
    return plt


def plot_metric_vs_epoch(
    epochs: Sequence[int],
    values: Sequence[float],
    metric_name: str,
    output_dir: Path,
    ylabel: str | None = None,
    formats: Iterable[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
    std: Sequence[float] | None = None,
    figsize: tuple = (8, 5),
) -> List[Path]:
    """Plot a single scalar metric against epoch and save in the requested formats."""
    plt = _get_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, values, marker="o", linewidth=1.5, markersize=4, color="#1f4e79")
    if std is not None:
        lower = [v - s for v, s in zip(values, std)]
        upper = [v + s for v, s in zip(values, std)]
        ax.fill_between(epochs, lower, upper, alpha=0.2, color="#1f4e79")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or metric_name)
    ax.set_title(f"{metric_name} vs. Training Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    saved_paths = []
    safe_name = metric_name.lower().replace(" ", "_").replace("/", "_")
    for fmt in formats:
        out_path = output_dir / f"{safe_name}_vs_epoch.{fmt}"
        fig.savefig(out_path, dpi=dpi, format=fmt)
        saved_paths.append(out_path)
    plt.close(fig)
    logger.info("Saved %s plot(s): %s", metric_name, [str(p) for p in saved_paths])
    return saved_paths


def plot_all_metrics(
    metric_series: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
    formats: Iterable[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
) -> None:
    """
    Batch-plot every metric in `metric_series`, a dict of the form:

        {
          "FID": {"epoch": [...], "value": [...], "std": [...] (optional)},
          "CER": {"epoch": [...], "value": [...]},
          ...
        }
    """
    for metric_name, series in metric_series.items():
        epochs = series.get("epoch", [])
        values = series.get("value", [])
        std = series.get("std")
        if not epochs or not values:
            logger.warning("Skipping empty series for metric '%s'", metric_name)
            continue
        plot_metric_vs_epoch(epochs, values, metric_name, output_dir, formats=formats, dpi=dpi, std=std)
