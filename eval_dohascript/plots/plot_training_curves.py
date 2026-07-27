"""
plots/plot_training_curves.py
==============================
Two complementary visualizations of training dynamics:

1. `plot_multi_metric_grid` — a grid of subplots, one per metric, sharing
   the epoch axis, useful as a single "training dashboard" figure.
2. `plot_qualitative_evolution` — a strip of example generated images at
   increasing epochs (e.g. 0 -> 5000 -> 10000 -> 20000 -> 50000), cropped
   and scaled identically, to visualize qualitative improvement.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from utils import load_image_gray

logger = logging.getLogger("plots.training_curves")


def _get_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "font.family": "serif"})
    return plt


def plot_multi_metric_grid(
    metric_series: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    n_cols: int = 3,
    dpi: int = 300,
) -> Path:
    """Render every metric's epoch curve as a subplot in one dashboard figure."""
    plt = _get_matplotlib()
    names = list(metric_series.keys())
    n = len(names)
    if n == 0:
        raise ValueError("No metrics provided to plot_multi_metric_grid.")

    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, name in zip(axes_flat, names):
        series = metric_series[name]
        ax.plot(series.get("epoch", []), series.get("value", []), marker="o", markersize=3, linewidth=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved training dashboard to %s", output_path)
    return output_path


def _center_crop(gray: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = gray.shape
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return gray[top:top + crop_h, left:left + crop_w]


def plot_qualitative_evolution(
    epoch_to_example: Dict[int, Path],
    output_path: Path,
    crop_size: tuple = (128, 512),
    dpi: int = 300,
) -> Path:
    """
    Save a single-row strip figure showing one representative generated
    image per selected epoch, all center-cropped to the same size so
    they are directly visually comparable.
    """
    plt = _get_matplotlib()
    epochs = sorted(epoch_to_example)
    if not epochs:
        raise ValueError("No epochs provided to plot_qualitative_evolution.")

    crop_h, crop_w = crop_size
    fig, axes = plt.subplots(1, len(epochs), figsize=(3 * len(epochs), 3.5))
    axes = np.atleast_1d(axes)

    for ax, epoch in zip(axes, epochs):
        try:
            gray = load_image_gray(epoch_to_example[epoch])
            crop = _center_crop(gray, crop_h, crop_w)
            ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to render qualitative example for epoch %d: %s", epoch, exc)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_title(f"Epoch {epoch}", fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved qualitative evolution strip to %s", output_path)
    return output_path
