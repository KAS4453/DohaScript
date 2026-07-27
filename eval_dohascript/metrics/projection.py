"""
metrics/projection.py
======================
Horizontal and vertical projection profiles of the binarized ink mask,
used both as scalar summary features (projection "energy") and as
saved plots for qualitative line/word-spacing analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import load_image_gray, ensure_dir
from metrics.stroke_density import binarize_otsu

logger = logging.getLogger("metrics.projection")


def compute_projection_profiles(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (horizontal_profile, vertical_profile): row-sums and column-sums of the ink mask."""
    mask = binarize_otsu(gray)
    horizontal = mask.sum(axis=1).astype(np.float64)  # per row
    vertical = mask.sum(axis=0).astype(np.float64)    # per column
    return horizontal, vertical


def projection_energy(gray: np.ndarray) -> Dict[str, float]:
    """Scalar summary: variance of each profile, a proxy for how structured the ink layout is."""
    h, v = compute_projection_profiles(gray)
    return {
        "horizontal_energy": float(np.var(h)),
        "vertical_energy": float(np.var(v)),
    }


def save_projection_plot(gray: np.ndarray, output_path: Path, dpi: int = 300) -> None:
    """Save a figure with the image, its horizontal profile, and its vertical profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, v = compute_projection_profiles(gray)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].plot(h, np.arange(len(h)))
    axes[1].invert_yaxis()
    axes[1].set_title("Horizontal projection")
    axes[1].set_xlabel("Ink pixel count")
    axes[1].set_ylabel("Row")

    axes[2].plot(v)
    axes[2].set_title("Vertical projection")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Ink pixel count")

    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def compute_projection_per_epoch(
    epoch_groups: Dict[int, List[Path]],
    figures_dir: Path | None = None,
    max_plots_per_epoch: int = 2,
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        h_energy, v_energy = [], []
        for idx, p in enumerate(paths):
            try:
                gray = load_image_gray(p)
                energy = projection_energy(gray)
                h_energy.append(energy["horizontal_energy"])
                v_energy.append(energy["vertical_energy"])
                if figures_dir is not None and idx < max_plots_per_epoch:
                    out = Path(figures_dir) / "projections" / f"epoch_{epoch}_{p.stem}.png"
                    save_projection_plot(gray, out)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Projection failed for %s: %s", p.name, exc)
        if not h_energy:
            continue
        rows.append(
            {
                "epoch": epoch,
                "horizontal_energy_mean": float(np.mean(h_energy)),
                "vertical_energy_mean": float(np.mean(v_energy)),
                "n_images": len(h_energy),
            }
        )
    return rows
