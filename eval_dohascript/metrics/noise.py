"""
metrics/noise.py
=================
Fast, reference-free noise-level estimation using the Immerkaer (1996)
Laplacian-based estimator: convolves the image with a noise-sensitive
kernel and derives a closed-form sigma estimate from the response.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.signal import convolve2d

from utils import load_image_gray

logger = logging.getLogger("metrics.noise")

# Immerkaer's noise-estimation kernel
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimate the standard deviation of additive Gaussian noise in `gray`
    using Immerkaer's fast single-image estimator:

        sigma = sqrt(pi / 2) * (1 / (6 * (W - 2) * (H - 2))) * sum(|I * kernel|)
    """
    h, w = gray.shape
    if h <= 2 or w <= 2:
        return 0.0
    response = convolve2d(gray.astype(np.float64), _NOISE_KERNEL, mode="same", boundary="symm")
    sigma = np.sqrt(np.pi / 2) * np.mean(np.abs(response)) / 6.0
    return float(sigma)


def compute_noise_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        values = []
        for p in paths:
            try:
                gray = load_image_gray(p)
                values.append(estimate_noise_sigma(gray))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Noise estimation failed for %s: %s", p.name, exc)
        if not values:
            continue
        row = {
            "epoch": epoch,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_images": len(values),
        }
        rows.append(row)
        logger.info("Epoch %d: noise sigma mean=%.4f std=%.4f", epoch, row["mean"], row["std"])

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "mean", "std", "n_images"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
