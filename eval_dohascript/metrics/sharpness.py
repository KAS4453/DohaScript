"""
metrics/sharpness.py
=====================
Image sharpness via the variance of the Laplacian, a standard proxy for
focus/detail sharpness. Higher variance generally implies a crisper image.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils import load_image_gray

logger = logging.getLogger("metrics.sharpness")


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Compute the variance of the Laplacian of a grayscale image."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_sharpness_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        values = []
        for p in paths:
            try:
                gray = load_image_gray(p)
                values.append(variance_of_laplacian(gray))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sharpness failed for %s: %s", p.name, exc)
        if not values:
            continue
        arr = np.asarray(values)
        row = {
            "epoch": epoch,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "n_images": len(values),
        }
        rows.append(row)
        logger.info("Epoch %d: sharpness mean=%.2f median=%.2f std=%.2f", epoch, row["mean"], row["median"], row["std"])

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "mean", "median", "std", "n_images"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
