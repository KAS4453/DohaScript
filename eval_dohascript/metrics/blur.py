"""
metrics/blur.py
================
Three complementary no-reference blur metrics:
  * Laplacian energy (variance of Laplacian, same core idea as sharpness.py
    but reported alongside the other two for a combined "blur profile")
  * Tenengrad (mean squared gradient magnitude via Sobel)
  * Brenner's focus measure (squared difference of pixels 2 apart)

Higher values of all three indicate a less blurry (sharper) image.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils import load_image_gray

logger = logging.getLogger("metrics.blur")


def laplacian_energy(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.mean(lap ** 2))


def tenengrad(gray: np.ndarray, ksize: int = 3, threshold: float = 0.0) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag_sq = gx ** 2 + gy ** 2
    mask = grad_mag_sq > threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(grad_mag_sq[mask]))


def brenner(gray: np.ndarray) -> float:
    gray_f = gray.astype(np.float64)
    shifted = np.roll(gray_f, -2, axis=1)
    diff = (shifted - gray_f) ** 2
    diff[:, -2:] = 0  # invalid wrap-around columns
    return float(np.sum(diff))


def compute_blur_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        lap_vals, ten_vals, bren_vals = [], [], []
        for p in paths:
            try:
                gray = load_image_gray(p)
                lap_vals.append(laplacian_energy(gray))
                ten_vals.append(tenengrad(gray))
                bren_vals.append(brenner(gray))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Blur metrics failed for %s: %s", p.name, exc)
        if not lap_vals:
            continue
        row = {
            "epoch": epoch,
            "laplacian_mean": float(np.mean(lap_vals)),
            "tenengrad_mean": float(np.mean(ten_vals)),
            "brenner_mean": float(np.mean(bren_vals)),
            "n_images": len(lap_vals),
        }
        rows.append(row)
        logger.info(
            "Epoch %d: Laplacian=%.2f Tenengrad=%.2f Brenner=%.2f",
            epoch, row["laplacian_mean"], row["tenengrad_mean"], row["brenner_mean"],
        )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "laplacian_mean", "tenengrad_mean", "brenner_mean", "n_images"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
