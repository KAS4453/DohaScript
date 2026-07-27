"""
metrics/stroke_density.py
==========================
Stroke density: fraction of foreground (ink) pixels after Otsu
binarization. Also estimates average stroke width via the distance
transform of the binarized stroke mask, since both are naturally
computed from the same thresholded image.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils import load_image_gray

logger = logging.getLogger("metrics.stroke_density")


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold a grayscale image; returns a binary mask where 1 = ink (foreground)."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (mask > 0).astype(np.uint8)


def stroke_density(gray: np.ndarray) -> float:
    """Foreground pixel ratio after Otsu thresholding."""
    mask = binarize_otsu(gray)
    return float(np.mean(mask))


def estimate_stroke_width(gray: np.ndarray) -> float:
    """
    Estimate the average stroke width using the distance transform of the
    binarized ink mask: for each foreground pixel, the distance to the
    nearest background pixel approximates the stroke's local half-width;
    doubling the mean distance along the mask's morphological skeleton
    approximates the average full stroke width.
    """
    mask = binarize_otsu(gray)
    if mask.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask.astype(bool))
        widths = dist[skel]
        if widths.size == 0:
            return float(2 * dist[mask.astype(bool)].mean())
        return float(2 * widths.mean())
    except ImportError:
        # Fallback without scikit-image: use raw foreground distances.
        return float(2 * dist[mask.astype(bool)].mean())


def compute_stroke_density_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        density_vals, width_vals = [], []
        for p in paths:
            try:
                gray = load_image_gray(p)
                density_vals.append(stroke_density(gray))
                width_vals.append(estimate_stroke_width(gray))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Stroke density failed for %s: %s", p.name, exc)
        if not density_vals:
            continue
        row = {
            "epoch": epoch,
            "density_mean": float(np.mean(density_vals)),
            "density_std": float(np.std(density_vals)),
            "stroke_width_mean": float(np.mean(width_vals)),
            "n_images": len(density_vals),
        }
        rows.append(row)
        logger.info(
            "Epoch %d: stroke density=%.4f (+-%.4f) stroke width=%.2fpx",
            epoch, row["density_mean"], row["density_std"], row["stroke_width_mean"],
        )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "density_mean", "density_std", "stroke_width_mean", "n_images"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
