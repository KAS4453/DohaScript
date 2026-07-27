"""
metrics/entropy.py
===================
Shannon entropy of the pixel-intensity histogram, plus histogram-derived
statistics (mean intensity, std, contrast, dynamic range) and connected
component analysis, since all three are cheap, closely related summaries
of a single grayscale image and are typically reported together in a
handwriting-quality table.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils import load_image_gray
from metrics.stroke_density import binarize_otsu

logger = logging.getLogger("metrics.entropy")


def shannon_entropy(gray: np.ndarray) -> float:
    """Shannon entropy (bits) of the 256-bin pixel intensity histogram."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    prob = hist / max(hist.sum(), 1)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def histogram_statistics(gray: np.ndarray) -> Dict[str, float]:
    """Mean intensity, std, contrast (std/mean), and dynamic range (max-min)."""
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    contrast = float(std / mean) if mean > 0 else 0.0
    dynamic_range = float(np.max(gray) - np.min(gray))
    return {"mean_intensity": mean, "std_intensity": std, "contrast": contrast, "dynamic_range": dynamic_range}


def connected_components_stats(gray: np.ndarray) -> Dict[str, float]:
    """Count, largest, and average size of connected ink components (handwriting completeness proxy)."""
    mask = binarize_otsu(gray)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    if areas.size == 0:
        return {"num_components": 0, "largest_component_area": 0.0, "avg_component_area": 0.0}
    return {
        "num_components": int(areas.size),
        "largest_component_area": float(np.max(areas)),
        "avg_component_area": float(np.mean(areas)),
    }


def compute_entropy_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        entropy_vals = []
        for p in paths:
            try:
                gray = load_image_gray(p)
                entropy_vals.append(shannon_entropy(gray))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Entropy failed for %s: %s", p.name, exc)
        if not entropy_vals:
            continue
        row = {
            "epoch": epoch,
            "mean": float(np.mean(entropy_vals)),
            "std": float(np.std(entropy_vals)),
            "n_images": len(entropy_vals),
        }
        rows.append(row)
        logger.info("Epoch %d: entropy mean=%.4f std=%.4f", epoch, row["mean"], row["std"])

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "mean", "std", "n_images"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows


def compute_histogram_and_components_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        hist_accum = {"mean_intensity": [], "std_intensity": [], "contrast": [], "dynamic_range": []}
        cc_accum = {"num_components": [], "largest_component_area": [], "avg_component_area": []}
        for p in paths:
            try:
                gray = load_image_gray(p)
                h = histogram_statistics(gray)
                c = connected_components_stats(gray)
                for k, v in h.items():
                    hist_accum[k].append(v)
                for k, v in c.items():
                    cc_accum[k].append(v)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Histogram/CC stats failed for %s: %s", p.name, exc)
        if not hist_accum["mean_intensity"]:
            continue
        row = {"epoch": epoch}
        row.update({f"{k}_mean": float(np.mean(v)) for k, v in hist_accum.items()})
        row.update({f"{k}_mean": float(np.mean(v)) for k, v in cc_accum.items()})
        row["n_images"] = len(hist_accum["mean_intensity"])
        rows.append(row)

    if output_csv is not None and rows:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
