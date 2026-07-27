"""
metrics/edge_density.py
========================
Edge density via Canny edge detection: the fraction of pixels flagged
as edges, tracked per epoch as a proxy for stroke-boundary sharpness
and structural complexity.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils import load_image_gray

logger = logging.getLogger("metrics.edge_density")


def edge_density(gray: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> float:
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return float(np.mean(edges > 0))


def compute_edge_density_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        values = []
        for p in paths:
            try:
                gray = load_image_gray(p)
                values.append(edge_density(gray))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Edge density failed for %s: %s", p.name, exc)
        if not values:
            continue
        row = {
            "epoch": epoch,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_images": len(values),
        }
        rows.append(row)
        logger.info("Epoch %d: edge density mean=%.4f std=%.4f", epoch, row["mean"], row["std"])

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "mean", "std", "n_images"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
