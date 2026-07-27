"""
metrics/skeleton.py
====================
Morphological skeleton statistics: total stroke length, branch points,
endpoints, and skeletal stroke density, using scikit-image's
`skeletonize`. These are useful structural descriptors of handwriting
topology (e.g. detecting broken or fused strokes as training proceeds).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.ndimage import convolve

from utils import load_image_gray
from metrics.stroke_density import binarize_otsu

logger = logging.getLogger("metrics.skeleton")

try:
    from skimage.morphology import skeletonize
except ImportError as e:  # pragma: no cover
    skeletonize = None
    _IMPORT_ERROR = e


# 3x3 neighbor-count kernel (center excluded)
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def skeleton_statistics(gray: np.ndarray) -> Dict[str, float]:
    """
    Compute skeleton-derived statistics for one image:
      - total_length: pixel count of the skeleton (proxy for stroke length)
      - branch_points: skeleton pixels with >=3 neighbors
      - endpoints: skeleton pixels with exactly 1 neighbor
      - skeleton_density: skeleton pixels / total ink pixels
    """
    if skeletonize is None:
        raise ImportError("scikit-image is required for skeleton statistics.") from _IMPORT_ERROR

    mask = binarize_otsu(gray).astype(bool)
    if mask.sum() == 0:
        return {"total_length": 0.0, "branch_points": 0, "endpoints": 0, "skeleton_density": 0.0}

    skel = skeletonize(mask)
    neighbor_counts = convolve(skel.astype(np.uint8), _NEIGHBOR_KERNEL, mode="constant", cval=0)
    neighbor_counts = neighbor_counts * skel  # only count at skeleton pixels

    total_length = float(np.sum(skel))
    branch_points = int(np.sum(neighbor_counts >= 3))
    endpoints = int(np.sum(neighbor_counts == 1))
    skeleton_density = float(total_length / mask.sum())

    return {
        "total_length": total_length,
        "branch_points": branch_points,
        "endpoints": endpoints,
        "skeleton_density": skeleton_density,
    }


def compute_skeleton_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        accum = {"total_length": [], "branch_points": [], "endpoints": [], "skeleton_density": []}
        for p in paths:
            try:
                gray = load_image_gray(p)
                stats = skeleton_statistics(gray)
                for k, v in stats.items():
                    accum[k].append(v)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skeleton stats failed for %s: %s", p.name, exc)
        if not accum["total_length"]:
            continue
        row = {"epoch": epoch}
        row.update({f"{k}_mean": float(np.mean(v)) for k, v in accum.items()})
        row["n_images"] = len(accum["total_length"])
        rows.append(row)
        logger.info(
            "Epoch %d: skeleton length=%.1f branches=%.1f endpoints=%.1f",
            epoch, row["total_length_mean"], row["branch_points_mean"], row["endpoints_mean"],
        )

    if output_csv is not None and rows:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
