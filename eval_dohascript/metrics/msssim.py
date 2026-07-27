"""
metrics/msssim.py
==================
Multi-Scale Structural Similarity (MS-SSIM) among generated images,
computed pairwise within each epoch to measure intra-epoch diversity
(low MS-SSIM = more diverse samples; high = mode collapse risk).
"""

from __future__ import annotations

import csv
import itertools
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils import load_image_gray

logger = logging.getLogger("metrics.msssim")

try:
    from skimage.metrics import structural_similarity as _ssim
except ImportError as e:  # pragma: no cover
    _ssim = None
    _IMPORT_ERROR = e


def _to_common_size(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize the larger image down to the smaller one's shape for fair comparison."""
    import cv2

    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    a = cv2.resize(img_a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def ms_ssim(img_a: np.ndarray, img_b: np.ndarray, levels: int = 5) -> float:
    """
    Multi-scale SSIM implemented as the geometric mean of single-scale SSIM
    computed at `levels` progressively down-sampled resolutions (a common
    simplification of Wang et al. 2003 when a dedicated MS-SSIM library is
    not desired as a hard dependency).
    """
    if _ssim is None:
        raise ImportError("scikit-image is required for MS-SSIM.") from _IMPORT_ERROR

    import cv2

    a, b = _to_common_size(img_a, img_b)
    scores = []
    weights = []
    for level in range(levels):
        min_dim = min(a.shape[0], a.shape[1])
        win = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win < 3:
            break
        score = _ssim(a, b, data_range=255, win_size=win)
        scores.append(max(score, 1e-8))
        weights.append(1.0 / (level + 1))  # simple decaying weight per scale
        a = cv2.resize(a, (max(a.shape[1] // 2, 1), max(a.shape[0] // 2, 1)))
        b = cv2.resize(b, (max(b.shape[1] // 2, 1), max(b.shape[0] // 2, 1)))
        if a.shape[0] < 8 or a.shape[1] < 8:
            break
    if not scores:
        return float("nan")
    weights = np.array(weights) / np.sum(weights)
    log_scores = np.log(np.clip(scores, 1e-8, None))
    return float(np.exp(np.sum(weights * log_scores)))


def compute_msssim_for_epoch(paths: List[Path], max_pairs: int = 500, seed: int = 42) -> Dict[str, float]:
    """Compute pairwise MS-SSIM statistics for one epoch's set of images."""
    if len(paths) < 2:
        return {"mean": float("nan"), "std": float("nan"), "n_pairs": 0}

    all_pairs = list(itertools.combinations(range(len(paths)), 2))
    if len(all_pairs) > max_pairs:
        rng = random.Random(seed)
        all_pairs = rng.sample(all_pairs, max_pairs)

    images = {i: load_image_gray(paths[i]) for i in {idx for pair in all_pairs for idx in pair}}

    scores = []
    for i, j in all_pairs:
        try:
            scores.append(ms_ssim(images[i], images[j]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MS-SSIM failed on pair (%s, %s): %s", paths[i].name, paths[j].name, exc)

    if not scores:
        return {"mean": float("nan"), "std": float("nan"), "n_pairs": 0}

    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "n_pairs": len(scores)}


def compute_msssim_per_epoch(
    epoch_groups: Dict[int, List[Path]], max_pairs: int = 500, output_csv: Path | None = None
) -> List[Dict[str, float]]:
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        stats = compute_msssim_for_epoch(paths, max_pairs)
        row = {"epoch": epoch, **stats}
        rows.append(row)
        logger.info("Epoch %d: MS-SSIM mean=%.4f std=%.4f (n_pairs=%d)", epoch, stats["mean"], stats["std"], stats["n_pairs"])

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "mean", "std", "n_pairs"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
