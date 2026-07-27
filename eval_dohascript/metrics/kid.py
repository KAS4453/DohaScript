"""
metrics/kid.py
===============
Kernel Inception Distance (KID), computed per epoch using `torch-fidelity`.

KID (Binkowski et al., 2018) is an unbiased alternative to FID based on the
squared Maximum Mean Discrepancy between Inception feature distributions.
torch-fidelity reports both the mean and standard deviation across its
internal subset sampling procedure.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("metrics.kid")

try:
    import torch_fidelity
except ImportError as e:  # pragma: no cover
    torch_fidelity = None
    _IMPORT_ERROR = e


def compute_kid_for_folders(
    real_dir: Path,
    generated_dir: Path,
    device: str = "cuda",
    subset_size: int = 100,
    subsets: int = 100,
) -> Dict[str, float]:
    """Compute KID mean/std between two image folders."""
    if torch_fidelity is None:
        raise ImportError("torch-fidelity is required for KID computation.") from _IMPORT_ERROR

    import torch

    cuda = device == "cuda" and torch.cuda.is_available()
    if device == "cuda" and not cuda:
        logger.warning("CUDA requested but not available; falling back to CPU for KID.")

    from utils import list_images

    n_real = len(list_images(Path(real_dir)))
    n_gen = len(list_images(Path(generated_dir)))
    effective_subset_size = min(subset_size, n_real, n_gen)
    if effective_subset_size < 2:
        raise ValueError(
            f"KID needs at least 2 images per side, got real={n_real}, generated={n_gen} "
            f"in {generated_dir}. If your generator only emits one sample per epoch checkpoint, "
            "pool neighboring epochs together first (see utils.window_epoch_groups / the "
            "--fid-window CLI flag, which also applies to KID)."
        )
    if effective_subset_size < subset_size:
        logger.warning(
            "Requested kid_subset_size=%d exceeds available images (real=%d, generated=%d); "
            "using subset_size=%d instead.", subset_size, n_real, n_gen, effective_subset_size,
        )

    metrics = torch_fidelity.calculate_metrics(
        input1=str(generated_dir),
        input2=str(real_dir),
        cuda=cuda,
        kid=True,
        fid=False,
        isc=False,
        verbose=False,
        kid_subset_size=effective_subset_size,
        kid_subsets=subsets,
    )
    return {
        "kid_mean": float(metrics["kernel_inception_distance_mean"]),
        "kid_std": float(metrics["kernel_inception_distance_std"]),
    }


def compute_kid_per_epoch(
    real_dir: Path,
    epoch_dirs: Dict[int, Path],
    device: str = "cuda",
    subset_size: int = 100,
    subsets: int = 100,
    output_csv: Path | None = None,
) -> List[Dict[str, float]]:
    """
    Compute KID(real, generated_at_epoch) for every epoch.

    `epoch_dirs` must map epoch -> a directory containing only that epoch's
    generated images (torch-fidelity operates on directories, not lists of
    files, so callers should stage per-epoch subfolders; see main.py's
    `stage_epoch_subfolders` helper).
    """
    rows: List[Dict[str, float]] = []
    for epoch in sorted(epoch_dirs):
        gen_dir = epoch_dirs[epoch]
        try:
            result = compute_kid_for_folders(real_dir, gen_dir, device, subset_size, subsets)
        except Exception as exc:  # noqa: BLE001
            logger.exception("KID failed at epoch %d: %s", epoch, exc)
            result = {"kid_mean": float("nan"), "kid_std": float("nan")}
        row = {"epoch": epoch, **result}
        rows.append(row)
        logger.info("Epoch %d: KID mean=%.6f std=%.6f", epoch, row["kid_mean"], row["kid_std"])

    if output_csv is not None:
        write_kid_csv(rows, output_csv)
    return rows


def write_kid_csv(rows: List[Dict[str, float]], output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "kid_mean", "kid_std"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote KID results to %s", output_csv)
