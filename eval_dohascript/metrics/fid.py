"""
metrics/fid.py
===============
Frechet Inception Distance (FID) between the real and generated
handwriting-image distributions, computed per epoch.

Wraps `pytorch-fid` (Seitzer, 2020) for the actual Inception-V3 feature
extraction and Frechet distance computation, adding:
  * GPU / CPU auto-fallback
  * batch-size control
  * a simple Python API on top of pytorch-fid's internals so callers
    don't need to shell out to its CLI
  * per-epoch evaluation against a fixed "real" reference folder
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("metrics.fid")

try:
    import torch
    from pytorch_fid.fid_score import calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3
except ImportError as e:  # pragma: no cover
    torch = None
    calculate_frechet_distance = None
    InceptionV3 = None
    _IMPORT_ERROR = e


def _resolve_device(requested: str) -> "torch.device":
    if torch is None:
        raise ImportError("pytorch-fid / torch is required for FID computation.") from _IMPORT_ERROR
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        logger.warning("CUDA requested but not available; falling back to CPU for FID.")
    return torch.device("cpu")


def _get_inception_model(device, dims: int = 2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def _compute_activation_statistics(
    image_paths: List[Path], model, device, batch_size: int = 32, dims: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (mu, sigma) of Inception activations for a list of images."""
    from pytorch_fid.fid_score import get_activations

    if len(image_paths) < 2:
        raise ValueError(
            f"Need at least 2 images to estimate a covariance matrix for FID, got {len(image_paths)}. "
            "If your generator only emits one sample per epoch checkpoint, pool neighboring epochs "
            "together first (see utils.window_epoch_groups / the --fid-window CLI flag)."
        )

    acts = get_activations([str(p) for p in image_paths], model, batch_size, dims, device)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def compute_fid_for_folders(
    real_dir: Path,
    generated_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
    dims: int = 2048,
    staging_dir: Path | None = None,
    canonical_size: tuple = (299, 299),
) -> float:
    """Compute a single FID score between two folders of images."""
    from utils import list_images, stage_resized_copies

    dev = _resolve_device(device)
    model = _get_inception_model(dev, dims)

    real_paths = list_images(real_dir)
    gen_paths = list_images(generated_dir)
    if not real_paths:
        raise FileNotFoundError(f"No real images found in {real_dir}")
    if not gen_paths:
        raise FileNotFoundError(f"No generated images found in {generated_dir}")

    stage_root = Path(staging_dir) if staging_dir is not None else Path("_staging_fid_single")
    real_staged = stage_resized_copies(real_paths, stage_root / "real", canonical_size)
    gen_staged = stage_resized_copies(gen_paths, stage_root / "generated", canonical_size)

    mu1, sigma1 = _compute_activation_statistics(real_staged, model, dev, batch_size, dims)
    mu2, sigma2 = _compute_activation_statistics(gen_staged, model, dev, batch_size, dims)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return float(fid_value)


def compute_fid_per_epoch(
    real_dir: Path,
    epoch_groups: Dict[int, List[Path]],
    device: str = "cuda",
    batch_size: int = 32,
    dims: int = 2048,
    output_csv: Path | None = None,
    staging_dir: Path | None = None,
    canonical_size: tuple = (299, 299),
) -> List[Dict[str, float]]:
    """
    Compute FID(real, generated_at_epoch) for every epoch in `epoch_groups`.
    Real-image statistics are computed once and reused for every epoch
    (fixed reference distribution).

    Handwriting crops are rarely uniform in size, and pytorch-fid's default
    DataLoader will crash trying to stack a batch of differently-sized
    tensors ("stack expects each tensor to be equal size"). To avoid that,
    every image is first letterboxed onto a fixed `canonical_size` canvas
    in a staging directory before activations are computed.
    """
    from utils import list_images, stage_resized_copies

    dev = _resolve_device(device)
    model = _get_inception_model(dev, dims)

    real_paths = list_images(real_dir)
    if not real_paths:
        raise FileNotFoundError(f"No real images found in {real_dir}")

    stage_root = Path(staging_dir) if staging_dir is not None else Path("_staging_fid")
    real_staged = stage_resized_copies(real_paths, stage_root / "real", canonical_size)

    logger.info("Computing reference (real) Inception statistics from %d images", len(real_staged))
    mu_real, sigma_real = _compute_activation_statistics(real_staged, model, dev, batch_size, dims)

    rows: List[Dict[str, float]] = []
    for epoch in sorted(epoch_groups):
        paths = epoch_groups[epoch]
        if not paths:
            continue
        try:
            gen_staged = stage_resized_copies(paths, stage_root / f"epoch_{epoch}", canonical_size)
            mu_gen, sigma_gen = _compute_activation_statistics(gen_staged, model, dev, batch_size, dims)
            fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        except Exception as exc:  # noqa: BLE001
            logger.exception("FID failed at epoch %d: %s", epoch, exc)
            fid_value = float("nan")
        rows.append({"epoch": epoch, "fid": float(fid_value), "n_images": len(paths)})
        logger.info("Epoch %d: FID=%.4f (n=%d)", epoch, fid_value, len(paths))

    if output_csv is not None:
        write_fid_csv(rows, output_csv)
    return rows


def write_fid_csv(rows: List[Dict[str, float]], output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "fid", "n_images"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote FID results to %s", output_csv)
