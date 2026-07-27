"""
utils.py
========
Shared utility functions used across the evaluation framework:
logging setup, filename/epoch parsing, image loading, and small
statistics helpers (mean/std/95% CI) reused by several metric modules.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple  # noqa: F401

import numpy as np

logger = logging.getLogger("utils")

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str, log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a logger that writes to stdout and, optionally, a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Filename / epoch parsing
# ---------------------------------------------------------------------------
_DEFAULT_PATTERN = re.compile(r"epoch_(\d+)-(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)", re.IGNORECASE)


def parse_epoch_from_filename(filename: str, pattern: str | None = None) -> Tuple[int, int]:
    """
    Extract (epoch, global_iteration) from a filename such as
    ``epoch_1000-23000.png`` -> (1000, 23000).

    Raises ValueError if the filename does not match the expected pattern.
    """
    regex = re.compile(pattern, re.IGNORECASE) if pattern else _DEFAULT_PATTERN
    match = regex.search(filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match epoch pattern.")
    epoch = int(match.group(1))
    iteration = int(match.group(2))
    return epoch, iteration


def list_images(directory: Path, exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> List[Path]:
    """Return a sorted list of image files in `directory`."""
    directory = Path(directory)
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()]
    return sorted(files)


def group_images_by_epoch(directory: Path, pattern: str | None = None) -> Dict[int, List[Path]]:
    """
    Scan `directory` and group image paths by the epoch parsed from
    their filenames. Files that fail to parse are skipped with a warning.
    """
    logger = logging.getLogger("utils")
    groups: Dict[int, List[Path]] = defaultdict(list)
    for path in list_images(directory):
        try:
            epoch, _ = parse_epoch_from_filename(path.name, pattern)
        except ValueError:
            logger.warning("Skipping unparsable filename: %s", path.name)
            continue
        groups[epoch].append(path)
    return dict(sorted(groups.items()))


def window_epoch_groups(
    epoch_groups: Dict[int, List[Path]],
    window: int = 1,
    min_samples: int = 2,
) -> Dict[int, List[Path]]:
    """
    Pool nearby epochs' images together so each "epoch point" has enough
    samples for distribution-based metrics like FID/KID, which need a
    real *distribution* (multiple samples) per side, not a single image.

    If checkpoints only emit one generated image per epoch (as is common
    for GAN training logs), computing FID/KID per individual epoch is
    mathematically invalid — a covariance matrix (FID) or MMD estimate
    (KID) can't be formed from n=1. This groups every epoch's own image
    together with `window // 2` neighboring epochs on each side, trading
    temporal resolution for a valid sample count.

    `window=1` (the default) disables pooling and returns `epoch_groups`
    unchanged — use this only when each epoch already has multiple
    generated samples.

    Epochs that still fall short of `min_samples` after pooling (e.g. the
    very first/last epochs when `window` extends past the available
    range) are dropped, with a warning, rather than being silently passed
    through to a metric call that would just crash on them.
    """
    logger = logging.getLogger("utils")
    if window <= 1:
        return epoch_groups

    sorted_epochs = sorted(epoch_groups)
    half = window // 2
    pooled: Dict[int, List[Path]] = {}
    dropped = 0

    for i, epoch in enumerate(sorted_epochs):
        lo, hi = max(0, i - half), min(len(sorted_epochs), i + half + 1)
        neighbor_epochs = sorted_epochs[lo:hi]
        paths: List[Path] = []
        for ne in neighbor_epochs:
            paths.extend(epoch_groups[ne])
        if len(paths) < min_samples:
            dropped += 1
            continue
        pooled[epoch] = paths

    if dropped:
        logger.warning(
            "Dropped %d epoch(s) that had fewer than %d pooled samples even "
            "after windowing (window=%d); increase --fid-window or ignore "
            "the edge epochs.", dropped, min_samples, window,
        )
    return pooled


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------
def load_image_gray(path: Path) -> np.ndarray:
    """Load an image as a single-channel uint8 grayscale numpy array."""
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        return img
    if Image is not None:
        return np.array(Image.open(path).convert("L"))
    raise ImportError("Neither opencv-python nor Pillow is available to load images.")


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as an HxWx3 uint8 RGB numpy array."""
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if Image is not None:
        return np.array(Image.open(path).convert("RGB"))
    raise ImportError("Neither opencv-python nor Pillow is available to load images.")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Compute mean, standard deviation, and a normal-approximation confidence
    interval half-width for a sequence of scalar values.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    # z-score for the requested confidence level (only 90/95/99 supported
    # explicitly; default to the 95% z-score otherwise).
    z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_table.get(confidence, 1.96)
    ci = z * std / np.sqrt(arr.size) if arr.size > 0 else float("nan")

    return {"mean": mean, "std": std, "ci95": float(ci), "n": int(arr.size)}


def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_resized_copies(
    paths: Iterable[Path],
    dest_dir: Path,
    size: Tuple[int, int] = (299, 299),
    background: int = 255,
    overwrite: bool = False,
) -> List[Path]:
    """
    Copy each image in `paths` into `dest_dir`, resized (aspect-ratio
    preserved, letterboxed onto a fixed-size canvas) to a uniform `size`.

    Several downstream tools (pytorch-fid's default DataLoader, and
    torch-fidelity) collate a batch of images into a single tensor and
    will crash with a "stack expects each tensor to be equal size" error
    if the source images have varying dimensions -- which handwriting
    word/line crops of different lengths always will. Staging fixed-size
    copies first sidesteps that without patching the third-party libraries.
    """
    if Image is None:
        raise ImportError("Pillow is required to stage resized image copies.")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_w, target_h = size
    staged: List[Path] = []

    skipped: List[Path] = []
    for src in paths:
        out_path = dest_dir / src.name
        if out_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            out_path = out_path.with_suffix(".png")
        if out_path.exists() and not overwrite:
            staged.append(out_path)
            continue

        try:
            img = Image.open(src).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable image %s: %s", src, exc)
            skipped.append(src)
            continue

        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (target_w, target_h), (background, background, background))
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        canvas.paste(resized, offset)
        canvas.save(out_path)
        staged.append(out_path)

    if skipped:
        logger.warning(
            "Skipped %d unreadable/corrupted image(s) out of %d during staging.",
            len(skipped), len(skipped) + len(staged),
        )

    return staged

