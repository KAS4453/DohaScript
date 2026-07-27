"""
metrics/quality.py
===================
No-reference perceptual image quality metrics: BRISQUE, NIQE, and PIQE.

These rely on optional third-party packages (`piq` for BRISQUE/NIQE via
PyTorch, `piqe` or a MATLAB-derived reimplementation for PIQE). Because
availability varies a lot across environments, each metric is guarded
individually: if a backing library is missing, that metric is skipped
(logged, not silently dropped) rather than crashing the whole pipeline.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils import load_image_gray

logger = logging.getLogger("metrics.quality")


def _try_brisque(gray: np.ndarray) -> Optional[float]:
    try:
        import torch
        import piq

        tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        return float(piq.brisque(tensor, data_range=1.0))
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("BRISQUE failed: %s", exc)
        return None


def _try_niqe(gray: np.ndarray) -> Optional[float]:
    try:
        import torch
        import piq

        tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        return float(piq.niqe(tensor, data_range=1.0))
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("NIQE failed: %s", exc)
        return None


def _try_piqe(gray: np.ndarray) -> Optional[float]:
    try:
        import piqe as piqe_module  # third-party 'piqe' package, if installed

        score, _, _, _ = piqe_module.piqe(gray)
        return float(score)
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("PIQE failed: %s", exc)
        return None


_AVAILABILITY_LOGGED = False


def _log_availability_once() -> None:
    global _AVAILABILITY_LOGGED
    if _AVAILABILITY_LOGGED:
        return
    for name, fn in (("BRISQUE/NIQE (piq)", _try_brisque), ("PIQE (piqe)", _try_piqe)):
        dummy = np.zeros((32, 32), dtype=np.uint8)
        available = fn(dummy) is not None
        logger.info("%s available: %s", name, available)
    _AVAILABILITY_LOGGED = True


def compute_quality_per_epoch(
    epoch_groups: Dict[int, List[Path]], output_csv: Path | None = None
) -> List[Dict[str, float]]:
    _log_availability_once()
    rows = []
    for epoch, paths in sorted(epoch_groups.items()):
        brisque_vals, niqe_vals, piqe_vals = [], [], []
        for p in paths:
            try:
                gray = load_image_gray(p)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load %s for quality metrics: %s", p.name, exc)
                continue
            b = _try_brisque(gray)
            n = _try_niqe(gray)
            q = _try_piqe(gray)
            if b is not None:
                brisque_vals.append(b)
            if n is not None:
                niqe_vals.append(n)
            if q is not None:
                piqe_vals.append(q)

        row = {"epoch": epoch, "n_images": len(paths)}
        row["brisque_mean"] = float(np.mean(brisque_vals)) if brisque_vals else float("nan")
        row["niqe_mean"] = float(np.mean(niqe_vals)) if niqe_vals else float("nan")
        row["piqe_mean"] = float(np.mean(piqe_vals)) if piqe_vals else float("nan")
        rows.append(row)
        logger.info(
            "Epoch %d: BRISQUE=%.3f NIQE=%.3f PIQE=%.3f",
            epoch, row["brisque_mean"], row["niqe_mean"], row["piqe_mean"],
        )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "brisque_mean", "niqe_mean", "piqe_mean", "n_images"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return rows
