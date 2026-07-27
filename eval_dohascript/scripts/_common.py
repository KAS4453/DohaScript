"""
scripts/_common.py
===================
Shared bootstrap code for the per-metric runner scripts in this folder.

Each script in `scripts/` runs exactly ONE metric, independent of every
other metric. This module just factors out the boring, identical parts
(CLI flags, Config construction, epoch scanning) so that every script's
"own logic" section stays short and easy to read on its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make the project root importable when a script is run directly, e.g.
#   python scripts/run_fid.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config, get_default_config  # noqa: E402
from utils import setup_logger, group_images_by_epoch  # noqa: E402


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Base flags every single-metric script accepts (paths + device)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--generated", type=str, default=None, help="Directory of generated images (epoch_*.png).")
    parser.add_argument("--real", type=str, default=None, help="Directory of real reference images.")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results.")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Compute device.")
    return parser


def build_config(args: argparse.Namespace) -> Config:
    cfg = get_default_config()
    if getattr(args, "generated", None):
        cfg.generated_dir = Path(args.generated)
    if getattr(args, "real", None):
        cfg.real_dir = Path(args.real)
    if getattr(args, "output", None):
        cfg.output_dir = Path(args.output)
    if getattr(args, "device", None):
        cfg.device = args.device
    if getattr(args, "sarvam_api_key", None):
        cfg.sarvam_api_key = args.sarvam_api_key
    cfg.__post_init__()
    return cfg


def load_epoch_groups(cfg: Config, logger_name: str) -> Tuple[Dict[int, List[Path]], "logging.Logger"]:
    """Scan `cfg.generated_dir` and return {epoch: [image paths]}, or exit cleanly if empty."""
    logger = setup_logger(logger_name, cfg.logs_dir)
    logger.info("Scanning generated images in %s", cfg.generated_dir)
    epoch_groups = group_images_by_epoch(cfg.generated_dir, cfg.filename_regex)
    if not epoch_groups:
        logger.error("No generated images found (or none matched the epoch filename pattern) in %s", cfg.generated_dir)
        raise SystemExit(1)
    logger.info("Found %d epochs, %d total images.", len(epoch_groups), sum(len(v) for v in epoch_groups.values()))
    return epoch_groups, logger


def plot_single_metric(cfg: Config, label: str, epochs: List[int], values: List[float]) -> None:
    """Save a standalone curve for this one metric (no cross-metric dashboard)."""
    if not epochs:
        return
    from plots import plot_metrics
    plot_metrics.plot_all_metrics(
        {label: {"epoch": epochs, "value": values}},
        cfg.figures_dir / "metric_curves", cfg.plot_formats, cfg.plot_dpi,
    )
