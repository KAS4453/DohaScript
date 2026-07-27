"""
scripts/run_entropy.py
======================
Run ONLY the Image Entropy metric.

Usage
-----
    python scripts/run_entropy.py
    python scripts/run_entropy.py --generated img3
"""

from __future__ import annotations

import importlib

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute Image Entropy per epoch, and only Image Entropy.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_entropy")

    mod = importlib.import_module("metrics.entropy")
    fn = getattr(mod, "compute_entropy_per_epoch")
    rows = fn(epoch_groups, output_csv=cfg.tables_dir / "entropy.csv")

    if rows:
        plot_single_metric(
            cfg, "Image Entropy",
            [r["epoch"] for r in rows],
            [r.get("mean", float("nan")) for r in rows],
        )
    logger.info("Image Entropy complete: %d rows -> %s", len(rows), cfg.tables_dir / "entropy.csv")


if __name__ == "__main__":
    main()
