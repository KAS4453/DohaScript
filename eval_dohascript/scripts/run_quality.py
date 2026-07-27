"""
scripts/run_quality.py
======================
Run ONLY the BRISQUE metric.

Usage
-----
    python scripts/run_quality.py
    python scripts/run_quality.py --generated img3
"""

from __future__ import annotations

import importlib

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute BRISQUE per epoch, and only BRISQUE.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_quality")

    mod = importlib.import_module("metrics.quality")
    fn = getattr(mod, "compute_quality_per_epoch")
    rows = fn(epoch_groups, output_csv=cfg.tables_dir / "quality.csv")

    if rows:
        plot_single_metric(
            cfg, "BRISQUE",
            [r["epoch"] for r in rows],
            [r.get("brisque_mean", float("nan")) for r in rows],
        )
    logger.info("BRISQUE complete: %d rows -> %s", len(rows), cfg.tables_dir / "quality.csv")


if __name__ == "__main__":
    main()
