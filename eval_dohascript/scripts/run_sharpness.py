"""
scripts/run_sharpness.py
========================
Run ONLY the Sharpness metric.

Usage
-----
    python scripts/run_sharpness.py
    python scripts/run_sharpness.py --generated img3
"""

from __future__ import annotations

import importlib

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute Sharpness per epoch, and only Sharpness.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_sharpness")

    mod = importlib.import_module("metrics.sharpness")
    fn = getattr(mod, "compute_sharpness_per_epoch")
    rows = fn(epoch_groups, output_csv=cfg.tables_dir / "sharpness.csv")

    if rows:
        plot_single_metric(
            cfg, "Sharpness",
            [r["epoch"] for r in rows],
            [r.get("mean", float("nan")) for r in rows],
        )
    logger.info("Sharpness complete: %d rows -> %s", len(rows), cfg.tables_dir / "sharpness.csv")


if __name__ == "__main__":
    main()
