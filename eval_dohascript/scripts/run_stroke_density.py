"""
scripts/run_stroke_density.py
=============================
Run ONLY the Stroke Density metric.

Usage
-----
    python scripts/run_stroke_density.py
    python scripts/run_stroke_density.py --generated img3
"""

from __future__ import annotations

import importlib

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute Stroke Density per epoch, and only Stroke Density.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_stroke_density")

    mod = importlib.import_module("metrics.stroke_density")
    fn = getattr(mod, "compute_stroke_density_per_epoch")
    rows = fn(epoch_groups, output_csv=cfg.tables_dir / "stroke_density.csv")

    if rows:
        plot_single_metric(
            cfg, "Stroke Density",
            [r["epoch"] for r in rows],
            [r.get("density_mean", float("nan")) for r in rows],
        )
    logger.info("Stroke Density complete: %d rows -> %s", len(rows), cfg.tables_dir / "stroke_density.csv")


if __name__ == "__main__":
    main()
