"""
scripts/run_skeleton.py
=======================
Run ONLY the Skeleton Length metric.

Usage
-----
    python scripts/run_skeleton.py
    python scripts/run_skeleton.py --generated img3
"""

from __future__ import annotations

import importlib

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute Skeleton Length per epoch, and only Skeleton Length.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_skeleton")

    mod = importlib.import_module("metrics.skeleton")
    fn = getattr(mod, "compute_skeleton_per_epoch")
    rows = fn(epoch_groups, output_csv=cfg.tables_dir / "skeleton.csv")

    if rows:
        plot_single_metric(
            cfg, "Skeleton Length",
            [r["epoch"] for r in rows],
            [r.get("total_length_mean", float("nan")) for r in rows],
        )
    logger.info("Skeleton Length complete: %d rows -> %s", len(rows), cfg.tables_dir / "skeleton.csv")


if __name__ == "__main__":
    main()
