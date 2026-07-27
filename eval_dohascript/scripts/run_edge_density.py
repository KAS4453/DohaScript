"""
scripts/run_edge_density.py
===========================
Run ONLY the Edge Density metric.

Usage
-----
    python scripts/run_edge_density.py
    python scripts/run_edge_density.py --generated img3
"""

from __future__ import annotations

import importlib

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute Edge Density per epoch, and only Edge Density.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_edge_density")

    mod = importlib.import_module("metrics.edge_density")
    fn = getattr(mod, "compute_edge_density_per_epoch")
    rows = fn(epoch_groups, output_csv=cfg.tables_dir / "edge_density.csv")

    if rows:
        plot_single_metric(
            cfg, "Edge Density",
            [r["epoch"] for r in rows],
            [r.get("mean", float("nan")) for r in rows],
        )
    logger.info("Edge Density complete: %d rows -> %s", len(rows), cfg.tables_dir / "edge_density.csv")


if __name__ == "__main__":
    main()
