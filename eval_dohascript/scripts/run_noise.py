"""
scripts/run_noise.py
====================
Run ONLY the Noise (sigma) metric.

Usage
-----
    python scripts/run_noise.py
    python scripts/run_noise.py --generated img3
"""

from __future__ import annotations

import importlib

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute Noise (sigma) per epoch, and only Noise (sigma).")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_noise")

    mod = importlib.import_module("metrics.noise")
    fn = getattr(mod, "compute_noise_per_epoch")
    rows = fn(epoch_groups, output_csv=cfg.tables_dir / "noise.csv")

    if rows:
        plot_single_metric(
            cfg, "Noise (sigma)",
            [r["epoch"] for r in rows],
            [r.get("mean", float("nan")) for r in rows],
        )
    logger.info("Noise (sigma) complete: %d rows -> %s", len(rows), cfg.tables_dir / "noise.csv")


if __name__ == "__main__":
    main()
