"""
scripts/run_blur.py
===================
Run ONLY the Blur (Laplacian) metric.

Usage
-----
    python scripts/run_blur.py
    python scripts/run_blur.py --generated img3
"""

from __future__ import annotations

import importlib

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute Blur (Laplacian) per epoch, and only Blur (Laplacian).")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_blur")

    mod = importlib.import_module("metrics.blur")
    fn = getattr(mod, "compute_blur_per_epoch")
    rows = fn(epoch_groups, output_csv=cfg.tables_dir / "blur.csv")

    if rows:
        plot_single_metric(
            cfg, "Blur (Laplacian)",
            [r["epoch"] for r in rows],
            [r.get("laplacian_mean", float("nan")) for r in rows],
        )
    logger.info("Blur (Laplacian) complete: %d rows -> %s", len(rows), cfg.tables_dir / "blur.csv")


if __name__ == "__main__":
    main()
