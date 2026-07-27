"""
scripts/run_msssim.py
======================
Run ONLY the MS-SSIM metric.

Usage
-----
    python scripts/run_msssim.py
    python scripts/run_msssim.py --generated img3
"""

from __future__ import annotations

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute MS-SSIM per epoch, and only MS-SSIM.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_msssim")

    from metrics import msssim as msssim_mod
    rows = msssim_mod.compute_msssim_per_epoch(epoch_groups, cfg.msssim_max_pairs, cfg.tables_dir / "msssim.csv")

    plot_single_metric(cfg, "MS-SSIM", [r["epoch"] for r in rows], [r["mean"] for r in rows])
    logger.info("MS-SSIM complete: %d rows -> %s", len(rows), cfg.tables_dir / "msssim.csv")


if __name__ == "__main__":
    main()
