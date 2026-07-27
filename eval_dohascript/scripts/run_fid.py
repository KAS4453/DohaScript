"""
scripts/run_fid.py
===================
Run ONLY the FID metric. Nothing else in the framework is touched.

Usage
-----
    python scripts/run_fid.py
    python scripts/run_fid.py --generated img3 --real dataset_real
    python scripts/run_fid.py --fid-window 51   # pool 51 consecutive epochs
                                                 # (needed if your generator
                                                 # emits 1 image per epoch)
"""

from __future__ import annotations

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric
from utils import window_epoch_groups


def main() -> None:
    parser = build_arg_parser("Compute FID per epoch, and only FID.")
    parser.add_argument(
        "--fid-window", type=int, default=1,
        help="Pool this many consecutive epochs' generated images together (default 1 = no pooling).",
    )
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_fid")

    if args.fid_window > 1:
        epoch_groups = window_epoch_groups(epoch_groups, window=args.fid_window)
        logger.info("FID: pooling window=%d -> %d epoch points with >=2 samples.", args.fid_window, len(epoch_groups))

    from metrics import fid as fid_mod
    rows = fid_mod.compute_fid_per_epoch(
        cfg.real_dir, epoch_groups, cfg.device, cfg.batch_size, cfg.fid_dims,
        output_csv=cfg.tables_dir / "FID.csv",
        staging_dir=cfg.output_dir / "_staging_fid",
    )

    plot_single_metric(cfg, "FID", [r["epoch"] for r in rows], [r["fid"] for r in rows])
    logger.info("FID complete: %d rows -> %s", len(rows), cfg.tables_dir / "FID.csv")


if __name__ == "__main__":
    main()
