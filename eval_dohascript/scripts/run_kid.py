"""
scripts/run_kid.py
===================
Run ONLY the KID metric. Nothing else in the framework is touched.

Usage
-----
    python scripts/run_kid.py
    python scripts/run_kid.py --generated img3 --real dataset_real
    python scripts/run_kid.py --fid-window 51   # pool 51 consecutive epochs
                                                 # (needed if your generator
                                                 # emits 1 image per epoch)
"""

from __future__ import annotations

from _common import build_arg_parser, build_config, load_epoch_groups
from utils import window_epoch_groups, list_images, stage_resized_copies


def main() -> None:
    parser = build_arg_parser("Compute KID per epoch, and only KID.")
    parser.add_argument(
        "--fid-window", type=int, default=1,
        help="Pool this many consecutive epochs' generated images together (default 1 = no pooling).",
    )
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_kid")

    if args.fid_window > 1:
        epoch_groups = window_epoch_groups(epoch_groups, window=args.fid_window)
        logger.info("KID: pooling window=%d -> %d epoch points with >=2 samples.", args.fid_window, len(epoch_groups))

    staging_root = cfg.output_dir / "_staging_kid"
    epoch_dirs = {}
    for epoch, paths in epoch_groups.items():
        epoch_dir = staging_root / f"epoch_{epoch}"
        stage_resized_copies(paths, epoch_dir)
        epoch_dirs[epoch] = epoch_dir

    real_dir_staged = staging_root / "real"
    stage_resized_copies(list_images(cfg.real_dir), real_dir_staged)

    from metrics import kid as kid_mod
    rows = kid_mod.compute_kid_per_epoch(
        real_dir_staged, epoch_dirs, cfg.device, cfg.kid_subset_size, cfg.kid_subsets,
        output_csv=cfg.tables_dir / "KID.csv",
    )

    if rows:
        from plots import plot_metrics
        plot_metrics.plot_all_metrics(
            {"KID": {
                "epoch": [r["epoch"] for r in rows],
                "value": [r["kid_mean"] for r in rows],
                "std": [r["kid_std"] for r in rows],
            }},
            cfg.figures_dir / "metric_curves", cfg.plot_formats, cfg.plot_dpi,
        )
    logger.info("KID complete: %d rows -> %s", len(rows), cfg.tables_dir / "KID.csv")


if __name__ == "__main__":
    main()
