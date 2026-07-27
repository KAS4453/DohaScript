"""
scripts/run_projection.py
==========================
Run ONLY the horizontal/vertical projection-profile metric.

Usage
-----
    python scripts/run_projection.py
    python scripts/run_projection.py --generated img3
"""

from __future__ import annotations

import csv

from _common import build_arg_parser, build_config, load_epoch_groups, plot_single_metric


def main() -> None:
    parser = build_arg_parser("Compute projection profiles per epoch, and only projection.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_projection")

    from metrics import projection as projection_mod
    rows = projection_mod.compute_projection_per_epoch(epoch_groups, cfg.figures_dir)

    with open(cfg.tables_dir / "projection.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    plot_single_metric(
        cfg, "Projection Energy",
        [r["epoch"] for r in rows], [r["horizontal_energy_mean"] for r in rows],
    )
    logger.info("Projection complete: %d rows -> %s", len(rows), cfg.tables_dir / "projection.csv")


if __name__ == "__main__":
    main()
