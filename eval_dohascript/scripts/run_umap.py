"""
scripts/run_umap.py
====================
Run ONLY the UMAP (real vs generated embeddings) plot.

Usage
-----
    python scripts/run_umap.py
    python scripts/run_umap.py --generated img3 --real dataset_real
"""

from __future__ import annotations

from _common import build_arg_parser, build_config


def main() -> None:
    parser = build_arg_parser("Plot UMAP of real vs. generated embeddings, and only that.")
    args = parser.parse_args()

    cfg = build_config(args)

    from utils import setup_logger
    logger = setup_logger("run_umap", cfg.logs_dir)

    from plots import plot_umap
    out_path = cfg.figures_dir / "umap_real_vs_generated.png"
    plot_umap.plot_umap_real_vs_generated(
        cfg.real_dir, cfg.generated_dir, out_path,
        cfg.embedding_backbone, cfg.device, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.random_seed, cfg.plot_dpi,
    )
    logger.info("UMAP complete -> %s", out_path)


if __name__ == "__main__":
    main()
