"""
scripts/run_tsne.py
====================
Run ONLY the t-SNE (real vs generated embeddings) plot.

Usage
-----
    python scripts/run_tsne.py
    python scripts/run_tsne.py --generated img3 --real dataset_real
"""

from __future__ import annotations

from _common import build_arg_parser, build_config


def main() -> None:
    parser = build_arg_parser("Plot t-SNE of real vs. generated embeddings, and only that.")
    args = parser.parse_args()

    cfg = build_config(args)

    from utils import setup_logger
    logger = setup_logger("run_tsne", cfg.logs_dir)

    from plots import plot_tsne
    out_path = cfg.figures_dir / "tsne_real_vs_generated.png"
    plot_tsne.plot_tsne_real_vs_generated(
        cfg.real_dir, cfg.generated_dir, out_path,
        cfg.embedding_backbone, cfg.device, cfg.tsne_perplexity, cfg.tsne_n_iter, cfg.random_seed, cfg.plot_dpi,
    )
    logger.info("t-SNE complete -> %s", out_path)


if __name__ == "__main__":
    main()
