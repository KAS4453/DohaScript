"""
plots/plot_umap.py
===================
UMAP visualization of real-vs-generated image embeddings, as a
complement to t-SNE (plot_tsne.py) — UMAP tends to preserve more
global structure and runs faster on larger sample counts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils import list_images
from plots.plot_tsne import extract_embeddings

logger = logging.getLogger("plots.umap")


def plot_umap_real_vs_generated(
    real_dir: Path,
    generated_dir: Path,
    output_path: Path,
    backbone: str = "inception",
    device: str = "cuda",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    dpi: int = 300,
) -> Path:
    """Compute embeddings for both folders, run UMAP jointly, and plot a labeled scatter."""
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_paths = list_images(real_dir)
    gen_paths = list_images(generated_dir)
    if not real_paths or not gen_paths:
        raise FileNotFoundError("Both real_dir and generated_dir must contain images for UMAP.")

    real_emb = extract_embeddings(real_paths, backbone, device)
    gen_emb = extract_embeddings(gen_paths, backbone, device)

    all_emb = np.concatenate([real_emb, gen_emb], axis=0)
    labels = np.array(["Real"] * len(real_emb) + ["Generated"] * len(gen_emb))

    effective_neighbors = min(n_neighbors, max(2, len(all_emb) - 1))
    reducer = umap.UMAP(n_neighbors=effective_neighbors, min_dist=min_dist, random_state=seed)
    coords = reducer.fit_transform(all_emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in (("Real", "#1f4e79"), ("Generated", "#c0392b")):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6, label=label, color=color)
    ax.set_title("UMAP: Real vs. Generated Handwriting Embeddings")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved UMAP plot to %s", output_path)
    return output_path
