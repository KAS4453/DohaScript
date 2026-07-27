"""
plots/plot_tsne.py
===================
t-SNE visualization of real-vs-generated image embeddings, extracted
via an Inception-V3 or CLIP backbone (config.embedding_backbone).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from utils import list_images, load_image_rgb

logger = logging.getLogger("plots.tsne")


def extract_embeddings(paths: List[Path], backbone: str = "inception", device: str = "cuda", batch_size: int = 32) -> np.ndarray:
    """Extract a fixed-length embedding per image using Inception-V3 pooled features or CLIP."""
    import torch

    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    if backbone == "clip":
        import clip  # openai-clip

        model, preprocess = clip.load("ViT-B/32", device=dev)
        model.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                from PIL import Image
                imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(dev)
                feats = model.encode_image(imgs)
                embeddings.append(feats.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    # default: Inception-V3 pooled features via pytorch-fid's wrapper
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import get_activations

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(dev)
    model.eval()
    return get_activations([str(p) for p in paths], model, batch_size, 2048, dev)


def plot_tsne_real_vs_generated(
    real_dir: Path,
    generated_dir: Path,
    output_path: Path,
    backbone: str = "inception",
    device: str = "cuda",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
    dpi: int = 300,
) -> Path:
    """Compute embeddings for both folders, run t-SNE jointly, and plot a labeled scatter."""
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_paths = list_images(real_dir)
    gen_paths = list_images(generated_dir)
    if not real_paths or not gen_paths:
        raise FileNotFoundError("Both real_dir and generated_dir must contain images for t-SNE.")

    real_emb = extract_embeddings(real_paths, backbone, device)
    gen_emb = extract_embeddings(gen_paths, backbone, device)

    all_emb = np.concatenate([real_emb, gen_emb], axis=0)
    labels = np.array(["Real"] * len(real_emb) + ["Generated"] * len(gen_emb))

    effective_perplexity = min(perplexity, max(5, (len(all_emb) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=effective_perplexity, n_iter=n_iter, random_state=seed, init="pca")
    coords = tsne.fit_transform(all_emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in (("Real", "#1f4e79"), ("Generated", "#c0392b")):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6, label=label, color=color)
    ax.set_title("t-SNE: Real vs. Generated Handwriting Embeddings")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved t-SNE plot to %s", output_path)
    return output_path
