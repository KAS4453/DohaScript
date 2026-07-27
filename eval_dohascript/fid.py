#!/usr/bin/env python3
"""
fid.py — Handwriting-FID (HFID) evaluation pipeline

Compares a folder of REAL Devanagari handwritten images against a folder
of GENERATED images using embeddings from a handwriting-recognition model
(NOT Inception-v3 / ImageNet features).

--------------------------------------------------------------------------
IMPORTANT / HONEST LIMITATION (please read before citing this in a paper)
--------------------------------------------------------------------------
There is currently no publicly available Hugging Face encoder trained
specifically on Indic / Devanagari handwriting recognition. Realistic
candidates and why they fall short:

  - "microsoft/trocr-base-handwritten" / "microsoft/trocr-small-handwritten"
    are trained on IAM (English handwriting only).
  - Public PARSeq checkpoints are trained on Latin-script scene text.
  - CRNN checkpoints on the Hub are almost all Latin-script (IAM/synthetic
    English) too.

None of these have ever seen Devanagari glyphs, so their features encode
stroke/ink texture and Latin glyph-shape priors, not Devanagari structure.
Using one of them and calling it "Handwriting FID for Devanagari" without
disclosure would be a misleading methodology claim in a KDD submission.

Given that, this script defaults to "microsoft/trocr-base-handwritten"
(its ViT encoder, mean-pooled) per the instruction to fall back to
TrOCR/PARSeq when no Indic-specific model exists — but it:
  1. Prints a runtime warning about this.
  2. Records the exact model id used in metrics.json / the text report,
     so the limitation is auditable rather than hidden.
  3. Exposes --encoder so this can point at a fine-tuned Devanagari HTR
     encoder the moment one exists (recommended before using HFID numbers
     in the paper itself).

Usage:
    python evaluate.py --real images2 --generated img5 --output results

Key flags:
    --encoder microsoft/trocr-base-handwritten   (HF model id or local path)
    --device auto|cuda|cpu
    --n-bootstrap 1000
    --kid-subsets 10
    --kid-subset-size 100
"""

import argparse
import csv
import json
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Handwriting-FID (HFID) / KID / PRDC / MMD evaluation of "
                     "generated vs. real Devanagari handwriting images using "
                     "handwriting-recognition embeddings (not Inception)."
    )
    p.add_argument("--real", required=True, help="Directory of real images.")
    p.add_argument("--generated", required=True, help="Directory of generated images.")
    p.add_argument("--output", required=True, help="Output directory for results/plots.")
    p.add_argument("--encoder", default="microsoft/trocr-base-handwritten",
                    help="HF model id (or local path) of the handwriting-recognition "
                         "model whose encoder is used for embeddings. See the module "
                         "docstring for why no true Devanagari-specific public model "
                         "currently exists.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-bootstrap", type=int, default=1000,
                    help="Bootstrap resamples for HFID 95%% CI.")
    p.add_argument("--kid-subset-size", type=int, default=100)
    p.add_argument("--kid-subsets", type=int, default=10,
                    help="Number of random subsets to repeat KID over.")
    p.add_argument("--n-neighbors", type=int, default=5,
                    help="k for nearest-neighbor retrieval grid and PRDC.")
    p.add_argument("--image-ext", default=".png,.jpg,.jpeg")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_images(folder, exts):
    """Return sorted list of image paths under `folder` matching `exts`.
    No labels, no pairing, no filename matching between real/generated."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    exts = {e.strip().lower() for e in exts}
    files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No images with extensions {sorted(exts)} found under {folder}")
    return files


def get_device(requested):
    import torch
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: --device cuda requested but CUDA unavailable; using CPU.",
                  file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_handwriting_encoder(model_id, device):
    """
    Loads a handwriting-recognition model's vision encoder and its matching
    image processor. Returns (encode_fn, embedding_dim, processor).

    encode_fn(list_of_PIL_images) -> np.ndarray (N, D), gradients disabled.

    Supports TrOCR-style VisionEncoderDecoderModel checkpoints (default) and
    falls back to a generic AutoModel/AutoImageProcessor pair for other
    encoder-only HTR checkpoints (e.g. a custom fine-tuned Devanagari model
    passed via --encoder).
    """
    import torch

    print(f"Loading handwriting-recognition encoder: {model_id}", file=sys.stderr)
    print("NOTE: see module docstring — no public Devanagari-specific HTR "
          "encoder currently exists on the Hub; verify this choice before "
          "using HFID numbers in a publication.", file=sys.stderr)

    try:
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        processor = TrOCRProcessor.from_pretrained(model_id)
        full_model = VisionEncoderDecoderModel.from_pretrained(model_id)
        encoder = full_model.encoder.to(device).eval()
        embedding_dim = encoder.config.hidden_size

        @torch.no_grad()
        def encode_fn(pil_images):
            inputs = processor(images=pil_images, return_tensors="pt").pixel_values.to(device)
            out = encoder(pixel_values=inputs).last_hidden_state  # (N, tokens, D)
            pooled = out.mean(dim=1)  # mean-pool over patch tokens
            return pooled.detach().cpu().numpy()

        return encode_fn, embedding_dim, processor

    except Exception as e:
        print(f"TrOCR-style loading failed ({e}); falling back to generic "
              f"AutoModel/AutoImageProcessor.", file=sys.stderr)
        from transformers import AutoModel, AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        embedding_dim = model.config.hidden_size

        @torch.no_grad()
        def encode_fn(pil_images):
            inputs = processor(images=pil_images, return_tensors="pt").pixel_values.to(device)
            out = model(pixel_values=inputs).last_hidden_state
            pooled = out.mean(dim=1)
            return pooled.detach().cpu().numpy()

        return encode_fn, embedding_dim, processor


def _load_pil(path):
    from PIL import Image
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def extract_embeddings(paths, encode_fn, batch_size, desc=""):
    """extract_embeddings(): one embedding per image, inference mode, batched."""
    feats = []
    kept_paths = []
    n = len(paths)
    for start in range(0, n, batch_size):
        chunk_paths = paths[start:start + batch_size]
        imgs = []
        for p in chunk_paths:
            try:
                imgs.append(_load_pil(p))
                kept_paths.append(p)
            except Exception as e:
                print(f"WARNING: skipping unreadable image {p}: {e}", file=sys.stderr)
        if not imgs:
            continue
        feats.append(encode_fn(imgs))
        done = min(start + batch_size, n)
        print(f"  [{desc}] {done}/{n}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    if not feats:
        raise RuntimeError(f"No usable images for '{desc}'.")
    return np.concatenate(feats, axis=0), kept_paths


def _stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f"WARNING: sqrtm gave non-trivial imaginary part "
                  f"(max |imag|={np.max(np.abs(covmean.imag)):.2e}); using real part.",
                  file=sys.stderr)
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_hfid(real_feats, gen_feats):
    """FID equation, unmodified, computed on handwriting-encoder embeddings."""
    mu_r, sigma_r = _stats(real_feats)
    mu_g, sigma_g = _stats(gen_feats)
    return _frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


def bootstrap_hfid(real_feats, gen_feats, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_real, n_gen = real_feats.shape[0], gen_feats.shape[0]
    values = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        r_idx = rng.integers(0, n_real, size=n_real)
        g_idx = rng.integers(0, n_gen, size=n_gen)
        values[i] = compute_hfid(real_feats[r_idx], gen_feats[g_idx])
        if (i + 1) % 20 == 0 or i == n_bootstrap - 1:
            print(f"  [bootstrap] {i + 1}/{n_bootstrap}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    mean = float(np.mean(values))
    std = float(np.std(values))
    ci_low, ci_high = (float(x) for x in np.percentile(values, [2.5, 97.5]))
    return {
        "mean": mean, "std": std, "ci_95_low": ci_low, "ci_95_high": ci_high,
        "n_bootstrap": n_bootstrap, "values": values.tolist(),
    }


def _poly_kernel(x, y, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x.dot(y.T) + coef0) ** degree


def _kid_mmd(x, y):
    m, n = x.shape[0], y.shape[0]
    k_xx = _poly_kernel(x, x)
    k_yy = _poly_kernel(y, y)
    k_xy = _poly_kernel(x, y)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    term_xx = k_xx.sum() / (m * (m - 1))
    term_yy = k_yy.sum() / (n * (n - 1))
    term_xy = 2.0 * k_xy.sum() / (m * n)
    return term_xx + term_yy - term_xy


def compute_kid(real_feats, gen_feats, subset_size=100, n_subsets=10, seed=42):
    """KID: unbiased polynomial-kernel MMD^2 estimator (Binkowski et al., 2018),
    repeated over `n_subsets` random subsets; reports mean and std."""
    rng = np.random.default_rng(seed)
    n_real, n_gen = real_feats.shape[0], gen_feats.shape[0]
    subset_size = max(2, min(subset_size, n_real, n_gen))
    values = np.empty(n_subsets, dtype=np.float64)
    for i in range(n_subsets):
        r_idx = rng.choice(n_real, size=subset_size, replace=False)
        g_idx = rng.choice(n_gen, size=subset_size, replace=False)
        values[i] = _kid_mmd(real_feats[r_idx], gen_feats[g_idx])
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "subset_size": int(subset_size), "n_subsets": int(n_subsets),
        "values": values.tolist(),
    }


def compute_mmd(real_feats, gen_feats, gamma=None):
    """Standard biased RBF-kernel MMD^2 between the two full feature sets
    (median-heuristic bandwidth), complementing the KID subset estimator."""
    from scipy.spatial.distance import cdist

    if gamma is None:
        combined = np.concatenate([real_feats, gen_feats], axis=0)
        d2 = cdist(combined, combined, "sqeuclidean")
        median_d2 = np.median(d2[d2 > 0])
        gamma = 1.0 / (2.0 * median_d2) if median_d2 > 0 else 1.0 / real_feats.shape[1]

    def rbf(a, b):
        d2 = cdist(a, b, "sqeuclidean")
        return np.exp(-gamma * d2)

    k_rr = rbf(real_feats, real_feats)
    k_gg = rbf(gen_feats, gen_feats)
    k_rg = rbf(real_feats, gen_feats)

    mmd2 = k_rr.mean() + k_gg.mean() - 2 * k_rg.mean()
    return {"mmd2": float(mmd2), "gamma": float(gamma)}


def _pairwise_dist(a, b):
    from scipy.spatial.distance import cdist
    return cdist(a, b, "euclidean")


def _kth_nn_radius(dist_matrix, k):
    sorted_d = np.sort(dist_matrix, axis=1)
    return sorted_d[:, k]


def compute_prdc(real_feats, gen_feats, k=5):
    """
    Precision/Recall/Density/Coverage using k-NN radii, following the
    manifold-estimation approach of Naeem et al. (2020).
    """
    k = max(1, min(k, real_feats.shape[0] - 1, gen_feats.shape[0] - 1))

    d_rr = _pairwise_dist(real_feats, real_feats)
    d_gg = _pairwise_dist(gen_feats, gen_feats)
    d_rg = _pairwise_dist(real_feats, gen_feats)  # (n_real, n_gen)

    real_radii = _kth_nn_radius(d_rr, k)
    gen_radii = _kth_nn_radius(d_gg, k)

    within_real_ball = d_rg.T <= real_radii[None, :]  # (n_gen, n_real)
    precision = float(np.mean(within_real_ball.any(axis=1)))

    within_gen_ball = d_rg <= gen_radii[None, :]  # (n_real, n_gen)
    recall = float(np.mean(within_gen_ball.any(axis=1)))

    density = float(np.mean(within_real_ball.sum(axis=1)) / k)

    coverage = float(np.mean((d_rg <= real_radii[:, None]).any(axis=1)))

    return {"precision": precision, "recall": recall, "density": density,
            "coverage": coverage, "k": int(k)}


def plot_pca(real_feats, gen_feats, output_path, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    combined = np.concatenate([real_feats, gen_feats], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    proj = pca.fit_transform(combined)
    n_real = real_feats.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(*proj[:n_real].T, s=16, alpha=0.6, label=f"Real (n={n_real})", c="#1f77b4")
    ax.scatter(*proj[n_real:].T, s=16, alpha=0.6,
               label=f"Generated (n={gen_feats.shape[0]})", c="#d62728")
    vr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({vr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({vr[1]*100:.1f}%)")
    ax.set_title("PCA of handwriting-encoder embeddings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_tsne(real_feats, gen_feats, output_path, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    combined = np.concatenate([real_feats, gen_feats], axis=0)
    perplexity = max(5, min(30, combined.shape[0] // 4))
    proj = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=seed, learning_rate="auto").fit_transform(combined)
    n_real = real_feats.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(*proj[:n_real].T, s=16, alpha=0.6, label=f"Real (n={n_real})", c="#1f77b4")
    ax.scatter(*proj[n_real:].T, s=16, alpha=0.6,
               label=f"Generated (n={gen_feats.shape[0]})", c="#d62728")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of handwriting-encoder embeddings (perplexity={perplexity})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_umap(real_feats, gen_feats, output_path, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import umap
    except ImportError:
        print("WARNING: umap-learn not installed (`pip install umap-learn`); "
              "skipping UMAP plot.", file=sys.stderr)
        return False

    combined = np.concatenate([real_feats, gen_feats], axis=0)
    n_neighbors = max(2, min(15, combined.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed)
    proj = reducer.fit_transform(combined)
    n_real = real_feats.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(*proj[:n_real].T, s=16, alpha=0.6, label=f"Real (n={n_real})", c="#1f77b4")
    ax.scatter(*proj[n_real:].T, s=16, alpha=0.6,
               label=f"Generated (n={gen_feats.shape[0]})", c="#d62728")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of handwriting-encoder embeddings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return True


def _cosine_matrix(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an.dot(bn.T)


def plot_cosine_similarity(real_feats, gen_feats, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rr = _cosine_matrix(real_feats, real_feats)
    gg = _cosine_matrix(gen_feats, gen_feats)
    rg = _cosine_matrix(real_feats, gen_feats)

    rr_vals = rr[np.triu_indices_from(rr, k=1)]
    gg_vals = gg[np.triu_indices_from(gg, k=1)]
    rg_vals = rg.flatten()

    fig, ax = plt.subplots(figsize=(7, 6))
    bins = 60
    ax.hist(rr_vals, bins=bins, alpha=0.5, density=True, label="real vs real")
    ax.hist(gg_vals, bins=bins, alpha=0.5, density=True, label="generated vs generated")
    ax.hist(rg_vals, bins=bins, alpha=0.5, density=True, label="real vs generated")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise cosine similarity (handwriting embeddings)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "real_vs_real": {"mean": float(rr_vals.mean()), "std": float(rr_vals.std())},
        "gen_vs_gen": {"mean": float(gg_vals.mean()), "std": float(gg_vals.std())},
        "real_vs_gen": {"mean": float(rg_vals.mean()), "std": float(rg_vals.std())},
    }


def plot_feature_norm_distribution(real_feats, gen_feats, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_norms = np.linalg.norm(real_feats, axis=1)
    gen_norms = np.linalg.norm(gen_feats, axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(real_norms, bins=40, alpha=0.6, density=True, label="Real")
    ax.hist(gen_norms, bins=40, alpha=0.6, density=True, label="Generated")
    ax.set_xlabel("Embedding L2 norm")
    ax.set_ylabel("Density")
    ax.set_title("Feature norm distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "real": {"mean": float(real_norms.mean()), "std": float(real_norms.std())},
        "generated": {"mean": float(gen_norms.mean()), "std": float(gen_norms.std())},
    }


def plot_hfid_bootstrap(bootstrap_values, output_path, point_estimate=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(bootstrap_values)
    ci_low, ci_high = np.percentile(values, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(values, bins=40, alpha=0.75, color="#4c72b0")
    ax.axvline(ci_low, color="black", linestyle="--", linewidth=1, label="95% CI")
    ax.axvline(ci_high, color="black", linestyle="--", linewidth=1)
    if point_estimate is not None:
        ax.axvline(point_estimate, color="#d62728", linewidth=1.5, label="HFID (point estimate)")
    ax.set_xlabel("Bootstrapped HFID")
    ax.set_ylabel("Count")
    ax.set_title("HFID bootstrap distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def nearest_neighbor_visualization(real_feats, real_paths, gen_feats, gen_paths,
                                    output_path, k=5, max_rows=8, seed=42):
    """
    For a sample of generated images, retrieve their k nearest real images in
    embedding space and save a grid: each row = [generated | top-k real].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(seed)
    n_gen = gen_feats.shape[0]
    n_rows = min(max_rows, n_gen)
    row_idx = rng.choice(n_gen, size=n_rows, replace=False)

    dists = _pairwise_dist(gen_feats[row_idx], real_feats)  # (n_rows, n_real)
    topk_idx = np.argsort(dists, axis=1)[:, :k]

    fig, axes = plt.subplots(n_rows, k + 1, figsize=(2.0 * (k + 1), 2.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, gi in enumerate(row_idx):
        ax = axes[row, 0]
        ax.imshow(Image.open(gen_paths[gi]).convert("RGB"))
        ax.set_title("Generated" if row == 0 else "", fontsize=9)
        ax.axis("off")
        for col, ri in enumerate(topk_idx[row]):
            ax = axes[row, col + 1]
            ax.imshow(Image.open(real_paths[ri]).convert("RGB"))
            ax.set_title(f"NN-{col + 1}" if row == 0 else "", fontsize=9)
            ax.axis("off")

    fig.suptitle(f"Generated \u2192 top-{k} nearest real neighbors (embedding space)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _package_versions():
    versions = {"python": platform.python_version()}
    for pkg in ["torch", "torchvision", "transformers", "numpy", "scipy",
                "matplotlib", "sklearn", "umap"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def save_results(metrics, output_dir):
    output_dir = Path(output_dir)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    flat = {
        "n_real_images": metrics["config"]["n_real_images"],
        "n_generated_images": metrics["config"]["n_generated_images"],
        "encoder": metrics["config"]["encoder"],
        "hfid": metrics["hfid"]["value"],
        "hfid_ci_95_low": metrics["hfid"]["bootstrap"]["ci_95_low"],
        "hfid_ci_95_high": metrics["hfid"]["bootstrap"]["ci_95_high"],
        "kid_mean": metrics["kid"]["mean"],
        "kid_std": metrics["kid"]["std"],
        "mmd2": metrics["mmd"]["mmd2"],
        "prdc_precision": metrics["prdc"]["precision"],
        "prdc_recall": metrics["prdc"]["recall"],
        "prdc_density": metrics["prdc"]["density"],
        "prdc_coverage": metrics["prdc"]["coverage"],
        "cosine_real_vs_real_mean": metrics["cosine_similarity"]["real_vs_real"]["mean"],
        "cosine_gen_vs_gen_mean": metrics["cosine_similarity"]["gen_vs_gen"]["mean"],
        "cosine_real_vs_gen_mean": metrics["cosine_similarity"]["real_vs_gen"]["mean"],
        "runtime_seconds": metrics["runtime_seconds"],
        "seed": metrics["config"]["seed"],
    }
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(flat.keys())
        writer.writerow(flat.values())

    lines = []
    lines.append("=" * 70)
    lines.append("HANDWRITING-FID EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("LIMITATION NOTICE:")
    lines.append(
        "  No publicly available Hugging Face encoder is trained specifically\n"
        "  on Devanagari handwriting. This report's embeddings come from\n"
        f"  '{metrics['config']['encoder']}', which is trained on English (IAM)\n"
        "  handwritten text (or whatever custom encoder was passed via --encoder).\n"
        "  Treat HFID/KID/PRDC/MMD numbers below as encoder-dependent and\n"
        "  disclose this choice in any paper draft."
    )
    lines.append("")
    lines.append(f"Real images:      {metrics['config']['n_real_images']}")
    lines.append(f"Generated images: {metrics['config']['n_generated_images']}")
    lines.append(f"Encoder:          {metrics['config']['encoder']}")
    lines.append(f"Embedding dim:    {metrics['config']['embedding_dim']}")
    lines.append(f"Device:           {metrics['config']['device']}")
    lines.append(f"Seed:             {metrics['config']['seed']}")
    lines.append("")
    lines.append(f"HFID:             {metrics['hfid']['value']:.4f}")
    lines.append(f"HFID 95% CI:      [{metrics['hfid']['bootstrap']['ci_95_low']:.4f}, "
                 f"{metrics['hfid']['bootstrap']['ci_95_high']:.4f}] "
                 f"(n_bootstrap={metrics['hfid']['bootstrap']['n_bootstrap']})")
    lines.append(f"KID:              {metrics['kid']['mean']:.6f} +/- {metrics['kid']['std']:.6f} "
                 f"(subset_size={metrics['kid']['subset_size']}, "
                 f"n_subsets={metrics['kid']['n_subsets']})")
    lines.append(f"MMD^2 (RBF):      {metrics['mmd']['mmd2']:.6f} (gamma={metrics['mmd']['gamma']:.3e})")
    lines.append("")
    lines.append("PRDC:")
    for k in ["precision", "recall", "density", "coverage"]:
        lines.append(f"  {k.capitalize():<10}: {metrics['prdc'][k]:.4f}")
    lines.append("")
    lines.append("Cosine similarity (mean +/- std):")
    for k, v in metrics["cosine_similarity"].items():
        lines.append(f"  {k:<15}: {v['mean']:.4f} +/- {v['std']:.4f}")
    lines.append("")
    lines.append(f"Runtime: {metrics['runtime_seconds']:.1f} s")
    lines.append("")
    lines.append("Package versions:")
    for k, v in metrics["package_versions"].items():
        lines.append(f"  {k:<15}: {v}")
    lines.append("")
    lines.append("Output files:")
    for k, v in metrics["plots"].items():
        lines.append(f"  {k:<25}: {v}")
    lines.append("=" * 70)

    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write("\n".join(lines))

    return output_dir / "metrics.json", output_dir / "metrics.csv", output_dir / "evaluation_report.txt"


def main():
    t_start = time.time()
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = [e if e.startswith(".") else f".{e}" for e in args.image_ext.split(",")]

    print("Scanning image directories (unpaired, no labels, no filename matching)...",
          file=sys.stderr)
    real_paths = load_images(args.real, exts)
    gen_paths = load_images(args.generated, exts)
    print(f"  real: {len(real_paths)} images from {args.real}", file=sys.stderr)
    print(f"  generated: {len(gen_paths)} images from {args.generated}", file=sys.stderr)

    device = get_device(args.device)
    print(f"Using device: {device}", file=sys.stderr)

    encode_fn, embedding_dim, _ = build_handwriting_encoder(args.encoder, device)

    print("Extracting real embeddings...", file=sys.stderr)
    real_feats, real_paths_kept = extract_embeddings(real_paths, encode_fn, args.batch_size, "real")
    print("Extracting generated embeddings...", file=sys.stderr)
    gen_feats, gen_paths_kept = extract_embeddings(gen_paths, encode_fn, args.batch_size, "generated")
    print(f"Embedding shapes: real={real_feats.shape}, generated={gen_feats.shape}", file=sys.stderr)

    np.save(output_dir / "features_real.npy", real_feats)
    np.save(output_dir / "features_generated.npy", gen_feats)

    print("Computing HFID...", file=sys.stderr)
    hfid_value = compute_hfid(real_feats, gen_feats)

    print(f"Bootstrapping HFID (n={args.n_bootstrap})...", file=sys.stderr)
    hfid_bootstrap = bootstrap_hfid(real_feats, gen_feats, n_bootstrap=args.n_bootstrap, seed=args.seed)

    print(f"Computing KID over {args.kid_subsets} subsets...", file=sys.stderr)
    kid = compute_kid(real_feats, gen_feats, subset_size=args.kid_subset_size,
                       n_subsets=args.kid_subsets, seed=args.seed)

    print("Computing MMD...", file=sys.stderr)
    mmd = compute_mmd(real_feats, gen_feats)

    print("Computing PRDC...", file=sys.stderr)
    prdc = compute_prdc(real_feats, gen_feats, k=args.n_neighbors)

    print("Plotting PCA...", file=sys.stderr)
    plot_pca(real_feats, gen_feats, output_dir / "pca_embeddings.png", seed=args.seed)

    print("Plotting t-SNE...", file=sys.stderr)
    plot_tsne(real_feats, gen_feats, output_dir / "tsne_embeddings.png", seed=args.seed)

    print("Plotting UMAP...", file=sys.stderr)
    umap_ok = plot_umap(real_feats, gen_feats, output_dir / "umap_embeddings.png", seed=args.seed)

    print("Plotting cosine similarity...", file=sys.stderr)
    cosine_stats = plot_cosine_similarity(real_feats, gen_feats, output_dir / "cosine_similarity.png")

    print("Plotting feature norm distribution...", file=sys.stderr)
    norm_stats = plot_feature_norm_distribution(
        real_feats, gen_feats, output_dir / "feature_norm_distribution.png")

    print("Plotting HFID bootstrap distribution...", file=sys.stderr)
    plot_hfid_bootstrap(hfid_bootstrap["values"], output_dir / "hfid_bootstrap.png",
                         point_estimate=hfid_value)

    print("Building nearest-neighbor retrieval grid...", file=sys.stderr)
    nearest_neighbor_visualization(
        real_feats, real_paths_kept, gen_feats, gen_paths_kept,
        output_dir / "retrieval_examples.png", k=args.n_neighbors, seed=args.seed)

    runtime_seconds = time.time() - t_start

    metrics = {
        "config": {
            "real_dir": str(args.real),
            "generated_dir": str(args.generated),
            "encoder": args.encoder,
            "embedding_dim": int(embedding_dim),
            "n_real_images": int(real_feats.shape[0]),
            "n_generated_images": int(gen_feats.shape[0]),
            "device": str(device),
            "seed": args.seed,
        },
        "hfid": {
            "value": hfid_value,
            "bootstrap": {k: v for k, v in hfid_bootstrap.items() if k != "values"},
        },
        "kid": {k: v for k, v in kid.items() if k != "values"},
        "mmd": mmd,
        "prdc": prdc,
        "cosine_similarity": cosine_stats,
        "feature_norms": norm_stats,
        "runtime_seconds": runtime_seconds,
        "package_versions": _package_versions(),
        "plots": {
            "hfid_bootstrap": str(output_dir / "hfid_bootstrap.png"),
            "pca_embeddings": str(output_dir / "pca_embeddings.png"),
            "tsne_embeddings": str(output_dir / "tsne_embeddings.png"),
            "umap_embeddings": str(output_dir / "umap_embeddings.png") if umap_ok else "skipped (umap-learn not installed)",
            "cosine_similarity": str(output_dir / "cosine_similarity.png"),
            "feature_norm_distribution": str(output_dir / "feature_norm_distribution.png"),
            "retrieval_examples": str(output_dir / "retrieval_examples.png"),
        },
    }

    json_path, csv_path, report_path = save_results(metrics, output_dir)

    print("\n=== Summary ===", file=sys.stderr)
    print(f"HFID: {hfid_value:.4f}  95% CI [{hfid_bootstrap['ci_95_low']:.4f}, "
          f"{hfid_bootstrap['ci_95_high']:.4f}]", file=sys.stderr)
    print(f"KID:  {kid['mean']:.6f} +/- {kid['std']:.6f}", file=sys.stderr)
    print(f"MMD^2: {mmd['mmd2']:.6f}", file=sys.stderr)
    print(f"PRDC: precision={prdc['precision']:.3f} recall={prdc['recall']:.3f} "
          f"density={prdc['density']:.3f} coverage={prdc['coverage']:.3f}", file=sys.stderr)
    print(f"Runtime: {runtime_seconds:.1f}s", file=sys.stderr)
    print(f"Results: {json_path}, {csv_path}, {report_path}", file=sys.stderr)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
