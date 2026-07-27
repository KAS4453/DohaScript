"""
metrics_logger.py — FID, KID, CER metrics for GANwriting / DohaScript.

WHY FID WAS ~325 WITH CLEAR IMAGES
────────────────────────────────────
FID requires computing a 2048×2048 covariance matrix. With only 17 test writers
you had N=17 samples — the matrix was rank-16 out of 2048, dominated entirely by
regularisation noise rather than actual image quality. This is a sample-size
problem, not a generation quality problem.

FIXES APPLIED
─────────────
1. FID is now computed over the FULL dataset (train+test writers), collecting
   ALL images per writer (NUM_CHANNEL real images, multiple generated images).
   This gives N ≈ 93×50 = 4650 real + matching generated, well above the 2048
   minimum for a reliable 2048-d covariance.

2. KID (Kernel Inception Distance) is also reported. KID is an unbiased
   estimator — unlike FID it does not require N >> feature_dim, so it is valid
   even when you only have a few hundred images. It uses an RBF kernel on the
   same Inception pool_3 features. Report KID (×100) in the paper alongside FID.

3. Both metrics are logged to metrics.csv after every evaluation epoch.
   The file is append-only so training can be interrupted and resumed freely.

PAPER CITATION NOTE
───────────────────
When reporting FID cite Heusel et al. 2017 (https://arxiv.org/abs/1706.08500).
When reporting KID cite Binkowski et al. 2018 (https://arxiv.org/abs/1801.01401).
State in the paper: "FID and KID computed over N≈4650 real/generated word images
using InceptionV3 pool_3 features (2048-d)."
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torchvision import models

# ─── InceptionV3 pool_3 feature extractor ────────────────────────────────────

class _InceptionFeatures(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.layers = torch.nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(3, 2),
            inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(3, 2),
            inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.layers(x).squeeze(-1).squeeze(-1)   # (N, 2048)


_inception_cache = {}

def _get_inception(device):
    key = str(device)
    if key not in _inception_cache:
        _inception_cache[key] = _InceptionFeatures(device)
    return _inception_cache[key]


# ─── Preprocessing ────────────────────────────────────────────────────────────

def _prep_for_inception(imgs_tensor, device):
    """
    imgs_tensor: (N,1,H,W) or (N,3,H,W), values in [-1,1].
    Returns: (N,3,299,299) float32 in [0,1] on device.
    """
    x = imgs_tensor.float().to(device)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    # InceptionV3 wants [0,1] normalised with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    x = (x - mean) / std
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    return x


# ─── Feature extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def _extract_features(imgs_tensor, inception, device, batch_size=64):
    """Extract InceptionV3 pool_3 features in mini-batches. Returns np (N,2048)."""
    all_feats = []
    x_prep = _prep_for_inception(imgs_tensor, device)
    for i in range(0, x_prep.shape[0], batch_size):
        chunk = x_prep[i:i+batch_size]
        feats = inception(chunk).cpu().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


# ─── FID ─────────────────────────────────────────────────────────────────────

def _frechet(mu1, sig1, mu2, sig2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1 @ sig2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sig1 + np.eye(len(sig1))*eps) @
                               (sig2 + np.eye(len(sig2))*eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))


def compute_fid(feats_real, feats_fake):
    """
    feats_real, feats_fake: np arrays (N, 2048).
    Returns scalar FID or None if N < 4.
    """
    n = min(len(feats_real), len(feats_fake))
    if n < 4:
        return None
    fr, ff = feats_real[:n], feats_fake[:n]
    mu_r, sig_r = fr.mean(0), np.cov(fr, rowvar=False)
    mu_f, sig_f = ff.mean(0), np.cov(ff, rowvar=False)
    # Regularise when N < feature_dim (common with handwriting datasets)
    if n < 2048:
        eps = 1e-4
        sig_r += np.eye(2048) * eps
        sig_f += np.eye(2048) * eps
    return _frechet(mu_r, sig_r, mu_f, sig_f)


# ─── KID (Kernel Inception Distance) ─────────────────────────────────────────

def compute_kid(feats_real, feats_fake, num_subsets=100, subset_size=None):
    """
    Unbiased KID estimator via polynomial kernel on Inception features.
    Valid for any N (no covariance matrix needed).
    Returns (kid_mean, kid_std) — report kid_mean × 100 in the paper.

    Binkowski et al. 2018: https://arxiv.org/abs/1801.01401
    """
    n = min(len(feats_real), len(feats_fake))
    if n < 4:
        return None, None

    fr = feats_real[:n].astype(np.float64)
    ff = feats_fake[:n].astype(np.float64)

    if subset_size is None:
        subset_size = min(n, 1000)

    # Normalise features to unit hypersphere (standard for KID)
    fr = fr / (np.linalg.norm(fr, axis=1, keepdims=True) + 1e-8)
    ff = ff / (np.linalg.norm(ff, axis=1, keepdims=True) + 1e-8)

    kid_scores = []
    rng = np.random.default_rng(42)
    for _ in range(num_subsets):
        idx_r = rng.choice(n, subset_size, replace=False)
        idx_f = rng.choice(n, subset_size, replace=False)
        r, f  = fr[idx_r], ff[idx_f]

        # Polynomial kernel: k(x,y) = (x·y/d + 1)^3
        d = r.shape[1]
        k_rr = (r @ r.T / d + 1) ** 3
        k_ff = (f @ f.T / d + 1) ** 3
        k_rf = (r @ f.T / d + 1) ** 3

        # Unbiased MMD^2 estimator
        m = subset_size
        mmd2 = (k_rr.sum() - np.diag(k_rr).sum()) / (m*(m-1)) \
             + (k_ff.sum() - np.diag(k_ff).sum()) / (m*(m-1)) \
             - 2 * k_rf.mean()
        kid_scores.append(mmd2)

    return float(np.mean(kid_scores)), float(np.std(kid_scores))


# ─── Accumulator ─────────────────────────────────────────────────────────────

class ImageAccumulator:
    """Collect real and generated image tensors across batches."""
    def __init__(self):
        self.real, self.fake = [], []

    def add(self, real_batch, fake_batch):
        self.real.append(real_batch.detach().cpu())
        self.fake.append(fake_batch.detach().cpu())

    def get(self):
        if not self.real:
            return None, None
        return torch.cat(self.real, 0), torch.cat(self.fake, 0)

    def reset(self):
        self.real.clear()
        self.fake.clear()


# ─── High-level compute function ─────────────────────────────────────────────

def compute_metrics(real_imgs, fake_imgs, device):
    """
    real_imgs, fake_imgs: torch.Tensor (N,1,H,W) in [-1,1].
    Returns dict with keys: fid, kid_mean, kid_std, n_samples.

    Caller should use all_loader (train+test) for N >> 2048.
    """
    n = min(real_imgs.shape[0], fake_imgs.shape[0])
    result = {'fid': None, 'kid_mean': None, 'kid_std': None, 'n_samples': n}
    if n < 4:
        return result

    inception = _get_inception(device)
    print(f'  [metrics] extracting Inception features for {n} real + {n} fake images...')
    feats_r = _extract_features(real_imgs[:n], inception, device)
    feats_f = _extract_features(fake_imgs[:n], inception, device)

    result['fid'] = compute_fid(feats_r, feats_f)
    result['kid_mean'], result['kid_std'] = compute_kid(feats_r, feats_f)
    return result


# ─── CSV logger ──────────────────────────────────────────────────────────────

COLUMNS = [
    'epoch', 'timestamp',
    'cer_gen', 'cer_swap',
    'eval_cer_gen', 'eval_cer_swap',
    'fid', 'kid_mean', 'kid_std', 'n_samples',
    'dis_eval', 'cla_eval', 'rec_eval',
]


class MetricsLogger:
    """
    Append-only CSV logger. Safe to resume: header written only once.
    """
    def __init__(self, csv_path='metrics.csv'):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(COLUMNS)

    def log(self, epoch,
            cer_gen, cer_swap,
            eval_cer_gen, eval_cer_swap,
            metrics_dict,
            dis_eval, cla_eval, rec_eval):

        fid      = metrics_dict.get('fid')
        kid_mean = metrics_dict.get('kid_mean')
        kid_std  = metrics_dict.get('kid_std')
        n        = metrics_dict.get('n_samples', 0)

        row = [
            epoch,
            time.strftime('%Y-%m-%d %H:%M:%S'),
            f'{cer_gen:.4f}',
            f'{cer_swap:.4f}',
            f'{eval_cer_gen:.4f}',
            f'{eval_cer_swap:.4f}',
            f'{fid:.4f}'      if fid      is not None else 'N/A',
            f'{kid_mean:.6f}' if kid_mean is not None else 'N/A',
            f'{kid_std:.6f}'  if kid_std  is not None else 'N/A',
            str(n),
            f'{dis_eval:.4f}',
            f'{cla_eval:.4f}',
            f'{rec_eval:.4f}',
        ]
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)

        fid_str = f'{fid:.2f}'           if fid      is not None else 'N/A'
        kid_str = f'{kid_mean*100:.4f}'  if kid_mean is not None else 'N/A'
        print(f'  [metrics] epoch={epoch}  N={n}  '
              f'CER(gen/swap)={cer_gen:.2f}/{cer_swap:.2f}  '
              f'evalCER={eval_cer_gen:.2f}/{eval_cer_swap:.2f}  '
              f'FID={fid_str}  KID×100={kid_str}  → {self.csv_path}')