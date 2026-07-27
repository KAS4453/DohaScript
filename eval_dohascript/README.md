# GANwriting-Devanagari Evaluation Framework

A modular evaluation suite for a writer-conditioned handwritten Devanagari
word/line generation GAN, tracking distributional, perceptual, structural,
and OCR-based metrics across training epochs (checkpoints saved every 500
iterations, up to 50,000 epochs).

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional packages (`piq`, `piqe`, CLIP) power BRISQUE/NIQE/PIQE and the
CLIP embedding backbone; if they are not installed, the corresponding
metrics are skipped with a log message rather than crashing the run.

Set your Sarvam OCR credentials before running OCR-based metrics:

```bash
export SARVAM_API_KEY="your-key-here"     # Windows: set SARVAM_API_KEY=...
```

If you already have a working Sarvam OCR client in your project, drop it
on the Python path as `sarvam_ocr_client.py` exposing either a
`recognize(image_path) -> dict` function or a `SarvamOCRClient` class with
a `.recognize(image_path, language_code=...)` method — `metrics/ocr.py`
will auto-detect and reuse it instead of its own REST fallback.

## 2. Directory Structure

```
evaluation/
├── config.py                    # Paths, ground-truth text, hyperparameters
├── requirements.txt
├── main.py                      # CLI entry point / orchestration
├── utils.py                     # Logging, filename parsing, image IO, stats
├── metrics/
│   ├── fid.py                   # Frechet Inception Distance
│   ├── kid.py                   # Kernel Inception Distance
│   ├── ocr.py                   # CER / WER / char & word accuracy (Sarvam OCR)
│   ├── msssim.py                # Multi-scale SSIM (intra-epoch diversity)
│   ├── entropy.py               # Shannon entropy, histogram stats, connected components
│   ├── sharpness.py             # Variance of Laplacian
│   ├── stroke_density.py        # Otsu foreground ratio + stroke width
│   ├── blur.py                  # Laplacian / Tenengrad / Brenner
│   ├── projection.py            # Horizontal/vertical projection profiles
│   ├── skeleton.py              # Skeleton length, branch points, endpoints
│   ├── edge_density.py          # Canny edge density
│   ├── noise.py                 # Immerkaer noise-sigma estimation
│   └── quality.py               # BRISQUE / NIQE / PIQE (optional deps)
├── plots/
│   ├── plot_metrics.py          # Generic metric-vs-epoch plots
│   ├── plot_training_curves.py  # Multi-metric dashboard + qualitative strip
│   ├── plot_tsne.py             # t-SNE of real vs. generated embeddings
│   └── plot_umap.py             # UMAP of real vs. generated embeddings
└── results/
    ├── tables/                  # CSV / JSON outputs
    ├── figures/                 # PNG / PDF / SVG plots
    └── logs/                    # Run logs
```

## 3. Usage

Run everything with defaults from `config.py`:

```bash
python main.py
```

Override the generated/real image directories:

```bash
python main.py --generated img3 --real dataset_real
```

Run a single metric family (useful for iterating or for expensive stages
like OCR/FID):

```bash
python main.py --only fid
python main.py --only ocr
python main.py --only fid,kid,ocr
```

Other flags: `--output`, `--device {cuda,cpu}`, `--sarvam-api-key`,
`--max-ocr-images-per-epoch` (caps Sarvam OCR calls per epoch for cost
control; default 5).

### Filename convention

Generated files are expected to look like `epoch_<epoch>-<iteration>.png`
(e.g. `epoch_1000-23000.png`). The epoch is parsed automatically via the
regex in `config.filename_regex`; unparsable filenames are skipped with a
warning rather than aborting the run.

## 4. Outputs

- `results/tables/metrics.csv` — a single wide table, one row per epoch,
  with every metric's columns prefixed by metric name (outer-joined).
- `results/tables/summary.csv` / `summary.json` — per-metric run summary
  (which metrics ran, how many epochs each covered).
- `results/tables/FID.csv`, `KID.csv`, `ocr_results.csv`,
  `ocr_summary.json`, `msssim.csv`, `sharpness.csv`, `blur.csv`,
  `stroke_density.csv`, `entropy.csv`, `skeleton.csv`, `projection.csv`,
  `edge_density.csv`, `noise.csv`, `quality.csv` — one CSV per metric
  family.
- `results/figures/metric_curves/*.{png,pdf,svg}` — one 300-dpi
  metric-vs-epoch plot per tracked metric.
- `results/figures/training_dashboard.png` — all metrics in one grid.
- `results/figures/qualitative_evolution.png` — a same-crop, same-scale
  strip of example outputs across epochs (e.g. 0 → 5000 → 10000 → 20000 →
  50000).
- `results/figures/tsne_real_vs_generated.png`,
  `umap_real_vs_generated.png` — embedding-space real-vs-generated plots.
- `results/figures/projections/*.png` — per-image horizontal/vertical
  projection profile plots.

## 5. How each metric is computed

| Metric | Method |
|---|---|
| FID | Inception-V3 pool3 activations (via `pytorch-fid`); Frechet distance between real and generated Gaussian-fitted feature distributions, per epoch against a fixed real reference set. |
| KID | Squared MMD between Inception features with a polynomial kernel (via `torch-fidelity`), reporting subset-sampled mean/std. |
| CER | Levenshtein edit distance over characters, divided by reference character count. |
| WER | Levenshtein edit distance over whitespace-tokenized words, divided by reference word count. |
| Character Accuracy | `1 - CER`. |
| Word Accuracy | Exact correct-word count (via edit-distance backtrace alignment) divided by total reference words. |
| MS-SSIM | Geometric mean of single-scale SSIM across progressively downsampled resolutions, computed pairwise within an epoch's generated samples. |
| Sharpness | Variance of the image Laplacian. |
| Blur | Laplacian energy, Tenengrad (Sobel gradient energy), and Brenner's focus measure. |
| Stroke Density | Foreground-pixel ratio after Otsu binarization. |
| Stroke Width | `2 x mean(distance-transform values on the morphological skeleton)`. |
| Image Entropy | Shannon entropy of the 256-bin pixel-intensity histogram. |
| Histogram Statistics | Mean intensity, std, contrast (`std/mean`), dynamic range (`max-min`). |
| Connected Components | Count, largest area, and average area of 8-connected ink components after Otsu thresholding. |
| Skeleton Statistics | `skimage.morphology.skeletonize`; total skeleton pixel count (stroke length proxy), branch points (≥3 skeleton neighbors), endpoints (1 skeleton neighbor). |
| Projection Profiles | Row-sum / column-sum of the binarized ink mask; scalar "energy" = profile variance. |
| Edge Density | Fraction of Canny-detected edge pixels. |
| Noise | Immerkaer (1996) closed-form Laplacian-kernel noise-sigma estimator. |
| Image Quality (BRISQUE/NIQE/PIQE) | No-reference quality scores via `piq` (BRISQUE, NIQE) and `piqe` (PIQE), where installed. |
| t-SNE / UMAP | Inception-V3 (or CLIP, if configured) embeddings of real and generated images, jointly projected to 2D. |

## 6. References

```bibtex
@inproceedings{heusel2017fid,
  title={GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
}

@inproceedings{binkowski2018kid,
  title={Demystifying MMD GANs},
  author={Bi{\'n}kowski, Miko{\l}aj and Sutherland, Danica J and Arbel, Michael and Gretton, Arthur},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}

@article{zhang2018lpips,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  journal={CVPR},
  year={2018}
}

@article{wang2003msssim,
  title={Multiscale Structural Similarity for Image Quality Assessment},
  author={Wang, Zhou and Simoncelli, Eero P and Bovik, Alan C},
  journal={Asilomar Conference on Signals, Systems \& Computers},
  year={2003}
}

@article{mittal2012brisque,
  title={No-Reference Image Quality Assessment in the Spatial Domain},
  author={Mittal, Anish and Moorthy, Anush K and Bovik, Alan C},
  journal={IEEE Transactions on Image Processing},
  year={2012}
}

@article{mittal2013niqe,
  title={Making a "Completely Blind" Image Quality Analyzer},
  author={Mittal, Anish and Soundararajan, Rajiv and Bovik, Alan C},
  journal={IEEE Signal Processing Letters},
  year={2013}
}

@inproceedings{venkatanath2015piqe,
  title={Blind Image Quality Evaluation Using Perception Based Features},
  author={Venkatanath, N and Praneeth, D and Bh, Maruthi Chandrasekhar and Channappayya, Sumohana S and Medasani, Swarup S},
  booktitle={National Conference on Communications (NCC)},
  year={2015}
}

@article{vandermaaten2008tsne,
  title={Visualizing Data using t-SNE},
  author={van der Maaten, Laurens and Hinton, Geoffrey},
  journal={Journal of Machine Learning Research},
  year={2008}
}

@article{mcinnes2018umap,
  title={UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction},
  author={McInnes, Leland and Healy, John and Melville, James},
  journal={arXiv preprint arXiv:1802.03426},
  year={2018}
}

@misc{sarvamocr,
  title={Sarvam OCR},
  author={{Sarvam AI}},
  howpublished={\url{https://www.sarvam.ai/}},
  year={2024}
}
```

## 7. Notes on reproducibility

- `config.random_seed` (default 42) seeds t-SNE, UMAP, and MS-SSIM pair
  subsampling.
- FID/KID always compute real-image statistics once and reuse them for
  every epoch, so all epochs are compared against an identical reference
  distribution.
- Every metric module degrades gracefully per-image (logs a warning and
  skips) rather than aborting the whole run on a single corrupt/unreadable
  file.
