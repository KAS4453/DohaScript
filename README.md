# Continuous Handwritten Devanagari Corpus — Data Pipeline & Baseline Experiments

This repository contains the **research-grade preprocessing, quality validation, line segmentation, OCR benchmarking, and handwriting generation pipeline** developed for a large-scale multi-writer dataset of continuous handwritten Hindi (Devanagari) text.

The codebase provides:

* Automated handwriting image quality filtering (blur + CNN-based refinement)
* Robust page-level **line segmentation** for handwritten Hindi documents
* Difficulty labeling (**Easy / Medium / Complex**) based on segmentation stability
* **OCR benchmarking** across Tesseract, TrOCR, and Sarvam AI, with a synthetic-data control experiment
* A **word-level handwriting generation baseline** (adapted GANwriting) for writer-conditioned style synthesis
* Statistical reporting for paper-ready results
* Publication-quality figure generation

---

## Project

The corpus is a controlled parallel handwriting dataset where **531 writers** transcribed the same six traditional Hindi dohas (couplets). This setup enables analysis of handwriting variation independent of lexical content.

The dataset and experiments are described in the accompanying paper:

> *A Parallel Multi-Writer Benchmark for Continuous Handwritten Devanagari Recognition* (531 writers, continuous handwritten couplets).

The repository now spans four stages of the research pipeline:

* **Data validation**: quality assessment via Laplacian variance blur scores and CNN-based filtering
* **Structural profiling**: line segmentation difficulty labeling over all pages
* **Recognition benchmarking**: OCR evaluation (Tesseract, TrOCR, Sarvam AI) on real vs. synthetic handwriting
* **Generative modeling**: a word-level GANwriting baseline for writer-style-conditioned synthesis

---

## Repository Structure

```bash
.
├── line_segmentation_research.py     # Main segmentation + scoring pipeline
├── generate_paper_figures.py         # Script for publication-ready plots
├── quality.ipynb                     # CNN-based blur classification notebook
├── adapted-GANwriting/               # Devanagari-adapted GANwriting model, training scripts
├── eval_dohascript/                  # Evaluation of Generated Images
├── ocr_benchmark (1).py              # OCR benchmarking (Tesseract, TrOCR, Sarvam AI) + synthetic reference eval
└── README.md
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install numpy opencv-python pandas scipy matplotlib seaborn
```

(Optional, for CNN notebook and generation baseline):

```bash
pip install torch torchvision
```

(Optional, for OCR benchmarking):

```bash
pip install pytesseract transformers
```

---

## Running the Line Segmentation Pipeline

The main script performs segmentation across all handwritten pages:

```bash
python line_segmentation_research.py
```

### Input

* Place dataset images inside:

```bash
Combined/
```

Each image is expected to contain **12 handwritten text lines**.

### Output

Results are stored in:

```bash
segmentation_results/
```

Including:

* Line-marked visualizations
* Per-page segmentation metrics
* Difficulty labels
* LaTeX tables

---

## Segmentation Methods Implemented

The pipeline combines three classical approaches:

### 1. Horizontal Projection

Detects valleys in row-wise pixel density.

### 2. Contour-Based Grouping

Uses dilation + bounding box merging.

### 3. Morphological Segmentation

Handles shirorekha removal and connected components.

### Hybrid Selection

The final method is chosen based on which produces line counts closest to the expected 12:

```python
hybrid_segmentation(image)
```

---

## 📊 Segmentation Quality Metrics

For each page, the pipeline computes:

* Number of lines detected
* Line count error (|detected − 12|)
* Line height variance
* Inter-line spacing uniformity
* Coverage ratio
* Straightness score

An overall segmentation score is produced:

```text
Score range: 0 – 100
```

---

## 🏷 Difficulty Classification

Each page is labeled as:

| Difficulty | Criteria                                  |
| ---------- | ------------------------------------------ |
| Easy       | Perfect or near-perfect segmentation       |
| Medium     | Minor spacing/height irregularities        |
| Complex    | Overlaps, irregular baselines, high error  |

Implemented in:

```python
classify_difficulty(score, line_diff)
```

Result summary (N=531 pages): 20.7% Easy, 26.6% Medium, 52.7% Complex, with 29.57% of pages achieving perfect (12/12 line) segmentation.

---

## 🧪 New: OCR Benchmark Experiments

To quantify how well existing OCR systems handle continuous handwritten Devanagari, the pipeline now benchmarks:

* **Tesseract** (four configurations: default, OEM 3/PSM 6, PSM 4, Hindi+English)
* **TrOCR** (`microsoft/trocr-base-handwritten`, zero-shot cross-script)
* **Sarvam AI** Document Intelligence (contemporary Indic-specific OCR API)

Run with:

```bash
cd eval_dohascript/
python ocr_benchmark.py
```

Metrics reported: **Character Error Rate (CER)** and **Word Error Rate (WER)**, computed via edit distance against the fixed six-doha ground truth.

**Key result:** Sarvam AI is the strongest system evaluated (CER = 0.138, WER = 0.272), substantially outperforming Tesseract (best CER = 0.594) and zero-shot TrOCR (CER = 0.995), but still misrecognizes more than 1 in 4 words — indicating continuous handwritten Devanagari recognition remains largely unsolved.

### Synthetic Reference Control

To isolate whether OCR errors stem from genuine handwriting complexity or system limitations, a 60-page synthetic reference set (same six-doha text, generated via prompt-based handwriting synthesis with varied stroke/slant/spacing) is evaluated alongside the real dataset:

```bash
cd eval_dohascript/
python synthetic_reference_eval.py
```

Sarvam AI achieves CER = 0.065 / WER = 0.142 on synthetic pages vs. CER = 0.138 / WER = 0.272 on real pages — a ~2× gap confirming that authentic, writer-attributed handwriting is meaningfully harder than clean synthetic approximations, and that benchmarks relying solely on synthetic data would overestimate real-world accuracy.

---

## 🧪 New: Handwriting Generation Baseline (GANwriting)

A word-level generative baseline, adapted from GANwriting, is included to demonstrate that the dataset's shared-lexicon design supports writer-conditioned style synthesis:

```bash
cd adapted-GANwriting/
python train.py
```

Adaptations for Devanagari:

* Vocabulary extended to independent vowels, consonants, matras, and conjuncts
* Text rendering modified to compose the shirorekha and vertically-stacked conjuncts
* Input resolution widened for the larger horizontal extent of Hindi words

**Key results (50,000 training epochs):**

* CER on generated words drops from 100% → 32.31%; CER on **swapped-word** generation (novel lexical content, same writer style) drops to 15.58% — lower than direct generation, confirming style is preserved independent of content.
* Handwriting-domain distributional metrics (HFID = 47.09, KID = 0.234, real–gen cosine similarity = 0.844) show generated samples occupy the same broad embedding region as real handwriting.
* Training is stable across all 50,000 adversarial steps with no discriminator/generator collapse.

This baseline is not intended as state-of-the-art synthesis, but establishes the dataset as a reproducible benchmark for generative handwriting and style-transfer research on low-resource Indic scripts. Page-level generation is left as future work (see paper Appendix E for rationale on word-level scope).

---

## Publication-Quality Figures

After segmentation, OCR benchmarking, and generation experiments complete, run:

```bash
python generate_paper_figures.py
```

Generates:

* Score distribution
* Error distribution
* Method comparison
* Difficulty analysis
* Feature correlation heatmap
* Processing time plots
* OCR CER/WER comparison charts
* Generation training curves (CER, HFID, KID over training)

Saved to:

```bash
segmentation_results/figures/
```

---

## Quality Filtering (CNN Notebook)

The notebook `quality.ipynb` implements:

* Laplacian blur score computation
* Binary + four-class CNN quality classifiers
* Dataset refinement into high-quality core subset
* Inter-rater validation against independent human judgment (Cohen's κ = 0.89 for quality, κ = 0.84 for segmentation difficulty)

Binary filtering retained:

* **288 high-quality pages (54.2%)**

---

## Key Contributions

* First large-scale continuous handwritten Hindi dataset with writer-attributed, page-level real handwriting
* Automated quality separation beyond fixed blur thresholds, validated against human raters
* Full-page line segmentation difficulty profiling
* Multi-system OCR benchmarking (Tesseract, TrOCR, Sarvam AI) establishing that continuous Devanagari recognition remains an open problem
* Synthetic-vs-real OCR comparison isolating genuine handwriting difficulty from system limitations
* A word-level generative baseline demonstrating writer-style disentanglement from lexical content
* Research-ready statistics, tables, and plots

---

## Citation

If you use this pipeline or dataset, please cite:

```bibtex
@article{devanagariBenchmark2027,
  title={A Parallel Multi-Writer Benchmark for Continuous Handwritten Devanagari Recognition},
  author={Singh, Kunwar Arpit and Prakash, Ankush and Lone, Haroon R.},
  year={2027}
}
```

If you find this useful, consider starring the repo!
