# DohaScript Line Segmentation & Quality Analysis Pipeline

This repository contains the **research-grade preprocessing, quality validation, and line segmentation pipeline** developed for the **DohaScript dataset**: a large-scale multi-writer dataset of continuous handwritten Hindi (Devanagari) text.

The codebase provides:

* Automated handwriting image quality filtering (blur + CNN-based refinement)
* Robust page-level **line segmentation** for handwritten Hindi documents
* Difficulty labeling (**Easy / Medium / Complex**) based on segmentation stability
* Statistical reporting and LaTeX table generation for paper-ready results
* Publication-quality figure generation

---

## Project

DohaScript is a controlled parallel handwriting dataset where **531 writers** transcribed the same six Hindi dohas. This setup enables analysis of handwriting variation independent of lexical content.

The dataset is described in the accompanying paper:

> *DohaScript: A Large-Scale Multi-Writer Dataset for Continuous Handwritten Hindi Text* (531 writers, continuous handwritten couplets).

The repository focuses on the **data validation stage**:

* Quality assessment via Laplacian variance blur scores
* CNN-based filtering to retain high-quality samples
* Line segmentation difficulty profiling over all pages

---

## Repository Structure

```bash
.
├── line_segmentation_research.py     # Main segmentation + scoring pipeline
├── generate_paper_figures.py         # Script for publication-ready plots
├── quality.ipynb                    # CNN-based blur classification notebook
└── README.md
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install numpy opencv-python pandas scipy matplotlib seaborn
```

(Optional, for CNN notebook):

```bash
pip install torch torchvision
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
| ---------- | ----------------------------------------- |
| Easy       | Perfect or near-perfect segmentation      |
| Medium     | Minor spacing/height irregularities       |
| Complex    | Overlaps, irregular baselines, high error |

Implemented in:

```python
classify_difficulty(score, line_diff)
```

---

## 📑 Research Outputs

### CSV Report

```bash
segmentation_results/segmentation_results_detailed.csv
```

Contains per-image:

* Method selected
* Processing time
* Segmentation score
* Difficulty label

### LaTeX Tables

Automatically generated:

```bash
segmentation_results/latex_tables.tex
```

Includes:

* Overall performance summary
* Difficulty distribution
* Method comparison
* Error statistics

---

## Publication-Quality Figures

After segmentation completes, run:

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

Binary filtering retained:

* **288 high-quality pages (54.2%)**

---

## Key Contributions

* First large-scale continuous handwritten Hindi dataset validation
* Automated quality separation beyond fixed blur thresholds
* Full-page line segmentation difficulty profiling
* Research-ready statistics, tables, and plots

---

## Citation

If you use this pipeline or dataset, please cite:

```bibtex
@article{dohaScript2026,
  title={DohaScript: A Large-Scale Multi-Writer Dataset for Continuous Handwritten Hindi Text},
  author={Singh, Kunwar Arpit and Prakash, Ankush and Lone, Haroon R.},
  year={2026}
}
```

If you find this useful, consider starring the repo!
