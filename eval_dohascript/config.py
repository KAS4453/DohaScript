"""
config.py
=========
Central configuration for the GANwriting Devanagari evaluation framework.

All paths, constants, and tunables live here so that every module in the
package (and the CLI in main.py) shares a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Ground-truth text
# ---------------------------------------------------------------------------
# Every generated image is conditioned on (a line drawn from) this corpus.
# It is kept as a single module-level constant so that the OCR evaluation
# module can reuse it without re-reading a file from disk.
GROUND_TRUTH_TEXT: str = (
    "गुरु गोविंद दोऊ खड़े, काके लागूं पांय।\n"
    "बलिहारी गुरु आपने, गोविंद दियो बताय॥\n\n"
    "धीरे-धीरे रे मना, धीरे सब कुछ होय।\n"
    "माली सींचे सौ घड़ा, ऋतु आए फल होय॥\n\n"
    "दया धर्म का मूल है, पाप मूल अभिमान।\n"
    "तुलसी दया न छाँड़िये, जब लग घट में प्राण॥\n\n"
    "पोथी पढ़ि पढ़ि जग मुआ, पंडित भया न कोय।\n"
    "ढाई आखर प्रेम का, पढ़े सो पंडित होय॥\n\n"
    "सांच बराबर तप नहीं, झूठ बराबर पाप।\n"
    "जाके हिरदै सांच है, ताके हिरदै आप॥\n\n"
    "क्षेत्रपाल गुरु ज्ञान का, शुद्ध रखे विचार।\n"
    "षट्दर्शन सब जानिए, सद्गुरु ही आधार॥"
)

# Individual lines (dohas), useful when a generated image corresponds to a
# single line rather than the full corpus. Index 0 is the first line, etc.
GROUND_TRUTH_LINES: List[str] = [
    line.strip() for line in GROUND_TRUTH_TEXT.split("\n") if line.strip()
]


@dataclass
class Config:
    """Container for all runtime configuration."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    generated_dir: Path = Path(
        r"D:\Code&Study\Program Files\IISER Books\Projects\LaurelHardyFedProject"
        r"\research-GANwriting\img3"
    )
    real_dir: Path = Path("dataset_real")
    output_dir: Path = Path("results")
    tables_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # ------------------------------------------------------------------
    # Filename parsing
    # ------------------------------------------------------------------
    # Filenames look like: epoch_0-0.png, epoch_500-11500.png, ...
    # Group 1 = epoch number, Group 2 = global iteration number.
    filename_regex: str = r"epoch_(\d+)-(\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff)"

    # ------------------------------------------------------------------
    # Sarvam OCR
    # ------------------------------------------------------------------
    sarvam_api_key: str = field(default_factory=lambda: os.environ.get("SARVAM_API_KEY", ""))
    sarvam_language_code: str = "hi-IN"
    ocr_max_retries: int = 3
    ocr_retry_backoff_seconds: float = 2.0
    ocr_concurrent_requests: int = 4

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    device: str = "cuda"  # falls back to "cpu" automatically if unavailable
    num_workers: int = max(1, (os.cpu_count() or 2) - 1)
    batch_size: int = 32
    random_seed: int = 42

    # ------------------------------------------------------------------
    # Metric-specific knobs
    # ------------------------------------------------------------------
    fid_dims: int = 2048  # Inception feature dimensionality
    kid_subset_size: int = 100
    kid_subsets: int = 100
    msssim_max_pairs: int = 500  # cap pairwise comparisons for speed
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    embedding_backbone: str = "inception"  # "inception" or "clip"

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plot_dpi: int = 300
    plot_formats: List[str] = field(default_factory=lambda: ["png", "pdf", "svg"])
    figure_width_in: float = 8.0
    figure_height_in: float = 5.0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.logs_dir = self.output_dir / "logs"
        for d in (self.output_dir, self.tables_dir, self.figures_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_default_config() -> Config:
    """Return a fresh default :class:`Config` instance."""
    return Config()
