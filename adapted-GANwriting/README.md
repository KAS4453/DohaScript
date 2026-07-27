# GANwriting — DohaScript (Devanagari) Setup

## Files changed from original

| File | Status |
|------|--------|
| `load_data.py` | **Rewritten** — Devanagari vocab, DohaScript groundtruth format, auto writer mapping |
| `modules_tro.py` | **Modified** — `write_image()` uses PIL for Devanagari text rendering |
| `main_run.py` | **Modified** — imports from new `load_data`, no IAM-specific pairs file |
| `loss_tro.py` | **Minor fix** — CER filter uses `>= num_tokens` instead of per-token loop |
| `blocks.py` | Unchanged — copy from original zip |
| `network_tro.py` | Unchanged — copy from original zip |
| `vgg_tro_channel3_modi.py` | Unchanged — copy from original zip |
| `recognizer/` | Unchanged — copy entire folder from original zip |

## Directory layout

```
project/
├── load_data.py
├── modules_tro.py
├── main_run.py
├── loss_tro.py
├── network_tro.py
├── blocks.py
├── vgg_tro_channel3_modi.py
├── groundtruth.txt          ← tab-separated: images/WRITER_line_XX_word_YY.png \t word
├── images/                  ← word-level PNG crops
│   ├── F-18-BENGALURU_line_01_word_01.png
│   └── ...
├── save_weights/            ← created automatically
├── imgs/                    ← debug visualisations, created automatically
└── recognizer/
    └── models/
        ├── attention.py
        ├── decoder.py
        ├── encoder_vgg.py
        ├── seq2seq.py
        └── vgg_tro_channel3.py
```

## Configuration (edit load_data.py top section)

```python
DATASET_ROOT = '.'           # root containing images/ and groundtruth.txt
GT_FILE      = 'groundtruth.txt'
TRAIN_RATIO  = 0.85          # fraction of writers used for training
RANDOM_SEED  = 42
```

## Installation

```bash
pip install torch torchvision opencv-python Pillow python-Levenshtein
# For Devanagari text in debug images:
pip install Pillow
# Download NotoSansDevanagari-Regular.ttf to the project root, or set
# DEVANAGARI_FONT in modules_tro.py to your font path.
```

## Training

```bash
# From scratch
python main_run.py 0

# Resume from epoch 1000
python main_run.py 1000
```

## Key constants (auto-derived from groundtruth.txt)

| Constant | Value | Note |
|----------|-------|------|
| `NUM_WRITERS` | 110 | auto-counted |
| `vocab_size` | 59 | 56 chars + 3 special tokens |
| `MAX_CHARS` | 12 | longest Devanagari word is 10 codepoints |
| `IMG_HEIGHT` | 64 | same as original |
| `IMG_WIDTH` | 256 | wider than original 216 |
| `OUTPUT_MAX_LEN` | 14 | MAX_CHARS + GO + END |

## Notes

- **OOV=True** (default): the generator is trained to write *any* Devanagari word
  from the corpus, not just words seen per writer. This is the recommended mode.
- The text corpus for OOV sampling is built from all unique words in `groundtruth.txt`.
- `pairs_idx_wid_iam.py` is **not needed** — writer↔integer mapping is built automatically.
- Suspicious/failed segments from your pipeline are ignored since they don't have
  corresponding image files; `read_image()` returns a zero canvas for missing files.
