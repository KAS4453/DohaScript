"""
=============================================================================
  Hindi Handwritten OCR Benchmarking Pipeline
  ─────────────────────────────────────────────
  Compares Tesseract (multiple configs) vs Indic/Transformer OCR on 531
  handwritten Hindi (Devanagari) page images.


  Usage:
      python ocr_benchmark.py --input ./Combined --output ./results
      python ocr_benchmark.py --input ./Combined --output ./results --no-preprocess
      python ocr_benchmark.py --input ./Combined --output ./results --workers 4
=============================================================================
"""


import argparse
import csv
import hashlib
import html
import io
import json
import logging
import os
import sys
import time
import unicodedata
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import cv2
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SARVAM_API_KEYS: List[str] = [
    os.environ.get("SARVAM_KEY_7", "sk_t8pwexi5_w3GssCdfJe5VCujYrPfTnZjN"),   # ← Backup key 6
    os.environ.get("SARVAM_KEY_8", "sk_aqpe5dfl_QuzvSoxz72UU6RmTii6wGyT5"),   # ← Backup key 7

]


# Filter out empty / placeholder entries so the rotation list is clean.
_ACTIVE_SARVAM_KEYS: List[str] = [
    k for k in SARVAM_API_KEYS if k.strip() and k != "YOUR_SARVAM_API_KEY"
]

# Index of the key that is currently in use (shared within a single process).
_sarvam_key_index: int = 0
# ─────────────────────────────────────────────────────────────────────────────


# ── optional heavy imports (fail gracefully) ────────────────────────────────
try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    warnings.warn("pytesseract not found – Tesseract methods will be skipped.")


try:
    from jiwer import cer, wer
    JIWER_OK = True
except ImportError:
    JIWER_OK = False
    warnings.warn("jiwer not found – falling back to manual CER/WER.")


try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image as PILImage
    TROCR_OK = True
except ImportError:
    TROCR_OK = False
    warnings.warn("transformers / Pillow not found – TrOCR method will be skipped.")


try:
    from sarvamai import SarvamAI
    SARVAM_OK = True
except ImportError:
    SARVAM_OK = False
    warnings.warn("sarvamai not found – Sarvam OCR method will be skipped. "
                  "Install with: pip install sarvamai")



# ─────────────────────────────────────────────────────────────────────────────
# 0.  GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────


HINDI_TEXT = (
    "गुरु गोविंद दोऊ खड़े, काके लागूं पांय। "
    "बलिहारी गुरु आपने, गोविंद दियो बताय॥ "
    "धीरे-धीरे रे मना, धीरे सब कुछ होय। "
    "माली सींचे सौ घड़ा, ऋतु आए फल होय॥ "
    "दया धर्म का मूल है, पाप मूल अभिमान। "
    "तुलसी दया न छाँड़िये, जब लग घट में प्राण॥ "
    "पोथी पढ़ि पढ़ि जग मुआ, पंडित भया न कोय। "
    "ढाई आखर प्रेम का, पढ़े सो पंडित होय॥ "
    "सांच बराबर तप नहीं, झूठ बराबर पाप। "
    "जाके हिरदै सांच है, ताके हिरदै आप॥ "
    "क्षेत्रपाल गुरु ज्ञान का, शुद्ध रखे विचार। "
    "षट्दर्शन सब जानिए, सद्गुरु ही आधार॥"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOGGING
# ─────────────────────────────────────────────────────────────────────────────


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)



# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────


def load_image(path: str) -> np.ndarray:
    """Load image from disk (BGR)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img



def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR → grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img



def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Binarise with Otsu's method.


    Formula:
        T* = argmin_T [ w0(T)·σ0²(T) + w1(T)·σ1²(T) ]
    OpenCV computes this automatically.
    """
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary



def remove_noise(binary: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Morphological opening to remove small noise blobs."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened



def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement (adaptive histogram equalisation)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)



def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate image to correct skew using moment-based angle estimation."""
    coords = np.column_stack(np.where(gray < 128))   # foreground pixels
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated



def preprocess(
    path: str,
    do_contrast: bool = True,
    do_deskew: bool = False,
    do_noise: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline; returns a cleaned grayscale image."""
    img   = load_image(path)
    gray  = to_grayscale(img)
    if do_contrast:
        gray = enhance_contrast(gray)
    binary = otsu_threshold(gray)
    if do_deskew:
        binary = deskew(binary)
    if do_noise:
        binary = remove_noise(binary)
    return binary



# ─────────────────────────────────────────────────────────────────────────────
# 3.  TEXT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────


def normalize(text: str, strip_punct: bool = False) -> str:
    """
    Normalise OCR output and ground truth for fair comparison.


    Steps:
      1. Unicode NFKC normalisation (handles composed / decomposed Devanagari)
      2. Collapse multiple whitespace → single space
      3. Strip leading / trailing whitespace
      4. Optionally remove punctuation (ASCII + common Devanagari dandas)
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    if strip_punct:
        import re
        text = re.sub(r"[।॥,.\-!?\"'()।॥]", "", text)
        text = " ".join(text.split())
    return text.strip()



# ─────────────────────────────────────────────────────────────────────────────
# 4.  METRICS
# ─────────────────────────────────────────────────────────────────────────────


def _edit_distance(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m]



def compute_cer(prediction: str, reference: str) -> float:
    """
    Character Error Rate (CER):
        CER = (S_c + D_c + I_c) / N_c
    where S_c, D_c, I_c = char substitutions, deletions, insertions
    and   N_c = total chars in reference.
    """
    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    if JIWER_OK:
        try:
            return float(cer(reference, prediction))
        except Exception:
            pass
    dist = _edit_distance(prediction, reference)
    return dist / len(reference)



def compute_wer(prediction: str, reference: str) -> float:
    """
    Word Error Rate (WER):
        WER = (S_w + D_w + I_w) / N_w
    where S_w, D_w, I_w = word substitutions, deletions, insertions
    and   N_w = total words in reference.
    """
    if JIWER_OK:
        try:
            return float(wer(reference, prediction))
        except Exception:
            pass
    ref_words  = reference.split()
    pred_words = prediction.split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    dist = _edit_distance(pred_words, ref_words)
    return dist / len(ref_words)



# ─────────────────────────────────────────────────────────────────────────────
# 5.  OCR ENGINES
# ─────────────────────────────────────────────────────────────────────────────


# ── 5A. Tesseract ────────────────────────────────────────────────────────────


TESSERACT_CONFIGS = {
    "tess_default":   "--lang hin",
    "tess_oem3_psm6": "--oem 3 --psm 6 -l hin",
    "tess_psm4":      "--oem 3 --psm 4 -l hin",
    "tess_hin_eng":   "--oem 3 --psm 6 -l hin+eng",
}



def run_tesseract(image: np.ndarray, config_str: str) -> str:
    """Run pytesseract on a pre-processed image array."""
    if not TESSERACT_OK:
        return ""
    try:
        text = pytesseract.image_to_string(image, config=config_str)
        return text.strip()
    except Exception as exc:
        log.debug("Tesseract error (%s): %s", config_str, exc)
        return ""



# ── 5B. TrOCR / Transformer OCR ─────────────────────────────────────────────


_TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"
_trocr_processor  = None
_trocr_model      = None



def _load_trocr():
    """Lazy-load TrOCR processor and model (download once)."""
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        log.info("Loading TrOCR model '%s' …", _TROCR_MODEL_NAME)
        _trocr_processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL_NAME)
        _trocr_model     = VisionEncoderDecoderModel.from_pretrained(
            _TROCR_MODEL_NAME
        )
        _trocr_model.eval()
        log.info("TrOCR model loaded.")



def run_trocr(image_path: str) -> str:
    """Run TrOCR on an image file (RGB PIL)."""
    if not TROCR_OK:
        return ""
    try:
        _load_trocr()
        pil_img = PILImage.open(image_path).convert("RGB")
        pixel_values = _trocr_processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values
        import torch
        with torch.no_grad():
            generated_ids = _trocr_model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4
            )
        text = _trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()
    except Exception as exc:
        log.debug("TrOCR error: %s", exc)
        return ""



# ── 5C. Sarvam OCR ───────────────────────────────────────────────────────────


class _HTMLTextExtractor(HTMLParser):
    """Minimal stdlib HTML → plain-text extractor (no extra deps)."""


    def __init__(self):
        super().__init__()
        self._parts: List[str] = []
        self._skip_tags = {"script", "style"}
        self._current_skip = 0


    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._current_skip += 1


    def handle_endtag(self, tag):
        if tag in self._skip_tags and self._current_skip > 0:
            self._current_skip -= 1


    def handle_data(self, data):
        if self._current_skip == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)


    def get_text(self) -> str:
        return " ".join(self._parts)



def _extract_text_from_html(html_bytes: bytes) -> str:
    """Parse HTML bytes and return plain text."""
    try:
        html_str = html_bytes.decode("utf-8", errors="replace")
        # Unescape HTML entities (e.g. &amp; → &, &#2325; → क)
        html_str = html.unescape(html_str)
        parser = _HTMLTextExtractor()
        parser.feed(html_str)
        return parser.get_text()
    except Exception as exc:
        log.debug("HTML text extraction error: %s", exc)
        return ""



def _extract_text_from_zip(zip_bytes: bytes) -> str:
    """
    Sarvam returns a ZIP archive containing one or more HTML files.
    We read every .html file in the archive and concatenate their text.
    """
    texts: List[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            html_names = sorted(
                n for n in zf.namelist() if n.lower().endswith(".html")
            )
            if not html_names:
                # Fallback: try any text-like file
                html_names = [
                    n for n in zf.namelist()
                    if not n.endswith("/")
                ]
            for name in html_names:
                with zf.open(name) as f:
                    texts.append(_extract_text_from_html(f.read()))
    except zipfile.BadZipFile as exc:
        log.debug("Sarvam ZIP parse error: %s", exc)
    return " ".join(texts)


# ── Sarvam key-rotation helpers ──────────────────────────────────────────────

# One cached SarvamAI client per key (keyed by the API key string itself).
_sarvam_clients: Dict[str, "SarvamAI"] = {}


def _is_credit_exhausted_error(exc: Exception) -> bool:
    """
    Return True when the exception signals that the current API key has
    run out of credits / quota so we should rotate to the next key.

    Sarvam may raise HTTP 402, 429 or embed keywords like
    "quota", "credit", "limit", "insufficient" in the message.
    Adjust the list below if the SDK uses different wording.
    """
    msg = str(exc).lower()
    credit_keywords = (
        "quota",
        "credit",
        "insufficient",
        "limit exceeded",
        "payment",
        "402",
        "429",
        "out of credits",
        "balance",
        "exhausted",
    )
    return any(kw in msg for kw in credit_keywords)


def _get_sarvam_client(key: str) -> "SarvamAI":
    """
    Return a cached SarvamAI client for *key*, creating it on first call.
    Raises ValueError if the key is empty or still a placeholder.
    """
    if not key or key == "YOUR_SARVAM_API_KEY":
        raise ValueError(
            "Sarvam API key is empty or not set. "
            "Open ocr_benchmark.py and fill in your actual key(s) in "
            "the SARVAM_API_KEYS list near the top of the file."
        )
    if key not in _sarvam_clients:
        _sarvam_clients[key] = SarvamAI(api_subscription_key=key)
        log.info("Sarvam client initialised for key …%s", key[-6:])
    return _sarvam_clients[key]


def _current_sarvam_key() -> str:
    """Return the API key that is currently active."""
    if not _ACTIVE_SARVAM_KEYS:
        raise RuntimeError(
            "No valid Sarvam API keys found. "
            "Add at least one key to SARVAM_API_KEYS in ocr_benchmark.py."
        )
    return _ACTIVE_SARVAM_KEYS[_sarvam_key_index]


def _rotate_sarvam_key() -> bool:
    """
    Advance _sarvam_key_index to the next available key.

    Returns True  if a new key is now active.
    Returns False if all keys have been exhausted.
    """
    global _sarvam_key_index
    if _sarvam_key_index + 1 < len(_ACTIVE_SARVAM_KEYS):
        _sarvam_key_index += 1
        log.warning(
            "Sarvam key [%d/%d] exhausted — rotating to key [%d/%d] …%s",
            _sarvam_key_index,          # old (1-based display)
            len(_ACTIVE_SARVAM_KEYS),
            _sarvam_key_index + 1,      # new (1-based display)
            len(_ACTIVE_SARVAM_KEYS),
            _ACTIVE_SARVAM_KEYS[_sarvam_key_index][-6:],
        )
        return True
    log.error(
        "All %d Sarvam API key(s) have been exhausted. "
        "No further Sarvam OCR calls will be made.",
        len(_ACTIVE_SARVAM_KEYS),
    )
    return False


# ─────────────────────────────────────────────────────────────────────────────


def run_sarvam(image_path: str) -> str:
    """
    Run Sarvam Document Intelligence OCR on a single image file.

    Key-rotation logic
    ──────────────────
    The function attempts the call with the current key.  If the API
    returns a credit / quota error it:
      1. Marks the current key as exhausted via _rotate_sarvam_key().
      2. Retries the *same* image with the next key.
    This continues until either the call succeeds or all keys are used up.

    Flow per key attempt:
      1. Create a job (language=hi-IN for Hindi, output_format=html)
      2. Upload the image file directly
      3. Start the job and wait for completion
      4. Download the output ZIP
      5. Extract plain text from the HTML inside the ZIP
    """
    if not SARVAM_OK:
        return ""

    while True:                             # loop over keys until success / give-up
        try:
            key    = _current_sarvam_key()
            client = _get_sarvam_client(key)

            # Step 1 – create job
            job = client.document_intelligence.create_job(
                language="hi-IN",
                output_format="html",
            )
            log.debug(
                "Sarvam job created: %s  (file=%s, key=…%s)",
                job.job_id, image_path, key[-6:],
            )

            # Step 2 – upload the image (Sarvam accepts JPEG/PNG directly)
            job.upload_file(image_path)

            # Step 3 – start & wait
            job.start()
            status = job.wait_until_complete()
            log.debug(
                "Sarvam job %s finished: state=%s", job.job_id, status.job_state
            )

            if status.job_state.lower() not in {"completed", "success"}:
                log.warning(
                    "Sarvam job %s ended in unexpected state: %s",
                    job.job_id, status.job_state,
                )
                return ""

            # Step 4 – download ZIP to an in-memory buffer
            zip_buffer = io.BytesIO()
            job.download_output(zip_buffer)          # pass file-like object
            zip_bytes = zip_buffer.getvalue()

            # Fallback: some SDK versions require a path string
            if not zip_bytes:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    job.download_output(tmp_path)
                    with open(tmp_path, "rb") as f:
                        zip_bytes = f.read()
                finally:
                    os.unlink(tmp_path)

            # Step 5 – extract text
            text = _extract_text_from_zip(zip_bytes)
            return text.strip()

        except Exception as exc:
            if _is_credit_exhausted_error(exc):
                # Try to rotate to the next key and retry the same image.
                rotated = _rotate_sarvam_key()
                if rotated:
                    continue          # ← retry the while-loop with the new key
                else:
                    # All keys exhausted — return empty string gracefully.
                    return ""
            else:
                # Non-credit error (network blip, bad image, etc.) – log & bail.
                log.debug("Sarvam OCR error for %s: %s", image_path, exc)
                return ""



# ─────────────────────────────────────────────────────────────────────────────
# 6.  CACHING
# ─────────────────────────────────────────────────────────────────────────────


_CACHE_FILE = ".ocr_cache.json"



def _image_hash(path: str) -> str:
    """MD5 of first 64 KB – fast fingerprint to detect changed files."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()



def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}



def save_cache(cache: dict, cache_path: str) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)



# ─────────────────────────────────────────────────────────────────────────────
# 7.  PER-IMAGE WORKER
# ─────────────────────────────────────────────────────────────────────────────


def process_image(
    image_path: str,
    use_preprocess: bool,
    ground_truth_norm: str,
    cache: Optional[dict] = None,
) -> List[Dict]:
    """
    Run all OCR methods on a single image.
    Returns a list of result dicts (one per method).
    """
    filename = os.path.basename(image_path)
    img_hash = _image_hash(image_path)
    results  = []


    # ── preprocessing ────────────────────────────────────────────────────────
    if use_preprocess:
        try:
            processed_img = preprocess(image_path)
        except Exception as exc:
            log.warning("Preprocessing failed for %s: %s", filename, exc)
            processed_img = to_grayscale(load_image(image_path))
    else:
        processed_img = to_grayscale(load_image(image_path))


    # ── Tesseract methods ─────────────────────────────────────────────────────
    if TESSERACT_OK:
        for method_name, cfg in TESSERACT_CONFIGS.items():
            cache_key = f"{img_hash}::{method_name}"
            if cache is not None and cache_key in cache:
                raw_pred = cache[cache_key]
            else:
                raw_pred = run_tesseract(processed_img, cfg)
                if cache is not None:
                    cache[cache_key] = raw_pred


            norm_pred = normalize(raw_pred)
            cer_score = compute_cer(norm_pred, ground_truth_norm)
            wer_score = compute_wer(norm_pred, ground_truth_norm)


            results.append({
                "filename":   filename,
                "method":     method_name,
                "prediction": raw_pred,
                "norm_pred":  norm_pred,
                "CER":        round(cer_score, 4),
                "WER":        round(wer_score, 4),
            })


    # ── TrOCR method ─────────────────────────────────────────────────────────
    if TROCR_OK:
        cache_key = f"{img_hash}::trocr"
        if cache is not None and cache_key in cache:
            raw_pred = cache[cache_key]
        else:
            raw_pred = run_trocr(image_path)
            if cache is not None:
                cache[cache_key] = raw_pred


        norm_pred = normalize(raw_pred)
        cer_score = compute_cer(norm_pred, ground_truth_norm)
        wer_score = compute_wer(norm_pred, ground_truth_norm)


        results.append({
            "filename":   filename,
            "method":     "trocr",
            "prediction": raw_pred,
            "norm_pred":  norm_pred,
            "CER":        round(cer_score, 4),
            "WER":        round(wer_score, 4),
        })


    # ── Sarvam OCR method ─────────────────────────────────────────────────────
    # NOTE: Sarvam receives the original image file (not the pre-processed
    # numpy array) because its cloud pipeline does its own pre-processing.
    if SARVAM_OK:
        cache_key = f"{img_hash}::sarvam"
        if cache is not None and cache_key in cache:
            raw_pred = cache[cache_key]
        else:
            raw_pred = run_sarvam(image_path)
            if cache is not None:
                cache[cache_key] = raw_pred


        norm_pred = normalize(raw_pred)
        cer_score = compute_cer(norm_pred, ground_truth_norm)
        wer_score = compute_wer(norm_pred, ground_truth_norm)


        results.append({
            "filename":   filename,
            "method":     "sarvam",
            "prediction": raw_pred,
            "norm_pred":  norm_pred,
            "CER":        round(cer_score, 4),
            "WER":        round(wer_score, 4),
        })


    if not results:
        log.warning("No OCR engine available for %s", filename)


    return results



# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────


def plot_cer_histogram(df: pd.DataFrame, output_dir: str) -> None:
    """Histogram of CER per method."""
    methods = df["method"].unique()
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(5 * len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        data = df[df["method"] == method]["CER"]
        ax.hist(data, bins=20, edgecolor="black", color="#4C72B0", alpha=0.8)
        ax.set_title(method, fontsize=10)
        ax.set_xlabel("CER")
        ax.set_ylabel("Count")
        mean_val = data.mean()
        ax.axvline(mean_val, color="red", linestyle="--",
                   label=f"mean={mean_val:.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("CER Distribution per OCR Method", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "cer_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Saved: %s", path)



def plot_boxplot(df: pd.DataFrame, output_dir: str) -> None:
    """Side-by-side boxplots for CER and WER across methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    methods = sorted(df["method"].unique())


    for ax, metric in [(ax1, "CER"), (ax2, "WER")]:
        data_list = [df[df["method"] == m][metric].values for m in methods]
        bp = ax.boxplot(data_list, labels=methods, patch_artist=True)
        colours = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
        ax.set_title(f"{metric} by Method", fontsize=12)
        ax.set_ylabel(metric)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.4)


    fig.suptitle("OCR Benchmark — CER & WER Comparison", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "boxplot_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Saved: %s", path)



def plot_mean_bar(df: pd.DataFrame, output_dir: str) -> None:
    """Simple bar chart of mean CER and WER per method."""
    summary = df.groupby("method")[["CER", "WER"]].mean().reset_index()
    x      = np.arange(len(summary))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, summary["CER"], width, label="CER", color="#4C72B0")
    ax.bar(x + width/2, summary["WER"], width, label="WER", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=20, ha="right")
    ax.set_ylabel("Error Rate")
    ax.set_title("Mean CER / WER per OCR Method", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(output_dir, "mean_error_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Saved: %s", path)



# ─────────────────────────────────────────────────────────────────────────────
# 9.  ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────


def analyse(df: pd.DataFrame, ground_truth_norm: str) -> None:
    """
    Print a structured analysis section:
      • Winner table
      • Top-5 worst predictions per method
      • Common failure pattern notes
    """
    sep = "─" * 70
    print(f"\n{'═'*70}")
    print("  ANALYSIS")
    print(f"{'═'*70}\n")


    # ── 9.1 Summary table ────────────────────────────────────────────────────
    summary = df.groupby("method")[["CER", "WER"]].agg(["mean", "std"])
    summary.columns = ["CER_mean", "CER_std", "WER_mean", "WER_std"]
    summary = summary.sort_values("CER_mean")


    print("┌─ Mean ± Std Error per Method ──────────────────────────────────┐")
    print(f"  {'Method':<22} {'CER mean':>10} {'CER std':>9} "
          f"{'WER mean':>10} {'WER std':>9}")
    print(f"  {sep}")
    for method, row in summary.iterrows():
        print(f"  {method:<22} {row.CER_mean:>10.4f} {row.CER_std:>9.4f} "
              f"{row.WER_mean:>10.4f} {row.WER_std:>9.4f}")
    print("└────────────────────────────────────────────────────────────────┘\n")


    best_method = summary.index[0]
    print(f"  ✅  BEST method (lowest mean CER): {best_method!r}\n")


    # ── 9.2 Worst 5 predictions (per method) ─────────────────────────────────
    for method in df["method"].unique():
        worst = (
            df[df["method"] == method]
            .nlargest(5, "CER")[["filename", "CER", "WER", "norm_pred"]]
        )
        print(f"  {'─'*66}")
        print(f"  Top-5 worst predictions — {method}")
        print(f"  {'─'*66}")
        for _, row in worst.iterrows():
            print(f"  [{row.filename}]  CER={row.CER:.3f}  WER={row.WER:.3f}")
            pred_snippet = row.norm_pred[:120].replace("\n", " ")
            print(f"    PRED : {pred_snippet}")
            gt_snippet   = ground_truth_norm[:120]
            print(f"    GT   : {gt_snippet}")
            print()


    # ── 9.3 Common failure pattern commentary ────────────────────────────────
    print(f"\n{'═'*70}")
    print("  FAILURE PATTERN ANALYSIS  (Devanagari-specific)")
    print(f"{'═'*70}")
    patterns = {
        "Matras (vowel diacritics)": (
            "Short vowel signs (ि, ी, ु, ू, ा …) are often missed or\n"
            "  substituted, especially when handwriting slants.\n"
            "  Effect: High CER, words become unrecognisable."
        ),
        "Conjunct consonants (Samyuktakshara)": (
            "Stacked forms (क्ष, त्र, ज्ञ, ट्ट …) are difficult for\n"
            "  models not trained on Devanagari; often split into\n"
            "  component characters or skipped entirely.\n"
            "  Effect: Elevated WER on rare conjuncts."
        ),
        "Halant / Virama (्)": (
            "The halant suppresses the inherent vowel and forms half-\n"
            "  consonants. OCR systems frequently output the full\n"
            "  consonant instead of the halant form.\n"
            "  Effect: Doubled characters, raised CER."
        ),
        "Anusvara / Chandrabindu (ं / ँ)": (
            "Nasalisation marks written above the shirorekha are\n"
            "  frequently omitted or confused with nukta (़).\n"
            "  Effect: Semantic errors, raised WER."
        ),
        "Shirorekha segmentation": (
            "Some engines segment at the headline, treating the upper\n"
            "  and lower portions of each akshara independently.\n"
            "  Effect: Garbled character sequences."
        ),
        "Inter-word spacing": (
            "Handwritten Hindi often has variable spacing; sandhi\n"
            "  (word fusion) may cause merges or incorrect splits.\n"
            "  Effect: WER spikes even when CER is low."
        ),
    }
    for pattern, desc in patterns.items():
        print(f"\n  ▸ {pattern}")
        for line in desc.split("\n"):
            print(f"    {line}")


    # ── 9.4 Recommendations ───────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  RECOMMENDATIONS")
    print(f"{'═'*70}")
    print("""
  1. Fine-tune Tesseract on a Devanagari handwriting corpus using
     tesstrain to improve conjunct and matra recognition.


  2. Use a pre-trained Indic OCR model (e.g. ai4bharat/IndicOCRBench
     or iNLTK / Dhruva) which is trained on Hindi scripts.


  3. Apply morphological closing before Tesseract to reconnect
     broken shirorekha segments.


  4. Post-process with a Hindi language model (n-gram or BERT-based)
     to correct phonetically similar errors.


  5. Consider an ensemble: Tesseract (raw) + transformer + Sarvam → voting.


  6. For Sarvam, set language="hi-IN" (already configured) to get the
     best Devanagari-specific model.  Sarvam's cloud pipeline handles
     its own pre-processing, so the raw image is uploaded directly
     without the local Otsu / CLAHE pipeline.
""")



# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hindi Handwritten OCR Benchmarking Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        default="C:\\Users\\IISER\\OneDrive\\Ankush\\Generation\\ocr\\Combined",
        help="Folder containing handwritten Hindi images (default: C:\Users\IISER\OneDrive\Ankush\Generation\ocr\Combined)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./results",
        help="Output folder for CSV and plots (default: ./results)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip image preprocessing (pass raw grayscale to OCR)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N images (0 = all; useful for testing)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching"
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()


    # ── validate input folder ─────────────────────────────────────────────────
    input_dir = Path(args.input)
    if not input_dir.exists():
        log.error("Input folder not found: %s", input_dir)
        sys.exit(1)


    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_paths = sorted(
        str(p) for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )
    if not image_paths:
        log.error("No images found in %s", input_dir)
        sys.exit(1)


    if args.limit > 0:
        image_paths = image_paths[: args.limit]


    log.info("Found %d images in '%s'", len(image_paths), input_dir)


    # ── output folder ─────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)


    # ── normalise ground truth once ───────────────────────────────────────────
    gt_norm = normalize(HINDI_TEXT)
    log.info("Ground truth length (chars): %d", len(gt_norm))


    # ── cache ─────────────────────────────────────────────────────────────────
    cache_path = str(output_dir / _CACHE_FILE)
    cache      = {} if args.no_cache else load_cache(cache_path)


    # ── check available engines ───────────────────────────────────────────────
    if not TESSERACT_OK and not TROCR_OK and not SARVAM_OK:
        log.error("No OCR engine available. "
                  "Install pytesseract, transformers, or sarvamai.")
        sys.exit(1)


    if SARVAM_OK:
        log.info(
            "Sarvam OCR enabled  (language=hi-IN, output_format=html, "
            "%d key(s) configured)",
            len(_ACTIVE_SARVAM_KEYS),
        )


    # ── process images ────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    use_preprocess = not args.no_preprocess
    t0 = time.time()


    if args.workers > 1:
        log.info("Running with %d workers …", args.workers)
        # NOTE: multiprocessing cannot share cache dict; each worker works
        # independently and we merge after.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_image, path, use_preprocess, gt_norm, None
                ): path
                for path in image_paths
            }
            done = 0
            for future in as_completed(futures):
                path = futures[future]
                try:
                    all_results.extend(future.result())
                except Exception as exc:
                    log.error("Worker failed for %s: %s", path, exc)
                done += 1
                if done % 50 == 0 or done == len(image_paths):
                    log.info("  … %d / %d images done", done, len(image_paths))
    else:
        for idx, path in enumerate(image_paths, 1):
            try:
                results = process_image(path, use_preprocess, gt_norm, cache)
                all_results.extend(results)
            except Exception as exc:
                log.error("Failed on %s: %s", path, exc)
            if idx % 50 == 0 or idx == len(image_paths):
                log.info("  … %d / %d images done", idx, len(image_paths))


    elapsed = time.time() - t0
    log.info("OCR complete in %.1f s", elapsed)


    # ── save cache ────────────────────────────────────────────────────────────
    if not args.no_cache:
        save_cache(cache, cache_path)


    if not all_results:
        log.error("No results collected.")
        sys.exit(1)


    # ── build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)


    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = str(output_dir / "results.csv")
    df[["filename", "method", "prediction", "CER", "WER"]].to_csv(
        csv_path, index=False, encoding="utf-8-sig"   # BOM for Excel compat.
    )
    log.info("Results saved: %s", csv_path)


    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  OCR BENCHMARK SUMMARY")
    print(f"{'═'*60}")
    summary = df.groupby("method")[["CER", "WER"]].mean().round(4)
    print(summary.to_string())
    print(f"{'═'*60}\n")


    summary_path = str(output_dir / "summary.csv")
    summary.to_csv(summary_path)
    log.info("Summary saved: %s", summary_path)


    # ── visualisations ────────────────────────────────────────────────────────
    plot_cer_histogram(df, str(output_dir))
    plot_boxplot(df, str(output_dir))
    plot_mean_bar(df, str(output_dir))


    # ── analysis section ──────────────────────────────────────────────────────
    analyse(df, gt_norm)


    log.info("All outputs written to: %s", output_dir)



if __name__ == "__main__":
    main()