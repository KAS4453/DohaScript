"""
metrics/ocr.py
===============
OCR-based evaluation of generated handwriting: CER, WER, character
accuracy, and word accuracy, using Sarvam OCR.

Design note
-----------
Sarvam does not expose a plain "/v1/ocr" REST endpoint (that 404s). OCR
goes through the Document Intelligence job API instead, via the official
`sarvamai` SDK: create a job, upload the image, start it, wait for
completion, then download a ZIP containing HTML/Markdown + a JSON page
dump. This is the same flow your OCR benchmarking script already uses
successfully, applied here directly (no project-local client discovery
needed).

Install the SDK with: pip install sarvamai
"""

from __future__ import annotations

import csv
import html
import io
import logging
import tempfile
import time
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils import mean_std_ci, save_json

logger = logging.getLogger("metrics.ocr")

try:
    from sarvamai import SarvamAI
    SARVAM_SDK_OK = True
except ImportError:
    SARVAM_SDK_OK = False
    logger.warning("sarvamai not installed — run `pip install sarvamai` to enable OCR.")


# ---------------------------------------------------------------------------
# Minimal HTML -> text extraction (Sarvam's job output is an HTML/ZIP bundle)
# ---------------------------------------------------------------------------
class _HTMLTextExtractor(HTMLParser):
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
    try:
        html_str = html.unescape(html_bytes.decode("utf-8", errors="replace"))
        parser = _HTMLTextExtractor()
        parser.feed(html_str)
        return parser.get_text()
    except Exception as exc:  # noqa: BLE001
        logger.debug("HTML text extraction error: %s", exc)
        return ""


def _extract_text_from_zip(zip_bytes: bytes) -> str:
    """Sarvam's job output ZIP contains one or more HTML files; concatenate them."""
    texts: List[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            html_names = sorted(n for n in zf.namelist() if n.lower().endswith(".html"))
            if not html_names:
                html_names = [n for n in zf.namelist() if not n.endswith("/")]
            for name in html_names:
                with zf.open(name) as f:
                    texts.append(_extract_text_from_html(f.read()))
    except zipfile.BadZipFile as exc:
        logger.debug("Sarvam ZIP parse error: %s", exc)
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Sarvam client (single API key, cached so repeated calls reuse the client)
# ---------------------------------------------------------------------------
_sarvam_clients: Dict[str, "SarvamAI"] = {}


def _get_sarvam_client(api_key: str) -> "SarvamAI":
    if not api_key:
        raise ValueError(
            "No Sarvam API key configured. Set the SARVAM_API_KEY environment "
            "variable, or pass --sarvam-api-key on the command line."
        )
    if api_key not in _sarvam_clients:
        _sarvam_clients[api_key] = SarvamAI(api_subscription_key=api_key)
    return _sarvam_clients[api_key]


def _call_sarvam_document_intelligence(image_path: Path, api_key: str, language_code: str) -> Dict:
    """Run OCR via Sarvam's Document Intelligence job API (create -> upload -> start -> wait -> download)."""
    if not SARVAM_SDK_OK:
        raise RuntimeError("sarvamai package is not installed. Run `pip install sarvamai`.")

    client = _get_sarvam_client(api_key)

    job = client.document_intelligence.create_job(
        language=language_code,
        output_format="html",
    )
    job.upload_file(str(image_path))
    job.start()
    status = job.wait_until_complete()

    if status.job_state not in {"Completed", "PartiallyCompleted"}:
        raise RuntimeError(f"Sarvam job ended in state {status.job_state!r} for {image_path.name}")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        job.download_output(tmp_path)
        zip_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    text = _extract_text_from_zip(zip_bytes)
    return {"text": text, "confidence": None, "raw": {"job_state": status.job_state}}


def run_ocr(image_path: Path, api_key: str, language_code: str, max_retries: int = 3, backoff: float = 2.0) -> Dict:
    """Run OCR on a single image with retry-with-backoff."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return _call_sarvam_document_intelligence(image_path, api_key, language_code)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("OCR attempt %d/%d failed for %s: %s", attempt, max_retries, image_path.name, exc)
            time.sleep(backoff * attempt)
    raise RuntimeError(f"OCR failed after {max_retries} attempts for {image_path}") from last_error


# ---------------------------------------------------------------------------
# Metric computation: CER / WER / accuracies
# ---------------------------------------------------------------------------
def _levenshtein(a: List[str] | str, b: List[str] | str) -> int:
    """Classic O(len(a)*len(b)) edit distance over a sequence of tokens/chars."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit_distance(chars) / len(reference_chars)."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    dist = _levenshtein(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = edit_distance(words) / len(reference_words)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = _levenshtein(ref_words, hyp_words)
    return dist / len(ref_words)


def word_accuracy(reference: str, hypothesis: str) -> float:
    """
    Word accuracy = (# correctly matched words) / (# reference words),
    where matches are computed positionally after alignment via the
    Levenshtein backtrace (a word counts correct if it survives with zero
    substitution/deletion cost). We approximate this efficiently using
    1 - WER clipped to [0, 1], which is the standard convention when a
    full alignment-based match count isn't required. For an exact
    correct-word count, use `word_accuracy_exact`.
    """
    return max(0.0, 1.0 - word_error_rate(reference, hypothesis))


def word_accuracy_exact(reference: str, hypothesis: str) -> float:
    """
    Exact word accuracy = correct_words / total_words, where "correct"
    is determined via a Needleman-Wunsch style alignment that counts
    substitutions/deletions as incorrect and matches as correct.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 1.0 if m == 0 else 0.0

    # DP table storing (cost, match_count) is overkill; instead compute an
    # alignment via standard edit-distance backtrace.
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrace to count exact matches.
    i, j = n, m
    correct = 0
    while i > 0 and j > 0:
        if ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            correct += 1
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1  # substitution
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1  # deletion
        else:
            j -= 1  # insertion
    return correct / n


@dataclass
class OCRResult:
    epoch: int
    filename: str
    cer: float
    wer: float
    char_accuracy: float
    word_accuracy: float
    confidence: Optional[float]
    hypothesis_text: str


def evaluate_ocr_for_images(
    image_epoch_pairs: List[tuple],
    reference_text: str,
    api_key: str,
    language_code: str = "hi-IN",
    max_retries: int = 3,
    backoff: float = 2.0,
) -> List[OCRResult]:
    """
    Run OCR + compute CER/WER/accuracy for a list of (path, epoch) pairs
    against a single shared reference text (the ground-truth Doha corpus).
    """
    results: List[OCRResult] = []
    for path, epoch in image_epoch_pairs:
        try:
            ocr_out = run_ocr(path, api_key, language_code, max_retries, backoff)
        except Exception as exc:  # noqa: BLE001
            logger.error("Giving up on %s after retries: %s", path, exc)
            ocr_out = {"text": "", "confidence": None}

        hyp_text = ocr_out.get("text", "") or ""
        cer = character_error_rate(reference_text, hyp_text)
        wer = word_error_rate(reference_text, hyp_text)
        char_acc = 1.0 - cer
        word_acc = word_accuracy_exact(reference_text, hyp_text)

        results.append(
            OCRResult(
                epoch=epoch,
                filename=path.name,
                cer=cer,
                wer=wer,
                char_accuracy=char_acc,
                word_accuracy=word_acc,
                confidence=ocr_out.get("confidence"),
                hypothesis_text=hyp_text,
            )
        )
        logger.info(
            "OCR %s (epoch %d): CER=%.4f WER=%.4f CharAcc=%.4f WordAcc=%.4f",
            path.name, epoch, cer, wer, char_acc, word_acc,
        )
    return results


def summarize_ocr_results(results: List[OCRResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate mean/std/95%CI for each metric, overall and per epoch."""
    summary: Dict[str, Dict[str, float]] = {}
    summary["overall"] = {
        "cer": mean_std_ci([r.cer for r in results]),
        "wer": mean_std_ci([r.wer for r in results]),
        "char_accuracy": mean_std_ci([r.char_accuracy for r in results]),
        "word_accuracy": mean_std_ci([r.word_accuracy for r in results]),
    }

    per_epoch: Dict[int, List[OCRResult]] = {}
    for r in results:
        per_epoch.setdefault(r.epoch, []).append(r)

    for epoch, rs in sorted(per_epoch.items()):
        summary[f"epoch_{epoch}"] = {
            "cer": mean_std_ci([r.cer for r in rs]),
            "wer": mean_std_ci([r.wer for r in rs]),
            "char_accuracy": mean_std_ci([r.char_accuracy for r in rs]),
            "word_accuracy": mean_std_ci([r.word_accuracy for r in rs]),
        }
    return summary


def write_ocr_csv(results: List[OCRResult], output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "filename", "cer", "wer", "char_accuracy", "word_accuracy", "confidence", "hypothesis_text"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "epoch": r.epoch,
                    "filename": r.filename,
                    "cer": r.cer,
                    "wer": r.wer,
                    "char_accuracy": r.char_accuracy,
                    "word_accuracy": r.word_accuracy,
                    "confidence": r.confidence if r.confidence is not None else "",
                    "hypothesis_text": r.hypothesis_text,
                }
            )
    logger.info("Wrote OCR results to %s", output_csv)


def write_ocr_summary_json(summary: Dict[str, Dict[str, float]], output_json: Path) -> None:
    save_json(summary, output_json)
    logger.info("Wrote OCR summary to %s", output_json)
