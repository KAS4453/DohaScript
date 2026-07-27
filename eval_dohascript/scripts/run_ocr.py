"""
scripts/run_ocr.py
===================
Run ONLY the Sarvam-OCR-based CER/WER evaluation. Nothing else in the
framework is touched, and no other metric's compute cost (FID/KID/etc.)
is incurred.

Usage
-----
    python scripts/run_ocr.py
    python scripts/run_ocr.py --generated img3 --sarvam-api-key sk-...
    python scripts/run_ocr.py --max-ocr-images-per-epoch 3
"""

from __future__ import annotations

from _common import build_arg_parser, build_config, load_epoch_groups
from config import GROUND_TRUTH_TEXT


def main() -> None:
    parser = build_arg_parser("Compute OCR (CER/WER) per epoch, and only OCR.")
    parser.add_argument("--sarvam-api-key", type=str, default=None, help="Sarvam OCR API key (overrides env var).")
    parser.add_argument("--max-ocr-images-per-epoch", type=int, default=5, help="Cap OCR calls per epoch (cost control).")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_ocr")

    pairs = []
    for epoch, paths in epoch_groups.items():
        for p in paths[: args.max_ocr_images_per_epoch]:
            pairs.append((p, epoch))

    from metrics import ocr as ocr_mod
    results = ocr_mod.evaluate_ocr_for_images(
        pairs, GROUND_TRUTH_TEXT, cfg.sarvam_api_key, cfg.sarvam_language_code,
        cfg.ocr_max_retries, cfg.ocr_retry_backoff_seconds,
    )
    ocr_mod.write_ocr_csv(results, cfg.tables_dir / "ocr_results.csv")
    summary = ocr_mod.summarize_ocr_results(results)
    ocr_mod.write_ocr_summary_json(summary, cfg.tables_dir / "ocr_summary.json")

    per_epoch_rows = [
        {
            "epoch": int(key.split("_")[1]),
            "cer_mean": val["cer"]["mean"],
            "wer_mean": val["wer"]["mean"],
            "char_accuracy_mean": val["char_accuracy"]["mean"],
            "word_accuracy_mean": val["word_accuracy"]["mean"],
        }
        for key, val in summary.items() if key.startswith("epoch_")
    ]

    if per_epoch_rows:
        from plots import plot_metrics
        epochs = [r["epoch"] for r in per_epoch_rows]
        plot_metrics.plot_all_metrics(
            {
                "CER": {"epoch": epochs, "value": [r["cer_mean"] for r in per_epoch_rows]},
                "WER": {"epoch": epochs, "value": [r["wer_mean"] for r in per_epoch_rows]},
                "Character Accuracy": {"epoch": epochs, "value": [r["char_accuracy_mean"] for r in per_epoch_rows]},
                "Word Accuracy": {"epoch": epochs, "value": [r["word_accuracy_mean"] for r in per_epoch_rows]},
            },
            cfg.figures_dir / "metric_curves", cfg.plot_formats, cfg.plot_dpi,
        )

    logger.info(
        "OCR complete: %d epoch rows -> %s / %s",
        len(per_epoch_rows), cfg.tables_dir / "ocr_results.csv", cfg.tables_dir / "ocr_summary.json",
    )


if __name__ == "__main__":
    main()
