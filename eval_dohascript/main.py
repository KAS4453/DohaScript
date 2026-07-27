"""
main.py
=======
Command-line entry point for the GANwriting Devanagari evaluation framework.

Examples
--------
    python main.py
    python main.py --generated img3 --real dataset_real
    python main.py --only fid
    python main.py --only ocr
    python main.py --only fid,kid,ocr
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

from config import Config, GROUND_TRUTH_TEXT, get_default_config
from utils import setup_logger, group_images_by_epoch, window_epoch_groups, save_json

logger = logging.getLogger("main")

ALL_METRICS = [
    "fid", "kid", "ocr", "msssim", "sharpness", "blur", "stroke_density",
    "entropy", "skeleton", "projection", "edge_density", "noise", "quality",
    "tsne", "umap", "qualitative",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a writer-conditioned handwritten Devanagari GAN across training epochs."
    )
    parser.add_argument("--generated", type=str, default=None, help="Directory of generated images (epoch_*.png).")
    parser.add_argument("--real", type=str, default=None, help="Directory of real reference images.")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated subset of metrics to run. Choices: {', '.join(ALL_METRICS)}",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Compute device.")
    parser.add_argument("--sarvam-api-key", type=str, default=None, help="Sarvam OCR API key (overrides env var).")
    parser.add_argument("--max-ocr-images-per-epoch", type=int, default=5, help="Cap OCR calls per epoch (cost control).")
    parser.add_argument(
        "--fid-window", type=int, default=1,
        help=(
            "Pool this many consecutive epochs' generated images together for FID/KID "
            "(default 1 = no pooling). Use >1 (e.g. 51) when your generator only emits one "
            "image per epoch checkpoint, since FID/KID need multiple samples per point to "
            "estimate a distribution — with window=1 in that case they will fail."
        ),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = get_default_config()
    if args.generated:
        cfg.generated_dir = Path(args.generated)
    if args.real:
        cfg.real_dir = Path(args.real)
    if args.output:
        cfg.output_dir = Path(args.output)
    if args.device:
        cfg.device = args.device
    if args.sarvam_api_key:
        cfg.sarvam_api_key = args.sarvam_api_key
    cfg.__post_init__()
    return cfg


def stage_epoch_subfolders(
    epoch_groups: Dict[int, List[Path]], staging_root: Path, canonical_size: tuple = (299, 299)
) -> Dict[int, Path]:
    """
    torch-fidelity (used for KID) operates on directories, and — like
    pytorch-fid — batches images into tensors internally, which crashes
    ("stack expects each tensor to be equal size") if the source crops
    have different dimensions, as handwriting word/line images do. This
    helper letterboxes every image onto a fixed `canonical_size` canvas
    and writes the result per epoch under `staging_root/epoch_<n>/`,
    rather than symlinking the raw (variable-size) originals.
    """
    from utils import stage_resized_copies

    staging_root = Path(staging_root)
    epoch_dirs: Dict[int, Path] = {}
    for epoch, paths in epoch_groups.items():
        epoch_dir = staging_root / f"epoch_{epoch}"
        stage_resized_copies(paths, epoch_dir, canonical_size)
        epoch_dirs[epoch] = epoch_dir
    return epoch_dirs


def merge_into_master_table(all_rows: Dict[str, List[dict]], output_csv: Path) -> None:
    """
    Outer-join every metric's per-epoch rows on `epoch` into a single wide
    `metrics.csv`, prefixing columns by metric name to avoid collisions.
    """
    merged: Dict[int, dict] = {}
    for metric_name, rows in all_rows.items():
        for row in rows:
            epoch = row["epoch"]
            merged.setdefault(epoch, {"epoch": epoch})
            for k, v in row.items():
                if k == "epoch":
                    continue
                merged[epoch][f"{metric_name}_{k}"] = v

    if not merged:
        logger.warning("No metric rows to merge into master table.")
        return

    all_columns = sorted({col for row in merged.values() for col in row}, key=lambda c: (c != "epoch", c))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for epoch in sorted(merged):
            writer.writerow(merged[epoch])
    logger.info("Wrote master metrics table to %s", output_csv)


def run(cfg: Config, selected_metrics: List[str], max_ocr_images_per_epoch: int, fid_window: int = 1) -> None:
    setup_logger("main", cfg.logs_dir)
    logger.info("Scanning generated images in %s", cfg.generated_dir)
    epoch_groups = group_images_by_epoch(cfg.generated_dir, cfg.filename_regex)
    if not epoch_groups:
        logger.error("No generated images found (or none matched the epoch filename pattern) in %s", cfg.generated_dir)
        return
    logger.info("Found %d epochs, %d total images.", len(epoch_groups), sum(len(v) for v in epoch_groups.values()))

    if fid_window > 1:
        fid_kid_epoch_groups = window_epoch_groups(epoch_groups, window=fid_window)
        logger.info(
            "FID/KID: pooling window=%d applied -> %d epoch points with >=2 samples.",
            fid_window, len(fid_kid_epoch_groups),
        )
    else:
        fid_kid_epoch_groups = epoch_groups

    all_rows: Dict[str, List[dict]] = {}
    metric_series_for_plots: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # FID
    # ------------------------------------------------------------------
    if "fid" in selected_metrics:
        from metrics import fid as fid_mod
        try:
            rows = fid_mod.compute_fid_per_epoch(
                cfg.real_dir, fid_kid_epoch_groups, cfg.device, cfg.batch_size, cfg.fid_dims,
                output_csv=cfg.tables_dir / "FID.csv",
                staging_dir=cfg.output_dir / "_staging_fid",
            )
            all_rows["fid"] = rows
            metric_series_for_plots["FID"] = {"epoch": [r["epoch"] for r in rows], "value": [r["fid"] for r in rows]}
        except Exception:
            logger.exception("FID stage failed.")

    # ------------------------------------------------------------------
    # KID
    # ------------------------------------------------------------------
    if "kid" in selected_metrics:
        from metrics import kid as kid_mod
        try:
            from utils import list_images, stage_resized_copies

            staging_root = cfg.output_dir / "_staging_kid"
            epoch_dirs = stage_epoch_subfolders(fid_kid_epoch_groups, staging_root)
            real_dir_staged = staging_root / "real"
            stage_resized_copies(list_images(cfg.real_dir), real_dir_staged)
            rows = kid_mod.compute_kid_per_epoch(
                real_dir_staged, epoch_dirs, cfg.device, cfg.kid_subset_size, cfg.kid_subsets,
                output_csv=cfg.tables_dir / "KID.csv",
            )
            all_rows["kid"] = rows
            metric_series_for_plots["KID"] = {
                "epoch": [r["epoch"] for r in rows],
                "value": [r["kid_mean"] for r in rows],
                "std": [r["kid_std"] for r in rows],
            }
        except Exception:
            logger.exception("KID stage failed.")

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    if "ocr" in selected_metrics:
        from metrics import ocr as ocr_mod
        try:
            pairs = []
            for epoch, paths in epoch_groups.items():
                for p in paths[:max_ocr_images_per_epoch]:
                    pairs.append((p, epoch))
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
            all_rows["ocr"] = per_epoch_rows
            metric_series_for_plots["CER"] = {"epoch": [r["epoch"] for r in per_epoch_rows], "value": [r["cer_mean"] for r in per_epoch_rows]}
            metric_series_for_plots["WER"] = {"epoch": [r["epoch"] for r in per_epoch_rows], "value": [r["wer_mean"] for r in per_epoch_rows]}
            metric_series_for_plots["Character Accuracy"] = {"epoch": [r["epoch"] for r in per_epoch_rows], "value": [r["char_accuracy_mean"] for r in per_epoch_rows]}
            metric_series_for_plots["Word Accuracy"] = {"epoch": [r["epoch"] for r in per_epoch_rows], "value": [r["word_accuracy_mean"] for r in per_epoch_rows]}
        except Exception:
            logger.exception("OCR stage failed.")

    # ------------------------------------------------------------------
    # MS-SSIM
    # ------------------------------------------------------------------
    if "msssim" in selected_metrics:
        from metrics import msssim as msssim_mod
        try:
            rows = msssim_mod.compute_msssim_per_epoch(epoch_groups, cfg.msssim_max_pairs, cfg.tables_dir / "msssim.csv")
            all_rows["msssim"] = rows
            metric_series_for_plots["MS-SSIM"] = {"epoch": [r["epoch"] for r in rows], "value": [r["mean"] for r in rows]}
        except Exception:
            logger.exception("MS-SSIM stage failed.")

    # ------------------------------------------------------------------
    # Sharpness / Blur / Stroke density / Entropy / Skeleton / Projection /
    # Edge density / Noise / Quality
    # ------------------------------------------------------------------
    simple_stages = {
        "sharpness": ("metrics.sharpness", "compute_sharpness_per_epoch", "sharpness.csv", "Sharpness", "mean"),
        "blur": ("metrics.blur", "compute_blur_per_epoch", "blur.csv", "Blur (Laplacian)", "laplacian_mean"),
        "stroke_density": ("metrics.stroke_density", "compute_stroke_density_per_epoch", "stroke_density.csv", "Stroke Density", "density_mean"),
        "entropy": ("metrics.entropy", "compute_entropy_per_epoch", "entropy.csv", "Image Entropy", "mean"),
        "skeleton": ("metrics.skeleton", "compute_skeleton_per_epoch", "skeleton.csv", "Skeleton Length", "total_length_mean"),
        "edge_density": ("metrics.edge_density", "compute_edge_density_per_epoch", "edge_density.csv", "Edge Density", "mean"),
        "noise": ("metrics.noise", "compute_noise_per_epoch", "noise.csv", "Noise (sigma)", "mean"),
        "quality": ("metrics.quality", "compute_quality_per_epoch", "quality.csv", "BRISQUE", "brisque_mean"),
    }
    import importlib
    for key, (module_path, fn_name, csv_name, plot_label, value_col) in simple_stages.items():
        if key not in selected_metrics:
            continue
        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, fn_name)
            rows = fn(epoch_groups, output_csv=cfg.tables_dir / csv_name)
            all_rows[key] = rows
            if rows:
                metric_series_for_plots[plot_label] = {
                    "epoch": [r["epoch"] for r in rows],
                    "value": [r.get(value_col, float("nan")) for r in rows],
                }
        except Exception:
            logger.exception("%s stage failed.", key)

    # Projection profiles (separate: also saves per-image plots, not just a CSV table)
    if "projection" in selected_metrics:
        from metrics import projection as projection_mod
        try:
            rows = projection_mod.compute_projection_per_epoch(epoch_groups, cfg.figures_dir)
            all_rows["projection"] = rows
            metric_series_for_plots["Projection Energy"] = {
                "epoch": [r["epoch"] for r in rows], "value": [r["horizontal_energy_mean"] for r in rows]
            }
            with open(cfg.tables_dir / "projection.csv", "w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
        except Exception:
            logger.exception("Projection stage failed.")

    # ------------------------------------------------------------------
    # t-SNE / UMAP
    # ------------------------------------------------------------------
    if "tsne" in selected_metrics:
        from plots import plot_tsne
        try:
            plot_tsne.plot_tsne_real_vs_generated(
                cfg.real_dir, cfg.generated_dir, cfg.figures_dir / "tsne_real_vs_generated.png",
                cfg.embedding_backbone, cfg.device, cfg.tsne_perplexity, cfg.tsne_n_iter, cfg.random_seed, cfg.plot_dpi,
            )
        except Exception:
            logger.exception("t-SNE stage failed.")

    if "umap" in selected_metrics:
        from plots import plot_umap
        try:
            plot_umap.plot_umap_real_vs_generated(
                cfg.real_dir, cfg.generated_dir, cfg.figures_dir / "umap_real_vs_generated.png",
                cfg.embedding_backbone, cfg.device, cfg.umap_n_neighbors, cfg.umap_min_dist, cfg.random_seed, cfg.plot_dpi,
            )
        except Exception:
            logger.exception("UMAP stage failed.")

    # ------------------------------------------------------------------
    # Qualitative evolution strip
    # ------------------------------------------------------------------
    if "qualitative" in selected_metrics:
        from plots import plot_training_curves
        try:
            sorted_epochs = sorted(epoch_groups)
            n_examples = min(5, len(sorted_epochs))
            step = max(1, len(sorted_epochs) // max(n_examples - 1, 1))
            chosen = sorted_epochs[::step][:n_examples]
            if sorted_epochs[-1] not in chosen:
                chosen[-1] = sorted_epochs[-1]
            example_map = {epoch: epoch_groups[epoch][0] for epoch in chosen}
            plot_training_curves.plot_qualitative_evolution(
                example_map, cfg.figures_dir / "qualitative_evolution.png", dpi=cfg.plot_dpi
            )
        except Exception:
            logger.exception("Qualitative evolution stage failed.")

    # ------------------------------------------------------------------
    # Aggregate outputs
    # ------------------------------------------------------------------
    merge_into_master_table(all_rows, cfg.tables_dir / "metrics.csv")

    if metric_series_for_plots:
        from plots import plot_metrics, plot_training_curves
        plot_metrics.plot_all_metrics(metric_series_for_plots, cfg.figures_dir / "metric_curves", cfg.plot_formats, cfg.plot_dpi)
        try:
            plot_training_curves.plot_multi_metric_grid(
                metric_series_for_plots, cfg.figures_dir / "training_dashboard.png", dpi=cfg.plot_dpi
            )
        except Exception:
            logger.exception("Dashboard plot failed.")

    summary = {
        metric: {
            "n_epochs": len(rows),
            "epochs": [r["epoch"] for r in rows],
        }
        for metric, rows in all_rows.items()
    }
    save_json(summary, cfg.tables_dir / "summary.json")
    with open(cfg.tables_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "n_epochs"])
        for metric, rows in all_rows.items():
            writer.writerow([metric, len(rows)])

    logger.info("Evaluation complete. Results written to %s", cfg.output_dir)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    selected = args.only.split(",") if args.only else ALL_METRICS
    selected = [s.strip() for s in selected]
    invalid = set(selected) - set(ALL_METRICS)
    if invalid:
        raise SystemExit(f"Unknown metric(s) in --only: {invalid}. Choices: {ALL_METRICS}")
    run(cfg, selected, args.max_ocr_images_per_epoch, args.fid_window)


if __name__ == "__main__":
    main()
