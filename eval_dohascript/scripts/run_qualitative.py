"""
scripts/run_qualitative.py
===========================
Run ONLY the qualitative evolution strip (a handful of sample images
across training epochs, side by side).

Usage
-----
    python scripts/run_qualitative.py
    python scripts/run_qualitative.py --generated img3
"""

from __future__ import annotations

from _common import build_arg_parser, build_config, load_epoch_groups


def main() -> None:
    parser = build_arg_parser("Plot the qualitative evolution strip, and only that.")
    args = parser.parse_args()

    cfg = build_config(args)
    epoch_groups, logger = load_epoch_groups(cfg, "run_qualitative")

    sorted_epochs = sorted(epoch_groups)
    n_examples = min(5, len(sorted_epochs))
    step = max(1, len(sorted_epochs) // max(n_examples - 1, 1))
    chosen = sorted_epochs[::step][:n_examples]
    if sorted_epochs[-1] not in chosen:
        chosen[-1] = sorted_epochs[-1]
    example_map = {epoch: epoch_groups[epoch][0] for epoch in chosen}

    from plots import plot_training_curves
    out_path = cfg.figures_dir / "qualitative_evolution.png"
    plot_training_curves.plot_qualitative_evolution(example_map, out_path, dpi=cfg.plot_dpi)
    logger.info("Qualitative evolution strip complete -> %s", out_path)


if __name__ == "__main__":
    main()
