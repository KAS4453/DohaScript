"""
Publication-quality plot of GANWriting training dynamics.

Fixes the "too noisy to publish" issue by:
  - showing raw per-checkpoint values as faint thin lines (context only)
  - overlaying a rolling-mean smoothed trend line (the actual takeaway)
  - shading a rolling min/max (or std) band around the smoothed line
  - dropping per-point markers (they clutter at this density)

Usage:
    python plot_training_curves_clean.py training_log.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})

WINDOW = 15  # rolling window in *number of checkpoints* (not epochs); tune as needed


def load_data(path):
    df = pd.read_csv(path)
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


def safe_logx_epoch(epoch):
    e = epoch.astype(float).copy()
    e[e == 0] = 1.0
    return e


def smoothed(series, window=WINDOW):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def band(series, window=WINDOW):
    roll = series.rolling(window=window, center=True, min_periods=1)
    return roll.min(), roll.max()


def plot_main_figure(df, out_path="training_curves.png", window=WINDOW):
    x = safe_logx_epoch(df["epoch"])

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)

    # ---------------- Top panel: CER ----------------
    ax = axes[0]

    for col, color, label in [
        ("eval_cer_gen", "#1f77b4", "CER (gen)"),
        ("eval_cer_swap", "#d62728", "CER (swap)"),
    ]:
        raw = df[col]
        smooth = smoothed(raw, window)
        lo, hi = band(raw, window)

        # faint raw signal for context
        ax.plot(x, raw, color=color, lw=0.6, alpha=0.25)
        # shaded local range
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        # bold smoothed trend (the actual claim)
        ax.plot(x, smooth, color=color, lw=2.2, label=label)

    ax.set_ylabel("CER (%)")
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Training dynamics: CER and distributional distance")

    # ---------------- Bottom panel: distance proxy ----------------
    ax = axes[1]
    raw = df["fid"]
    smooth = smoothed(raw, window)
    lo, hi = band(raw, window)

    ax.plot(x, raw, color="#2ca02c", lw=0.6, alpha=0.25)
    ax.fill_between(x, lo, hi, color="#2ca02c", alpha=0.12, linewidth=0)
    ax.plot(x, smooth, color="#2ca02c", lw=2.2, label="Training-time distance (Inception-FID)")

    ax.set_ylabel("Distance")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_xscale("log")
    ax.legend(loc="upper right", frameon=False)

    for a in axes:
        a.grid(True, which="both", axis="x", alpha=0.2)
        a.grid(True, which="major", axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def print_milestone_summary(df):
    print("\n--- Milestone check against paper prose ---")
    e0 = df[df["epoch"] == 0].iloc[0]
    print(f"Epoch 0:      eval_cer_gen={e0['eval_cer_gen']:.2f}%  "
          f"eval_cer_swap={e0['eval_cer_swap']:.2f}%  fid={e0['fid']:.2f}")

    near400 = df.iloc[(df["epoch"] - 400).abs().argsort()[:1]].iloc[0]
    print(f"Near epoch 400 (closest={int(near400['epoch'])}): "
          f"eval_cer_gen={near400['eval_cer_gen']:.2f}%  fid={near400['fid']:.2f}")

    last = df.iloc[-1]
    print(f"Final epoch {int(last['epoch'])}: "
          f"eval_cer_gen={last['eval_cer_gen']:.2f}%  eval_cer_swap={last['eval_cer_swap']:.2f}%  "
          f"fid={last['fid']:.2f}")
    print("--------------------------------------------\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_training_curves_clean.py <path_to_training_log.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = load_data(csv_path)

    print_milestone_summary(df)
    plot_main_figure(df, out_path="training_curves.png")