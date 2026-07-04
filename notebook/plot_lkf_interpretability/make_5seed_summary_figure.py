#!/usr/bin/env python3
"""Create an aggregate-only 5-seed LKF interpretability figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 5-seed aggregate LKF summary figure.")
    parser.add_argument("--input_dir", default="results/lkf_interpretability_20260630")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def pivot_weight(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot_table(
        index="filter_index",
        columns="frequency",
        values="normalized_magnitude",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    freq = pivot.columns.to_numpy(dtype=float)
    data = pivot.to_numpy(dtype=float)
    order = np.argsort(freq[np.nanargmax(data, axis=1)])
    return freq, data[order], order


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_dir / "lkf_5seed_summary"

    weight = pd.read_csv(input_dir / "lkf_weight_response.csv")
    base_curve = pd.read_csv(input_dir / "aggregate_base_response_mean_std.csv")
    group = pd.read_csv(input_dir / "aggregate_snr_family_response_mean_std.csv")

    trained = weight[(weight["branch"] == "base") & (weight["checkpoint"] != "initial")].copy()
    trained_mean = (
        trained.groupby(["filter_index", "frequency"], as_index=False)["normalized_magnitude"]
        .mean()
    )
    trained_std = (
        trained.groupby(["filter_index", "frequency"], as_index=False)["normalized_magnitude"]
        .std()
        .fillna(0.0)
    )
    freq, mean_data, order = pivot_weight(trained_mean)
    std_pivot = trained_std.pivot_table(
        index="filter_index",
        columns="frequency",
        values="normalized_magnitude",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    std_data = std_pivot.to_numpy(dtype=float)[order]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(12.6, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    ax_mean = fig.add_subplot(gs[0, 0])
    ax_std = fig.add_subplot(gs[0, 1])
    ax_env = fig.add_subplot(gs[1, 0])
    ax_group = fig.add_subplot(gs[1, 1])

    im_mean = ax_mean.imshow(
        mean_data,
        aspect="auto",
        origin="lower",
        extent=[freq.min(), freq.max(), 0, mean_data.shape[0] - 1],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax_mean.set_title("A. Learned LKF response, 5-seed mean", fontsize=11, weight="bold")
    ax_mean.set_xlabel("Normalized frequency")
    ax_mean.set_ylabel("Filter index")
    fig.colorbar(im_mean, ax=ax_mean, shrink=0.86, label="Magnitude")

    vmax = max(float(np.nanmax(std_data)), 1e-6)
    im_std = ax_std.imshow(
        std_data,
        aspect="auto",
        origin="lower",
        extent=[freq.min(), freq.max(), 0, std_data.shape[0] - 1],
        cmap="cividis",
        vmin=0,
        vmax=vmax,
    )
    ax_std.set_title("B. Cross-seed variability", fontsize=11, weight="bold")
    ax_std.set_xlabel("Normalized frequency")
    ax_std.set_ylabel("Filter index")
    fig.colorbar(im_std, ax=ax_std, shrink=0.86, label="Std.")

    x = base_curve["frequency"].to_numpy(dtype=float)
    mean = base_curve["mean"].to_numpy(dtype=float)
    std = base_curve["std"].to_numpy(dtype=float)
    ax_env.plot(x, mean, color="#0f5c7a", linewidth=2.2, label="5-seed mean")
    ax_env.fill_between(x, mean - std, mean + std, color="#0f5c7a", alpha=0.18, label="±1 std")
    ax_env.set_title("C. Aggregate response envelope", fontsize=11, weight="bold")
    ax_env.set_xlabel("Normalized frequency")
    ax_env.set_ylabel("Mean normalized magnitude")
    ax_env.set_ylim(0, 1.05)
    ax_env.grid(True, alpha=0.25)
    ax_env.legend(frameon=False)

    colors = {
        "high_snr_analog": "#7a1f1f",
        "high_snr_digital": "#d66a2c",
        "low_snr_analog": "#1d4e89",
        "low_snr_digital": "#2a9d8f",
    }
    labels = {
        "high_snr_analog": "High SNR, analog",
        "high_snr_digital": "High SNR, digital",
        "low_snr_analog": "Low SNR, analog",
        "low_snr_digital": "Low SNR, digital",
    }
    for name, group_df in group.groupby("group"):
        group_df = group_df.sort_values("frequency")
        gx = group_df["frequency"].to_numpy(dtype=float)
        gy = group_df["mean"].to_numpy(dtype=float)
        gs_ = group_df["std"].to_numpy(dtype=float)
        ax_group.plot(gx, gy, linewidth=2.0, color=colors.get(name), label=labels.get(name, name))
        ax_group.fill_between(gx, gy - gs_, gy + gs_, color=colors.get(name), alpha=0.10)
    ax_group.set_title("D. LKF output spectra by SNR and modulation family", fontsize=11, weight="bold")
    ax_group.set_xlabel("Normalized frequency")
    ax_group.set_ylabel("Normalized output power")
    ax_group.set_ylim(0, 1.05)
    ax_group.grid(True, alpha=0.25)
    ax_group.legend(frameon=False, fontsize=8)

    fig.suptitle("LKF interpretability: 5-seed aggregate spectral response", fontsize=14, weight="bold")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {output_prefix.with_suffix('.pdf')}")
    print(f"Saved {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
