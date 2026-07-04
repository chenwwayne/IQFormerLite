#!/usr/bin/env python3
"""Build one paper-review summary figure from LKF interpretability CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact LKF interpretability summary figure.")
    parser.add_argument("--input_dir", default="results/lkf_interpretability_20260630")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def pivot_response(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def plot_heatmap(ax: plt.Axes, freq: np.ndarray, data: np.ndarray, title: str, cmap: str, vmax: float = 1.0):
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[freq.min(), freq.max(), 0, data.shape[0] - 1],
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Filter index")
    return im


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_dir / "lkf_interpretability_summary"

    weight = pd.read_csv(input_dir / "lkf_weight_response.csv")
    group = pd.read_csv(input_dir / "aggregate_snr_family_response_mean_std.csv")
    base_curve = pd.read_csv(input_dir / "aggregate_base_response_mean_std.csv")

    base = weight[weight["branch"] == "base"].copy()
    initial = base[base["checkpoint"] == "initial"]
    trained = base[base["checkpoint"] != "initial"]

    init_freq, init_data, _ = pivot_response(initial)
    trained_mean = (
        trained.groupby(["filter_index", "frequency"], as_index=False)["normalized_magnitude"]
        .mean()
    )
    trained_std = (
        trained.groupby(["filter_index", "frequency"], as_index=False)["normalized_magnitude"]
        .std()
        .fillna(0.0)
    )
    tr_freq, tr_data, order = pivot_response(trained_mean)
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
    fig = plt.figure(figsize=(13.2, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])

    ax_init = fig.add_subplot(gs[0, 0])
    ax_trained = fig.add_subplot(gs[0, 1])
    ax_std = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, 0])
    ax_group = fig.add_subplot(gs[1, 1:])

    im0 = plot_heatmap(ax_init, init_freq, init_data, "A. Random initialization", "magma")
    im1 = plot_heatmap(ax_trained, tr_freq, tr_data, "B. Trained LKF, 5-seed mean", "viridis")
    im2 = plot_heatmap(ax_std, tr_freq, std_data, "C. Cross-seed variability", "cividis", vmax=max(float(np.nanmax(std_data)), 1e-6))
    fig.colorbar(im0, ax=ax_init, shrink=0.88, label="Magnitude")
    fig.colorbar(im1, ax=ax_trained, shrink=0.88, label="Magnitude")
    fig.colorbar(im2, ax=ax_std, shrink=0.88, label="Std.")

    freq = base_curve["frequency"].to_numpy(dtype=float)
    mean = base_curve["mean"].to_numpy(dtype=float)
    std = base_curve["std"].to_numpy(dtype=float)
    ax_curve.plot(freq, mean, color="#0f5c7a", linewidth=2.2)
    ax_curve.fill_between(freq, mean - std, mean + std, color="#0f5c7a", alpha=0.18)
    ax_curve.set_title("D. Mean response envelope", fontsize=11, weight="bold")
    ax_curve.set_xlabel("Normalized frequency")
    ax_curve.set_ylabel("Mean normalized magnitude")
    ax_curve.set_ylim(0, 1.05)
    ax_curve.grid(True, alpha=0.25)

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
        x = group_df["frequency"].to_numpy(dtype=float)
        y = group_df["mean"].to_numpy(dtype=float)
        s = group_df["std"].to_numpy(dtype=float)
        ax_group.plot(x, y, linewidth=2.0, color=colors.get(name), label=labels.get(name, name))
        ax_group.fill_between(x, y - s, y + s, color=colors.get(name), alpha=0.10)
    ax_group.set_title("E. LKF output spectra grouped by SNR and modulation family", fontsize=11, weight="bold")
    ax_group.set_xlabel("Normalized frequency")
    ax_group.set_ylabel("Normalized output power")
    ax_group.set_ylim(0, 1.05)
    ax_group.grid(True, alpha=0.25)
    ax_group.legend(frameon=False, ncol=2, loc="upper right")

    fig.suptitle(
        "LKF interpretability: learned spectral structure and input-response stability",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {output_prefix.with_suffix('.pdf')}")
    print(f"Saved {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
