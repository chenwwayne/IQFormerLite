#!/usr/bin/env python3
"""Create title-free split figures for 5-seed LKF interpretability results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create split LKF figures without in-figure titles.")
    parser.add_argument("--input_dir", default="results/lkf_interpretability_20260630")
    parser.add_argument("--output_dir", default=None)
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


def savefig(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "paper_ready"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    im = ax.imshow(
        mean_data,
        aspect="auto",
        origin="lower",
        extent=[freq.min(), freq.max(), 0, mean_data.shape[0] - 1],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Filter index")
    fig.colorbar(im, ax=ax, shrink=0.86, label="Magnitude")
    fig.tight_layout()
    savefig(fig, output_dir / "lkf_A_5seed_mean_learned_filter_response", args.dpi)

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    im = ax.imshow(
        std_data,
        aspect="auto",
        origin="lower",
        extent=[freq.min(), freq.max(), 0, std_data.shape[0] - 1],
        cmap="cividis",
        vmin=0,
        vmax=max(float(np.nanmax(std_data)), 1e-6),
    )
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Filter index")
    fig.colorbar(im, ax=ax, shrink=0.86, label="Std.")
    fig.tight_layout()
    savefig(fig, output_dir / "lkf_B_cross_seed_variability", args.dpi)

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    x = base_curve["frequency"].to_numpy(dtype=float)
    mean = base_curve["mean"].to_numpy(dtype=float)
    std = base_curve["std"].to_numpy(dtype=float)
    ax.plot(x, mean, color="#0f5c7a", linewidth=2.2, label="5-seed mean")
    ax.fill_between(x, mean - std, mean + std, color="#0f5c7a", alpha=0.18, label="±1 std")
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Mean normalized magnitude")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, output_dir / "lkf_C_aggregate_response_envelope", args.dpi)

    fig, ax = plt.subplots(figsize=(5.8, 3.3))
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
        ax.plot(gx, gy, linewidth=2.0, color=colors.get(name), label=labels.get(name, name))
        ax.fill_between(gx, gy - gs_, gy + gs_, color=colors.get(name), alpha=0.10)
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Normalized output power")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, output_dir / "lkf_D_snr_modulation_family_output_spectra", args.dpi)

    print(f"Saved split figures to {output_dir}")


if __name__ == "__main__":
    main()
