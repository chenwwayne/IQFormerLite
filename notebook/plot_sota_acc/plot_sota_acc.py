#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


DATASET_LABELS = {
    "2016.10a": "RadioML2016.10A",
    "2016.10b": "RadioML2016.10B",
}
OUTPUT_NAMES = {
    "2016.10a": "RML201610a_sota_acc",
    "2016.10b": "RML201610b_sota_acc",
}
MODEL_ORDER = (
    "MCLDNN",
    "MCFormer",
    "PET-CGDNN",
    "AMC-Net",
    "FEA-T",
    "IQFormer",
    "IQFormerLite",
)
COLORS = {
    "MCLDNN": "#4C78A8",
    "MCFormer": "#59A14F",
    "PET-CGDNN": "#F28E8B",
    "AMC-Net": "#E377C2",
    "FEA-T": "#76B7B2",
    "IQFormer": "#4E5AB8",
    "IQFormerLite": "#9C1C1C",
}
MARKERS = {
    "MCLDNN": "o",
    "MCFormer": "s",
    "PET-CGDNN": "D",
    "AMC-Net": "^",
    "FEA-T": "v",
    "IQFormer": ">",
    "IQFormerLite": "<",
}
SHADED_MODELS = {"IQFormer", "IQFormerLite"}


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate five-seed SNR-accuracy plots for paper Figure 4.")
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=code_root / "results" / "table2_5seed_20260626" / "plot_data" / "figure4_snr_accuracy_mean_std.csv",
    )
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parent / "result")
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=code_root / "paper" / "Emerald_Publishing_V2_wo_author" / "Figures",
    )
    return parser.parse_args()


def draw_curve(ax: plt.Axes, frame: pd.DataFrame, model: str, marker_size: float) -> None:
    frame = frame.sort_values("SNR")
    x = frame["SNR"].to_numpy(dtype=float)
    mean = frame["mean"].to_numpy(dtype=float)
    std = frame["std"].to_numpy(dtype=float)
    color = COLORS[model]
    ax.plot(
        x,
        mean,
        label=model,
        marker=MARKERS[model],
        color=color,
        linewidth=2.0,
        markersize=marker_size,
    )
    if model in SHADED_MODELS:
        ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.14, linewidth=0)


def plot_dataset(data: pd.DataFrame, dataset: str, output_dirs: tuple[Path, Path]) -> None:
    subset = data.loc[data["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_ORDER:
        model_data = subset.loc[subset["model"] == model]
        if model_data.empty:
            raise ValueError(f"Missing Figure 4 data for {dataset}/{model}")
        draw_curve(ax, model_data, model, marker_size=6)

    ax.set_xlim(-21, 19)
    ax.set_ylim(0, 1.01)
    ax.set_xticks(range(-20, 19, 4))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_title(DATASET_LABELS[dataset], fontsize=17)
    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, loc="upper left", ncol=1, framealpha=0.9)

    inset = inset_axes(ax, width="46%", height="43%", loc="lower right", borderpad=1.0)
    high = subset.loc[subset["SNR"] >= 0]
    for model in MODEL_ORDER:
        draw_curve(inset, high.loc[high["model"] == model], model, marker_size=3.5)
    shaded = high.loc[high["model"].isin(SHADED_MODELS)]
    lower = min(float(high["mean"].min()), float((shaded["mean"] - shaded["std"]).min()))
    upper = max(float(high["mean"].max()), float((shaded["mean"] + shaded["std"]).max()))
    inset.set_xlim(-0.5, 18.5)
    inset.set_ylim(max(0, math.floor((lower - 0.01) * 20) / 20), min(1.005, math.ceil((upper + 0.01) * 20) / 20))
    inset.set_xticks([0, 6, 12, 18])
    inset.grid(True, linestyle="--", alpha=0.4)
    inset.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.91)
    stem = OUTPUT_NAMES[dataset]
    result_dir, paper_dir = output_dirs
    fig.savefig(result_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(result_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(result_dir / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(paper_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(paper_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.paper_figures_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data_csv)
    for dataset in DATASET_LABELS:
        plot_dataset(data, dataset, (args.result_dir, args.paper_figures_dir))
    print(f"Generated Figure 4 plots in {args.paper_figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
