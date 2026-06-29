#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


DATASET_LABELS = {
    "2016.10a": "RadioML2016.10A",
    "2016.10b": "RadioML2016.10B",
}
DATASET_STEMS = {
    "2016.10a": "RML201610a",
    "2016.10b": "RML201610b",
}
MODELS = ("IQFormer", "IQFormerLite")
MODULATION_ORDER = (
    "8PSK",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "AM-DSB",
    "AM-SSB",
    "WBFM",
)
MARKERS = ("o", "s", "D", "^", "v", ">", "<", "p", "*", "h", "H")
COLORS = dict(zip(MODULATION_ORDER, plt.get_cmap("tab20").colors[: len(MODULATION_ORDER)]))


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate five-seed modulation plots for paper Figure 5.")
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=code_root / "results" / "table2_5seed_20260626" / "plot_data" / "figure5_modulation_accuracy_mean_std.csv",
    )
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parent / "mod_acc_result")
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=code_root / "paper" / "Emerald_Publishing_V2_wo_author" / "Figures",
    )
    return parser.parse_args()


def plot_panel(data: pd.DataFrame, dataset: str, model: str, result_dir: Path, paper_dir: Path) -> None:
    subset = data.loc[(data["dataset"] == dataset) & (data["model"] == model)]
    available = set(subset["modulation"])
    modulations = [modulation for modulation in MODULATION_ORDER if modulation in available]
    if not modulations:
        raise ValueError(f"Missing Figure 5 data for {dataset}/{model}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for index, modulation in enumerate(modulations):
        curve = subset.loc[subset["modulation"] == modulation].sort_values("SNR")
        ax.plot(
            curve["SNR"],
            curve["mean"],
            label=modulation,
            marker=MARKERS[index],
            color=COLORS[modulation],
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xlim(-21, 19)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(-20, 19, 4))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_title(f"{model} on {DATASET_LABELS[dataset]}", fontsize=16)
    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.9)
    fig.tight_layout()

    stem = f"{DATASET_STEMS[dataset]}_{model}_mod_acc"
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
        for model in MODELS:
            plot_panel(data, dataset, model, args.result_dir, args.paper_figures_dir)
    print(f"Generated Figure 5 plots in {args.paper_figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
