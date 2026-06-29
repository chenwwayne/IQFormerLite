#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASETS = ("2016.10a", "2016.10b")
MODEL_ORDER = (
    "MCLDNN",
    "MCFormer",
    "PET-CGDNN",
    "AMC-Net",
    "FEA-T",
    "IQFormer",
    "IQFormerLite",
)
BASELINE_LABELS = {
    "MCLDNN": "MCLDNN",
    "MCFormer": "MCFormer",
    "PETCGDNN": "PET-CGDNN",
    "AMCNET": "AMC-Net",
    "FEA_T128": "FEA-T",
}


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build five-seed data for paper Figures 4 and 5.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=code_root / "results" / "table2_5seed_20260626",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=code_root / "results" / "table2_5seed_20260626" / "plot_data",
    )
    parser.add_argument("--legacy-output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def seed_number(path: Path) -> int:
    return int(path.parent.name.removeprefix("seed"))


def read_test_acc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if set(frame.columns) != {"SNR", "0"}:
        raise ValueError(f"Unexpected Test_ACC.csv columns in {path}: {list(frame.columns)}")
    frame = frame.loc[frame["SNR"].astype(str) != "Avg"].copy()
    frame["SNR"] = pd.to_numeric(frame["SNR"], errors="raise").astype(int)
    frame["accuracy"] = pd.to_numeric(frame["0"], errors="raise")
    expected = list(range(-20, 20, 2))
    if frame["SNR"].tolist() != expected:
        raise ValueError(f"Unexpected SNR sequence in {path}")
    return frame[["SNR", "accuracy"]]


def collect_figure4(results_root: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    baseline_root = results_root / "baseline" / "raw_runs"
    iq_root = results_root / "iqformer_vs_lite" / "raw_runs"

    for dataset in DATASETS:
        for model_dir in sorted((baseline_root / dataset).iterdir()):
            label = BASELINE_LABELS.get(model_dir.name)
            if label is None:
                continue
            for csv_path in sorted(model_dir.glob("seed*/Test_ACC.csv"), key=seed_number):
                frame = read_test_acc(csv_path)
                frame.insert(0, "seed", seed_number(csv_path))
                frame.insert(0, "model", label)
                frame.insert(0, "dataset", dataset)
                records.append(frame)

        for model in ("IQFormer", "IQFormerLite"):
            for csv_path in sorted((iq_root / dataset / model).glob("seed*/Test_ACC.csv"), key=seed_number):
                frame = read_test_acc(csv_path)
                frame.insert(0, "seed", seed_number(csv_path))
                frame.insert(0, "model", model)
                frame.insert(0, "dataset", dataset)
                records.append(frame)

    per_seed = pd.concat(records, ignore_index=True)
    summary = (
        per_seed.groupby(["dataset", "model", "SNR"], sort=False)["accuracy"]
        .agg(n="count", mean="mean", std=lambda values: values.std(ddof=1))
        .reset_index()
    )
    bad = summary.loc[summary["n"] != 5]
    if not bad.empty:
        raise ValueError(f"Figure 4 groups without five seeds:\n{bad.to_string(index=False)}")
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    return summary.sort_values(["dataset", "model", "SNR"]).reset_index(drop=True)


def read_modulation_acc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "SNR" not in frame.columns:
        raise ValueError(f"Missing SNR column in {path}")
    frame["SNR"] = pd.to_numeric(frame["SNR"], errors="raise").astype(int)
    if frame["SNR"].tolist() != list(range(-20, 20, 2)):
        raise ValueError(f"Unexpected SNR sequence in {path}")
    return frame.melt(id_vars="SNR", var_name="modulation", value_name="accuracy")


def collect_figure5(results_root: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    iq_root = results_root / "iqformer_vs_lite" / "raw_runs"
    for dataset in DATASETS:
        for model in ("IQFormer", "IQFormerLite"):
            paths = sorted((iq_root / dataset / model).glob("seed*/Test_mod_SNR.csv"), key=seed_number)
            for csv_path in paths:
                frame = read_modulation_acc(csv_path)
                frame.insert(0, "seed", seed_number(csv_path))
                frame.insert(0, "model", model)
                frame.insert(0, "dataset", dataset)
                records.append(frame)

    per_seed = pd.concat(records, ignore_index=True)
    summary = (
        per_seed.groupby(["dataset", "model", "modulation", "SNR"], sort=False)["accuracy"]
        .agg(n="count", mean="mean", std=lambda values: values.std(ddof=1))
        .reset_index()
    )
    bad = summary.loc[summary["n"] != 5]
    if not bad.empty:
        raise ValueError(f"Figure 5 groups without five seeds:\n{bad.to_string(index=False)}")
    return summary.sort_values(["dataset", "model", "modulation", "SNR"]).reset_index(drop=True)


def write_legacy_wide_csvs(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        dataset_name = dataset.replace(".", "")
        subset = summary.loc[summary["dataset"] == dataset]
        for metric, suffix in (("mean", ""), ("std", "_std")):
            wide = subset.pivot(index="SNR", columns="model", values=metric).reset_index()
            columns = ["SNR", *[model for model in MODEL_ORDER if model in wide.columns]]
            wide[columns].to_csv(output_dir / f"RML{dataset_name}{suffix}.csv", index=False)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure4 = collect_figure4(args.results_root)
    figure5 = collect_figure5(args.results_root)
    figure4_path = args.output_dir / "figure4_snr_accuracy_mean_std.csv"
    figure5_path = args.output_dir / "figure5_modulation_accuracy_mean_std.csv"
    figure4.to_csv(figure4_path, index=False)
    figure5.to_csv(figure5_path, index=False)
    write_legacy_wide_csvs(figure4, args.legacy_output_dir)

    print(f"Wrote {figure4_path} ({len(figure4)} rows)")
    print(f"Wrote {figure5_path} ({len(figure5)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
