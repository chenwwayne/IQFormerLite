#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VARIANTS = {
    "2016.10a": ["none", "conv", "base_only", "rbf_only", "bspline", "full"],
    "2016.10b": ["conv", "rbf_only", "full"],
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the five-seed LKF ablation.")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/lkf_ablation_20260630"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--full-archive-root",
        type=Path,
        default=Path("results/table2_5seed_20260626/iqformer_vs_lite/raw_runs"),
    )
    parser.add_argument(
        "--complexity-csv",
        type=Path,
        default=Path("results/lkf_ablation_20260630/lkf_ablation_complexity.csv"),
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def run_dir(args: argparse.Namespace, dataset: str, variant: str, seed: int) -> Path:
    comment = f"IQFormerLite_lkf-{variant}_seed{seed}"
    tag = f"model_{dataset}_{args.num_epochs}_{args.batch_size}_{args.lr}_{comment}"
    return args.logs_root / tag


def read_accuracy(path: Path) -> tuple[float, float, float]:
    frame = pd.read_csv(path)
    value_col = "0" if "0" in frame.columns else frame.columns[1]
    labels = frame["SNR"].astype(str)
    overall_rows = frame.loc[labels == "Avg", value_col]
    if overall_rows.empty:
        raise ValueError(f"Missing Avg row in {path}")
    numeric_snr = pd.to_numeric(frame["SNR"], errors="coerce")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    low = values.loc[(numeric_snr >= -20) & (numeric_snr <= 0)].mean()
    high = values.loc[(numeric_snr >= 0) & (numeric_snr <= 18)].mean()
    return float(overall_rows.iloc[0]), float(low), float(high)


def format_pm(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    missing = []

    for dataset, variants in VARIANTS.items():
        for variant in variants:
            for seed in args.seeds:
                directory = run_dir(args, dataset, variant, seed)
                acc_path = directory / "Test_ACC.csv"
                if variant == "full" and not acc_path.exists():
                    directory = args.full_archive_root / dataset / "IQFormerLite" / f"seed{seed}"
                    acc_path = directory / "Test_ACC.csv"
                if not acc_path.exists():
                    missing.append(str(acc_path))
                    continue
                overall, low, high = read_accuracy(acc_path)
                records.append({
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "overall": overall,
                    "low_snr": low,
                    "high_snr": high,
                    "run_dir": str(directory),
                })

    per_seed = pd.DataFrame(records)
    per_seed.to_csv(args.output_dir / "lkf_ablation_per_seed.csv", index=False)
    if per_seed.empty:
        print("No completed LKF ablation runs found.")
        return 1

    summary_rows = []
    for (dataset, variant), group in per_seed.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant, "n": len(group)}
        for metric in ("overall", "low_snr", "high_snr"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
            row[f"{metric}_mean_pm_std"] = format_pm(row[f"{metric}_mean"], row[f"{metric}_std"])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if not args.complexity_csv.exists():
        raise FileNotFoundError(
            f"Missing complexity file {args.complexity_csv}; run scripts/profile_lkf_ablation.py first"
        )
    complexity = pd.read_csv(args.complexity_csv)
    summary = summary.merge(complexity, on=["dataset", "variant"], how="left", validate="one_to_one")
    if summary[["params", "raw_flops", "basis_ops", "flops"]].isna().any().any():
        raise ValueError("Complexity CSV does not cover every summarized dataset/variant")
    full_means = summary.loc[summary["variant"] == "full"].set_index("dataset")["overall_mean"]
    summary["delta_vs_full_pp"] = summary.apply(
        lambda row: (row["overall_mean"] - full_means.loc[row["dataset"]]) * 100,
        axis=1,
    )
    summary.to_csv(args.output_dir / "lkf_ablation_summary.csv", index=False)

    lines = [
        "# LKF Ablation Summary",
        "",
        "Accuracy is the mean ± sample standard deviation over five training seeds.",
        "FLOPs use the existing paper convention plus estimated basis-construction operations.",
        "These are server-side complexity estimates, not RK3588 deployment measurements.",
        "",
        "| Dataset | Variant | n | Params (M) | FLOPs (M) | Basis ops (M) | Low SNR | High SNR | Overall | Delta vs Full (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {int(row['n'])} | "
            f"{row['params'] / 1e6:.4f} | {row['flops'] / 1e6:.4f} | {row['basis_ops'] / 1e6:.4f} | "
            f"{row['low_snr_mean_pm_std']} | {row['high_snr_mean_pm_std']} | "
            f"{row['overall_mean_pm_std']} | {row['delta_vs_full_pp']:+.2f} |"
        )
    (args.output_dir / "lkf_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if missing:
        print("Missing result files:")
        for path in missing:
            print(f"  - {path}")
        if args.fail_on_missing:
            return 2
    print(summary[["dataset", "variant", "n", "overall_mean_pm_std", "delta_vs_full_pp"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
