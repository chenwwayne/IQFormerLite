#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Conv(k=3) against k=31 controls.")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/lkf_conv_k3_20260701"))
    parser.add_argument("--reference-dir", type=Path, default=Path("results/lkf_ablation_20260630"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def read_accuracy(path: Path) -> tuple[float, float, float]:
    frame = pd.read_csv(path)
    values = pd.to_numeric(frame["0"], errors="coerce")
    snr = pd.to_numeric(frame["SNR"], errors="coerce")
    overall = values.loc[frame["SNR"].astype(str) == "Avg"]
    if overall.empty:
        raise ValueError(f"Missing Avg row in {path}")
    low = values.loc[(snr >= -20) & (snr <= 0)].mean()
    high = values.loc[(snr >= 0) & (snr <= 18)].mean()
    return float(overall.iloc[0]), float(low), float(high)


def format_pm(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    missing = []
    for dataset in ("2016.10a", "2016.10b"):
        for seed in args.seeds:
            tag = f"model_{dataset}_60_1024_0.001_IQFormerLite_lkf-conv_k3_seed{seed}"
            path = args.logs_root / tag / "Test_ACC.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            overall, low, high = read_accuracy(path)
            records.append({"dataset": dataset, "variant": "conv_k3", "seed": seed,
                            "overall": overall, "low_snr": low, "high_snr": high,
                            "run_dir": str(path.parent)})

    new_runs = pd.DataFrame(records)
    new_runs.to_csv(args.output_dir / "conv_k3_per_seed.csv", index=False)
    if missing and args.fail_on_missing:
        print("Missing runs:\n" + "\n".join(missing))
        return 2

    reference = pd.read_csv(args.reference_dir / "lkf_ablation_per_seed.csv")
    reference = reference.loc[reference["variant"].isin(["conv", "full"])].copy()
    reference["variant"] = reference["variant"].map({"conv": "conv_k31", "full": "full_k31"})
    combined = pd.concat([new_runs, reference], ignore_index=True)

    complexity_k3 = pd.read_csv(args.output_dir / "conv_k3_complexity.csv")
    complexity_ref = pd.read_csv(args.reference_dir / "lkf_ablation_complexity.csv")
    complexity_ref = complexity_ref.loc[complexity_ref["variant"].isin(["conv", "full"]),
                                        ["dataset", "variant", "params", "flops"]].copy()
    complexity_ref["variant"] = complexity_ref["variant"].map({"conv": "conv_k31", "full": "full_k31"})
    complexity = pd.concat([complexity_k3, complexity_ref], ignore_index=True)

    summary_rows = []
    for (dataset, variant), group in combined.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant, "n": len(group)}
        for metric in ("overall", "low_snr", "high_snr"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
            row[f"{metric}_mean_pm_std"] = format_pm(row[f"{metric}_mean"], row[f"{metric}_std"])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).merge(complexity, on=["dataset", "variant"], validate="one_to_one")
    summary.to_csv(args.output_dir / "conv_k3_comparison_summary.csv", index=False)

    pivot = combined.pivot(index=["dataset", "seed"], columns="variant", values="overall")
    pair_rows = []
    t_critical_df4 = 2.7764451051977987
    for dataset in ("2016.10a", "2016.10b"):
        dataset_values = pivot.loc[dataset]
        for reference_name in ("conv_k31", "full_k31"):
            delta = (dataset_values["conv_k3"] - dataset_values[reference_name]) * 100
            mean = delta.mean()
            std = delta.std(ddof=1)
            margin = t_critical_df4 * std / math.sqrt(len(delta))
            pair_rows.append({"dataset": dataset, "variant_a": "conv_k3", "variant_b": reference_name,
                              "delta_a_minus_b_pp": mean, "paired_delta_std_pp": std,
                              "ci95_low_pp": mean - margin, "ci95_high_pp": mean + margin})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(args.output_dir / "conv_k3_pairwise.csv", index=False)

    lines = ["# Conv(k=3) Five-Seed Comparison", "",
             "| Dataset | Variant | Params (M) | FLOPs (M) | Low SNR | High SNR | Overall |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in summary.iterrows():
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['params']/1e6:.4f} | "
                     f"{row['flops']/1e6:.4f} | {row['low_snr_mean_pm_std']} | "
                     f"{row['high_snr_mean_pm_std']} | {row['overall_mean_pm_std']} |")
    lines.extend(["", "## Paired differences", "",
                  "| Dataset | Variant A | Variant B | Delta A-B (pp) | 95% CI (pp) |",
                  "| --- | --- | --- | ---: | ---: |"])
    for _, row in pairs.iterrows():
        lines.append(f"| {row['dataset']} | {row['variant_a']} | {row['variant_b']} | "
                     f"{row['delta_a_minus_b_pp']:+.3f} | "
                     f"[{row['ci95_low_pp']:+.3f}, {row['ci95_high_pp']:+.3f}] |")
    lines.extend(["", "Conv(k=3) is a conventional small-receptive-field baseline. It is not a "
                  "receptive-field-matched replacement for the existing Conv(k=31) control."])
    (args.output_dir / "conv_k3_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary[["dataset", "variant", "overall_mean_pm_std"]].to_string(index=False))
    print(pairs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
