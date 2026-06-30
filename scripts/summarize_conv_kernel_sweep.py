#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def read_acc(path: Path) -> tuple[float, float, float]:
    frame = pd.read_csv(path)
    values = pd.to_numeric(frame["0"], errors="coerce")
    snr = pd.to_numeric(frame["SNR"], errors="coerce")
    overall = values.loc[frame.SNR.astype(str) == "Avg"]
    if overall.empty:
        raise ValueError(f"Missing Avg row: {path}")
    return float(overall.iloc[0]), float(values[(snr >= -20) & (snr <= 0)].mean()), \
        float(values[(snr >= 0) & (snr <= 18)].mean())


def pm(mean: float, std: float) -> str:
    return f"{mean*100:.2f} ± {std*100:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ordinary-convolution kernel sweep.")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/conv_kernel_sweep_20260701"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("results/lkf_ablation_20260630"))
    parser.add_argument("--k3-dir", type=Path, default=Path("results/lkf_conv_k3_20260701"))
    parser.add_argument("--new-kernels", nargs="+", type=int, default=[7, 15, 51])
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for dataset in ("2016.10a", "2016.10b"):
        for kernel in args.new_kernels:
            for seed in range(1, 6):
                tag = f"model_{dataset}_60_1024_0.001_IQFormerLite_lkf-conv_k{kernel}_seed{seed}"
                path = args.logs_root / tag / "Test_ACC.csv"
                if not path.exists():
                    missing.append(str(path)); continue
                overall, low, high = read_acc(path)
                rows.append({"dataset": dataset, "variant": f"conv_k{kernel}", "seed": seed,
                             "overall": overall, "low_snr": low, "high_snr": high,
                             "run_dir": str(path.parent)})
    if missing and args.fail_on_missing:
        print("Missing runs:\n" + "\n".join(missing)); return 2

    new = pd.DataFrame(rows)
    k3 = pd.read_csv(args.k3_dir / "conv_k3_per_seed.csv")
    ref = pd.read_csv(args.ablation_dir / "lkf_ablation_per_seed.csv")
    conv31 = ref[ref.variant.eq("conv")].copy(); conv31["variant"] = "conv_k31"
    full = ref[ref.variant.eq("full")].copy(); full["variant"] = "full_lkf_k31"
    combined = pd.concat([k3, new, conv31, full], ignore_index=True)
    combined.to_csv(args.output_dir / "per_seed.csv", index=False)

    complexity = pd.read_csv(args.output_dir / "complexity.csv")
    full_complexity = pd.read_csv(args.ablation_dir / "lkf_ablation_complexity.csv")
    full_complexity = full_complexity[full_complexity.variant.eq("full")][
        ["dataset", "params", "flops"]].copy()
    full_complexity["variant"] = "full_lkf_k31"
    complexity = pd.concat([complexity, full_complexity], ignore_index=True)

    summaries = []
    for (dataset, variant), group in combined.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant, "n": len(group)}
        for metric in ("overall", "low_snr", "high_snr"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
            row[f"{metric}_pm"] = pm(row[f"{metric}_mean"], row[f"{metric}_std"])
        summaries.append(row)
    summary = pd.DataFrame(summaries).merge(complexity, on=["dataset", "variant"], validate="one_to_one")
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    pivot = combined.pivot(index=["dataset", "seed"], columns="variant", values="overall")
    pairs = []
    for dataset in ("2016.10a", "2016.10b"):
        values = pivot.loc[dataset]
        for variant in [f"conv_k{k}" for k in [3, 7, 15, 31, 51]]:
            delta = (values[variant] - values["full_lkf_k31"]) * 100
            mean, std = delta.mean(), delta.std(ddof=1)
            margin = 2.7764451051977987 * std / math.sqrt(5)
            pairs.append({"dataset": dataset, "variant": variant,
                          "delta_vs_full_pp": mean, "ci95_low_pp": mean-margin,
                          "ci95_high_pp": mean+margin})
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(args.output_dir / "paired_vs_full.csv", index=False)

    lines = ["# Ordinary Convolution Kernel Sweep", "",
             "| Dataset | Variant | Params (M) | FLOPs (M) | Low SNR | High SNR | Overall |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    order = {f"conv_k{k}": k for k in [3, 7, 15, 31, 51]}; order["full_lkf_k31"] = 999
    summary["order"] = summary.variant.map(order)
    for _, row in summary.sort_values(["dataset", "order"]).iterrows():
        lines.append(f"| {row.dataset} | {row.variant} | {row.params/1e6:.4f} | "
                     f"{row.flops/1e6:.4f} | {row.low_snr_pm} | {row.high_snr_pm} | {row.overall_pm} |")
    lines.extend(["", "## Paired difference versus Full LKF(k=31)", "",
                  "| Dataset | Conv variant | Delta (pp) | 95% CI (pp) |",
                  "| --- | --- | ---: | ---: |"])
    for _, row in pair_frame.iterrows():
        lines.append(f"| {row.dataset} | {row.variant} | {row.delta_vs_full_pp:+.3f} | "
                     f"[{row.ci95_low_pp:+.3f}, {row.ci95_high_pp:+.3f}] |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.sort_values(["dataset", "order"])[["dataset","variant","overall_pm"]].to_string(index=False))
    print(pair_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
