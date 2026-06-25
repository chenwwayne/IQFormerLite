#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "MCLDNN": "MCLDNN",
    "MCFormer": "MCFormer",
    "PETCGDNN": "PET-CGDNN",
    "AMCNET": "AMC-Net",
    "FEA_T128": "FEA-T",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline runs for Table II.")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/table2_5seed_20260626/baseline"))
    parser.add_argument("--datasets", nargs="+", default=["2016.10a", "2016.10b"])
    parser.add_argument("--models", nargs="+", default=["MCLDNN", "MCFormer", "PETCGDNN", "AMCNET", "FEA_T128"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=str, default="0.001")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def read_accs(path: Path) -> tuple[float, float, float]:
    df = pd.read_csv(path)
    acc_col = "0" if "0" in df.columns else df.columns[1]
    snr = df["SNR"].astype(str)
    numeric = pd.to_numeric(df.loc[snr != "Avg", "SNR"], errors="coerce")
    acc = pd.to_numeric(df.loc[snr != "Avg", acc_col], errors="raise")
    low = float(acc.loc[(numeric >= -20) & (numeric <= 0)].mean())
    high = float(acc.loc[(numeric >= 0) & (numeric <= 18)].mean())
    overall = float(df.loc[snr == "Avg", acc_col].iloc[0])
    return low, high, overall


def pct(value: float) -> float:
    return round(value * 100.0, 2)


def mean_std(values: list[float]) -> tuple[float, float]:
    series = pd.Series(values, dtype=float)
    return float(series.mean()), float(series.std(ddof=1)) if len(series) > 1 else 0.0


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing = []
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                comment = f"{model}_aux-none_seed{seed}"
                run_tag = f"model_{dataset}_{args.num_epochs}_{args.batch_size}_{args.lr}_{comment}"
                csv_path = args.logs_root / run_tag / "Test_ACC.csv"
                if not csv_path.exists():
                    missing.append(str(csv_path))
                    continue
                low, high, overall = read_accs(csv_path)
                records.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "label": MODEL_LABELS.get(model, model),
                        "seed": seed,
                        "low_acc": low,
                        "high_acc": high,
                        "overall_acc": overall,
                        "run_dir": str(csv_path.parent),
                    }
                )

    per_run = pd.DataFrame(records)
    per_run.to_csv(args.output_dir / "baseline_per_seed_accuracy.csv", index=False)

    if per_run.empty:
        print("No completed baseline runs found.")
        if missing:
            print("\nMissing runs:")
            print("\n".join(missing))
        return 2

    summary_rows = []
    for (dataset, model, label), group in per_run.groupby(["dataset", "model", "label"], sort=True):
        low_mean, low_std = mean_std(group["low_acc"].tolist())
        high_mean, high_std = mean_std(group["high_acc"].tolist())
        overall_mean, overall_std = mean_std(group["overall_acc"].tolist())
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "label": label,
                "n": len(group),
                "low_mean_pct": pct(low_mean),
                "low_std_pct": pct(low_std),
                "high_mean_pct": pct(high_mean),
                "high_std_pct": pct(high_std),
                "overall_mean_pct": pct(overall_mean),
                "overall_std_pct": pct(overall_std),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["model", "dataset"]).reset_index(drop=True)
    summary.to_csv(args.output_dir / "baseline_summary_accuracy.csv", index=False)

    lines = [
        "# Baseline Table II summary",
        "",
        "| Model | Dataset | n | -20..0 | 0..18 | Overall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {label} | {dataset} | {n} | {low:.2f} ± {low_std:.2f} | {high:.2f} ± {high_std:.2f} | {overall:.2f} ± {overall_std:.2f} |".format(
                label=row["label"],
                dataset="A" if row["dataset"] == "2016.10a" else "B",
                n=int(row["n"]),
                low=float(row["low_mean_pct"]),
                low_std=float(row["low_std_pct"]),
                high=float(row["high_mean_pct"]),
                high_std=float(row["high_std_pct"]),
                overall=float(row["overall_mean_pct"]),
                overall_std=float(row["overall_std_pct"]),
            )
        )
    (args.output_dir / "baseline_summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nWrote {args.output_dir / 'baseline_summary_accuracy.csv'}")
    print(f"Wrote {args.output_dir / 'baseline_summary_table.md'}")
    print(f"Wrote {args.output_dir / 'baseline_per_seed_accuracy.csv'}")

    if missing:
        print("\nMissing runs:")
        print("\n".join(missing))
        if args.fail_on_missing:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
