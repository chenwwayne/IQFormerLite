#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


RUN_TAG_RE = re.compile(
    r"^model_(?P<dataset>[^_]+)_(?P<epochs>\d+)_(?P<batch>\d+)_(?P<lr>[^_]+)_(?P<comment>.+)$"
)
COMMENT_RE = re.compile(
    r"^(?P<model>IQFormerLite|IQFormer)_aux-(?P<aux>[^_]+)_seed(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 5-seed AMR results.")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/table2_5seed_20260626/iqformer_vs_lite"))
    parser.add_argument("--datasets", nargs="+", default=["2016.10a", "2016.10b"])
    parser.add_argument("--models", nargs="+", default=["IQFormerLite", "IQFormer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def format_percent(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def read_test_acc(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    if "SNR" not in df.columns:
        raise ValueError(f"Missing SNR column in {csv_path}")
    acc_col = "0" if "0" in df.columns else df.columns[1]
    snr_series = df["SNR"].astype(str)
    avg_rows = df.loc[snr_series == "Avg", acc_col]
    if avg_rows.empty:
        raise ValueError(f"Missing Avg row in {csv_path}")
    return float(avg_rows.iloc[0])


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing = []

    for dataset in args.datasets:
        for model in args.models:
            aux_mode = "kan" if model == "IQFormerLite" else "stft"
            for seed in args.seeds:
                comment = f"{model}_aux-{aux_mode}_seed{seed}"
                run_tag = f"model_{dataset}_{args.num_epochs}_{args.batch_size}_{args.lr}_{comment}"
                run_dir = args.logs_root / run_tag
                csv_path = run_dir / "Test_ACC.csv"
                if not csv_path.exists():
                    missing.append(str(csv_path))
                    continue

                run_value = read_test_acc(csv_path)
                records.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "aux_mode": aux_mode,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "test_acc": run_value,
                    }
                )

    per_run_df = pd.DataFrame(records)
    per_run_csv = args.output_dir / "iqformer_per_seed_overall_accuracy.csv"
    per_run_df.to_csv(per_run_csv, index=False)

    if per_run_df.empty:
        print("No completed runs were found.")
        if missing:
            print("Missing files:")
            for item in missing:
                print(f"  - {item}")
        return 1

    summary_rows = []
    grouped = per_run_df.groupby(["dataset", "model"], sort=True)
    for (dataset, model), group in grouped:
        values = group["test_acc"].tolist()
        mean = float(pd.Series(values).mean())
        std = float(pd.Series(values).std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n": len(values),
                "mean": mean,
                "std": std,
                "mean_pm_std": format_percent(mean, std),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    summary_csv = args.output_dir / "iqformer_summary_overall_accuracy.csv"
    summary_md = args.output_dir / "iqformer_summary_overall_accuracy.md"
    summary_df.to_csv(summary_csv, index=False)

    md_lines = [
        "# 5-seed summary",
        "",
        "| Dataset | Model | n | Mean | Std | Mean ± Std |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        md_lines.append(
            "| {dataset} | {model} | {n} | {mean:.4f} | {std:.4f} | {mean_pm_std} |".format(
                dataset=row["dataset"],
                model=row["model"],
                n=int(row["n"]),
                mean=float(row["mean"]),
                std=float(row["std"]),
                mean_pm_std=row["mean_pm_std"],
            )
        )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {per_run_csv}")

    if missing:
        print("\nMissing runs:")
        for item in missing:
            print(f"  - {item}")
        if args.fail_on_missing:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
