import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/orangepi/IQFormerLite")
PYTHON_BIN = Path("/home/orangepi/miniconda3/envs/rknn/bin/python")
RUNTIME_LIB = ROOT / "rknn" / "runtime" / "lib"

PROJECTS = {
    "rknn_IQFormerLite": {
        "script": ROOT / "rknn" / "rknn_IQFormerLite" / "inference_on_rk3588.py",
        "models_dir": ROOT / "rknn" / "rknn_IQFormerLite" / "weights",
        "output_dir": ROOT / "rknn" / "rknn_IQFormerLite",
        "extra_args": [],
    },
    "rknn_IQFormer": {
        "script": ROOT / "rknn" / "rknn_IQFormer" / "inference_on_rk3588.py",
        "models_dir": ROOT / "rknn" / "rknn_IQFormer" / "weights" / "IQFormer",
        "output_dir": ROOT / "rknn" / "rknn_IQFormer",
        "extra_args": ["--pt_model_path", ""],
    },
}

METRICS = [
    "params_m",
    "flops_g",
    "cpu_model_size_kb",
    "rknn_model_size_kb",
    "cpu_latency_ms",
    "cpu_throughput",
    "npu_latency_batch_ms",
    "npu_latency_sample_ms",
    "npu_throughput",
    "npu_accuracy",
    "speedup",
    "memory_baseline_mb",
    "memory_peak_mb",
    "memory_delta_mb",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-seed RKNN benchmark aligned with main.py split.")
    parser.add_argument(
        "--projects",
        nargs="+",
        default=["rknn_IQFormerLite", "rknn_IQFormer"],
        choices=sorted(PROJECTS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--database_choose", type=str, default="2016.10a")
    parser.add_argument("--data", type=str, default=str(ROOT / "dataset" / "RML2016.10a.pkl"))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split_random_state", type=int, default=233)
    return parser.parse_args()


def to_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    summary_rows = []
    for model, model_rows in grouped.items():
        summary = {
            "model": model,
            "run_count": len(model_rows),
            "seed_list": "|".join(str(r["seed"]) for r in model_rows),
            "split_random_state": model_rows[0]["split_random_state"],
            "database_choose": model_rows[0]["database_choose"],
            "data": model_rows[0]["data"],
            "model_source": model_rows[0]["model_source"],
        }
        for metric in METRICS:
            values = [to_float(r.get(metric)) for r in model_rows]
            values = [v for v in values if v is not None]
            if not values:
                summary[f"{metric}_mean"] = ""
                summary[f"{metric}_std"] = ""
                continue
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            summary[f"{metric}_mean"] = round(mean, 6)
            summary[f"{metric}_std"] = round(std, 6)
        summary_rows.append(summary)
    return summary_rows


def build_env():
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    extra_paths = [
        str(ROOT),
        str(ROOT / "model"),
        str(ROOT / "utils"),
        str(ROOT / "model" / "torch-conv-kan"),
        str(ROOT / "model" / "torch-conv-kan" / "kan_convs"),
    ]
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra_paths + ([current_pythonpath] if current_pythonpath else []))
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([str(RUNTIME_LIB)] + ([current_ld] if current_ld else []))
    return env


def run_project(project_name, args):
    config = PROJECTS[project_name]
    output_dir = config["output_dir"]
    env = build_env()
    all_rows = []

    for seed in args.seeds:
        seed_csv = output_dir / f"benchmark_seed{seed}.csv"
        cmd = [
            str(PYTHON_BIN),
            str(config["script"]),
            "--models_dir",
            str(config["models_dir"]),
            "--output_csv",
            str(seed_csv),
            "--data",
            args.data,
            "--database_choose",
            args.database_choose,
            "--test_size",
            str(args.test_size),
            "--batch_size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--split_random_state",
            str(args.split_random_state),
        ] + config["extra_args"]
        print(f"[run] {project_name} seed={seed}")
        completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        print(f"[exit] {project_name} seed={seed} code={completed.returncode}")
        if not seed_csv.exists():
            raise RuntimeError(f"seed result not generated: {seed_csv}")
        rows = read_csv_rows(seed_csv)
        for row in rows:
            row["seed"] = seed
            row["split_random_state"] = args.split_random_state
            row["database_choose"] = args.database_choose
            row["data"] = args.data
            row["model_source"] = "shared_existing_model"
            all_rows.append(row)

    leading_columns = ["seed", "split_random_state", "database_choose", "data", "model_source"]
    trailing_columns = [name for name in all_rows[0].keys() if name not in leading_columns]
    combined_path = output_dir / "benchmark_5seeds.csv"
    write_csv(combined_path, all_rows, leading_columns + trailing_columns)

    summary_rows = build_summary(all_rows)
    summary_path = output_dir / "benchmark_5seeds_summary.csv"
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    print(f"[saved] {combined_path}")
    print(f"[saved] {summary_path}")


def main():
    args = parse_args()
    for project_name in args.projects:
        run_project(project_name, args)


if __name__ == "__main__":
    main()
