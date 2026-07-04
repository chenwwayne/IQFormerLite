#!/usr/bin/env python3
"""Export LKF frequency-response and input-response figures.

This script is intended for the IQFormerLite revision experiment:

1. Plot learned frequency responses from the KAN filterbank weights.
2. Plot LKF output spectra grouped by SNR and modulation family.

The KAN filterbank is nonlinear. The direct frequency-response plot therefore
uses the linear convolution weights of the base branch and separately archives
the spline branch weight spectra. The SNR/family plot is input driven: it
measures the spectrum of the trained ``kanstem`` output for grouped samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model"))

from IQFormerLite import IQFormerLite  # noqa: E402


ANALOG_FAMILY = {"AM-DSB", "AM-SSB", "WBFM"}
DIGITAL_FAMILY = {
    "8PSK",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
}


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export IQFormerLite LKF interpretability figures."
    )
    parser.add_argument("--database_path", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--database_choose", default="2016.10b", choices=["2016.10a", "2016.10b"])
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="Checkpoint path. Can be repeated.")
    parser.add_argument("--checkpoint_label", action="append", default=[],
                        help="Label for each --checkpoint, in the same order.")
    parser.add_argument("--include_random_init", action="store_true",
                        help="Also analyze a randomly initialized model as an initial reference.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "lkf_interpretability_202606xx"))
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_samples_per_group", type=int, default=512)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--minSNR", type=int, default=-20)
    parser.add_argument("--maxSNR", type=int, default=18)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--band_k", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--grid_size", type=int, default=4)
    parser.add_argument("--grid_range", type=float, nargs=2, default=[-2.0, 2.0])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_classes(database_choose: str) -> list[str]:
    if database_choose == "2016.10a":
        return [
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
        ]
    return [
        "8PSK",
        "BPSK",
        "CPFSK",
        "GFSK",
        "PAM4",
        "QAM16",
        "QAM64",
        "QPSK",
        "AM-DSB",
        "WBFM",
    ]


def load_test_split(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    classes = get_classes(args.database_choose)
    filename = "RML2016.10a.pkl" if args.database_choose == "2016.10a" else "RML2016.10b.dat"
    data_path = Path(args.database_path) / filename
    data = pd.read_pickle(data_path)

    samples_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    snr_out: list[np.ndarray] = []
    for (label, snr), samples in data.items():
        if snr < args.minSNR or snr > args.maxSNR or label not in classes:
            continue
        labels = np.full(len(samples), classes.index(label))
        snrs = np.full(len(samples), snr)
        _, x_test, _, y_test, _, snr_test = train_test_split(
            samples,
            labels,
            snrs,
            test_size=args.test_size,
            random_state=233,
            stratify=labels,
        )
        samples_out.extend(x_test)
        labels_out.extend(y_test)
        snr_out.extend(snr_test)

    return (
        np.asarray(samples_out, dtype=np.float32),
        np.asarray(labels_out, dtype=np.int64),
        np.asarray(snr_out, dtype=np.int64),
        classes,
    )


def build_model(args: argparse.Namespace, num_classes: int) -> IQFormerLite:
    return IQFormerLite(
        [2, 3, 2],
        embed_dims=[64, 64, 64],
        mlp_ratios=1,
        act_layer=torch.nn.GELU,
        num_classes=num_classes,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.2,
        drop_path_rate=0.0,
        use_layer_scale=False,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        vit_num=1,
        aux_mode="kan",
        band_k=args.band_k,
        kernel_size=args.kernel_size,
        grid_size=args.grid_size,
        grid_range=tuple(args.grid_range),
    )


def sanitize_state_dict(state: object) -> dict[str, torch.Tensor]:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a state_dict-like object.")
    if state and all(str(k).startswith("module.") for k in state.keys()):
        return {str(k).replace("module.", "", 1): v for k, v in state.items()}
    return state  # type: ignore[return-value]


def resolve_checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    if args.include_random_init:
        specs.append(CheckpointSpec("initial", None))
    for idx, ckpt in enumerate(args.checkpoint):
        path = Path(ckpt)
        label = args.checkpoint_label[idx] if idx < len(args.checkpoint_label) else default_checkpoint_label(path)
        specs.append(CheckpointSpec(label, path))
    if not specs:
        raise ValueError("Provide at least one --checkpoint or pass --include_random_init.")
    if args.checkpoint_label and len(args.checkpoint_label) != len(args.checkpoint):
        raise ValueError("--checkpoint_label count must match --checkpoint count.")
    return specs


def default_checkpoint_label(path: Path) -> str:
    if path.stem == "weight":
        return path.parent.name
    return path.stem


def safe_filename(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in label.strip())
    return safe or "checkpoint"


def fft_response(weight: torch.Tensor, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized RMS magnitude response over non-filter dimensions."""
    w = weight.detach().cpu().float().numpy()
    response = np.fft.rfft(w, n=n_fft, axis=-1)
    mag = np.sqrt(np.mean(np.abs(response) ** 2, axis=tuple(range(1, response.ndim - 1))))
    denom = np.maximum(mag.max(axis=1, keepdims=True), 1e-12)
    mag = mag / denom
    freq = np.fft.rfftfreq(n_fft, d=1.0)
    return freq, mag


def plot_heatmap(
    freq: np.ndarray,
    mag: np.ndarray,
    title: str,
    output_base: Path,
    sort_by_peak: bool = True,
) -> None:
    order = np.arange(mag.shape[0])
    if sort_by_peak:
        order = np.argsort(freq[np.argmax(mag, axis=1)])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(
        mag[order],
        aspect="auto",
        origin="lower",
        extent=[freq[0], freq[-1], 0, mag.shape[0] - 1],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("LKF filter index (sorted)")
    fig.colorbar(im, ax=ax, label="Normalized magnitude")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def group_name(label: str, snr: int) -> str:
    snr_bin = "low_snr" if snr <= 0 else "high_snr"
    if label in ANALOG_FAMILY:
        family = "analog"
    elif label in DIGITAL_FAMILY:
        family = "digital"
    else:
        family = "other"
    return f"{snr_bin}_{family}"


def iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


@torch.no_grad()
def collect_group_response(
    model: IQFormerLite,
    samples: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    classes: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(args.seed)
    names = np.asarray([classes[int(idx)] for idx in labels])
    groups = np.asarray([group_name(label, int(snr)) for label, snr in zip(names, snrs)])

    for group in sorted(set(groups.tolist())):
        indices = np.where(groups == group)[0]
        if len(indices) > args.max_samples_per_group:
            indices = rng.choice(indices, size=args.max_samples_per_group, replace=False)
        spectra = []
        for batch_idx in iter_batches(indices, args.batch_size):
            x = torch.from_numpy(samples[batch_idx]).to(args.device)
            x = model.BN(x)
            feat = model.kanstem(x)
            spec = torch.fft.rfft(feat, n=args.n_fft, dim=-1).abs().pow(2)
            spec = spec.mean(dim=(0, 1)).detach().cpu().numpy()
            spectra.append(spec)
        if not spectra:
            continue
        mean_spec = np.mean(np.stack(spectra, axis=0), axis=0)
        mean_spec = mean_spec / max(float(mean_spec.max()), 1e-12)
        freq = np.fft.rfftfreq(args.n_fft, d=1.0)
        for f, value in zip(freq, mean_spec):
            rows.append(
                {
                    "group": group,
                    "frequency": float(f),
                    "normalized_power": float(value),
                    "n_samples": int(len(indices)),
                }
            )
    return pd.DataFrame(rows)


def plot_group_response(df: pd.DataFrame, title: str, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for group, group_df in df.groupby("group"):
        ax.plot(group_df["frequency"], group_df["normalized_power"], label=group, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Normalized LKF output power")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_weight_response(df: pd.DataFrame, branch: str, output_base: Path) -> None:
    branch_df = df[df["branch"] == branch].copy()
    if branch_df.empty:
        return
    per_checkpoint = (
        branch_df.groupby(["checkpoint", "frequency"], as_index=False)["normalized_magnitude"]
        .mean()
    )
    agg = (
        per_checkpoint.groupby("frequency")["normalized_magnitude"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg.to_csv(output_base.with_suffix(".csv"), index=False)

    x = agg["frequency"].to_numpy(dtype=float)
    mean = agg["mean"].to_numpy(dtype=float)
    std = agg["std"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x, mean, color="#1f5f8b", linewidth=2.0, label="mean across checkpoints")
    ax.fill_between(x, mean - std, mean + std, color="#1f5f8b", alpha=0.18, label="±1 std")
    ax.set_title(f"Aggregate LKF {branch}-branch response")
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Mean normalized magnitude")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_group_response(dfs: list[pd.DataFrame], output_base: Path) -> None:
    if not dfs:
        return
    all_df = pd.concat(dfs, ignore_index=True)
    per_checkpoint = (
        all_df.groupby(["checkpoint", "group", "frequency"], as_index=False)["normalized_power"]
        .mean()
    )
    agg = (
        per_checkpoint.groupby(["group", "frequency"])["normalized_power"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg.to_csv(output_base.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for group, group_df in agg.groupby("group"):
        x = group_df["frequency"].to_numpy(dtype=float)
        mean = group_df["mean"].to_numpy(dtype=float)
        std = group_df["std"].to_numpy(dtype=float)
        ax.plot(x, mean, linewidth=1.8, label=group)
        ax.fill_between(x, mean - std, mean + std, alpha=0.10)
    ax.set_title("Aggregate LKF output spectra by SNR and modulation family")
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Mean normalized LKF output power")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_weight_response_csv(
    output_path: Path,
    label: str,
    branch: str,
    freq: np.ndarray,
    mag: np.ndarray,
) -> None:
    rows = []
    for filter_idx in range(mag.shape[0]):
        for f, value in zip(freq, mag[filter_idx]):
            rows.append(
                {
                    "checkpoint": label,
                    "branch": branch,
                    "filter_index": filter_idx,
                    "frequency": float(f),
                    "normalized_magnitude": float(value),
                }
            )
    mode = "a" if output_path.exists() else "w"
    header = not output_path.exists()
    pd.DataFrame(rows).to_csv(output_path, mode=mode, header=header, index=False)


def write_readme(output_dir: Path, args: argparse.Namespace, specs: list[CheckpointSpec]) -> None:
    payload = {
        "database_choose": args.database_choose,
        "database_path": args.database_path,
        "checkpoints": [{"label": s.label, "path": str(s.path) if s.path else "random_init"} for s in specs],
        "band_k": args.band_k,
        "kernel_size": args.kernel_size,
        "grid_size": args.grid_size,
        "grid_range": args.grid_range,
        "n_fft": args.n_fft,
        "max_samples_per_group": args.max_samples_per_group,
        "note": (
            "Frequency-response plots use KAN base/spline convolution weights. "
            "SNR/family plots are input-driven spectra of the trained kanstem output."
        ),
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# LKF Interpretability Results\n\n"
        "This directory stores IQFormerLite LKF interpretability artifacts.\n\n"
        "## Interpretation Boundary\n\n"
        "- The KAN filterbank is nonlinear; direct learned-filter frequency responses are computed from stored convolution weights.\n"
        "- Different-SNR curves are not dynamic kernel responses. They are spectra of LKF outputs for grouped input samples.\n"
        "- Use these figures to support stable spectral-structure claims, not real-device deployment claims.\n\n"
        "## Files\n\n"
        "- `lkf_weight_response.csv`: normalized frequency response for base and spline convolution weights.\n"
        "- `*_base_response.pdf/png`: paper-ready base-branch filter response heatmaps.\n"
        "- `*_spline_response.pdf/png`: spline-branch weight response heatmaps for audit/supplement use.\n"
        "- `*_snr_family_response.csv`: SNR/family grouped LKF output spectra.\n"
        "- `*_snr_family_response.pdf/png`: paper-ready grouped output-response curves.\n"
        "- `aggregate_*`: checkpoint-level mean/std summaries when multiple trained checkpoints are provided.\n"
        "- `run_config.json`: exact analysis inputs and parameters.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = resolve_checkpoint_specs(args)

    samples, labels, snrs, classes = load_test_split(args)
    write_readme(output_dir, args, specs)

    weight_csv = output_dir / "lkf_weight_response.csv"
    if weight_csv.exists():
        weight_csv.unlink()

    trained_group_dfs: list[pd.DataFrame] = []
    for spec in specs:
        file_label = safe_filename(spec.label)
        model = build_model(args, num_classes=len(classes)).to(args.device)
        if spec.path is not None:
            state = torch.load(spec.path, map_location=args.device)
            model.load_state_dict(sanitize_state_dict(state))
        model.eval()
        if not hasattr(model, "kanstem"):
            raise AttributeError("Model does not expose kanstem; use IQFormerLite with aux_mode='kan'.")

        base_weight = model.kanstem.fb.base_conv[0].weight
        spline_weight = model.kanstem.fb.spline_conv[0].weight

        base_freq, base_mag = fft_response(base_weight, args.n_fft)
        spline_freq, spline_mag = fft_response(spline_weight, args.n_fft)
        save_weight_response_csv(weight_csv, spec.label, "base", base_freq, base_mag)
        save_weight_response_csv(weight_csv, spec.label, "spline", spline_freq, spline_mag)

        plot_heatmap(
            base_freq,
            base_mag,
            f"LKF base-branch frequency response ({spec.label})",
            output_dir / f"{file_label}_base_response",
        )
        plot_heatmap(
            spline_freq,
            spline_mag,
            f"LKF spline-branch weight response ({spec.label})",
            output_dir / f"{file_label}_spline_response",
        )

        group_df = collect_group_response(model, samples, labels, snrs, classes, args)
        group_df.insert(0, "checkpoint", spec.label)
        group_csv = output_dir / f"{file_label}_snr_family_response.csv"
        group_df.to_csv(group_csv, index=False)
        if spec.path is not None:
            trained_group_dfs.append(group_df)
        plot_group_response(
            group_df,
            f"LKF output spectra by SNR and modulation family ({spec.label})",
            output_dir / f"{file_label}_snr_family_response",
        )

    weight_df = pd.read_csv(weight_csv)
    trained_labels = {spec.label for spec in specs if spec.path is not None}
    trained_weight_df = weight_df[weight_df["checkpoint"].isin(trained_labels)]
    if not trained_weight_df.empty:
        plot_aggregate_weight_response(
            trained_weight_df,
            "base",
            output_dir / "aggregate_base_response_mean_std",
        )
        plot_aggregate_weight_response(
            trained_weight_df,
            "spline",
            output_dir / "aggregate_spline_response_mean_std",
        )
    plot_aggregate_group_response(
        trained_group_dfs,
        output_dir / "aggregate_snr_family_response_mean_std",
    )

    print(f"LKF interpretability artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
