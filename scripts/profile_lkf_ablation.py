#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from model.IQFormerLite import IQFormerLite


VARIANTS = {
    "2016.10a": ["none", "conv", "base_only", "rbf_only", "bspline", "full"],
    "2016.10b": ["conv", "rbf_only", "full"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile all LKF ablation variants.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lkf_ablation_20260630/lkf_ablation_complexity.csv"),
    )
    return parser.parse_args()


def build_model(dataset: str, variant: str) -> IQFormerLite:
    aux_mode = "none" if variant == "none" else "kan"
    return IQFormerLite(
        [2, 3, 2],
        embed_dims=[64, 64, 64],
        mlp_ratios=1,
        num_classes=10 if dataset == "2016.10b" else 11,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.2,
        drop_path_rate=0.0,
        use_layer_scale=False,
        aux_mode=aux_mode,
        band_k=32,
        kernel_size=31,
        grid_size=4,
        grid_range=(-2.0, 2.0),
        lkf_variant="full" if variant == "none" else variant,
    )


def estimate_basis_ops(model: torch.nn.Module, batch_size: int, length: int) -> int:
    """Count scalar operations for basis construction under a documented convention."""
    total = 0
    for module in model.modules():
        name = module.__class__.__name__
        if name == "FastKANConv1DLayer" and module.branch_mode != "base_only":
            basis_values = batch_size * module.inputdim * length * module.grid_size
            total += basis_values * 5  # subtract, divide, square, negate, exp
        elif name == "KANConv1DLayer":
            order = module.spline_order
            intervals = module.grid_size + 2 * order
            input_values = batch_size * module.inputdim * length
            initial_comparisons = 2 * intervals
            recurrence_ops = 11 * sum(intervals - k for k in range(1, order + 1))
            total += input_values * (initial_comparisons + recurrence_ops)
    return total


def count_conv_linear_ops(model: torch.nn.Module, x: torch.Tensor) -> int:
    """Count Conv/Linear and BatchNorm FLOPs using the existing paper convention."""
    total = 0
    handles = []

    def conv1d_hook(module, _inputs, output):
        nonlocal total
        kernel = module.kernel_size[0]
        total += 2 * output.numel() * (module.in_channels // module.groups) * kernel

    def linear_hook(module, _inputs, output):
        nonlocal total
        total += 2 * output.numel() * module.in_features

    def batch_norm_hook(_module, _inputs, output):
        nonlocal total
        total += 4 * output.numel()

    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            handles.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, torch.nn.BatchNorm1d):
            handles.append(module.register_forward_hook(batch_norm_hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return total


def main() -> int:
    args = parse_args()
    records = []
    x = torch.randn(1, 2, 128)
    for dataset, variants in VARIANTS.items():
        for variant in variants:
            model = build_model(dataset, variant).eval()
            raw_flops = count_conv_linear_ops(model, x)
            basis_ops = estimate_basis_ops(model, batch_size=1, length=128)
            records.append({
                "dataset": dataset,
                "variant": variant,
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "raw_flops": int(raw_flops),
                "basis_ops": int(basis_ops),
                "flops": int(raw_flops + basis_ops),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
