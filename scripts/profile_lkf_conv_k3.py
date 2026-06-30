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
from scripts.profile_lkf_ablation import count_conv_linear_ops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the matched Conv(k=3) control.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lkf_conv_k3_20260701/conv_k3_complexity.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    x = torch.randn(1, 2, 128)
    records = []
    for dataset, num_classes in (("2016.10a", 11), ("2016.10b", 10)):
        model = IQFormerLite(
            [2, 3, 2], embed_dims=[64, 64, 64], mlp_ratios=1,
            num_classes=num_classes, down_patch_size=3, down_stride=2, down_pad=1,
            drop_rate=0.2, drop_path_rate=0.0, use_layer_scale=False,
            aux_mode="kan", band_k=32, kernel_size=3, grid_size=4,
            grid_range=(-2.0, 2.0), lkf_variant="conv",
        ).eval()
        output = model(x)
        if output.shape != (1, num_classes):
            raise AssertionError(f"Unexpected output shape for {dataset}: {tuple(output.shape)}")
        records.append({
            "dataset": dataset,
            "variant": "conv_k3",
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "flops": count_conv_linear_ops(model, x),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
