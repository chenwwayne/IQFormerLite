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


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile ordinary-convolution kernel sweep.")
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 7, 15, 31, 51])
    parser.add_argument("--output", type=Path,
                        default=Path("results/conv_kernel_sweep_20260701/complexity.csv"))
    args = parser.parse_args()
    x = torch.randn(1, 2, 128)
    rows = []
    for dataset, classes in (("2016.10a", 11), ("2016.10b", 10)):
        for kernel in args.kernels:
            model = IQFormerLite(
                [2, 3, 2], embed_dims=[64, 64, 64], mlp_ratios=1,
                num_classes=classes, down_patch_size=3, down_stride=2, down_pad=1,
                drop_rate=0.2, drop_path_rate=0.0, use_layer_scale=False,
                aux_mode="kan", band_k=32, kernel_size=kernel, grid_size=4,
                grid_range=(-2.0, 2.0), lkf_variant="conv",
            ).eval()
            output = model(x)
            if output.shape != (1, classes):
                raise AssertionError(f"Unexpected output shape: {dataset}, k={kernel}")
            rows.append({"dataset": dataset, "variant": f"conv_k{kernel}",
                         "params": sum(p.numel() for p in model.parameters()),
                         "flops": count_conv_linear_ops(model, x)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
