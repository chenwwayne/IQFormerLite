#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from model.IQFormerLite import IQFormerLite


def build_model(aux_mode: str, variant: str = "full") -> IQFormerLite:
    return IQFormerLite(
        [2, 3, 2],
        embed_dims=[64, 64, 64],
        mlp_ratios=1,
        num_classes=11,
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
        lkf_variant=variant,
    )


def main() -> int:
    torch.manual_seed(1)
    x = torch.randn(2, 2, 128)
    variants = ["conv", "base_only", "rbf_only", "bspline", "full"]

    for variant in variants:
        model = build_model("kan", variant)
        output = model(x)
        if output.shape != (2, 11):
            raise AssertionError(f"Unexpected output shape for {variant}: {tuple(output.shape)}")
        output.sum().backward()
        if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
            raise AssertionError(f"No gradients produced for {variant}")

        fb = model.kanstem.fb
        if variant == "base_only" and len(fb.spline_conv) != 0:
            raise AssertionError("base_only unexpectedly constructed RBF convolution parameters")
        if variant == "rbf_only" and len(fb.base_conv) != 0:
            raise AssertionError("rbf_only unexpectedly constructed base convolution parameters")
        print(f"{variant}: output={tuple(output.shape)}, params={sum(p.numel() for p in model.parameters())}")

    no_lkf = build_model("none")
    output = no_lkf(x)
    if output.shape != (2, 11) or hasattr(no_lkf, "kanstem"):
        raise AssertionError("w/o LKF model does not match the intended architecture ablation")
    print(f"none: output={tuple(output.shape)}, params={sum(p.numel() for p in no_lkf.parameters())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
