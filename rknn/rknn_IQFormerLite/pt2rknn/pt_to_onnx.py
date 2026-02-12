import argparse
import os
import sys
import torch

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "model"))

from IQFormer import IQFormer


def parse_args():
    parser = argparse.ArgumentParser("IQFormer PT to ONNX")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--database_choose", type=str, default="2016.10a")
    parser.add_argument("--aux_mode", type=str, default="kan", choices=["none", "stft", "conv", "kan"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--stft_bins", type=int, default=32)
    parser.add_argument("--stft_frames", type=int, default=None)
    parser.add_argument("--band_k", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--grid_range", type=float, nargs=2, default=[-2.0, 2.0])
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def build_model(args):
    if args.database_choose == "2016.10b":
        num_classes = 10
    elif args.database_choose == "2019":
        num_classes = 26
    else:
        num_classes = 11
    if args.database_choose in ["2016.10a", "2016.10b"]:
        model = IQFormer(
            [2, 3, 2],
            embed_dims=[64, 64, 64],
            mlp_ratios=1,
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
            aux_mode=args.aux_mode,
            band_k=args.band_k,
            kernel_size=args.kernel_size,
            grid_size=args.grid_size,
            grid_range=tuple(args.grid_range),
        )
    else:
        model = IQFormer(
            [3, 3, 3],
            embed_dims=[64, 64, 64],
            mlp_ratios=4,
            num_classes=num_classes,
            down_patch_size=3,
            down_stride=2,
            down_pad=1,
            drop_rate=0.2,
            drop_path_rate=0.2,
            use_layer_scale=False,
            layer_scale_init_value=1e-5,
            fork_feat=False,
            vit_num=1,
            aux_mode=args.aux_mode,
            band_k=args.band_k,
            kernel_size=args.kernel_size,
            grid_size=args.grid_size,
            grid_range=tuple(args.grid_range),
        )
    return model


def sanitize_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_stft_frames(length, nperseg=31, noverlap=30):
    hop = max(1, nperseg - noverlap)
    return max(1, (length - nperseg) // hop + 1)


def main():
    args = parse_args()
    model = build_model(args)
    state = torch.load(args.model_path, map_location="cpu")
    state = sanitize_state_dict(state)
    model.load_state_dict(state)
    model.eval()

    batch_size = args.batch_size
    length = args.length
    iq = torch.randn(batch_size, 2, length, dtype=torch.float32)
    if args.aux_mode == "stft":
        frames = args.stft_frames
        if frames is None:
            frames = infer_stft_frames(length)
        stft = torch.randn(batch_size, 1, args.stft_bins, frames, dtype=torch.float32)
        inputs = (iq, stft)
        input_names = ["iq", "stft"]
        dynamic_axes = {
            "iq": {0: "batch", 2: "length"},
            "stft": {0: "batch", 3: "frames"},
            "logits": {0: "batch"},
        }
    else:
        inputs = (iq,)
        input_names = ["iq"]
        dynamic_axes = {
            "iq": {0: "batch", 2: "length"},
            "logits": {0: "batch"},
        }

    os.makedirs(os.path.dirname(args.onnx_path) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            args.onnx_path,
            input_names=input_names,
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )


if __name__ == "__main__":
    main()
