import argparse
import csv
import os
import pickle
import re
import subprocess
import sys
import time
import types
import importlib.util

import numpy as np

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "utils"))
sys.path.append(os.path.join(base_dir, "model"))
torch_conv_kan_dir = os.path.join(base_dir, "model", "torch-conv-kan")
kan_convs_dir = os.path.join(torch_conv_kan_dir, "kan_convs")
kans_utils_path = os.path.join(torch_conv_kan_dir, "kans", "utils.py")
if os.path.isdir(torch_conv_kan_dir):
    sys.path.append(torch_conv_kan_dir)
if os.path.isdir(kan_convs_dir):
    sys.path.append(kan_convs_dir)
if os.path.isfile(kans_utils_path) and "kans" not in sys.modules:
    kans_spec = importlib.util.spec_from_file_location("kans", kans_utils_path)
    kans_module = importlib.util.module_from_spec(kans_spec)
    if kans_spec and kans_spec.loader:
        kans_spec.loader.exec_module(kans_module)
        sys.modules["kans"] = kans_module
fast_kan_conv_path = os.path.join(kan_convs_dir, "fast_kan_conv.py")
if os.path.isfile(fast_kan_conv_path):
    pkg_name = "kan_convs"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [kan_convs_dir]
        sys.modules[pkg_name] = pkg
    module_name = f"{pkg_name}.fast_kan_conv"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, fast_kan_conv_path)
        module = importlib.util.module_from_spec(spec)
        if spec and spec.loader:
            spec.loader.exec_module(module)
            sys.modules[module_name] = module

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None
    print("Error: rknn_toolkit_lite2 not found.")
    print("Please install it first (e.g., pip install rknn_toolkit_lite2).")

try:
    import psutil
except Exception:
    psutil = None

def set_seed(seed):
    np.random.seed(seed)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def infer_stft_frames(length, nperseg=31, noverlap=30):
    # With boundary='zeros' and padded=True (default in scipy.signal.stft),
    # the signal is padded with nperseg//2 zeros on both sides.
    # Total length L' = length + 2 * (nperseg // 2)
    # Frames = (L' - nperseg) // hop + 1
    hop = max(1, nperseg - noverlap)
    pad_len = 2 * (nperseg // 2)
    total_len = length + pad_len
    return (total_len - nperseg) // hop + 1

def standardize_samples(samples):
    if samples.ndim == 3 and samples.shape[1] != 2 and samples.shape[-1] == 2:
        samples = samples.transpose(0, 2, 1)
    return samples

def load_rml2016_dict(data_path, classes, min_snr, max_snr, seed, test_size):
    with open(data_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data = u.load()
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])
    rng = np.random.RandomState(seed)
    test_samples = []
    test_labels = []
    test_snrs = []
    for mod in mods:
        if mod not in classes:
            continue
        for snr in snrs:
            if (mod, snr) not in data:
                continue
            if snr < min_snr or snr > max_snr:
                continue
            s = data[(mod, snr)]
            if len(s) == 0:
                continue
            idx = rng.permutation(len(s))
            test_count = int(round(len(s) * test_size))
            if test_count < 1:
                test_count = 1
            test_idx = idx[:test_count]
            test_samples.append(s[test_idx])
            test_labels.append(np.full(test_count, classes.index(mod)))
            test_snrs.append(np.full(test_count, snr))
    if not test_samples:
        print("No samples found for the specified SNR range.")
        sys.exit(1)
    samples = np.vstack(test_samples).astype(np.float32)
    labels = np.concatenate(test_labels).astype(np.int64)
    snr = np.concatenate(test_snrs).astype(np.int64)
    samples = standardize_samples(samples)
    return samples, labels, snr

def get_npu_load():
    try:
        with open("/sys/kernel/debug/rknpu/load", "r", encoding="utf-8") as f:
            output = f.read().strip()
    except Exception:
        return None
    core_loads = re.findall(r"(?:Core|core)\s*\d+\s*:\s*(\d+)\s*%", output)
    if core_loads:
        return list(map(int, core_loads))
    single_load = re.search(r"NPU load:\s*(\d+)\s*%", output, re.IGNORECASE)
    if single_load:
        return [int(single_load.group(1))]
    percent_values = re.findall(r"(\d+)\s*%", output)
    if percent_values:
        return list(map(int, percent_values))
    return None

def get_process_rss_mb():
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def generate_simulated_data(sim_type, shape):
    if sim_type == "sine":
        length = shape[-1]
        t = np.linspace(0, 10 * np.pi, length)
        data = np.zeros(shape, dtype=np.float32)
        if len(shape) == 3:
            for i in range(shape[0]):
                data[i, 0, :] = np.cos(t)
                data[i, 1, :] = np.sin(t)
        elif len(shape) == 2:
            data[0, :] = np.cos(t)
            data[1, :] = np.sin(t)
        return data
    if sim_type == "constant":
        return np.ones(shape, dtype=np.float32)
    if sim_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    return np.random.randn(*shape).astype(np.float32)

def load_data(data_path, shape, sim_type, classes, min_snr, max_snr, seed, test_size):
    if data_path is None:
        return generate_simulated_data(sim_type, shape), None
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)
    if data_path.endswith(".pkl"):
        return load_rml2016_dict(data_path, classes, min_snr, max_snr, seed, test_size)
    if data_path.endswith(".h5"):
        import h5py
        f = h5py.File(data_path, "r")
        samples = f["samples"][:]
        if samples.shape[-1] == 2:
            samples = samples.transpose(0, 2, 1)
        labels = f["label"][:]
        f.close()
        return samples.astype(np.float32), labels, None
    print("Unsupported file format.")
    sys.exit(1)

def build_stft_samples(samples, stft_bins, stft_frames, nperseg, noverlap, nfft):
    try:
        from scipy.signal import stft
    except Exception:
        stft = None
    total = len(samples)
    if total == 0:
        return None
    length = samples.shape[-1]
    default_frames = infer_stft_frames(length, nperseg, noverlap)
    target_frames = stft_frames or default_frames
    if stft is None:
        return np.random.randn(total, 1, stft_bins, target_frames).astype(np.float32)
    stft_list = []
    for i in range(total):
        i_signal = samples[i, 0, :]
        _, _, zxx = stft(i_signal, 1.0, "blackman", nperseg, noverlap, nfft)
        spec = np.real(zxx)
        spec = spec[:stft_bins, :target_frames]
        if spec.shape[0] < stft_bins:
            pad_bins = stft_bins - spec.shape[0]
            spec = np.pad(spec, ((0, pad_bins), (0, 0)), mode="constant")
        if spec.shape[1] < target_frames:
            pad_frames = target_frames - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_frames)), mode="constant")
        if spec.shape[1] > target_frames:
            spec = spec[:, :target_frames]
        stft_list.append(np.expand_dims(spec.astype(np.float32), 0))
    return np.stack(stft_list, axis=0)

def show_outputs(output, top_k=5):
    output = output.flatten()
    probabilities = softmax(output)
    indices = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
    print("----- Top {} Predictions -----".format(top_k))
    for i in range(min(top_k, len(probabilities))):
        idx = indices[i]
        print(f"Class {idx}: {probabilities[idx]:.6f}")
    print("----------------------------")

def build_pytorch_model(args):
    try:
        import torch
        from IQFormer import IQFormer
    except Exception as e:
        return None, None, str(e)
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
            mlp_ratios=4,
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
        )
    model.eval()
    return model, torch, None

def compute_cpu_report(args, sample_length, batch_size, baseline_rss_mb=None):
    model, torch, err = build_pytorch_model(args)
    if model is None:
        return None, err, None
    start_rss = baseline_rss_mb if baseline_rss_mb is not None else get_process_rss_mb()
    if args.pt_model_path:
        state = torch.load(args.pt_model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and all(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model = model.to("cpu")
    x = torch.randn(batch_size, 2, sample_length, dtype=torch.float32)
    frames = args.stft_frames or infer_stft_frames(sample_length, args.stft_nperseg, args.stft_noverlap)
    batch_stft = torch.randn(batch_size, 1, args.stft_bins, frames, dtype=torch.float32)
    try:
        from model_report import build_model_report, build_model_report_for_dtype, format_bytes
    except Exception:
        return None, "model_report import failed", None
    report = build_model_report(model, x, batch_stft, torch.device("cpu"), args.aux_mode)
    report["model_size_fmt"] = format_bytes(report["model_size"])
    report["peak_activation_fmt"] = format_bytes(report["peak_activation"]) if report["peak_activation"] is not None else None
    report["peak_total_fmt"] = format_bytes(report["peak_total"]) if report["peak_total"] is not None else None
    end_rss = get_process_rss_mb()
    delta_mb = (end_rss - start_rss) if (end_rss is not None and start_rss is not None) else None
    return report, None, delta_mb


def compute_cpu_report_for_dtype(args, sample_length, batch_size, dtype):
    model, torch, err = build_pytorch_model(args)
    if model is None:
        return None, err
    model = model.to("cpu")
    x = torch.randn(batch_size, 2, sample_length, dtype=torch.float32)
    frames = args.stft_frames or infer_stft_frames(sample_length, args.stft_nperseg, args.stft_noverlap)
    batch_stft = torch.randn(batch_size, 1, args.stft_bins, frames, dtype=torch.float32)
    try:
        from model_report import build_model_report_for_dtype, format_bytes
    except Exception:
        return None, "model_report import failed"
    r = build_model_report_for_dtype(model, x, batch_stft, torch.device("cpu"), args.aux_mode, dtype)
    if not r.get("available", False):
        return None, f"cpu dtype {dtype} unavailable"
    r["model_size_fmt"] = format_bytes(r["model_size"])
    r["peak_activation_fmt"] = format_bytes(r["peak_activation"]) if r["peak_activation"] is not None else None
    r["peak_total_fmt"] = format_bytes(r["peak_total"]) if r["peak_total"] is not None else None
    return r, None


def map_cpu_dtype(model_name: str):
    name = model_name.lower()
    if "fp16" in name:
        return "fp16"
    if "int8" in name:
        return "int8"
    return "fp32"

def run_npu_inference(rknn_lite, samples, stft_samples, labels, batch_size, warmup, iters, core_mask, baseline_rss_mb=None):
    total = len(samples)
    if total == 0:
        return None
    rknn_lite.init_runtime(core_mask=core_mask)
    if warmup > 0:
        warm_batch = samples[:batch_size]
        warm_stft = stft_samples[:batch_size] if stft_samples is not None else None
        inputs = [warm_batch, warm_stft] if warm_stft is not None else [warm_batch]
        rknn_lite.inference(inputs=inputs)
    npu_load_samples = []
    correct = 0
    peak_rss = baseline_rss_mb if baseline_rss_mb is not None else get_process_rss_mb()
    start = time.perf_counter()
    for i in range(0, total, batch_size):
        batch = samples[i:i + batch_size]
        batch_stft = stft_samples[i:i + batch_size] if stft_samples is not None else None
        inputs = [batch, batch_stft] if batch_stft is not None else [batch]
        outputs = rknn_lite.inference(inputs=inputs)
        logits = outputs[0]
        preds = np.argmax(logits, axis=1)
        if labels is not None:
            batch_labels = labels[i:i + batch_size]
            correct += int(np.sum(preds == batch_labels))
        load = get_npu_load()
        if load is not None:
            npu_load_samples.append(load)
        rss_now = get_process_rss_mb()
        if rss_now is not None:
            if peak_rss is None or rss_now > peak_rss:
                peak_rss = rss_now
    end = time.perf_counter()
    elapsed = end - start
    batch_count = int(np.ceil(total / batch_size))
    avg_batch_ms = (elapsed / batch_count) * 1000
    avg_sample_ms = (elapsed / total) * 1000
    throughput = total / elapsed if elapsed > 0 else None
    acc = (correct / total) if labels is not None else None
    delta_mb = None
    if peak_rss is not None and baseline_rss_mb is not None:
        delta_mb = peak_rss - baseline_rss_mb
    return {
        "avg_batch_ms": avg_batch_ms,
        "avg_sample_ms": avg_sample_ms,
        "throughput": throughput,
        "accuracy": acc,
        "npu_load_samples": npu_load_samples,
        "peak_rss_mb": peak_rss,
        "delta_mb": delta_mb,
    }

def summarize_npu_load(load_samples):
    if not load_samples:
        return None
    arr = np.array(load_samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "mean": arr.mean(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
    }

def resolve_model_paths(args):
    if args.models:
        return [p.strip() for p in args.models.split(",") if p.strip()]
    default_names = ["IQFormer_fp32.rknn", "IQFormer_fp16.rknn", "IQFormer_int8.rknn"]
    candidates = []
    for name in default_names:
        path = os.path.join(args.models_dir, name)
        if os.path.isfile(path):
            candidates.append(path)
    if candidates:
        return candidates
    return [args.model]

def format_list_value(values):
    if values is None:
        return ""
    return "|".join([f"{v:.4f}" for v in values])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RKNN Lite2 Inference Script for IQFormer")
    parser.add_argument("--model", type=str, default="/home/orangepi/IQFormerLite/rknn_iqformer/weights/IQFormer/IQFormer_fp32.rknn")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default="/home/orangepi/IQFormerLite/rknn_iqformer/weights/IQFormer")
    parser.add_argument("--output_csv", type=str, default="/home/orangepi/IQFormerLite/rknn_iqformer/benchmark.csv")
    parser.add_argument("--data", type=str, default="/home/orangepi/datasets/RML2016.10a_dict.pkl")
    parser.add_argument("--simulate", type=str, default="random", choices=["random", "sine", "constant", "zeros"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--min_snr", type=int, default=-20)
    parser.add_argument("--max_snr", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--aux_mode", type=str, default="stft", choices=["stft"])
    parser.add_argument("--stft_bins", type=int, default=32)
    parser.add_argument("--stft_frames", type=int, default=None)
    parser.add_argument("--stft_nperseg", type=int, default=31)
    parser.add_argument("--stft_noverlap", type=int, default=30)
    parser.add_argument("--stft_nfft", type=int, default=128)
    parser.add_argument("--database_choose", type=str, default="2016.10a")
    parser.add_argument("--pt_model_path", type=str, default="/home/orangepi/IQFormerLite/rknn_iqformer/weights/IQFormer/weight.pt")
    parser.add_argument("--report_batch", type=int, default=1)
    default_core_mask = RKNNLite.NPU_CORE_0 if RKNNLite else 0
    parser.add_argument("--core_mask", type=int, default=default_core_mask)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--rknn_verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.database_choose.endswith("a"):
        classes = ["8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "AM-DSB", "AM-SSB", "WBFM"]
    else:
        classes = ["8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "AM-DSB", "WBFM"]

    samples, labels, snr = load_data(
        args.data,
        shape=(1, 2, 128),
        sim_type=args.simulate,
        classes=classes,
        min_snr=args.min_snr,
        max_snr=args.max_snr,
        seed=args.seed,
        test_size=args.test_size,
    )

    if samples.ndim == 2:
        samples = np.expand_dims(samples, 0)
    if samples.shape[1] != 2 and samples.shape[-1] == 2:
        samples = samples.transpose(0, 2, 1)
    samples = samples.astype(np.float32)

    print(f"Test samples: {len(samples)}")
    print(f"Input shape: {samples[0].shape}")

    stft_samples = build_stft_samples(
        samples,
        args.stft_bins,
        args.stft_frames,
        args.stft_nperseg,
        args.stft_noverlap,
        args.stft_nfft,
    )

    model_paths = resolve_model_paths(args)
    rows = []

    for model_path in model_paths:
        per_model_baseline_mb = get_process_rss_mb()
        npu_result = None
        if RKNNLite is not None and os.path.isfile(model_path):
            rknn_lite = RKNNLite(verbose=args.rknn_verbose)
            ret = rknn_lite.load_rknn(model_path)
            if ret == 0:
                npu_result = run_npu_inference(
                    rknn_lite,
                    samples,
                    stft_samples,
                    labels,
                    args.batch_size,
                    args.warmup,
                    args.iters,
                    args.core_mask,
                    per_model_baseline_mb,
                )
                rknn_lite.release()
            else:
                print(f"Load RKNN model failed: {model_path}")
        else:
            if RKNNLite is None:
                print("rknn_toolkit_lite2 not installed, will only report CPU metrics.")
            else:
                print(f"Model file not found: {model_path}")

        npu_load_summary = summarize_npu_load(npu_result["npu_load_samples"]) if npu_result is not None else None
        rss_mb = get_process_rss_mb()
        peak_mb = npu_result["peak_rss_mb"] if npu_result is not None else rss_mb
        cpu_dtype = map_cpu_dtype(os.path.basename(model_path))
        cpu_report, cpu_err = compute_cpu_report_for_dtype(args, samples.shape[-1], args.report_batch, cpu_dtype)
        rknn_size_kb = os.path.getsize(model_path) / 1024 if os.path.isfile(model_path) else None
        delta_mb = None
        if npu_result is not None and npu_result["delta_mb"] is not None:
            delta_mb = npu_result["delta_mb"]

        print(f"===== Report: {os.path.basename(model_path)} =====")
        print("===== Static Complexity =====")
        if cpu_report is None:
            print("Parameters: N/A")
            print("FLOPs: N/A")
            if cpu_err:
                print(f"CPU Report Error: {cpu_err}")
        else:
            params_m = cpu_report["params"] / 1e6
            flops_g = cpu_report["flops"] / 1e9 if cpu_report["flops"] is not None else None
            print(f"Parameters (M): {params_m:.3f}")
            print("FLOPs (G): {}".format(f"{flops_g:.3f}" if flops_g is not None else "N/A"))
            print(f"CPU Model Size: {cpu_report['model_size_fmt']}")
            if rknn_size_kb is not None:
                print(f"RKNN Model Size: {rknn_size_kb:.2f} KB")

        print("===== Dynamic Inference Performance =====")
        if npu_result is not None:
            print(f"NPU Latency per batch (ms): {npu_result['avg_batch_ms']:.3f}")
            print(f"NPU Latency per sample (ms): {npu_result['avg_sample_ms']:.3f}")
            if npu_result["throughput"] is not None:
                print(f"NPU Throughput (samples/s): {npu_result['throughput']:.2f}")
            if npu_result["accuracy"] is not None:
                print(f"NPU Top1 Accuracy: {npu_result['accuracy']:.4f}")
        else:
            print("NPU metrics: N/A (RKNN Lite2 not available or model load failed)")

        speedup = None
        if cpu_report is not None and npu_result is not None and cpu_report["latency_ms"] and npu_result["avg_sample_ms"]:
            speedup = cpu_report["latency_ms"] / npu_result["avg_sample_ms"]

        if cpu_report is not None:
            print(f"CPU Latency (ms): {cpu_report['latency_ms']:.3f}")
            if cpu_report["throughput"] is not None:
                print(f"CPU Throughput (samples/s): {cpu_report['throughput']:.2f}")
            if speedup is not None:
                print(f"Speedup (CPU/NPU): {speedup:.2f}x")
            else:
                print("Speedup (CPU/NPU): N/A")

        print("===== System Overhead =====")
        if npu_load_summary is None:
            print("NPU Load per Core: N/A")
        else:
            print(f"NPU Load Mean: {npu_load_summary['mean']}")
            print(f"NPU Load Max: {npu_load_summary['max']}")
        if per_model_baseline_mb is None:
            print("Memory Baseline (MB): N/A")
        else:
            print(f"Memory Baseline (MB): {per_model_baseline_mb:.2f}")
        if peak_mb is None:
            print("Memory Peak (MB): N/A")
        else:
            print(f"Memory Peak (MB): {peak_mb:.2f}")
        if delta_mb is None:
            print("Memory Delta (MB): N/A")
        else:
            print(f"Memory Delta (MB): {delta_mb:.2f}")

        row = {
            "model": os.path.basename(model_path),
            "params_m": round(cpu_report["params"] / 1e6, 6) if cpu_report else None,
            "flops_g": round(cpu_report["flops"] / 1e9, 6) if cpu_report and cpu_report["flops"] is not None else None,
            "cpu_model_size_kb": round(cpu_report["model_size"] / 1024, 6) if cpu_report else None,
            "rknn_model_size_kb": round(rknn_size_kb, 6) if rknn_size_kb is not None else None,
            "cpu_latency_ms": round(cpu_report["latency_ms"], 6) if cpu_report else None,
            "cpu_throughput": round(cpu_report["throughput"], 6) if cpu_report and cpu_report["throughput"] is not None else None,
            "npu_latency_batch_ms": round(npu_result["avg_batch_ms"], 6) if npu_result else None,
            "npu_latency_sample_ms": round(npu_result["avg_sample_ms"], 6) if npu_result else None,
            "npu_throughput": round(npu_result["throughput"], 6) if npu_result and npu_result["throughput"] is not None else None,
            "npu_accuracy": round(npu_result["accuracy"], 6) if npu_result and npu_result["accuracy"] is not None else None,
            "speedup": round(speedup, 6) if speedup is not None else None,
            "npu_load_mean": format_list_value(npu_load_summary["mean"]) if npu_load_summary else "",
            "npu_load_max": format_list_value(npu_load_summary["max"]) if npu_load_summary else "",
            "memory_baseline_mb": round(per_model_baseline_mb, 6) if per_model_baseline_mb is not None else None,
            "memory_peak_mb": round(peak_mb, 6) if peak_mb is not None else None,
            "memory_delta_mb": round(delta_mb, 6) if delta_mb is not None else None,
        }
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to: {args.output_csv}")
