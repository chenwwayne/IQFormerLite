import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_rml201610a(path):
    if not os.path.exists(path):
        print(f"数据文件不存在: {path}")
        sys.exit(1)
    return pd.read_pickle(path)


def pick_random_sample(data):
    keys = list(data.keys())
    if not keys:
        print("数据为空，无法抽样")
        sys.exit(1)
    label, snr = random.choice(keys)
    samples = data[(label, snr)]
    if len(samples) == 0:
        print("该类别与SNR无样本")
        sys.exit(1)
    idx = random.randrange(len(samples))
    sample = np.array(samples[idx])
    if sample.ndim != 2:
        print(f"样本维度不符合预期: {sample.shape}")
        sys.exit(1)
    if sample.shape[0] == 2:
        iq = sample
    elif sample.shape[1] == 2:
        iq = sample.T
    else:
        print(f"样本通道维不为2: {sample.shape}")
        sys.exit(1)
    return label, snr, idx, iq


def smooth_signal(signal, window):
    if window <= 1:
        return signal
    window = int(window)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal, kernel, mode="same")


def save_iq_plot(iq, label, snr, idx, out_dir, smooth_window):
    os.makedirs(out_dir, exist_ok=True)
    length = iq.shape[1]
    t = np.arange(length)
    iq_smooth = np.vstack([
        smooth_signal(iq[0], smooth_window),
        smooth_signal(iq[1], smooth_window),
    ])
    plt.figure(figsize=(10, 4))
    plt.plot(t, iq_smooth[0], label="I")
    plt.plot(t, iq_smooth[1], label="Q")
    plt.axis("off")
    plt.legend(loc="upper center", ncol=2, frameon=False)
    plt.tight_layout()
    safe_label = str(label).replace("/", "_").replace(" ", "_")
    filename = f"{safe_label}_snr{snr}_idx{idx}.png"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=os.path.join(os.path.dirname(__file__), "RML2016.10a.pkl"))
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "vis_output"))
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--smooth_window", type=int, default=9)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_rml201610a(args.data_path)
    label, snr, idx, iq = pick_random_sample(data)
    save_iq_plot(iq, label, snr, idx, args.out_dir, args.smooth_window)


if __name__ == "__main__":
    main()
