import argparse
import os
import pickle
import numpy as np


def load_groups(data_path, length):
    with open(data_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data = u.load()
    snrs = sorted(list(set(map(lambda x: x[1], data.keys()))))
    mods = sorted(list(set(map(lambda x: x[0], data.keys()))))
    groups = {}
    for mod in mods:
        for snr in snrs:
            if (mod, snr) in data:
                s = data[(mod, snr)].astype(np.float32)
                if s.ndim == 3 and s.shape[1] != 2 and s.shape[2] == 2:
                    s = s.transpose(0, 2, 1)
                if s.shape[-1] != length:
                    continue
                groups[(mod, snr)] = s
    return mods, snrs, groups


def build_stft(sample, stft_bins, stft_frames, nperseg, noverlap, nfft):
    try:
        from scipy.signal import stft
    except Exception:
        stft = None
    if stft is None:
        frames = stft_frames or sample.shape[-1]
        return np.random.randn(1, 1, stft_bins, frames).astype(np.float32)
    if sample.ndim == 3:
        i_signal = sample[0, 0, :]
    else:
        i_signal = sample[0, :]
    _, _, zxx = stft(i_signal, 200000, "blackman", nperseg, noverlap, nfft)
    frames = stft_frames or zxx.shape[1]
    spec = np.abs(zxx)
    spec = spec[:stft_bins, :frames]
    if spec.shape[0] < stft_bins:
        pad_bins = stft_bins - spec.shape[0]
        spec = np.pad(spec, ((0, pad_bins), (0, 0)), mode="constant")
    if spec.shape[1] < frames:
        pad_frames = frames - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_frames)), mode="constant")
    if spec.shape[1] > frames:
        spec = spec[:, :frames]
    return np.expand_dims(np.expand_dims(spec.astype(np.float32), 0), 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--iq_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_num", type=int, default=300)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--per_group", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stft_bins", type=int, default=32)
    parser.add_argument("--stft_frames", type=int, default=None)
    parser.add_argument("--stft_nperseg", type=int, default=31)
    parser.add_argument("--stft_noverlap", type=int, default=30)
    parser.add_argument("--stft_nfft", type=int, default=128)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dataset_txt = os.path.join(args.out_dir, "dataset.txt")
    selected = []

    if args.iq_dir:
        candidates = sorted(
            [
                os.path.join(args.iq_dir, f)
                for f in os.listdir(args.iq_dir)
                if f.startswith("iq_") and f.endswith(".npy")
            ]
        )
        for p in candidates[: args.max_num]:
            sample = np.load(p).astype(np.float32)
            if sample.ndim == 2:
                sample = np.expand_dims(sample, 0)
            if sample.shape[-1] != args.length:
                continue
            selected.append(sample)
    else:
        if args.data_path is None:
            raise ValueError("data_path is required when iq_dir is not set")
        mods, snrs, groups = load_groups(args.data_path, args.length)
        total_groups = len(mods) * len(snrs)
        if total_groups == 0:
            raise ValueError("No valid (mod, SNR) groups found")
        quota = args.per_group if args.per_group is not None else max(1, args.max_num // total_groups)
        for mod in mods:
            for snr in snrs:
                key = (mod, snr)
                if key not in groups:
                    continue
                s = groups[key]
                n = len(s)
                k = min(quota, n)
                idx = rng.choice(n, size=k, replace=False)
                for i in idx:
                    selected.append(s[i])
        if len(selected) > args.max_num:
            idx = rng.choice(len(selected), size=args.max_num, replace=False)
            selected = [selected[i] for i in idx]
        else:
            selected = selected[: args.max_num]

    num = len(selected)
    with open(dataset_txt, "w") as f:
        for i in range(num):
            p = os.path.join(args.out_dir, f"iq_{i:04d}.npy")
            stft_p = os.path.join(args.out_dir, f"stft_{i:04d}.npy")
            sample = selected[i].astype(np.float32)
            if sample.ndim == 2:
                sample = np.expand_dims(sample, 0)
            np.save(p, sample)
            stft_sample = build_stft(
                sample,
                args.stft_bins,
                args.stft_frames,
                args.stft_nperseg,
                args.stft_noverlap,
                args.stft_nfft,
            )
            np.save(stft_p, stft_sample)
            f.write(p + " " + stft_p + "\n")

    print(dataset_txt)


if __name__ == "__main__":
    main()
