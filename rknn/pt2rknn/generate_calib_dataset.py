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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_num", type=int, default=300)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--per_group", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    mods, snrs, groups = load_groups(args.data_path, args.length)
    rng = np.random.default_rng(args.seed)
    total_groups = len(mods) * len(snrs)
    if total_groups == 0:
        raise ValueError("No valid (mod, SNR) groups found")
    quota = args.per_group if args.per_group is not None else max(1, args.max_num // total_groups)

    os.makedirs(args.out_dir, exist_ok=True)
    dataset_txt = os.path.join(args.out_dir, "dataset.txt")

    selected = []
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
            sample = selected[i].astype(np.float32)
            if sample.ndim == 2:
                sample = np.expand_dims(sample, 0)
            np.save(p, sample)
            f.write(p + "\n")

    print(dataset_txt)


if __name__ == "__main__":
    main()
