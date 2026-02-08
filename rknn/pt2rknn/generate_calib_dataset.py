import argparse
import os
import pickle
import numpy as np


def load_rml2016_pkl(data_path):
    with open(data_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data = u.load()
    snrs = sorted(list(set(map(lambda x: x[1], data.keys()))))
    mods = sorted(list(set(map(lambda x: x[0], data.keys()))))
    samples = []
    for mod in mods:
        for snr in snrs:
            if (mod, snr) in data:
                s = data[(mod, snr)]
                samples.append(s)
    samples = np.vstack(samples)
    return samples.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_num", type=int, default=300)
    parser.add_argument("--length", type=int, default=128)
    args = parser.parse_args()

    samples = load_rml2016_pkl(args.data_path)
    if samples.ndim == 3 and samples.shape[1] != 2 and samples.shape[2] == 2:
        samples = samples.transpose(0, 2, 1)

    if samples.shape[-1] != args.length:
        raise ValueError(f"Sample length mismatch: got {samples.shape[-1]}, expected {args.length}")

    os.makedirs(args.out_dir, exist_ok=True)
    dataset_txt = os.path.join(args.out_dir, "dataset.txt")

    num = min(args.max_num, len(samples))
    with open(dataset_txt, "w") as f:
        for i in range(num):
            p = os.path.join(args.out_dir, f"iq_{i:04d}.npy")
            sample = samples[i].astype(np.float32)
            if sample.ndim == 2:
                sample = np.expand_dims(sample, 0)
            np.save(p, sample)
            f.write(p + "\n")

    print(dataset_txt)


if __name__ == "__main__":
    main()
