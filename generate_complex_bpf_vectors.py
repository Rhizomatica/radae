"""
Generate streaming test vectors for the pure-C complex_bpf port.

Binary output layout:
    int32    ntap
    float32  Fs_Hz
    float32  bandwidth_Hz
    float32  centre_freq_Hz
    int32    max_len
    int32    ncases
    repeated ncases times:
        int32        nin
        complex64[]  nin input samples
        complex64[]  nin expected output samples

Usage:
    python3 generate_complex_bpf_vectors.py out.bin

Copyright (c) 2026 Rhizomatica, BSD-3-Clause
"""

import argparse
import sys

import numpy as np

from radae import complex_bpf


def _build_signal(total_len, fs_hz, rng):
    t = np.arange(total_len, dtype=np.float32) / np.float32(fs_hz)
    env = (0.65
           + 0.25 * np.sin(2 * np.pi * 0.37 * t)
           + 0.10 * np.sin(2 * np.pi * 0.11 * t + 0.7))
    sig = (
        0.70 * np.exp(1j * 2 * np.pi * 300.0 * t) +
        0.35 * np.exp(1j * 2 * np.pi * 1180.0 * t) +
        0.20 * np.exp(-1j * 2 * np.pi * 640.0 * t)
    ).astype(np.complex64)
    noise = (0.03 / np.sqrt(2.0)) * (
        rng.standard_normal(total_len) + 1j * rng.standard_normal(total_len)
    )
    return (env.astype(np.float32) * sig + noise.astype(np.complex64)).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--ntap", type=int, default=101)
    parser.add_argument("--Fs_Hz", type=np.float32, default=8000.0)
    parser.add_argument("--bandwidth_Hz", type=np.float32, default=975.0)
    parser.add_argument("--centre_freq_Hz", type=np.float32, default=1468.75)
    parser.add_argument("--max_len", type=int, default=240)
    parser.add_argument("--ncases", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    lengths = rng.integers(1, args.max_len + 1, size=args.ncases, dtype=np.int32)
    if args.ncases >= 4:
        lengths[0] = args.max_len
        lengths[1] = args.max_len - 1
        lengths[2] = args.max_len // 2
        lengths[3] = 1

    total_len = int(np.sum(lengths, dtype=np.int64))
    x = _build_signal(total_len, float(args.Fs_Hz), rng)
    bpf = complex_bpf(args.ntap, float(args.Fs_Hz), float(args.bandwidth_Hz),
                      float(args.centre_freq_Hz), args.max_len)

    with open(args.output, "wb") as f:
        f.write(np.int32(args.ntap).tobytes())
        f.write(np.float32(args.Fs_Hz).tobytes())
        f.write(np.float32(args.bandwidth_Hz).tobytes())
        f.write(np.float32(args.centre_freq_Hz).tobytes())
        f.write(np.int32(args.max_len).tobytes())
        f.write(np.int32(args.ncases).tobytes())

        off = 0
        for nin in lengths:
            xin = x[off:off + nin]
            y = bpf.bpf(xin).astype(np.complex64)
            f.write(np.int32(nin).tobytes())
            f.write(xin.astype(np.complex64).tobytes())
            f.write(y.tobytes())
            off += int(nin)

    print(f"wrote {args.ncases} streaming cases to {args.output}")
    print(f"  ntap={args.ntap} max_len={args.max_len} total_len={total_len}")
    print(f"  bandwidth={float(args.bandwidth_Hz):.2f} centre={float(args.centre_freq_Hz):.2f}")


if __name__ == "__main__":
    sys.exit(main())
