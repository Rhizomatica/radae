"""
Generate streaming vectors for the C rx2_frontend port.

Binary output layout:
    int32    sym_len
    int32    max_nin
    int32    ncases
    repeated ncases times:
        int32        agc_en
        int32        nin
        complex64[]  nin input samples
        float32      expected gain
        complex64[]  3*sym_len expected rx_buf state

Usage:
    python3 generate_rx2_frontend_vectors.py out.bin

Copyright (c) 2026 Rhizomatica, BSD-3-Clause
"""

import argparse
import sys

import numpy as np


def compute_gain(rx_in, agc_target, agc_en):
    if not agc_en:
        return np.float32(1.0)
    gain = agc_target / (np.sqrt(np.mean(np.abs(rx_in) ** 2)) + 1e-6)
    return np.float32(np.clip(gain, 0.1, 10.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--sym-len", type=int, default=160)
    parser.add_argument("--max-nin", type=int, default=240)
    parser.add_argument("--ncases", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    agc_target = np.float32(10 ** (-3 / 20))
    rx_buf = np.zeros(3 * args.sym_len, dtype=np.complex64)

    lengths = rng.integers(1, args.max_nin + 1, size=args.ncases, dtype=np.int32)
    if args.ncases >= 6:
        lengths[:6] = np.array(
            [args.max_nin, args.max_nin - 1, args.sym_len, args.sym_len // 2, 1, args.sym_len + args.sym_len // 4],
            dtype=np.int32,
        )

    agc_flags = rng.integers(0, 2, size=args.ncases, dtype=np.int32)
    if args.ncases >= 8:
        agc_flags[:8] = np.array([1, 1, 0, 1, 0, 1, 1, 0], dtype=np.int32)

    with open(args.output, "wb") as f:
        f.write(np.int32(args.sym_len).tobytes())
        f.write(np.int32(args.max_nin).tobytes())
        f.write(np.int32(args.ncases).tobytes())

        for c, (nin, agc_en) in enumerate(zip(lengths, agc_flags)):
            scale = rng.uniform(0.01, 2.5)
            if c % 17 == 0:
                scale = 1e-4
            elif c % 23 == 0:
                scale = 25.0
            x = scale * (
                rng.standard_normal(int(nin)) + 1j * rng.standard_normal(int(nin))
            ) / np.sqrt(2.0)
            x = x.astype(np.complex64)

            gain = compute_gain(x, agc_target, bool(agc_en))
            rx_buf[: 3 * args.sym_len - nin] = rx_buf[nin:]
            rx_buf[3 * args.sym_len - nin :] = x * gain

            f.write(np.int32(agc_en).tobytes())
            f.write(np.int32(nin).tobytes())
            f.write(x.tobytes())
            f.write(np.float32(gain).tobytes())
            f.write(rx_buf.astype(np.complex64).tobytes())

    print(f"wrote {args.ncases} cases to {args.output}")
    print(f"  sym_len={args.sym_len} max_nin={args.max_nin}")


if __name__ == "__main__":
    sys.exit(main())
