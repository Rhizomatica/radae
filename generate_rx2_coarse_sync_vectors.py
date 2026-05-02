"""
Generate streaming vectors for the C rx2_coarse_sync port.

Binary output layout:
    int32      M
    int32      Ncp
    int32      sym_len
    int32      ncases
    int32      fix_delta_hat      # 0 = argmax (default); else pinned index
    float32    Fs
    float32    B_bpf
    repeated ncases times:
        complex64[]  3*sym_len rx_buf snapshot
        complex64[]  sym_len expected Ry_smooth
        int32        expected delta_hat_g
        float32      expected Ry_max
        float32      expected Ry_min
        int32        expected sig_det
        int32        expected sine_det
        float32      expected snr_est_dB

Usage:
    python3 generate_rx2_coarse_sync_vectors.py out.bin

Copyright (c) 2026 Rhizomatica, BSD-3-Clause
"""

import argparse
import sys

import numpy as np


ALPHA = 0.95
TSIG = 0.38
TSIN = 4.0
SNR_CORR_A = 1.24392558
SNR_CORR_B = 3.33253932


def compute_autocorr(rx_buf, M, Ncp, sym_len, Ry_smooth):
    Ry_norm = np.zeros(sym_len, dtype=np.complex64)
    for gamma in range(sym_len):
        idx = sym_len + gamma
        y_cp = rx_buf[idx - Ncp : idx]
        y_m = rx_buf[idx - Ncp + M : idx + M]
        Ry = np.dot(y_cp, np.conj(y_m))
        D = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m)) + 1e-12
        Ry_norm[gamma] = 2.0 * Ry / np.abs(D)
    Ry_smooth = ALPHA * Ry_smooth + (1.0 - ALPHA) * Ry_norm
    return Ry_smooth.astype(np.complex64)


def detect_signal(Ry_smooth, snr_offset_dB, fix_delta_hat=0):
    abs_Ry = np.abs(Ry_smooth)
    if fix_delta_hat:
        delta_hat_g = int(fix_delta_hat)
    else:
        delta_hat_g = int(np.argmax(abs_Ry))
    Ry_max = float(abs_Ry[delta_hat_g])
    Ry_min = float(abs_Ry[int(np.argmin(abs_Ry))])
    sig_det = int(Ry_max > TSIG)
    sine_det = int(Ry_max / (Ry_min + 1e-12) < TSIN)
    rho = float(np.clip(np.max(abs_Ry), 0.0, 1.0 - 1e-6))
    snr_raw = 10.0 * np.log10(rho / (1.0 - rho) + 1e-12) - snr_offset_dB
    snr_est_dB = float(SNR_CORR_A * snr_raw + SNR_CORR_B)
    return delta_hat_g, Ry_max, Ry_min, sig_det, sine_det, snr_est_dB


def build_stream(args, rng):
    body = (rng.standard_normal(args.M) + 1j * rng.standard_normal(args.M)).astype(np.complex64) / np.sqrt(2.0)
    body *= np.float32(0.8)
    sym = np.concatenate([body[-args.Ncp :], body]).astype(np.complex64)

    n_good = args.sym_len * 40
    n_tone = args.sym_len * 20
    n_noise = args.sym_len * 20

    good = np.tile(sym, n_good // args.sym_len + 1)[:n_good]
    tone_n = np.arange(n_tone, dtype=np.float32)
    tone = (0.9 * np.exp(1j * 2.0 * np.pi * 420.0 * tone_n / args.Fs)).astype(np.complex64)
    noise = (0.08 * (rng.standard_normal(n_noise) + 1j * rng.standard_normal(n_noise)) / np.sqrt(2.0)).astype(np.complex64)

    stream = np.concatenate(
        [
            np.zeros(args.sym_len * 6, dtype=np.complex64),
            good,
            0.03 * noise[: args.sym_len * 12],
            tone,
            noise,
            good[: args.sym_len * 12],
        ]
    ).astype(np.complex64)
    return stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--Ncp", type=int, default=32)
    parser.add_argument("--Fs", type=float, default=8000.0)
    parser.add_argument("--B-bpf", dest="B_bpf", type=float, default=2600.0)
    parser.add_argument("--ncases", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fix-delta-hat", dest="fix_delta_hat", type=int, default=0,
                        help="pin delta_hat_g to this index (0 = argmax, matching rx2.py)")
    args = parser.parse_args()

    args.sym_len = args.M + args.Ncp
    rng = np.random.default_rng(args.seed)
    stream = build_stream(args, rng)
    snr_offset_dB = 10.0 * np.log10(3000.0 / args.B_bpf)

    lengths = rng.integers(1, args.sym_len + args.sym_len // 2 + 1, size=args.ncases, dtype=np.int32)
    if args.ncases >= 8:
        lengths[:8] = np.array(
            [args.sym_len, args.sym_len, args.sym_len // 2, args.sym_len + args.sym_len // 4, 1, args.sym_len, 17, args.sym_len],
            dtype=np.int32,
        )

    rx_buf = np.zeros(3 * args.sym_len, dtype=np.complex64)
    Ry_smooth = np.zeros(args.sym_len, dtype=np.complex64)
    pos = 0

    if args.fix_delta_hat and not (0 <= args.fix_delta_hat < args.sym_len):
        print(f"fix_delta_hat must be in [0, {args.sym_len - 1}]", file=sys.stderr)
        return 1

    with open(args.output, "wb") as f:
        f.write(np.int32(args.M).tobytes())
        f.write(np.int32(args.Ncp).tobytes())
        f.write(np.int32(args.sym_len).tobytes())
        f.write(np.int32(args.ncases).tobytes())
        f.write(np.int32(args.fix_delta_hat).tobytes())
        f.write(np.float32(args.Fs).tobytes())
        f.write(np.float32(args.B_bpf).tobytes())

        for nin in lengths:
            if pos + nin > len(stream):
                pos = 0
            chunk = stream[pos : pos + nin]
            pos += nin

            rx_buf[: 3 * args.sym_len - nin] = rx_buf[nin:]
            rx_buf[3 * args.sym_len - nin :] = chunk

            Ry_smooth = compute_autocorr(rx_buf, args.M, args.Ncp, args.sym_len, Ry_smooth)
            delta_hat_g, Ry_max, Ry_min, sig_det, sine_det, snr_est_dB = detect_signal(
                Ry_smooth, snr_offset_dB, args.fix_delta_hat
            )

            f.write(rx_buf.tobytes())
            f.write(Ry_smooth.tobytes())
            f.write(np.int32(delta_hat_g).tobytes())
            f.write(np.float32(Ry_max).tobytes())
            f.write(np.float32(Ry_min).tobytes())
            f.write(np.int32(sig_det).tobytes())
            f.write(np.int32(sine_det).tobytes())
            f.write(np.float32(snr_est_dB).tobytes())

    print(f"wrote {args.ncases} cases to {args.output}")
    print(f"  M={args.M} Ncp={args.Ncp} sym_len={args.sym_len}")


if __name__ == "__main__":
    sys.exit(main())
