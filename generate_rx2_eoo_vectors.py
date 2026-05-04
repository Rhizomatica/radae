"""
Generate streaming vectors for the C rx2_eoo port.

Binary output layout:
    int32      M
    int32      Ncp
    int32      ncases
    complex64[] M pend time-domain symbol
    repeated ncases times:
        complex64[] M rx_sym_td
        float32     expected_eoo_corr
        float32     expected_eoo_smooth
        int32       expected_detect

Usage:
    python3 generate_rx2_eoo_vectors.py out.bin
"""

import argparse
import sys

import numpy as np
import torch

from radae import RADAE


def make_eoo_like(rng, pend_fd, M, Ncp):
    h = np.zeros(M, dtype=np.complex64)
    taps = rng.integers(1, min(5, Ncp) + 1)
    vals = (
        rng.standard_normal(taps) + 1j * rng.standard_normal(taps)
    ).astype(np.complex64) / np.sqrt(2.0 * taps)
    h[:taps] = vals
    if taps > 1:
        h[-(taps - 1) :] = 0.25 * np.conj(vals[1:][::-1])
    H_fd = np.fft.fft(h)
    rx_fd = pend_fd * H_fd
    return np.fft.ifft(rx_fd).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--num-features", type=int, default=21)
    parser.add_argument("--cp", type=float, default=0.004)
    parser.add_argument("--ncases", type=int, default=160)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = RADAE(
        args.num_features,
        args.latent_dim,
        EbNodB=100,
        Nzmf=1,
        rate_Fs=True,
        bottleneck=3,
        cyclic_prefix=args.cp,
        time_offset=-16,
        correct_time_offset=-8,
        stateful_decoder=False,
        w1_dec=128,
        w1_dec_stateful=128,
    )
    model.eval()

    pend = model.pend.detach().cpu().numpy().astype(np.complex64)
    pend_fd = np.fft.fft(pend)
    M = model.M
    Ncp = model.Ncp
    eoo_smooth = 0.0

    with open(args.output, "wb") as f:
        f.write(np.int32(M).tobytes())
        f.write(np.int32(Ncp).tobytes())
        f.write(np.int32(args.ncases).tobytes())
        f.write(pend.tobytes())

        for c in range(args.ncases):
            if c % 11 in (7, 8, 9):
                rx_sym_td = make_eoo_like(rng, pend_fd, M, Ncp)
                rx_sym_td += (
                    0.01
                    * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
                    / np.sqrt(2.0)
                ).astype(np.complex64)
            else:
                rx_sym_td = (
                    0.35
                    * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
                    / np.sqrt(2.0)
                ).astype(np.complex64)

            active = np.abs(pend_fd) > np.max(np.abs(pend_fd)) * 1e-3
            rx_fd = np.fft.fft(rx_sym_td)
            H_est = np.zeros(M, dtype=np.complex64)
            H_est[active] = rx_fd[active] / pend_fd[active]
            h_est = np.fft.ifft(H_est)
            e_total = np.sum(np.abs(h_est) ** 2) + 1e-12
            e_cp = np.sum(np.abs(h_est[:Ncp]) ** 2) + np.sum(np.abs(h_est[-Ncp:]) ** 2)
            eoo_corr = float(e_cp / e_total)
            eoo_smooth = 0.70 * eoo_smooth + 0.30 * eoo_corr
            detect = int(eoo_smooth > 0.75)

            f.write(rx_sym_td.astype(np.complex64).tobytes())
            f.write(np.float32(eoo_corr).tobytes())
            f.write(np.float32(eoo_smooth).tobytes())
            f.write(np.int32(detect).tobytes())

    print(f"wrote {args.ncases} cases to {args.output}")
    print(f"  M={M} Ncp={Ncp}")


if __name__ == "__main__":
    sys.exit(main())
