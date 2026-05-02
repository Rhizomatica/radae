"""
Generate streaming vectors for the C rx2_demod port.

Binary output layout:
    int32      M
    int32      Ncp
    int32      Ns
    int32      Nc
    int32      sym_len
    int32      latent_dim
    int32      ncases
    float32    Fs
    int32      time_offset
    int32      correct_time_offset
    float32[]  Nc carrier frequencies w
    repeated ncases times:
        complex64[]  3*sym_len rx_buf snapshot
        float32      delta_hat
        float32      freq_offset
        complex64    expected rx_phase
        complex64[]  M expected rx_sym_td
        complex64[]  Ns*sym_len expected rx_i
        float32[]    latent_dim expected z_hat

Usage:
    python3 generate_rx2_demod_vectors.py out.bin
"""

import argparse
import sys

import numpy as np
import torch

from radae import RADAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--num-features", type=int, default=21)
    parser.add_argument("--cp", type=float, default=0.004)
    parser.add_argument("--time-offset", type=int, default=-16)
    parser.add_argument("--correct-time-offset", type=int, default=-16)
    parser.add_argument("--ncases", type=int, default=160)
    parser.add_argument("--seed", type=int, default=11)
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
        time_offset=args.time_offset,
        correct_time_offset=args.correct_time_offset,
        stateful_decoder=False,
        w1_dec=128,
        w1_dec_stateful=128,
    )
    model.eval()

    M = model.M
    Ncp = model.Ncp
    Ns = model.Ns
    Nc = model.Nc
    Fs = float(model.Fs)
    sym_len = M + Ncp
    latent_dim = model.latent_dim
    w = model.w.detach().cpu().numpy().astype(np.float32)

    rx_phase = np.complex64(1.0 + 0.0j)
    rx_i = np.zeros(Ns * sym_len, dtype=np.complex64)
    rx_sym_td = np.zeros(M, dtype=np.complex64)
    rx_buf = np.zeros(3 * sym_len, dtype=np.complex64)

    with open(args.output, "wb") as f:
        f.write(np.int32(M).tobytes())
        f.write(np.int32(Ncp).tobytes())
        f.write(np.int32(Ns).tobytes())
        f.write(np.int32(Nc).tobytes())
        f.write(np.int32(sym_len).tobytes())
        f.write(np.int32(latent_dim).tobytes())
        f.write(np.int32(args.ncases).tobytes())
        f.write(np.float32(Fs).tobytes())
        f.write(np.int32(args.time_offset).tobytes())
        f.write(np.int32(args.correct_time_offset).tobytes())
        f.write(w.tobytes())

        for c in range(args.ncases):
            nin = int(rng.integers(1, sym_len + sym_len // 4 + 1))
            chunk = (
                0.45
                * (rng.standard_normal(nin) + 1j * rng.standard_normal(nin))
                / np.sqrt(2.0)
            ).astype(np.complex64)
            if c % 13 == 0:
                tone_n = np.arange(nin, dtype=np.float32)
                chunk += (0.6 * np.exp(1j * 2.0 * np.pi * 700.0 * tone_n / Fs)).astype(np.complex64)
            rx_buf[: 3 * sym_len - nin] = rx_buf[nin:]
            rx_buf[3 * sym_len - nin :] = chunk

            delta_hat = np.float32(rng.uniform(0.0, sym_len - 1.0) + rng.uniform(-0.49, 0.49))
            freq_offset = np.float32(rng.uniform(-25.0, 25.0))
            delta_hat_rx = int(float(delta_hat) - Ncp)
            omega = 2.0 * np.pi * float(freq_offset) / Fs
            phase_step = np.exp(-1j * omega).astype(np.complex64)

            rx_phase_vec = np.zeros(sym_len, dtype=np.complex64)
            st = sym_len + delta_hat_rx
            rotated = np.zeros(sym_len, dtype=np.complex64)
            for n in range(sym_len):
                rx_phase = np.complex64(rx_phase * np.complex64(phase_step))
                rx_phase_vec[n] = rx_phase
                rotated[n] = np.complex64(rx_phase_vec[n] * rx_buf[st + n])

            rx_i[:sym_len] = rx_i[sym_len:]
            rx_i[sym_len:] = rotated
            rx_sym_td[:] = rotated[Ncp:]

            with torch.inference_mode():
                z_hat = model.receiver(torch.tensor(rx_i, dtype=torch.complex64), run_decoder=False)
            z_hat = z_hat.detach().cpu().numpy().reshape(-1).astype(np.float32)

            f.write(rx_buf.astype(np.complex64).tobytes())
            f.write(np.float32(delta_hat).tobytes())
            f.write(np.float32(freq_offset).tobytes())
            f.write(np.asarray([rx_phase], dtype=np.complex64).tobytes())
            f.write(rx_sym_td.astype(np.complex64).tobytes())
            f.write(rx_i.astype(np.complex64).tobytes())
            f.write(z_hat.tobytes())

    print(f"wrote {args.ncases} cases to {args.output}")
    print(f"  M={M} Ncp={Ncp} Ns={Ns} Nc={Nc} latent_dim={latent_dim}")


if __name__ == "__main__":
    sys.exit(main())
