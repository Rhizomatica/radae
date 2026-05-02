"""
Generate deterministic API-level vectors for rade_rx_v2_pure_c().
"""

import argparse

import numpy as np
import torch

import radae.radae_base as rb
from radae_rxe2 import radae_rx_v2


def remove_weight_norm(model):
    def _remove(m):
        try:
            torch.nn.utils.remove_weight_norm(m)
        except ValueError:
            return
    model.apply(_remove)


def qpsk_symbols(rng, n_sym, nc):
    vals = (np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.float32(np.sqrt(2.0))).astype(np.complex64)
    idx = rng.integers(0, len(vals), size=(n_sym, nc))
    return vals[idx]


def make_symbol_stream(model, rng, n_noise_prefix=8, n_data=28, n_noise_suffix=8):
    sym_len = model.M + model.Ncp
    winv = model.Winv.detach().cpu()
    data_fd = qpsk_symbols(rng, n_data, model.Nc)
    data_td = []
    for carriers in data_fd:
        carriers_t = torch.from_numpy(carriers).to(torch.complex64).reshape(1, 1, model.Nc)
        sym = torch.matmul(carriers_t, winv).detach().cpu().numpy().reshape(model.M).astype(np.complex64)
        sym_cp = np.concatenate((sym[-model.Ncp:], sym)).astype(np.complex64)
        data_td.append(sym_cp)
    data_td = np.concatenate(data_td) if data_td else np.zeros(0, dtype=np.complex64)
    noise_prefix = (0.02 * (rng.standard_normal(n_noise_prefix * sym_len)
                            + 1j * rng.standard_normal(n_noise_prefix * sym_len))).astype(np.complex64)
    noise_suffix = (0.02 * (rng.standard_normal(n_noise_suffix * sym_len)
                            + 1j * rng.standard_normal(n_noise_suffix * sym_len))).astype(np.complex64)
    eoo = model.eoo_v2.detach().cpu().numpy().reshape(-1).astype(np.complex64)
    return np.concatenate((noise_prefix, data_td, eoo, noise_suffix)).astype(np.complex64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--model", type=str, default="250725/checkpoints/checkpoint_epoch_200.pth")
    parser.add_argument("--frame-sync", type=str, default="250725a_ml_sync")
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    orig_n = rb.n
    rb.n = lambda x: x

    rx = radae_rx_v2(
        model_name=args.model,
        frame_sync_model_name=args.frame_sync,
        verbose=False,
    )
    remove_weight_norm(rx.model)
    rx.model.core_decoder_statefull_load_state_dict()
    rx.model.eval()

    floats_out = np.zeros(rx.get_n_floats_out(), dtype=np.float32)
    stream = make_symbol_stream(rx.model, rng)

    rows = []
    nin = rx.get_nin()
    prx = 0
    while prx + nin < len(stream):
        buf = stream[prx:prx + nin].astype(np.complex64)
        prx += nin
        valid = int(rx.do_radae_rx(buf, floats_out))
        rows.append({
            "nin": nin,
            "rx_in": buf,
            "valid": valid,
            "next_nin": rx.get_nin(),
            "sync": int(rx.get_sync()),
            "snr": int(rx.get_snrdB_3k_est()),
            "freq_offset": np.float32(rx.receiver.freq_offset),
            "features": floats_out.copy() if valid else np.zeros_like(floats_out),
        })
        nin = rx.get_nin()

    with open(args.output, "wb") as f:
        f.write(np.int32(rx.get_n_floats_out()).tobytes())
        f.write(np.int32(len(rows)).tobytes())
        for row in rows:
            f.write(np.int32(row["nin"]).tobytes())
            f.write(row["rx_in"].tobytes())
            f.write(np.int32(row["valid"]).tobytes())
            f.write(np.int32(row["next_nin"]).tobytes())
            f.write(np.int32(row["sync"]).tobytes())
            f.write(np.int32(row["snr"]).tobytes())
            f.write(np.float32(row["freq_offset"]).tobytes())
            f.write(row["features"].astype(np.float32).tobytes())

    rb.n = orig_n
    print(f"wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
