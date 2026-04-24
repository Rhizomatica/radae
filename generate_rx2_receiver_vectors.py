"""
Generate end-to-end symbol-step vectors for the thin C rx2_receiver coordinator.

The reference uses the deterministic decoder math path:
  - radae_base.n = lambda x: x
  - weight norm removed before stateful decoder transfer
"""

import argparse
from types import SimpleNamespace

import numpy as np
import torch

import radae.radae_base as rb
from radae import RADAE
from models_sync import FrameSyncNet
from rx2 import RADEv2Receiver


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


def state_to_int(state):
    return 0 if state == "idle" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--model", type=str, default="250725/checkpoints/checkpoint_epoch_200.pth")
    parser.add_argument("--frame-sync", type=str, default="250725a_ml_sync")
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--auxdata", type=int, default=1)
    parser.add_argument("--agc", type=int, default=1)
    parser.add_argument("--limit-pitch", type=int, default=1)
    parser.add_argument("--mute", type=int, default=1)
    parser.add_argument("--hangover", type=int, default=12)
    parser.add_argument("--timing-adj-at", type=int, default=10)
    parser.add_argument("--reset-output-on-resync", type=int, default=0)
    parser.add_argument("--time-offset", type=int, default=-16)
    parser.add_argument("--correct-time-offset", type=int, default=-16)
    parser.add_argument("--fix-delta-hat", type=int, default=0,
                        help="pin delta_hat_g (0 = argmax, matching rx2.py)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    orig_n = rb.n
    rb.n = lambda x: x

    num_features = 20 + (1 if args.auxdata else 0)
    model = RADAE(
        num_features,
        args.latent_dim,
        EbNodB=100,
        Nzmf=1,
        rate_Fs=True,
        bottleneck=3,
        cyclic_prefix=0.004,
        time_offset=args.time_offset,
        correct_time_offset=args.correct_time_offset,
        stateful_decoder=False,
        w1_dec=128,
        w1_dec_stateful=128,
        peak=True,
    )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict, strict=False)
    remove_weight_norm(model)
    model.core_decoder_statefull_load_state_dict()
    model.eval()

    frame_sync_nn = FrameSyncNet(args.latent_dim)
    frame_sync_nn.load_state_dict(torch.load(args.frame_sync, map_location="cpu", weights_only=True))
    frame_sync_nn.eval()

    rx_args = SimpleNamespace(
        agc=bool(args.agc),
        auxdata=bool(args.auxdata),
        limit_pitch=bool(args.limit_pitch),
        mute=bool(args.mute),
        fix_delta_hat=args.fix_delta_hat,
        hangover=args.hangover,
        reset_output_on_resync=bool(args.reset_output_on_resync),
        timing_adj_at=args.timing_adj_at,
        verbose=False,
    )
    receiver = RADEv2Receiver(model, frame_sync_nn, rx_args)

    rx = make_symbol_stream(model, rng)
    w = model.w.detach().cpu().numpy().astype(np.float32)
    pend = model.pend.detach().cpu().numpy().astype(np.complex64)
    B_bpf = np.float32(1.2 * (w[model.Nc - 1] - w[0]) * float(model.Fs) / (2.0 * np.pi))
    output_dim = num_features * model.dec_stride
    sym_len = receiver.sym_len

    rows = []
    nin = sym_len
    prx = 0
    while prx + nin < len(rx):
        receiver.s += 1
        st, en = prx, prx + nin
        prx += nin
        next_state, features_hat_slice, next_nin, sig_det, sine_det, gain = receiver._process_symbol(rx[st:en], nin)
        receiver.state = next_state
        decoded_valid = int(features_hat_slice is not None)
        features = np.zeros(output_dim, dtype=np.float32)
        if decoded_valid:
            features[:] = features_hat_slice.detach().cpu().numpy().reshape(-1).astype(np.float32)
            receiver.i += 1
        if receiver.s > receiver.args.timing_adj_at:
            receiver.timing_adj = 1

        rows.append({
            "nin": nin,
            "rx_in": rx[st:en].astype(np.complex64),
            "state": state_to_int(receiver.state),
            "count": receiver.count,
            "count1": receiver.count1,
            "n_acq": receiver.n_acq,
            "s": receiver.s,
            "i": receiver.i,
            "timing_adj": receiver.timing_adj,
            "sig_det": int(sig_det),
            "sine_det": int(sine_det),
            "decoded_valid": decoded_valid,
            "next_nin": next_nin,
            "new_sig_delta_hat": int(receiver.new_sig_delta_hat),
            "new_sig_f_hat": int(receiver.new_sig_f_hat),
            "gain": np.float32(gain),
            "delta_hat": np.float32(receiver.delta_hat),
            "delta_hat_g": int(receiver.delta_hat_g),
            "freq_offset": np.float32(receiver.freq_offset),
            "freq_offset_g": np.float32(receiver.freq_offset_g),
            "Ry_max": np.float32(receiver.Ry_max),
            "Ry_min": np.float32(receiver.Ry_min),
            "snr_est_dB": np.float32(receiver.snr_est_dB),
            "frame_sync_even": np.float32(receiver.frame_sync_even),
            "frame_sync_odd": np.float32(receiver.frame_sync_odd),
            "eoo_smooth": np.float32(receiver.eoo_smooth),
            "eoo_corr": np.float32(receiver._eoo_corr),
            "features": features,
        })
        nin = next_nin

    with open(args.output, "wb") as f:
        for value in (
            np.int32(args.auxdata),
            np.int32(args.agc),
            np.int32(args.limit_pitch),
            np.int32(args.mute),
            np.int32(args.hangover),
            np.int32(args.timing_adj_at),
            np.int32(args.reset_output_on_resync),
            np.int32(args.fix_delta_hat),
            np.int32(model.M),
            np.int32(model.Ncp),
            np.int32(model.Ns),
            np.int32(model.Nc),
            np.float32(model.Fs),
            B_bpf,
            np.int32(args.time_offset),
            np.int32(args.correct_time_offset),
        ):
            f.write(np.asarray(value).tobytes())
        f.write(w.tobytes())
        f.write(pend.tobytes())
        f.write(np.int32(output_dim).tobytes())
        f.write(np.int32(len(rows)).tobytes())
        for row in rows:
            f.write(np.int32(row["nin"]).tobytes())
            f.write(row["rx_in"].tobytes())
            for key in (
                "state", "count", "count1", "n_acq", "s", "i", "timing_adj",
                "sig_det", "sine_det", "decoded_valid", "next_nin",
                "new_sig_delta_hat", "new_sig_f_hat",
            ):
                f.write(np.int32(row[key]).tobytes())
            f.write(np.float32(row["gain"]).tobytes())
            f.write(np.float32(row["delta_hat"]).tobytes())
            f.write(np.int32(row["delta_hat_g"]).tobytes())
            for key in (
                "freq_offset", "freq_offset_g",
                "Ry_max", "Ry_min", "snr_est_dB",
                "frame_sync_even", "frame_sync_odd",
                "eoo_smooth", "eoo_corr",
            ):
                f.write(np.float32(row[key]).tobytes())
            f.write(row["features"].tobytes())

    rb.n = orig_n
    print(f"wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
