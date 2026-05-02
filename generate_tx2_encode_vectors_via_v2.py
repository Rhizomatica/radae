"""
Cross-check generator: same output format as generate_tx2_encode_vectors.py
(same on-the-wire schema test_tx2_encode reads), but uses David's
RADEv2Transmitter from radae_v2.py instead of radae_tx_v2 (radae_txe2.py)
as the Python oracle.  Two independent Python TX implementations
agreeing with our C is much stronger evidence than either alone.

RADEv2Transmitter does not apply the bottleneck=3 unit-magnitude
clipper itself, so we apply it here so the output matches what our C
tx2_encode produces post-clip.

Usage:
    PYTHONPATH=. python3 generate_tx2_encode_vectors_via_v2.py OUT.bin --ncases 32

Copyright (c) 2026 Rhizomatica, BSD-3-Clause
"""

import argparse
import sys

import numpy as np

# Disable the stochastic per-layer dither in radae.radae_base.n so we get
# a deterministic Python reference.  Same monkey-patch the
# radae_txe2.py-based generator uses.
import radae.radae_base as _rb
_rb.n = lambda x: x

import torch

from radae import RADAE
from radae_v2 import RADEv2Transmitter

NB_TOTAL_FEATURES = 36
NUM_USED_FEATURES = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument(
        "--model",
        type=str,
        default="250725/checkpoints/checkpoint_epoch_200.pth",
    )
    parser.add_argument("--ncases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Match the production radae_txe2.py constructor exactly so weights
    # load and the modem geometry agrees with tx2_model_data.h.
    model = RADAE(
        21,                       # num_features = 20 + auxdata
        56,                       # latent_dim
        EbNodB=100,
        Nzmf=1,
        rate_Fs=True,
        bottleneck=3,
        cyclic_prefix=0.004,
        time_offset=-16,
        correct_time_offset=-8,   # upstream's new V2 default
        w1_dec=128,
        w1_dec_stateful=128,
        peak=True,
        ssb_bpf=False,            # txbpf=False in production; oracle here
                                  # validates the no-BPF path only
    )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    state_dict      = checkpoint["state_dict"]
    model_dict      = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    model.core_encoder_statefull_load_state_dict()
    model.eval()

    tx = RADEv2Transmitter(model)
    Nzmf       = int(model.Nzmf)
    enc_stride = int(model.enc_stride)
    Ns         = int(model.Ns)
    M          = int(model.M)
    Ncp        = int(model.Ncp)
    Nmf        = Nzmf * Ns * (M + Ncp)
    n_features_in = Nzmf * enc_stride * NB_TOTAL_FEATURES

    Neoo = 6 * (M + Ncp)

    with open(args.output, "wb") as f:
        f.write(np.int32(1).tobytes())            # auxdata (fixed)
        f.write(np.int32(0).tobytes())            # txbpf_en = 0
        f.write(np.int32(n_features_in).tobytes())
        f.write(np.int32(Nmf).tobytes())
        f.write(np.int32(Neoo).tobytes())
        f.write(np.int32(args.ncases).tobytes())

        for _ in range(args.ncases):
            features_in = rng.normal(0.0, 0.8, size=n_features_in).astype(np.float32)

            # Pack as RADAE expects: (1, enc_stride, num_features)
            ff = torch.reshape(torch.tensor(features_in),
                               (1, enc_stride, NB_TOTAL_FEATURES))
            ff = ff[:, :, :NUM_USED_FEATURES]
            aux = -torch.ones((1, enc_stride, 1))
            ff = torch.cat([ff, aux], dim=2)

            iq = tx.transmit_frame(ff)             # (Nmf,) complex64

            # RADEv2Transmitter does NOT apply the bottleneck=3 unit-mag
            # clipper; our C tx2_encode does.  Apply it here so the
            # reference matches.
            mag = np.abs(iq)
            iq_clipped = np.where(mag > 0,
                                  iq / np.maximum(mag, 1e-12),
                                  np.complex64(1.0)).astype(np.csingle)

            f.write(features_in.tobytes())
            f.write(iq_clipped.tobytes())

        # EOO: emit the precomputed eoo_v2 waveform.  No clip applied
        # (radae_txe2.py doesn't clip eoo either).
        eoo = tx.eoo()
        assert eoo.shape == (Neoo,), eoo.shape
        f.write(eoo.tobytes())

    print(f"wrote {args.ncases} cases + 1 EOO frame to {args.output}")
    print(f"  oracle = RADEv2Transmitter (radae_v2.py) + post-clip")
    print(f"  n_features_in={n_features_in} Nmf={Nmf} Neoo={Neoo}")


if __name__ == "__main__":
    sys.exit(main())
