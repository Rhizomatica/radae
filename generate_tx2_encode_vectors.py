"""
Generate streaming reference vectors for the C tx2_encode port.

Each vector file drives test_tx2_encode with:
  - a header describing the session (auxdata, txbpf)
  - N (features_in, expected_iq_out) pairs produced by the Python
    radae_tx_v2 TX pipeline
  - one EOO frame captured after the last TX frame

Binary output layout:
    int32    auxdata
    int32    txbpf_en
    int32    n_features_in        (= Nzmf * enc_stride * 36)
    int32    Nmf
    int32    Neoo
    int32    ncases
    repeated ncases times:
        float32[n_features_in]   features_in
        complex64[Nmf]           expected tx_out
    complex64[Neoo]              expected EOO frame (after all TX frames)

Usage:
    PYTHONPATH=. python3 generate_tx2_encode_vectors.py OUT.bin --ncases 32
    PYTHONPATH=. python3 generate_tx2_encode_vectors.py OUT.bin --txbpf

Copyright (c) 2026 Rhizomatica, BSD-3-Clause
"""

import argparse
import sys

import numpy as np

# Disable the stochastic per-layer dithering n(x) = clamp(x + U(-1/254, 1/254))
# in radae_base before importing the TX module.  The C encoder is
# deterministic; bit-accurate parity is only possible against a
# deterministic Python reference.  Production TX still uses the noisy
# version (the encoder was trained with it); skipping it here only
# affects the parity-test reference, which is the intended deterministic
# oracle for the C port.  See C_TX_MIGRATION.md.
import radae.radae_base as _rb
_rb.n = lambda x: x

from radae_txe2 import radae_tx_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument(
        "--model",
        type=str,
        default="250725/checkpoints/checkpoint_epoch_200.pth",
        help="path to the 250725 .pth checkpoint",
    )
    parser.add_argument("--ncases", type=int, default=32,
                        help="number of feature frames to generate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--txbpf",
        dest="txbpf",
        action="store_true",
        help="enable the transmit complex BPF path",
    )
    parser.set_defaults(txbpf=False)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    tx = radae_tx_v2(model_name=args.model, txbpf_en=args.txbpf)
    n_features_in = int(tx.get_n_features_in())
    Nmf = int(tx.get_Nmf())
    Neoo = int(tx.get_Neoo())

    tx_out = np.zeros(Nmf, dtype=np.csingle)
    eoo_out = np.zeros(Neoo, dtype=np.csingle)

    with open(args.output, "wb") as f:
        f.write(np.int32(1).tobytes())                        # auxdata (fixed)
        f.write(np.int32(1 if args.txbpf else 0).tobytes())   # txbpf_en
        f.write(np.int32(n_features_in).tobytes())
        f.write(np.int32(Nmf).tobytes())
        f.write(np.int32(Neoo).tobytes())
        f.write(np.int32(args.ncases).tobytes())

        for _ in range(args.ncases):
            features_in = rng.normal(0.0, 0.8, size=n_features_in).astype(np.float32)
            tx.do_radae_tx(features_in, tx_out)
            f.write(features_in.tobytes())
            f.write(tx_out.tobytes())

        tx.do_eoo(eoo_out)
        f.write(eoo_out.tobytes())

    print(f"wrote {args.ncases} cases + 1 EOO frame to {args.output}")
    print(f"  auxdata=1 txbpf={int(args.txbpf)}")
    print(f"  n_features_in={n_features_in} Nmf={Nmf} Neoo={Neoo}")


if __name__ == "__main__":
    sys.exit(main())
