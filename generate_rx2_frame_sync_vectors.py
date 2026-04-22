"""
Generate streaming vectors for the C rx2_frame_sync port.

Binary output layout:
    int32      ncases
    repeated ncases times:
        int32       sym_index
        float32[]   z_hat (latent_dim)
        float32     expected_metric
        float32     expected_even
        float32     expected_odd
        int32       expected_valid
        float32[]   expected_az_hat (latent_dim)

Usage:
    python3 generate_rx2_frame_sync_vectors.py out.bin
"""

import argparse
import sys

import numpy as np
import torch

from radae import RADAE
from models_sync import FrameSyncNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--frame-sync", type=str, default="250725a_ml_sync")
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--ncases", type=int, default=192)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frame_sync_nn = FrameSyncNet(args.latent_dim)
    frame_sync_nn.load_state_dict(torch.load(args.frame_sync, weights_only=True, map_location=torch.device("cpu")))
    frame_sync_nn.eval()

    beta = 0.999
    frame_sync_even = 0.0
    frame_sync_odd = 0.0
    az_hat = np.zeros(args.latent_dim, dtype=np.float32)

    with open(args.output, "wb") as f:
        f.write(np.int32(args.ncases).tobytes())

        with torch.inference_mode():
            for c in range(args.ncases):
                sym_index = c + 1
                scale = rng.uniform(0.05, 2.5)
                z_hat = (scale * rng.standard_normal(args.latent_dim)).astype(np.float32)
                if c % 9 == 0:
                    z_hat += np.linspace(-0.75, 0.75, args.latent_dim, dtype=np.float32)

                z_tensor = torch.from_numpy(z_hat).reshape(1, 1, args.latent_dim)
                metric = float(frame_sync_nn(z_tensor)[0, 0, 0].detach())
                if sym_index % 2:
                    frame_sync_odd = beta * frame_sync_odd + (1.0 - beta) * metric
                    winning = frame_sync_odd > frame_sync_even
                else:
                    frame_sync_even = beta * frame_sync_even + (1.0 - beta) * metric
                    winning = frame_sync_even > frame_sync_odd

                if winning:
                    az_hat = z_hat.copy()

                f.write(np.int32(sym_index).tobytes())
                f.write(z_hat.tobytes())
                f.write(np.float32(metric).tobytes())
                f.write(np.float32(frame_sync_even).tobytes())
                f.write(np.float32(frame_sync_odd).tobytes())
                f.write(np.int32(int(winning)).tobytes())
                f.write(az_hat.astype(np.float32).tobytes())

    print(f"wrote {args.ncases} cases to {args.output}")
    print(f"  latent_dim={args.latent_dim}")


if __name__ == "__main__":
    sys.exit(main())
