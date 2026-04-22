"""
Generate (input, expected_output) pairs for the pure-C FrameSyncNet
bit-accuracy test.

Binary output layout, repeated N times:
    float32[input_dim]   - input vector
    float32              - PyTorch FrameSyncNet(...) output (sigmoid scalar)

Usage:
    python3 generate_frame_sync_vectors.py <fsync.pth> <out.bin>
                                           [--input-dim 56] [-n 1000]

Copyright (c) 2026 Rhizomatica, BSD-3-Clause
"""

import argparse
import sys

import numpy as np
import torch

from models_sync import FrameSyncNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--input-dim", type=int, default=56)
    parser.add_argument("-n", type=int, default=1000,
                        help="number of test vectors (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net = FrameSyncNet(args.input_dim)
    sd = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(sd, strict=True)
    net.eval()

    # Mix scales to exercise both saturated and mid-range sigmoid regions.
    scales = np.random.uniform(0.1, 3.0, size=args.n).astype(np.float32)
    inputs = (np.random.randn(args.n, args.input_dim).astype(np.float32)
              * scales[:, None])

    with torch.no_grad():
        # rx2.py calls with shape [1, 1, input_dim]; broadcast here as
        # [n, 1, input_dim] for batched inference, then squeeze.
        x = torch.from_numpy(inputs).unsqueeze(1)
        y = net(x).squeeze(-1).squeeze(-1).numpy().astype(np.float32)

    with open(args.output, "wb") as f:
        for i in range(args.n):
            f.write(inputs[i].tobytes())
            f.write(np.float32(y[i]).tobytes())

    print(f"wrote {args.n} pairs to {args.output}")
    print(f"  input_dim={args.input_dim}")
    print(f"  y min/mean/max = {y.min():.3e} / {y.mean():.3e} / {y.max():.3e}")


if __name__ == "__main__":
    sys.exit(main())
