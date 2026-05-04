"""
Generate streaming vectors for the C rx2_frame_sync port.

The Python decoder path normally injects random quantization noise through
radae_base.n(...), which makes exact parity impossible.  For this harness we
disable that shim so the reference matches the deterministic math exported to C.

Binary output layout:
    int32      auxdata
    int32      limit_pitch
    int32      mute
    int32      output_dim
    int32      ncases
    repeated ncases times:
        int32       sym_index
        int32       sig_det
        int32       sine_det
        float32[]   z_hat (latent_dim)
        float32     expected_metric
        float32     expected_even
        float32     expected_odd
        int32       expected_valid
        float32[]   expected_az_hat (latent_dim)
        float32[]   expected_features (deterministic Python baseline)

Usage:
    python3 generate_rx2_frame_sync_vectors.py out.bin
"""

import argparse
import sys

import numpy as np
import torch

import radae.radae_base as rb
from radae import RADAE
from models_sync import FrameSyncNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("--model", type=str, default="250725/checkpoints/checkpoint_epoch_200.pth")
    parser.add_argument("--frame-sync", type=str, default="250725a_ml_sync")
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--auxdata", type=int, default=1)
    parser.add_argument("--limit-pitch", type=int, default=1)
    parser.add_argument("--mute", type=int, default=1)
    parser.add_argument("--ncases", type=int, default=192)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    num_features = 20 + (1 if args.auxdata else 0)
    orig_n = rb.n
    rb.n = lambda x: x
    frame_sync_nn = FrameSyncNet(args.latent_dim)
    frame_sync_nn.load_state_dict(torch.load(args.frame_sync, weights_only=True, map_location=torch.device("cpu")))
    frame_sync_nn.eval()
    model = RADAE(
        num_features,
        args.latent_dim,
        EbNodB=100,
        Nzmf=1,
        rate_Fs=True,
        bottleneck=3,
        cyclic_prefix=0.004,
        time_offset=-16,
        correct_time_offset=-8,
        stateful_decoder=False,
        w1_dec=128,
        w1_dec_stateful=128,
        peak=True,
    )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict, strict=False)
    def _remove_weight_norm(m):
        try:
            torch.nn.utils.remove_weight_norm(m)
        except ValueError:
            return
    model.apply(_remove_weight_norm)
    model.core_decoder_statefull_load_state_dict()
    model.eval()

    beta = 0.999
    frame_sync_even = 0.0
    frame_sync_odd = 0.0
    az_hat = np.zeros(args.latent_dim, dtype=np.float32)
    output_dim = num_features * model.dec_stride

    with open(args.output, "wb") as f:
        f.write(np.int32(args.auxdata).tobytes())
        f.write(np.int32(args.limit_pitch).tobytes())
        f.write(np.int32(args.mute).tobytes())
        f.write(np.int32(output_dim).tobytes())
        f.write(np.int32(args.ncases).tobytes())

        with torch.inference_mode():
            for c in range(args.ncases):
                sym_index = c + 1
                scale = rng.uniform(0.05, 2.5)
                z_hat = (scale * rng.standard_normal(args.latent_dim)).astype(np.float32)
                if c % 9 == 0:
                    z_hat += np.linspace(-0.75, 0.75, args.latent_dim, dtype=np.float32)
                sig_det = int(rng.integers(0, 2))
                sine_det = int(rng.integers(0, 2)) if (c % 5 == 0) else 0

                z_tensor = torch.from_numpy(z_hat).reshape(1, 1, args.latent_dim)
                metric = float(frame_sync_nn(z_tensor)[0, 0, 0].detach())
                if sym_index % 2:
                    frame_sync_odd = beta * frame_sync_odd + (1.0 - beta) * metric
                    winning = frame_sync_odd > frame_sync_even
                else:
                    frame_sync_even = beta * frame_sync_even + (1.0 - beta) * metric
                    winning = frame_sync_even > frame_sync_odd

                expected_features = np.zeros(output_dim, dtype=np.float32)
                if winning:
                    az_hat = z_hat.copy()
                    features = model.core_decoder_statefull(
                        torch.reshape(torch.from_numpy(az_hat), (1, 1, args.latent_dim))
                    )
                    if args.limit_pitch:
                        features[:, :, 18].clamp_(min=-1.4)
                    if args.mute and (not sig_det or sine_det):
                        features[:, :, 0] = -5.0
                    expected_features[:] = features.detach().cpu().numpy().reshape(-1).astype(np.float32)

                f.write(np.int32(sym_index).tobytes())
                f.write(np.int32(sig_det).tobytes())
                f.write(np.int32(sine_det).tobytes())
                f.write(z_hat.tobytes())
                f.write(np.float32(metric).tobytes())
                f.write(np.float32(frame_sync_even).tobytes())
                f.write(np.float32(frame_sync_odd).tobytes())
                f.write(np.int32(int(winning)).tobytes())
                f.write(az_hat.astype(np.float32).tobytes())
                f.write(expected_features.tobytes())

    rb.n = orig_n

    print(f"wrote {args.ncases} cases to {args.output}")
    print(f"  latent_dim={args.latent_dim} output_dim={output_dim}")


if __name__ == "__main__":
    sys.exit(main())
