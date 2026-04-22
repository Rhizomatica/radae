"""
  Radio Autoencoder V2 streaming receiver, "embedded" version.

  Rate Fs complex float samples on stdin, FARGAN-compatible 36-dim
  features on stdout.  Designed to be piped into lpcnet_demo for
  real-time RADE V2 decoding from a SDR.

  Wraps RADEv2Receiver (from rx2.py) in a per-symbol streaming loop,
  mirroring the contract provided by radae_rxe.py (V1).

  Copyright (c) 2026 by Rhizomatica
  Based on rx2.py by David Rowe

/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

import os, sys, struct, argparse, types
import numpy as np
import torch

from radae import RADAE, complex_bpf
from models_sync import FrameSyncNet
from rx2 import ComfortNoiseGenerator, RADEv2Receiver


NB_TOTAL_FEATURES = 36      # FARGAN/lpcnet feature vector size
NUM_USED_FEATURES = 20      # features actually consumed by FARGAN; the rest are zero-padded


class radae_rx_v2:
    """Streaming RADE V2 receiver.

    One call to do_radae_rx() processes one OFDM symbol (nin complex samples).
    When the state machine is in 'sync' and frame sync wins, one latent
    vector worth of features is emitted via floats_out.

    The caller must read get_nin() complex samples from the input source
    each iteration, pass them in, and check the return code.
    """

    def __init__(self, model_name, frame_sync_model_name,
                 latent_dim=56, bottleneck=3, cp=0.004,
                 time_offset=-16, correct_time_offset=-16,
                 w1_dec=128, peak=True, auxdata=True,
                 bpf_en=True, agc_en=True, mute_en=True,
                 limit_pitch=True, hangover=75,
                 comfort_noise_en=True,
                 verbose=False, reset_output_on_resync=False):

        self.latent_dim  = latent_dim
        self.auxdata     = auxdata
        self.bpf_en      = bpf_en
        self.verbose     = verbose

        # Feature dim used inside the model (20 speech features + optional aux bit symbol)
        self.num_features = NUM_USED_FEATURES + (1 if auxdata else 0)

        # Load RADE V2 model.  Nzmf=1 matches rx2.py: the receiver emits per
        # latent vector, not in Nzmf-sized batches.
        self.model = RADAE(self.num_features, latent_dim,
                           EbNodB=100, Nzmf=1,
                           rate_Fs=True, bottleneck=bottleneck,
                           cyclic_prefix=cp,
                           time_offset=time_offset,
                           correct_time_offset=correct_time_offset,
                           stateful_decoder=False,
                           w1_dec=w1_dec, w1_dec_stateful=w1_dec,
                           peak=peak)
        checkpoint = torch.load(model_name, map_location='cpu', weights_only=True)

        # Load only weights whose shapes match the model.  Training used
        # different stateful-decoder widths in some checkpoints.
        state_dict      = checkpoint['state_dict']
        model_dict      = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        self.model.core_decoder_statefull_load_state_dict()
        self.model.eval()

        # Load frame-sync network
        self.frame_sync_nn = FrameSyncNet(latent_dim)
        sync_sd = torch.load(frame_sync_model_name, weights_only=True,
                             map_location=torch.device('cpu'))
        self.frame_sync_nn.load_state_dict(sync_sd)
        self.frame_sync_nn.eval()

        # Assemble an argparse-like object for RADEv2Receiver
        args = types.SimpleNamespace(
            agc=agc_en, mute=mute_en, limit_pitch=limit_pitch,
            hangover=hangover, verbose=verbose,
            reset_output_on_resync=reset_output_on_resync,
        )
        self.args     = args
        self.receiver = RADEv2Receiver(self.model, self.frame_sync_nn, args)

        # Streaming BPF.  Worst-case chunk size is sym_len + shift (shift=sym_len//4).
        self.sym_len = self.receiver.sym_len
        self.max_nin = self.sym_len + self.sym_len // 4

        if self.bpf_en:
            Ntap      = 101
            w         = np.array(self.model.w)
            Nc        = self.model.Nc
            bandwidth = 1.2 * (w[Nc - 1] - w[0]) * self.model.Fs / (2 * np.pi)
            centre    = (w[Nc - 1] + w[0]) * self.model.Fs / (2 * np.pi) / 2
            print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}",
                  file=sys.stderr)
            self.bpf = complex_bpf(Ntap, self.model.Fs, bandwidth, centre,
                                   self.max_nin)
        else:
            self.bpf = None

        # Output size: one latent vector decodes to dec_stride feature frames.
        self.dec_stride    = self.model.dec_stride
        self.n_floats_out  = self.dec_stride * NB_TOTAL_FEATURES

        # For the streaming loop we need enough "prefetch" to support
        # _adjust_timing() expanding a read by sym_len//4.
        self.nin = self.sym_len
        self.comfort_noise = ComfortNoiseGenerator(
            self.dec_stride, self.num_features, self.auxdata, comfort_noise_en
        )

    def _pack_features(self, features_hat_slice):
        pad = torch.zeros(
            (features_hat_slice.shape[0],
             features_hat_slice.shape[1],
             NB_TOTAL_FEATURES - self.num_features),
            dtype=features_hat_slice.dtype,
        )
        return torch.cat([features_hat_slice, pad], dim=-1)

    def _should_emit_comfort_noise(self, symbol_count):
        return self.comfort_noise.should_emit(symbol_count)

    def get_nin(self):
        return self.nin

    def get_nin_max(self):
        return self.max_nin

    def get_n_floats_out(self):
        return self.n_floats_out

    def get_sync(self):
        return self.receiver.state == "sync"

    def get_snrdB_3k_est(self):
        return float(self.receiver.snr_est_dB)

    def do_radae_rx(self, buffer_complex, floats_out):
        """Process one nin-sample chunk.

        Returns 1 if floats_out was populated with either a valid feature slice
        or a comfort-noise slice, 0 otherwise.
        """
        r = self.receiver

        with torch.inference_mode():
            rx_in = buffer_complex[:self.nin]
            if self.bpf is not None:
                rx_in = self.bpf.bpf(rx_in)

            r.s += 1
            prev_state = r.state

            next_state, features_hat_slice, new_nin, sig_det, sine_det, gain = \
                r._process_symbol(rx_in, self.nin)

            if self.verbose or r.state != prev_state:
                r._print_status(sig_det, sine_det, self.nin)

            r.state = next_state
            # Enable timing adjust after a few symbols so the IIR estimates settle.
            if r.s > 16:
                r.timing_adj = 1
            self.nin = new_nin

            if features_hat_slice is not None:
                self.comfort_noise.update(features_hat_slice, r.s)
                features_hat = self._pack_features(features_hat_slice)
                np.copyto(
                    floats_out,
                    features_hat.detach().cpu().numpy().reshape(-1).astype(np.float32),
                )
                r.i += 1
                return 1

            if not self._should_emit_comfort_noise(r.s):
                return 0

            features_hat = np.zeros((self.dec_stride, NB_TOTAL_FEATURES),
                                    dtype=np.float32)
            features_hat[:, :self.num_features] = self.comfort_noise.generate()
            np.copyto(floats_out, features_hat.reshape(-1))
            r.i += 1
            return 1


def _build_argparser():
    p = argparse.ArgumentParser(
        description='RADE V2 streaming receiver: IQ.f32 on stdin, '
                    'FARGAN features.f32 on stdout'
    )
    p.add_argument('--model_name', type=str, required=True,
                   help='path to RADE V2 model in .pth format')
    p.add_argument('--frame_sync_model_name', type=str, required=True,
                   help='path to ML frame-sync model in .pth format')
    p.add_argument('--latent-dim', dest='latent_dim', type=int, default=56)
    p.add_argument('--bottleneck', type=int, default=3)
    p.add_argument('--cp', type=float, default=0.004)
    p.add_argument('--time_offset', type=int, default=-16)
    p.add_argument('--correct_time_offset', type=int, default=-16)
    p.add_argument('--w1_dec', type=int, default=128)
    p.add_argument('--nopeak', dest='peak', action='store_false',
                   help='disable peak=True training flag (default enabled)')
    p.add_argument('--noauxdata', dest='auxdata', action='store_false',
                   help='disable aux data bits (default enabled)')
    p.add_argument('--no_bpf', dest='bpf', action='store_false',
                   help='disable input BPF')
    p.add_argument('--no_agc', dest='agc', action='store_false',
                   help='disable input AGC')
    p.add_argument('--no_mute', dest='mute', action='store_false',
                   help='disable feature-mute on signal loss')
    p.add_argument('--no_comfort_noise', dest='comfort_noise',
                   action='store_false',
                   help='disable comfort-noise output when no valid decode is available')
    p.add_argument('--no_stdout', dest='use_stdout', action='store_false',
                   help='disable stdout output (for profiling)')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='print per-symbol status to stderr')
    p.add_argument('--hangover', type=int, default=75)
    p.add_argument('--reset_output_on_resync', action='store_true')
    p.set_defaults(peak=True, auxdata=True, bpf=True, agc=True, mute=True,
                   comfort_noise=True, use_stdout=True)
    return p


if __name__ == '__main__':
    # Force CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = _build_argparser().parse_args()

    rx = radae_rx_v2(
        model_name=args.model_name,
        frame_sync_model_name=args.frame_sync_model_name,
        latent_dim=args.latent_dim,
        bottleneck=args.bottleneck,
        cp=args.cp,
        time_offset=args.time_offset,
        correct_time_offset=args.correct_time_offset,
        w1_dec=args.w1_dec,
        peak=args.peak,
        auxdata=args.auxdata,
        bpf_en=args.bpf,
        agc_en=args.agc,
        mute_en=args.mute,
        hangover=args.hangover,
        comfort_noise_en=args.comfort_noise,
        verbose=args.verbose,
        reset_output_on_resync=args.reset_output_on_resync,
    )

    floats_out = np.zeros(rx.get_n_floats_out(), dtype=np.float32)
    complex_nbytes = struct.calcsize("ff")   # 8 bytes per IQ sample

    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # The number of samples to read can grow on a timing-slip adjustment,
    # so we re-query nin every iteration.
    while True:
        n = rx.get_nin()
        want = n * complex_nbytes
        buf  = stdin.read(want)
        if len(buf) != want:
            break
        buffer_complex = np.frombuffer(buf, np.csingle)
        if rx.do_radae_rx(buffer_complex, floats_out) and args.use_stdout:
            stdout.write(floats_out.tobytes())
            stdout.flush()
