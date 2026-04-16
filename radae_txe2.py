"""
  Radio Autoencoder V2 streaming transmitter, "embedded" version.

  FARGAN-compatible 36-dim features on stdin, Rate Fs complex float
  IQ samples on stdout.  Designed to be piped from lpcnet_demo -features
  for real-time RADE V2 transmission to a SDR.

  Mirrors the streaming contract of radae_txe.py (V1) while using the
  V2 waveform: no pilot symbols, stateful encoder, hard clipper, and
  the V2 EOO sequence.

  Copyright (c) 2026 by Rhizomatica
  Based on radae_txe.py (V1) and inference.py by David Rowe

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

import os, sys, struct, argparse
import numpy as np
import torch

from radae import RADAE, complex_bpf


NB_TOTAL_FEATURES = 36      # FARGAN/lpcnet feature vector size
NUM_USED_FEATURES = 20      # features actually consumed by the encoder


class radae_tx_v2:
    """Streaming RADE V2 transmitter.

    One call to do_radae_tx() consumes one modem frame of features
    (n_floats_in floats = Nzmf*enc_stride*36 floats, i.e. 40 ms of audio
    with Nzmf=1) and produces Nmf complex IQ samples.
    """

    def __init__(self, model_name,
                 latent_dim=56, bottleneck=3, cp=0.004,
                 time_offset=-16, correct_time_offset=-16,
                 w1_dec=128, peak=True, auxdata=True,
                 txbpf_en=False):

        self.latent_dim = latent_dim
        self.auxdata    = auxdata
        self.bottleneck = bottleneck
        self.txbpf_en   = txbpf_en

        self.num_features = NUM_USED_FEATURES + (1 if auxdata else 0)

        # RADE V2 model.  Nzmf=1 matches per-latent streaming granularity.
        # pilots=False is the V2 default (barker pilots removed).
        self.model = RADAE(self.num_features, latent_dim,
                           EbNodB=100, Nzmf=1,
                           rate_Fs=True, bottleneck=bottleneck,
                           cyclic_prefix=cp,
                           time_offset=time_offset,
                           correct_time_offset=correct_time_offset,
                           w1_dec=w1_dec, w1_dec_stateful=w1_dec,
                           peak=peak)

        checkpoint = torch.load(model_name, map_location='cpu',
                                weights_only=True)
        # Shape-filtered load — mirrors radae_rxe2.py so we can load
        # checkpoints that differ only in stateful-decoder widths.
        state_dict      = checkpoint['state_dict']
        model_dict      = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        self.model.core_encoder_statefull_load_state_dict()
        self.model.eval()

        m = self.model

        # Per-frame sizes.
        self.n_floats_in = m.Nzmf * m.enc_stride * NB_TOTAL_FEATURES
        # V2 has no pilots, so: num_timesteps_at_rate_Fs = Nzmf*Ns*(M+Ncp)
        self.Nmf = int(m.Nzmf * m.Ns * (m.M + m.Ncp))
        # V2 EOO is 6 pend_cp symbols back-to-back (see radae.py eoo_v2).
        self.Neoo = int(6 * (m.M + m.Ncp))

        if self.txbpf_en:
            Ntap      = 101
            w         = np.array(m.w)
            Nc        = m.Nc
            bandwidth = 1.2 * (w[Nc - 1] - w[0]) * m.Fs / (2 * np.pi)
            centre    = (w[Nc - 1] + w[0]) * m.Fs / (2 * np.pi) / 2
            print(f"Tx BPF bandwidth: {bandwidth:f} centre: {centre:f}",
                  file=sys.stderr)
            self.txbpf = complex_bpf(Ntap, m.Fs, bandwidth, centre,
                                     max(self.Nmf, self.Neoo))
        else:
            self.txbpf = None

    def get_n_features_in(self):
        return self.model.Nzmf * self.model.enc_stride * NB_TOTAL_FEATURES

    def get_n_floats_in(self):
        return self.n_floats_in

    def get_Nmf(self):
        return self.Nmf

    def get_Neoo(self):
        return self.Neoo

    def do_radae_tx(self, buffer_f32, tx_out):
        m = self.model
        with torch.inference_mode():
            features = torch.reshape(
                torch.tensor(buffer_f32),
                (1, m.Nzmf * m.enc_stride, NB_TOTAL_FEATURES),
            )
            features = features[:, :, :NUM_USED_FEATURES]

            if self.auxdata:
                # Aux bit symbols: -1 for all four positions in a
                # symb_repeat=4 group (matches V1 radae_txe.py defaults,
                # a fixed beacon-style aux pattern).
                aux_symb = -torch.ones((1, features.shape[1], 1))
                symb_repeat = 4
                for i in range(1, symb_repeat):
                    aux_symb[0, i::symb_repeat, :] = aux_symb[0, ::symb_repeat, :]
                features = torch.cat([features, aux_symb], dim=2)

            # Stateful encoder: (1, Nzmf*enc_stride, feat) -> (1, Nzmf, latent_dim)
            z = m.core_encoder_statefull(features)

            # Map z to QPSK symbols.
            tx_sym = z[:, :, ::2] + 1j * z[:, :, 1::2]

            # Reshape to (batch, Nzmf*Ns, Nc).  No pilot insertion — V2 has pilots=False.
            num_timesteps_at_rate_Rs = m.Nzmf * m.Ns
            tx_sym = torch.reshape(tx_sym, (1, num_timesteps_at_rate_Rs, m.Nc))

            # IDFT to time domain.
            tx = torch.matmul(tx_sym, m.Winv)

            # Cyclic prefix.
            if m.Ncp:
                Ncp = m.Ncp
                tx_cp = torch.zeros(
                    (1, num_timesteps_at_rate_Rs, m.M + Ncp),
                    dtype=torch.complex64,
                )
                tx_cp[:, :, Ncp:] = tx
                tx_cp[:, :, :Ncp] = tx_cp[:, :, -Ncp:]
                tx = tx_cp

            tx = torch.reshape(tx, (1, num_timesteps_at_rate_Rs * (m.M + m.Ncp)))

            # Hard magnitude clipper — bottleneck=3 with tanh_clipper=False
            # (V2 training config). Constant-envelope after this stage.
            if self.bottleneck == 3:
                tx = torch.exp(1j * torch.angle(tx))

            tx = tx.cpu().detach().numpy().flatten().astype('csingle')
            if self.txbpf is not None:
                tx = self.txbpf.bpf(tx)
                tx = np.clip(np.abs(tx), a_min=0, a_max=1) * np.exp(1j * np.angle(tx))

            np.copyto(tx_out, tx)

    def do_eoo(self, tx_out):
        """Emit one V2 end-of-over frame (6 pend_cp symbols)."""
        eoo = self.model.eoo_v2
        eoo = eoo.cpu().detach().numpy().flatten().astype('csingle')
        if self.txbpf is not None:
            eoo = self.txbpf.bpf(eoo)
            eoo = np.clip(np.abs(eoo), a_min=0, a_max=1) * np.exp(1j * np.angle(eoo))
        np.copyto(tx_out, eoo)


def _build_argparser():
    p = argparse.ArgumentParser(
        description='RADE V2 streaming transmitter: features.f32 on stdin, '
                    'IQ.f32 on stdout'
    )
    p.add_argument('--model_name', type=str, required=True,
                   help='path to RADE V2 model in .pth format')
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
    p.add_argument('--txbpf', dest='txbpf', action='store_true',
                   help='enable transmit BPF (default disabled)')
    p.add_argument('--no_eoo', dest='send_eoo', action='store_false',
                   help='do not emit EOO frame on shutdown')
    p.set_defaults(peak=True, auxdata=True, txbpf=False, send_eoo=True)
    return p


if __name__ == '__main__':
    # Force CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = _build_argparser().parse_args()

    tx = radae_tx_v2(
        model_name=args.model_name,
        latent_dim=args.latent_dim,
        bottleneck=args.bottleneck,
        cp=args.cp,
        time_offset=args.time_offset,
        correct_time_offset=args.correct_time_offset,
        w1_dec=args.w1_dec,
        peak=args.peak,
        auxdata=args.auxdata,
        txbpf_en=args.txbpf,
    )

    tx_out  = np.zeros(tx.get_Nmf(),  dtype=np.csingle)
    eoo_out = np.zeros(tx.get_Neoo(), dtype=np.csingle)

    float_nbytes = struct.calcsize("f")
    want         = tx.get_n_floats_in() * float_nbytes

    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        while True:
            buf = stdin.read(want)
            if len(buf) != want:
                break
            buffer_f32 = np.frombuffer(buf, np.single)
            tx.do_radae_tx(buffer_f32, tx_out)
            stdout.write(tx_out.tobytes())
            stdout.flush()
    finally:
        if args.send_eoo:
            tx.do_eoo(eoo_out)
            stdout.write(eoo_out.tobytes())
            stdout.flush()
