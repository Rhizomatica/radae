"""
/*
  RADE V2 transmitter: speech features in, rate Fs IQ samples out.

  Copyright (c) 2025 by David Rowe */

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

import os
import argparse

import numpy as np
import torch

from radae import RADAE

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to RADE model in .pth format')
parser.add_argument('features', type=str, help='path to input feature file in .f32 format')
parser.add_argument('rx_out', type=str, help='path to output file of rate Fs IQ samples in ..IQIQ...f32 format')
parser.add_argument('--latent-dim', type=int, default=56, help='number of symbols produced by encoder (default 56)')
parser.add_argument('--cuda-visible-devices', type=str, default="", help='set to 0 to run using GPU rather than CPU')
parser.add_argument('--EbNodB', type=float, default=100, help='BPSK Eb/No in dB (default 100)')
parser.add_argument('--g_file', type=str, default="", help='path to rate Fs Doppler spread samples .f32 format')
parser.add_argument('--g_offset', type=float, default=0.0, help='start this many seconds into g_file (default 0)')
parser.add_argument('--phase_offset', type=float, default=0, help='phase offset in rads')
parser.add_argument('--freq_offset', type=float, default=0, help='freq offset in Hz')
parser.add_argument('--df_dt', type=float, default=0, help='rate of change of freq offset in Hz/s')
parser.add_argument('--correct_freq_offset', action='store_true', help='correct --freq_offset before decoding (default off)')
parser.add_argument('--cp', type=float, default=0.004, help='length of cyclic prefix in seconds (default 0.004)')
parser.add_argument('--time_offset', type=int, default=-16, help='time domain sampling offset in samples (default -16)')
parser.add_argument('--correct_time_offset', type=int, default=-8, help='freq domain time offset correction in samples (default -8)')
parser.add_argument('--ssb_bpf', action='store_true', help='SSB BPF simulation')
parser.add_argument('--rx_gain', type=float, default=1.0, help='gain applied to output IQ samples (default 1.0)')
parser.add_argument('--prepend_noise', type=float, default=1.0, help='seconds of noise before signal (default 1.0)')
parser.add_argument('--append_noise', type=float, default=1.0, help='seconds of noise after signal (default 1.0)')
parser.add_argument('--sine_amp', type=float, default=0.0, help='single freq interferer amplitude (default 0)')
parser.add_argument('--sine_freq', type=float, default=1000.0, help='single freq interferer freq in Hz (default 1000)')
parser.add_argument('--write_latent', type=str, default="", help='path to output latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--write_tx', type=str, default="", help='path to output pre-channel IQ samples in .f32 format')
parser.add_argument('--w1_enc', type=int, default=64, help='encoder GRU output dimension (default 64)')
parser.add_argument('--w2_enc', type=int, default=96, help='encoder conv output dimension (default 96)')
parser.add_argument('--w1_dec', type=int, default=128, help='decoder GRU output dimension (default 128)')
parser.add_argument('--w2_dec', type=int, default=32, help='decoder conv output dimension (default 32)')
parser.set_defaults(auxdata=True)
parser.add_argument('--no_auxdata', action='store_false', dest='auxdata', help='disable auxiliary data symbol (default enabled)')
parser.set_defaults(peak=True)
parser.add_argument('--no_peak', action='store_false', dest='peak', help='disable peak power in loss function (default enabled)')
parser.set_defaults(end_of_over_v2=True)
parser.add_argument('--no_eoo', action='store_false', dest='end_of_over_v2', help='disable end-of-over sequence (default enabled)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20
if args.auxdata:
   num_features += 1

model = RADAE(num_features, latent_dim, args.EbNodB, rate_Fs=True,
              phase_offset=args.phase_offset, freq_offset=args.freq_offset, df_dt=args.df_dt,
              cyclic_prefix=args.cp, time_offset=args.time_offset,
              correct_time_offset=args.correct_time_offset,
              correct_freq_offset=args.correct_freq_offset,
              bottleneck=3, ssb_bpf=args.ssb_bpf,
              w1_dec=args.w1_dec, w2_dec=args.w2_dec,
              w1_enc=args.w1_enc, w2_enc=args.w2_enc, peak=args.peak)
checkpoint = torch.load(args.model_name, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
checkpoint['state_dict'] = model.state_dict()

feature_file = args.features
features_in = np.reshape(np.fromfile(feature_file, dtype=np.float32), (1, -1, nb_total_features))
nb_features_rounded = model.num_10ms_times_steps_rounded_to_modem_frames(features_in.shape[1])
features = features_in[:, :nb_features_rounded, :]
features = features[:, :, :num_used_features]
if args.auxdata:
   aux_symb = -np.ones((1, features.shape[1], 1), dtype=np.float32)
   features = np.concatenate([features, aux_symb], axis=2)
features = torch.tensor(features)
print(f"Processing: {nb_features_rounded} feature vectors")

Rs = model.Rs
Nc = model.Nc
num_timesteps_at_rate_Rs = model.num_timesteps_at_rate_Rs(nb_features_rounded)
num_timesteps_at_rate_Fs = model.num_timesteps_at_rate_Fs(num_timesteps_at_rate_Rs)

# default rate Fs multipath model: AWGN (G1=1, G2=0)
G = torch.ones((1, num_timesteps_at_rate_Fs, 2), dtype=torch.complex64)
G[:, :, 1] = 0
if args.g_file:
   G = np.reshape(np.fromfile(args.g_file, dtype=np.csingle), (1, -1, 2))
   mp_gain = np.real(G[:, 0, 0])
   g_offset = int(args.g_offset * model.Fs)
   G = mp_gain * G[:, 1 + g_offset:, :]
   if G.shape[1] < num_timesteps_at_rate_Fs:
      print("Multipath Doppler spread file too short")
      quit()
   G = G[:, :num_timesteps_at_rate_Fs, :]
   G = torch.tensor(G)

# H not used in rate_Fs mode but required by model API
H = torch.ones((1, num_timesteps_at_rate_Rs, Nc))

if __name__ == '__main__':
   model.to(device)
   features = features.to(device)
   H = H.to(device)
   G = G.to(device)
   output = model(features, H, G)

   # print SNR stats
   tx = output["tx"].cpu().detach().numpy()
   S = np.mean(np.abs(tx)**2)
   N = output["sigma"]**2
   N = N.item()
   EbNo = 10**(args.EbNodB/10)
   B = 3000
   CNodB_meas = 10*np.log10(S*model.Fs/N)
   EbNodB_meas = CNodB_meas + 10*np.log10(model.M/(model.Fs*model.Nc*model.bps))
   SNRdB_meas = CNodB_meas - 10*np.log10(B)
   PAPRdB = 20*np.log10(np.max(np.abs(tx))/np.sqrt(S))
   SNR = EbNo*(model.Rb/B)
   SNRdB = 10*np.log10(SNR)
   CNodB = 10*np.log10(EbNo*model.Rb)
   print(f"          Eb/No   C/No     SNR3k  Rb'    Eq     PAPR")
   print(f"Target..: {args.EbNodB:6.2f}  {CNodB:6.2f}  {SNRdB:6.2f}  {int(model.Rb_dash):d}")
   print(f"Measured: {EbNodB_meas:6.2f}  {CNodB_meas:6.2f}  {SNRdB_meas:6.2f}                {PAPRdB:5.2f}")

   # build output IQ with optional EOO and noise padding
   rx = output["rx"]
   sigma = output["sigma"].item()

   if args.end_of_over_v2:
      eoo = model.eoo_v2
      freq = torch.zeros_like(eoo)
      freq[:, ] = model.freq_offset*torch.ones_like(eoo) + model.df_dt*torch.arange(eoo.shape[1])/model.Fs
      omega = freq*2*torch.pi/model.Fs
      lin_phase = torch.cumsum(omega, dim=1)
      lin_phase = torch.exp(1j*lin_phase)
      eoo = eoo*lin_phase*model.final_phase
      eoo = eoo + sigma*torch.randn_like(eoo)
      rx = torch.concatenate([rx, eoo], dim=1)
   if args.prepend_noise > 0.0:
      num_noise = int(model.Fs*args.prepend_noise)
      n = sigma*torch.randn(1, num_noise)
      rx = torch.concatenate([n, rx], dim=1)
   if args.append_noise > 0.0:
      num_noise = int(model.Fs*args.append_noise)
      n = sigma*torch.randn(1, num_noise)
      rx = torch.concatenate([rx, n], dim=1)
   if args.sine_amp > 0.0:
      s = args.sine_amp*torch.exp(1j*torch.arange(rx.shape[1])*2*torch.pi*args.sine_freq/model.Fs)
      rx[0, :] += s

   rx = args.rx_gain*rx.cpu().detach().numpy().flatten().astype('csingle')
   rx.tofile(args.rx_out)

   # write encoder input features for use with loss.py
   feat_out = features[:, :, :num_used_features]
   feat_out = torch.cat([feat_out, torch.zeros_like(feat_out)[:, :, :nb_total_features - num_used_features]], dim=-1)
   feat_out = feat_out.cpu().detach().numpy().flatten().astype('float32')
   feat_out.tofile("features_out.f32")

   if len(args.write_latent):
      z_hat = output["z_hat"].cpu().detach().numpy().flatten().astype('float32')
      z_hat.tofile(args.write_latent)

   if len(args.write_tx):
      tx.flatten().astype('csingle').tofile(args.write_tx)
