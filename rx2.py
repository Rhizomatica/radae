"""
/*
  RADE V2 receiver: rate Fs complex samples in, features out.

  No pilots, DSP acquisition, ML frame sync.

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

import os,sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE,complex_bpf
from models_sync import FrameSyncNet


class RADEv2Receiver:
   """RADE V2 acquisition and frame sync state machine.

   Processes rate-Fs complex IQ samples using cyclic-prefix autocorrelation
   for timing/frequency estimation and an ML network for frame alignment.
   """

   ALPHA = 0.95   # Ry_smooth IIR filter coefficient
   BETA  = 0.999  # delta_hat / freq_offset IIR filter coefficient
   TSIG  = 0.38   # signal-detection threshold on |Ry_smooth|
   TSIN  = 4.0    # sine-wave detection ratio threshold

   def __init__(self, model, frame_sync_nn, sequence_length, num_features, args):
      self.model          = model
      self.frame_sync_nn  = frame_sync_nn
      self.args           = args
      self.M              = model.M
      self.Ncp            = model.Ncp
      self.Ns             = model.Ns
      self.Fs             = float(model.Fs)
      self.sym_len        = self.Ncp + self.M

      # target RMS is PAPR (~3 dB) below peak of 1.0
      self.agc_target = 1.0 * 10 ** (-3 / 20)

      # State machine
      self.state      = "idle"
      self.count      = 0
      self.count1     = 0
      self.n_acq      = 0
      self.s          = 0   # symbol counter
      self.i          = 0   # output frame counter
      self.timing_adj = 0

      # Tracking estimates
      self.freq_offset       = 0.0
      self.delta_hat         = 0.0
      self.delta_hat_g       = 0
      self.freq_offset_g     = 0.0
      self.new_sig_delta_hat = False
      self.new_sig_f_hat     = False
      self.Ry_max            = 0.0
      self.Ry_min            = 0.0

      # Frame-sync discriminators (odd/even alignment)
      self.frame_sync_even = 0.0
      self.frame_sync_odd  = 0.0

      # Buffers
      self.rx_buf       = np.zeros(3 * self.sym_len, dtype=np.complex64)
      self.rx_i         = torch.zeros(self.Ns * self.sym_len, dtype=torch.complex64)
      self.rx_phase     = 1 + 1j * 0
      self.rx_phase_vec = np.zeros(self.sym_len, dtype=np.csingle)

      # Autocorrelation
      self.Ry_norm   = np.zeros(self.sym_len, dtype=np.complex64)
      self.Ry_smooth = np.zeros(self.sym_len, dtype=np.complex64)

      # Output latent vectors and decoded features
      self.z_hat        = torch.zeros((1, sequence_length, model.latent_dim), dtype=torch.float32)
      self.features_hat = torch.zeros((1, sequence_length * model.dec_stride, num_features))

      # Diagnostic logs (indexed by symbol number)
      sl = sequence_length
      self.state_log       = np.zeros(sl, dtype=np.int16)
      self.frame_sync_log  = np.zeros((sl, 2), dtype=np.float32)
      self.Ry_norm_log     = np.zeros((sl, self.sym_len), dtype=np.complex64)
      self.Ry_smooth_log   = np.zeros((sl, self.sym_len), dtype=np.complex64)
      self.sig_det_log     = np.zeros(sl, dtype=np.int16)
      self.delta_hat_log   = np.zeros(sl, dtype=np.float32)
      self.freq_offset_log = np.zeros(sl, dtype=np.float32)
      self.gain_log        = np.zeros(sl, dtype=np.float32)

   # ------------------------------------------------------------------
   # Public interface
   # ------------------------------------------------------------------

   def run(self, rx):
      """Process rx sample array. Returns z_hat (1, n_frames, latent_dim)."""
      nin = self.sym_len
      prx = 0

      while prx + nin < len(rx):
         self.s += 1
         st, en = prx, prx + nin
         prx   += nin

         prev_state = self.state
         next_state, az_hat, nin, sig_det, sine_det, gain = self._process_symbol(rx, st, en, nin)

         self._update_logs(sig_det, gain)

         if self.args.verbose or self.state != prev_state:
            self._print_status(sig_det, sine_det, nin)

         self.state = next_state

         if az_hat is not None:
            self.z_hat[0, self.i, :] = az_hat
            dec_st = self.model.dec_stride * self.i
            dec_en = self.model.dec_stride * (self.i + 1)
            az_hat = torch.reshape(az_hat,(1,1,model.latent_dim))
            self.features_hat[0, dec_st:dec_en, :] = self.model.core_decoder_statefull(az_hat)
            self.i += 1

         if self.s > self.args.timing_adj_at:
            self.timing_adj = 1

         if self.s == self.args.stop_at:
            quit()

      return self.z_hat[:, :self.i, :], self.features_hat[:, :self.i * self.model.dec_stride, :]

   # ------------------------------------------------------------------
   # Private helpers
   # ------------------------------------------------------------------

   def _process_symbol(self, rx, st, en, nin):
      """Run one symbol through gain, buffer, autocorr, detection and state machine.
      Returns (next_state, az_hat, nin, sig_det, sine_det)."""
      gain = self._compute_gain(rx, st, en)
      self._update_rx_buf(rx, st, en, nin, gain)
      nin = self.sym_len

      self._compute_autocorr()
      sig_det, sine_det = self._detect_signal()

      next_state = self.state
      az_hat = None
      if self.state == "idle":
         next_state = self._process_idle(sig_det, sine_det)
      elif self.state == "sync":
         next_state, az_hat = self._process_sync(sig_det, sine_det)
         nin = self._adjust_timing(nin)

      return next_state, az_hat, nin, sig_det, sine_det, gain

   def _compute_gain(self, rx, st, en):
      if not self.args.agc:
         return 1.0
      gain = self.agc_target / (np.sqrt(np.mean(np.abs(rx[st:en]) ** 2)) + 1e-6)
      return float(np.clip(gain, 0.1, 10.0))

   def _update_rx_buf(self, rx, st, en, nin, gain):
      self.rx_buf[:3 * self.sym_len - nin] = self.rx_buf[nin:]
      self.rx_buf[3 * self.sym_len - nin:] = rx[st:en] * gain

   def _compute_autocorr(self):
      M, Ncp, sym_len = self.M, self.Ncp, self.sym_len
      for gamma in range(sym_len):
         idx  = sym_len + gamma
         y_cp = self.rx_buf[idx - Ncp : idx]
         y_m  = self.rx_buf[idx - Ncp + M : idx + M]
         Ry   = np.dot(y_cp, np.conj(y_m))
         D    = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m)) + 1e-12
         self.Ry_norm[gamma] = 2.0 * Ry / np.abs(D)
      self.Ry_smooth = self.ALPHA * self.Ry_smooth + (1.0 - self.ALPHA) * self.Ry_norm

   def _detect_signal(self):
      abs_Ry           = np.abs(self.Ry_smooth)
      self.delta_hat_g = np.int16(np.argmax(abs_Ry))
      self.Ry_max      = abs_Ry[int(self.delta_hat_g)]
      self.Ry_min      = abs_Ry[int(np.argmin(abs_Ry))]
      sig_det  = self.Ry_max > self.TSIG
      sine_det = self.Ry_max / (self.Ry_min + 1e-12) < self.TSIN
      return sig_det, sine_det

   def _process_idle(self, sig_det, sine_det):
      if sig_det and not sine_det:
         self.count += 1
      else:
         self.count = 0

      if self.count == 5:
         delta_phi            = np.angle(self.Ry_smooth[int(self.delta_hat_g)])
         self.delta_hat       = self.delta_hat_g
         self.freq_offset     = -delta_phi * self.Fs / (2.0 * np.pi * self.M)
         self.count           = 0
         self.count1          = 0
         self.frame_sync_even = 0.0
         self.frame_sync_odd  = 0.0
         if self.args.reset_output_on_resync:
            self.i = 0
         self.n_acq += 1
         return "sync"

      return "idle"

   def _process_sync(self, sig_det, sine_det):
      next_state = "sync"

      # IIR-track timing and frequency offset
      delta_phi          = np.angle(self.Ry_smooth[self.delta_hat_g])
      self.freq_offset_g = -delta_phi * self.Fs / (2.0 * np.pi * self.M)
      self.delta_hat     = self.BETA * self.delta_hat + (1.0 - self.BETA) * self.delta_hat_g
      self.freq_offset   = self.BETA * self.freq_offset + (1.0 - self.BETA) * self.freq_offset_g

      # Check for sustained signal loss -> return to idle
      if not sig_det or sine_det:
         self.count += 1
      else:
         self.count = 0
      if self.count == self.args.hangover:
         next_state  = "idle"
         self.count  = 0
         self.count1 = 0

      # Check for a new/different signal -> re-acquire
      self.new_sig_delta_hat = np.abs(self.delta_hat_g - self.delta_hat) > self.Ncp
      self.new_sig_f_hat     = np.abs(self.freq_offset_g - self.freq_offset) > 5.0
      if sig_det and (self.new_sig_delta_hat or self.new_sig_f_hat):
         self.count1 += 1
      else:
         self.count1 = 0
      if self.count1 == 5:
         next_state  = "idle"
         self.count  = 0
         self.count1 = 0

      # Extract symbol and update frame sync (even when transitioning to idle)
      az_hat = self._extract_symbol()
      winning_az_hat = self._update_frame_sync(az_hat)

      return next_state, winning_az_hat

   def _extract_symbol(self):
      """Frequency-correct and extract one OFDM symbol; return latent z_hat."""
      delta_hat_rx = int(self.delta_hat - self.Ncp)
      omega = 2.0 * np.pi * self.freq_offset / self.Fs
      for n in range(self.sym_len):
         self.rx_phase        = self.rx_phase * np.exp(-1j * omega)
         self.rx_phase_vec[n] = self.rx_phase
      st = self.sym_len + delta_hat_rx
      en = st + self.sym_len
      self.rx_i[:self.sym_len] = self.rx_i[self.sym_len:]
      self.rx_i[self.sym_len:] = torch.tensor(
         self.rx_phase_vec * self.rx_buf[st:en], dtype=torch.complex64
      )
      return self.model.receiver(self.rx_i, run_decoder=False)

   def _update_frame_sync(self, az_hat):
      """Update odd/even metrics. Returns az_hat if it is the winning frame, else None."""
      metric = float(self.frame_sync_nn(az_hat)[0, 0, 0])
      gamma  = self.BETA
      if self.s % 2:
         self.frame_sync_odd = gamma * self.frame_sync_odd + (1 - gamma) * metric
         if self.frame_sync_odd > self.frame_sync_even:
            return az_hat
      else:
         self.frame_sync_even = gamma * self.frame_sync_even + (1 - gamma) * metric
         if self.frame_sync_even > self.frame_sync_odd:
            return az_hat
      return None

   def _adjust_timing(self, nin):
      """Shift delta_hat and Ry_smooth to keep timing away from buffer boundaries."""
      if not self.timing_adj:
         return nin
      shift = self.sym_len // 4
      if self.delta_hat > 3 * self.sym_len // 4:
         self.delta_hat -= shift
         tmp = np.array(self.Ry_smooth[:shift])
         self.Ry_smooth[:self.sym_len - shift] = self.Ry_smooth[shift:]
         self.Ry_smooth[self.sym_len - shift:]  = tmp
         nin = self.sym_len + shift
      if self.delta_hat < self.sym_len // 4:
         self.delta_hat += shift
         tmp = np.array(self.Ry_smooth[self.sym_len - shift:])
         self.Ry_smooth[shift:]  = self.Ry_smooth[:self.sym_len - shift]
         self.Ry_smooth[:shift]  = tmp
         nin = self.sym_len - shift
      return nin

   def _update_logs(self, sig_det, gain):
      s = self.s
      if s >= len(self.state_log):
         return
      self.state_log[s]         = 0 if self.state == "idle" else 1
      self.gain_log[s]          = gain
      self.Ry_norm_log[s]       = self.Ry_norm
      self.Ry_smooth_log[s]     = self.Ry_smooth
      self.sig_det_log[s]       = sig_det
      self.delta_hat_log[s]     = self.delta_hat
      self.freq_offset_log[s]   = self.freq_offset
      self.frame_sync_log[s, 0] = self.frame_sync_even
      self.frame_sync_log[s, 1] = self.frame_sync_odd

   def _print_status(self, sig_det, sine_det, nin):
      print(
         f"{self.s:4d} {self.i:4d} {self.state:5s} nin: {nin:3d} "
         f"sig: {int(sig_det):1d} sine: {int(sine_det):1d} "
         f"c: {self.count:2d} nsd: {int(self.new_sig_delta_hat):1d} "
         f"nsf: {int(self.new_sig_f_hat):1d} c1: {self.count1:2d} "
         f"fs: {int(self.frame_sync_odd > self.frame_sync_even):d} "
         f"delta_hat: {self.delta_hat:3.0f} delta_hat_g: {self.delta_hat_g:3.0f} "
         f"f_off: {self.freq_offset:5.2f} f_off_g: {self.freq_offset_g:5.2f} "
         f"Ry_max: {self.Ry_max:5.2f} Ry_min: {self.Ry_min:5.2f}",
         file=sys.stderr
      )


parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to RADE model in .pth format')
parser.add_argument('frame_sync_model_name', type=str, help='path to frame sync model in .pth format')
parser.add_argument('rx', type=str, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--bottleneck', type=int, default=3, help='1-1D rate Rs, 2-2D rate Rs, 3-2D rate Fs time domain (default 3)')
parser.add_argument('--cp', type=float, default=0.004, help='Length of cyclic prefix in seconds [--Ncp..0], (default 0.04)')
parser.add_argument('--no_bpf', action='store_false', dest='bpf', help='disable BPF')
parser.add_argument('--freq_offset', type=float, default=0, help='correct for this frequency offset')
parser.add_argument('--time_offset', type=int, default=-16, help='time domain sampling time offset in samples')
parser.add_argument('--correct_time_offset', type=int, default=-16, help='introduces a delay (or advance if -ve) in samples, applied in freq domain (default 0)')
parser.add_argument('--plots', action='store_true', help='display various plots')
parser.add_argument('--acq_test',  action='store_true', help='Acquisition test mode')
parser.add_argument('--acq_time_target', type=float, default=1.0, help='Acquisition test mode mean acquisition time target (default 1.0)')
parser.add_argument('--stateful',  action='store_true', help='use stateful core decoder')
parser.add_argument('--xcorr_dimension', type=int, help='Dimension of Input cross-correlation (fine timing)',default = 160,required = False)
parser.add_argument('--gru_dim', type=int, help='GRU Dimension (fine timing)',default = 64,required = False)
parser.add_argument('--output_dim', type=int, help='Output dimension (fine timing)',default = 160,required = False)
parser.add_argument('--write_Ry_smooth', type=str, default="", help='path to smoothed autocorrelation output feature file dim (seq_len,Ncp+M) .c64 format')
parser.add_argument('--write_delta_hat', type=str, default="", help='path to delta_hat output file dim (seq_len) in .int16 format')
parser.add_argument('--write_Ry_max', type=str, default="", help='path to Ty_max output file dim (seq_len) in .f32 format')
parser.add_argument('--write_sig_det', type=str, default="", help='path to signal detection flag output file dim (seq_len) in .int16 format')
parser.add_argument('--write_freq_offset', type=str, default="", help='path to freq offset est output file dim (seq_len) in .float32 format')
parser.add_argument('--write_delta_hat_rx', type=str, default="", help='path to delta_hat_rx file dim (seq_len) in .f32 format')
parser.add_argument('--write_state', type=str, default="", help='path to sync state machine output file dim (seq_len) in .int16 format')
parser.add_argument('--write_frame_sync', type=str, default="", help='path to frame sync output file dim (seq_len,2) in .int16 format')
parser.add_argument('--read_delta_hat', type=str, default="", help='path to delta_hat input file dim (seq_len) in .f32 format')
parser.add_argument('--fix_delta_hat', type=int,  default=0, help='disable timing estimation and used fixed delta_hat (default: use timing estimation)')
parser.add_argument('--write_gain', type=str, default="", help='path to AGC output file dim (seq_len) .f32 format')
parser.set_defaults(bpf=True)
parser.set_defaults(auxdata=True)
parser.set_defaults(verbose=True)
parser.add_argument('--pad_samples', type=int, default=0, help='Pad input with samples to simulate different timing offsets in rx signal')
parser.add_argument('--gain', type=float, default=1.0, help='manual gain control')
parser.add_argument('--agc', action='store_true', help='automatic gain control')
parser.add_argument('--w1_dec', type=int, default=96, help='Decoder GRU output dimension (default 96)')
parser.add_argument('--nofreq_offset', action='store_true', help='disable freq offset correction (default enabled)')
parser.add_argument('--test_mode', action='store_true', help='inject test delta sequence')
parser.add_argument('--hangover', type=int, default=75, help='Number of symbols of no signal before returning to noise state (default 75)')
parser.add_argument('--quiet', action='store_false', dest='verbose', help='inject test delta sequence')
parser.add_argument('--verbose', action='store_true', dest='verbose', help='inject test delta sequence')
parser.add_argument('--stop_at', type=int, default=0, help='exit program after this many symbols (default disabled)')
parser.add_argument('--timing_adj_at', type=int, default=0, help='enable timing adjust after this many symbols (default disabled)')
parser.add_argument('--reset_output_on_resync', action='store_true', help='only keep output from last resync (default disabled)')
parser.set_defaults(limit_pitch=True)
parser.add_argument('--nolimit_pitch', action='store_false', dest='limit_pitch', help='disable limiting (clip) lower end of pitch feature to prevent synthesis pops with some speakers/channels (default enabled)')
args = parser.parse_args()

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20
if args.auxdata:
    num_features += 1

# load RADE model
model = RADAE(num_features, latent_dim, EbNodB=100, Nzmf = 1,
              rate_Fs=True, bottleneck=args.bottleneck, cyclic_prefix=args.cp,
              time_offset=args.time_offset, correct_time_offset=args.correct_time_offset,
              stateful_decoder=args.stateful, w1_dec=args.w1_dec, w1_dec_stateful=args.w1_dec)
checkpoint = torch.load(args.model_name, map_location='cpu', weights_only=True)

# model was trained with core_decoder_stateful with different w1_dec_stateful.  So we remove
# mismatched size entries from dictionary so we can load model
state_dict = checkpoint['state_dict']
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)
model.core_decoder_statefull_load_state_dict()

model.eval()

# Load sync model
frame_sync_nn = FrameSyncNet(latent_dim)
frame_sync_nn.load_state_dict(torch.load(args.frame_sync_model_name,weights_only=True,map_location=torch.device('cpu')))
frame_sync_nn.eval()

M = model.M
Ncp = model.Ncp
Ns = model.Ns           # number of rate Rs symbols per modem frame
Nmf = int(Ns*(M+Ncp))   # number of samples in one modem frame
Nc = model.Nc
w = model.w.cpu().detach().numpy()
Fs = float(model.Fs)

# load rx rate_Fs samples
rx = np.fromfile(args.rx, dtype=np.csingle)*args.gain
w_off = 2*np.pi*args.freq_offset/Fs
rx = rx*np.exp(-1j*w_off*np.arange(len(rx)))

# ensure an integer number of frames
rx = np.concatenate((np.zeros(args.pad_samples, dtype=np.complex64),rx))

rx = rx[:Nmf*(len(rx)//Nmf)]
print(f"samples: {len(rx):d} Nmf: {Nmf:d} modem frames: {len(rx)//Nmf}")

# TODO: fix contrast of spectrogram - it's not very useful
if args.plots:
   fig, ax = plt.subplots(2, 1,figsize=(6,12))
   ax[0].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[0].set_title('Before BPF')
   ax[0].axis([0,len(rx)/model.Fs,0,3000])

# BPF to remove some of the noise and improve acquisition
Ntap = 0
if args.bpf:
   Ntap=101
   bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
   centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}")
   bpf = complex_bpf(Ntap, model.Fs, bandwidth, centre, len(rx))
   rx = bpf.bpf(rx)

if args.plots:
   ax[1].specgram(rx,NFFT=256,Fs=model.Fs)
   ax[1].axis([0,len(rx)/model.Fs,0,3000])
   ax[1].set_title('After BPF')
   plt.show(block=False)
   plt.pause(0.001)
   input("hit[enter] to end.")
   plt.close('all')

# Acquisition - timing, freq offset, and signal present estimates

sequence_length = len(rx)//(Ncp+M)
print(sequence_length)

receiver = RADEv2Receiver(model, frame_sync_nn, sequence_length, num_features, args)
z_hat, features_hat = receiver.run(rx)

if len(args.write_Ry_smooth):
   receiver.Ry_smooth_log.flatten().tofile(args.write_Ry_smooth)
if len(args.write_delta_hat):
   receiver.delta_hat_log.tofile(args.write_delta_hat)
if len(args.write_sig_det):
   receiver.sig_det_log.tofile(args.write_sig_det)
if len(args.write_freq_offset):
   receiver.freq_offset_log.tofile(args.write_freq_offset)
if len(args.write_gain):
   receiver.gain_log.tofile(args.write_gain)
if len(args.write_state):
   receiver.state_log.tofile(args.write_state)
if len(args.write_frame_sync):
   receiver.frame_sync_log.flatten().tofile(args.write_frame_sync)

rx = np.concatenate((rx,np.zeros(Ncp+M,dtype=np.complex64)))
z_hat.shape
print(f"n_acq: {receiver.n_acq:d}",file=sys.stderr)
print(f"latent vectors: {z_hat.shape[1]:d}",file=sys.stderr)

features_hat_out = np.zeros(0)
if z_hat.shape[1]:
   # limiting the lower end of the pitch feature removed pops
   if args.limit_pitch:
      features_hat[:,:,18].clamp_(min= -1.4)

   features_hat_out = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:nb_total_features-num_features]], dim=-1)
   features_hat_out = features_hat_out.cpu().detach().numpy().flatten().astype('float32')
features_hat_out.tofile(args.features_hat)

if len(args.write_latent):
   z_hat.cpu().detach().numpy().flatten().astype('float32').tofile(args.write_latent)
