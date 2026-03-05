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

# target RMS level is PAPR ~ 3 dB less than peak of 1.0
agc_target = 1.0*10**(-3/20)

alpha = 0.95
beta = 0.999
Tsig = 0.38
Tsin = 4.

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


# TODO: refactor, reorg all these variables and constants
# TODO: rationalise filter constants
# TODO: make logs optional, not part of core code
state = "idle"
count = 0
count1 = 0
state_log = np.zeros(sequence_length,dtype=np.int16)

frame_sync_log = np.zeros((sequence_length,2),dtype=np.float32)
frame_sync_even = 0.
frame_sync_odd = 0.

# off air samples for i-th frame
rx_i = torch.zeros((Ns*(Ncp+M)),dtype=torch.complex64)

rx_phase = 1 + 1j*0
rx_phase_vec = np.zeros(Ncp+M,np.csingle)

z_hat = torch.zeros((1,sequence_length, model.latent_dim), dtype=torch.float32)

i = 0
n_acq = 0

Ry_norm = np.zeros((Ncp+M),dtype=np.complex64)
Ry_norm_log = np.zeros((sequence_length,Ncp+M),dtype=np.complex64)
Ry_smooth = np.zeros((Ncp+M),dtype=np.complex64)
Ry_smooth_log = np.zeros((sequence_length,Ncp+M),dtype=np.complex64)

# these need to be floats as we IIR filter them over time
freq_offset = 0.0
delta_hat = 0.
freq_offset_log = np.zeros(sequence_length,dtype=np.float32)
delta_hat_log = np.zeros(sequence_length,dtype=np.float32)

gain_log = np.zeros(sequence_length,dtype=np.float32)

sig_det = 0
sig_det_log = np.zeros(sequence_length,dtype=np.int16)
delta_hat_g = 0.
freq_offset_g = 0.
new_sig_delta_hat = 0
new_sig_f_hat = 0

nin = Ncp+M
prx = 0
s = 0
timing_adj = 0

# M unused samples at start of buffer, but allows us to use index convention from write up
rx_buf = np.zeros(3*(Ncp+M),dtype=np.complex64)

while prx + nin < len(rx):
   s += 1
   st = prx
   en = st + nin
   prx += nin
   
   gain = 1.0
   if args.agc:
      gain = agc_target/(np.sqrt(np.mean(np.abs(rx[st:en])**2)) + 1E-6)
      gain = min(gain,10.0)
      gain = max(gain,0.1)
   rx_buf[:3*(Ncp+M)-nin] = rx_buf[nin:]
   rx_buf[3*(Ncp+M)-nin:] = rx[st:en]*gain
   nin = Ncp+M

   # Normalised autocorrelation function
   for gamma in np.arange(Ncp+M):
      st = Ncp+M+gamma
      y_cp = rx_buf[st-Ncp:st]
      y_m = rx_buf[st-Ncp+M:st+M]
      Ry = np.dot(y_cp, np.conj(y_m))
      D = np.dot(y_cp, np.conj(y_cp)) + np.dot(y_m, np.conj(y_m)) + 1E-12
      Ry_norm[gamma] = 2.*Ry/np.abs(D)

   # IIR smoothing
   Ry_smooth = Ry_smooth*alpha + Ry_norm*(1.-alpha)

   # Iterate acquisition (sync) state machine

   prev_state = state
   next_state = state

   delta_hat_g = np.int16(np.argmax(np.abs(Ry_smooth)))
   Ry_max = np.abs(Ry_smooth[int(delta_hat_g)])
   delta_hat_g1 = np.int16(np.argmin(np.abs(Ry_smooth)))
   Ry_min = np.abs(Ry_smooth[int(delta_hat_g1)])
   sig_det = Ry_max > Tsig
   sine_det = Ry_max/(Ry_min+1E-12) < Tsin
   
   if state == "idle":

      if sig_det and not sine_det:
         count += 1
      else:
         count = 0

      if count == 5:
         next_state = "sync"
         delta_hat = delta_hat_g
         delta_phi = np.angle(Ry_smooth[int(delta_hat_g)])
         freq_offset = -delta_phi*Fs/(2.*np.pi*M)
         count = 0
         count1 = 0
         if args.reset_output_on_resync:
            i = 0
         frame_sync_even = 0.
         frame_sync_odd = 0.
         n_acq += 1

   if state == "sync":

      delta_phi = np.angle(Ry_smooth[delta_hat_g])
      freq_offset_g = -delta_phi*Fs/(2.*np.pi*M)

      delta_hat = beta*delta_hat + (1.-beta)*delta_hat_g
      freq_offset = beta*freq_offset + (1.-beta)*freq_offset_g

      # if no signal of a sine wave we may have lost of RADE signal
      if not sig_det or sine_det:
         count += 1
      else:
         count = 0
      # hangover quite long so we can ride over fades (where sig_det will be patchy) without a re-sync
      if count == args.hangover:
         next_state = "idle"
         count = 0
         count1 = 0

      # trap consistent gross errors, e.g. signal1 closely followed by signal2, or a false sync corner case
      new_sig_delta_hat = np.abs(delta_hat_g - delta_hat) > Ncp
      new_sig_f_hat = np.abs(freq_offset_g -  freq_offset) > 5.
      if sig_det and (new_sig_delta_hat or new_sig_f_hat):
         count1 += 1
      else:
         count1 = 0
      
      # if a new signal is detected we can un-sync quickly
      if count1 == 5:
         next_state = "idle"
         count = 0
         count1 = 0
   
      # adjust timing to point to start of symbol
      delta_hat_rx = int(delta_hat-Ncp)

      # set up phase continous vector to correct freq offset
      freq_offset_rx = freq_offset
      w = 2*np.pi*freq_offset_rx/Fs
      for n in range(Ncp+M):
         rx_phase = rx_phase*np.exp(-1j*w)
         rx_phase_vec[n] = rx_phase

      # extract symbol into end of i-th frame
      st = Ncp+M + delta_hat_rx
      en = st + Ncp+M
      rx_i[:Ncp+M] = rx_i[Ncp+M:]
      rx_i[Ncp+M:] = torch.tensor(rx_phase_vec*rx_buf[st:en], dtype=torch.complex64)
      # run receiver to extract i-th freq domain OFDM symbols z_hat for one frame
      # Note this is run at symbol rate (twice frame rate) so we can get odd and even stats
      az_hat = model.receiver(rx_i,run_decoder=False)

      # update odd and even frame sync metrics
      frame_sync_metric_torch = frame_sync_nn(az_hat)
      frame_sync_metric = float(frame_sync_metric_torch[0,0,0])
      
      gamma = beta
      if s % 2:
         # odd frame alignment
         frame_sync_odd = gamma*frame_sync_odd + (1-gamma)*frame_sync_metric
         if frame_sync_odd > frame_sync_even:
            z_hat[0,i,:] = az_hat
            i += 1
      else:
         # even frame alignment
         frame_sync_even = gamma*frame_sync_even + (1-gamma)*frame_sync_metric
         if frame_sync_even > frame_sync_odd:
            z_hat[0,i,:] = az_hat
            i += 1
   
      if timing_adj:
         # keep timing centered in symbol to avoid wrap around issues near 0 or Ncp+M
         # which would cause loss of frame sync
         shift = (Ncp+M)//4
         if delta_hat > 3*(Ncp+M)//4:
            delta_hat -= shift
            # rotate Ry_smooth towards 0
            tmp = np.array(Ry_smooth[:shift])
            Ry_smooth[:Ncp+M-shift] = Ry_smooth[shift:]
            Ry_smooth[Ncp+M-shift:] = tmp
            nin = Ncp+M + shift
         if delta_hat < (Ncp+M)//4:
            delta_hat += shift
            # rotate Ry_smooth towards Ncp+M
            tmp = np.array(Ry_smooth[Ncp+M-shift:])
            Ry_smooth[shift:] = Ry_smooth[:Ncp+M-shift]
            Ry_smooth[:shift] = tmp
            nin = Ncp+M - shift
      
   if s > args.timing_adj_at:
      timing_adj = 1

   if s < sequence_length:
      if state == "idle":
         state_log[s] = 0
      else:
         state_log[s] = 1
      gain_log[s] = gain
      Ry_norm_log[s,:] = Ry_norm
      Ry_smooth_log[s,:] = Ry_smooth
      sig_det_log[s] = sig_det
      delta_hat_log[s] = delta_hat
      freq_offset_log[s] = freq_offset
      frame_sync_log[s,0] = frame_sync_even
      frame_sync_log[s,1] = frame_sync_odd

   if args.verbose or state != prev_state:
      print(f"{s:4d} {i:4d} {state:5s} nin: {nin:3d} sig: {sig_det:1d} sine: {sine_det:1d} c: {count:2d} nsd: {new_sig_delta_hat:1d} nsf: {new_sig_f_hat:1d} c1: {count1:2d} ", end='', file=sys.stderr)
      print(f"fs: {frame_sync_odd > frame_sync_even:d} ", end='', file=sys.stderr)
      print(f"delta_hat: {delta_hat:3.0f} delta_hat_g: {delta_hat_g:3.0f} ", end='',file=sys.stderr)
      print(f"f_off: {freq_offset:5.2f} f_off_g: {freq_offset_g:5.2f} Ry_max: {Ry_max:5.2f} Ry_min: {Ry_min:5.2f}", file=sys.stderr)

   state = next_state 

   if s == args.stop_at:
      quit()

# truncate from max length
z_hat = z_hat[:,:i,:]

if len(args.write_Ry_smooth):
   Ry_smooth_log.flatten().tofile(args.write_Ry_smooth)
if len(args.write_delta_hat):
   delta_hat_log.tofile(args.write_delta_hat)
if len(args.write_sig_det):
   sig_det_log.tofile(args.write_sig_det)
if len(args.write_freq_offset):
   freq_offset_log.tofile(args.write_freq_offset)
if len(args.write_gain):
   gain_log.tofile(args.write_gain)
if len(args.write_state):
   state_log.tofile(args.write_state)
if len(args.write_frame_sync):
   frame_sync_log.flatten().tofile(args.write_frame_sync)

rx = np.concatenate((rx,np.zeros(Ncp+M,dtype=np.complex64)))
z_hat.shape
print(f"n_acq: {n_acq:d}",file=sys.stderr)
print(f"latent vectors: {z_hat.shape[1]:d}",file=sys.stderr)

features_hat = np.zeros(0)
if z_hat.shape[1]:
   # run RADE decoder
   features_hat = torch.zeros((1,z_hat.shape[1]*model.dec_stride,num_features))

   features_hat = torch.zeros_like(features_hat)
   for i in range(z_hat.shape[1]):
      features_hat[0,model.dec_stride*i:model.dec_stride*(i+1),:] = model.core_decoder_statefull(z_hat[:,i:i+1,:])

   # limiting the lower end of the pitch feature removed pops 
   if args.limit_pitch:
      features_hat[:,:,18].clamp_(min= -1.4)

   features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:nb_total_features-num_features]], dim=-1)
   features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
features_hat.tofile(args.features_hat)

if len(args.write_latent):
   z_hat.cpu().detach().numpy().flatten().astype('float32').tofile(args.write_latent)

