"""
/*
  RADE V2 shared components.

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

import sys
import numpy as np
import torch


class ComfortNoiseGenerator:
   """Synth low-level noise frames so the speaker doesn't go dead silent
   between decoded frames in lossy / hangover conditions.  Imported by
   radae_rxe2.py and rx2.py main()."""

   HOLD_OUTPUTS = 50

   def __init__(self, dec_stride, num_features, auxdata, enabled=True):
      self.enabled = enabled
      self.dec_stride = dec_stride
      self.num_features = num_features
      self.auxdata = auxdata
      self.emit_parity = 0
      self.remaining_outputs = 0
      self.rng = np.random.default_rng()
      self.profile = np.zeros((dec_stride, num_features), dtype=np.float32)
      self.profile[:, 0] = -5.0
      self.profile[:, 18] = -1.4
      self.profile[:, 19] = -1.0

   def update(self, features_hat_slice, symbol_count):
      frames = features_hat_slice.detach().cpu().numpy()[0].astype(np.float32)
      target = np.array(frames, copy=True)
      target[:, 0] = np.clip(np.minimum(target[:, 0] - 2.0, -4.0), -6.0, -4.0)
      target[:, 18] = -1.4
      target[:, 19] = -1.0
      if self.auxdata and target.shape[1] > 20:
         target[:, 20] = 0.0
      self.profile = (0.96 * self.profile + 0.04 * target).astype(np.float32)
      self.emit_parity = symbol_count & 1
      self.remaining_outputs = self.HOLD_OUTPUTS

   def should_emit(self, symbol_count):
      if not self.enabled:
         return False
      if self.remaining_outputs <= 0:
         return False
      return (symbol_count & 1) == self.emit_parity

   def generate(self):
      frames = np.array(self.profile, copy=True)
      frames[:, 0] = np.clip(
         frames[:, 0] + 0.08 * self.rng.standard_normal(self.dec_stride),
         -6.0, -3.8
      )
      frames[:, 1:6] += 0.02 * self.rng.standard_normal((self.dec_stride, 5))
      frames[:, 18] = -1.4
      frames[:, 19] = np.clip(
         -0.92 + 0.03 * self.rng.standard_normal(self.dec_stride),
         -1.0, -0.75
      )
      if self.auxdata and frames.shape[1] > 20:
         frames[:, 20] = 0.0
      if self.remaining_outputs > 0:
         self.remaining_outputs -= 1
      return frames.astype(np.float32)


class RADEv2Receiver:
   """RADE V2 acquisition and frame sync state machine.

   Processes rate-Fs complex IQ samples using cyclic-prefix autocorrelation
   for timing/frequency estimation and an ML network for frame alignment.
   """

   ALPHA     = 0.95   # Ry_smooth IIR filter coefficient
   BETA      = 0.999  # delta_hat / freq_offset IIR filter coefficient
   TSIG      = 0.38   # signal-detection threshold on |Ry_smooth|
   TSIN      = 4.0    # sine-wave detection ratio threshold
   TEOO      = 0.75   # smoothed pend correlation threshold for EOO detection
   ALPHA_EOO = 0.70   # IIR filter coefficient for EOO pend correlation smoother

   def __init__(self, model, frame_sync_nn, args):
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

      # Autocorrelation (full CP - drives timing/sync)
      self.Ry_norm   = np.zeros(self.sym_len, dtype=np.complex64)
      self.Ry_smooth = np.zeros(self.sym_len, dtype=np.complex64)
      self.az_hat    = None

      # EOO detection: freq-corrected M-sample time domain symbol
      self.rx_sym_td  = np.zeros(self.M, dtype=np.complex64)
      self.eoo_count  = 0   # consecutive pend correlation hits
      self.eoo_smooth = 0.0 # IIR-smoothed pend correlation
      self._eoo_corr  = 0.0 # most recent instantaneous pend correlation

      # BPF bandwidth (matches the filter applied in main before the receiver)
      w = model.w.cpu().detach().numpy()
      self.B_bpf = 1.2 * (w[model.Nc - 1] - w[0]) * self.Fs / (2.0 * np.pi)

      # SNR estimate from CP autocorrelation peak.
      # CP correlator sees noise in B_bpf, not 3000 Hz, so subtract the offset
      # to express snr_est_dB as SNR3k (C/No/3000).
      # Linear correction fitted to minimise error across AWGN, MPG, MPP channels.
      self.snr_offset_dB = 10.0 * np.log10(3000.0 / self.B_bpf)
      self.snr_corr_a = 1.24392558
      self.snr_corr_b = 3.33253932
      self.snr_est_dB = 0.0

   # ------------------------------------------------------------------
   # Helpers
   # ------------------------------------------------------------------

   def _process_symbol(self, rx_in, nin):
      """Run one symbol through gain, buffer, autocorr, detection and state machine.
      Returns (next_state, features_hat, nin, sig_det, sine_det, gain).
      features_hat is the decoded output for the winning frame, or None."""
      gain = self._compute_gain(rx_in)
      self._update_rx_buf(rx_in, nin, gain)
      nin = self.sym_len

      self._compute_autocorr()
      sig_det, sine_det = self._detect_signal()

      next_state = self.state
      features_hat = None
      if self.state == "idle":
         next_state = self._process_idle(sig_det, sine_det)
      elif self.state == "sync":
         next_state, features_hat = self._process_sync(sig_det, sine_det)
         nin = self._adjust_timing(nin)

      return next_state, features_hat, nin, sig_det, sine_det, gain

   def _compute_gain(self, rx_in):
      if not self.args.agc:
         return 1.0
      gain = self.agc_target / (np.sqrt(np.mean(np.abs(rx_in) ** 2)) + 1e-6)
      return float(np.clip(gain, 0.1, 10.0))

   def _update_rx_buf(self, rx_in, nin, gain):
      self.rx_buf[:3 * self.sym_len - nin] = self.rx_buf[nin:]
      self.rx_buf[3 * self.sym_len - nin:] = rx_in * gain

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
      abs_Ry = np.abs(self.Ry_smooth)
      if self.args.fix_delta_hat:
         self.delta_hat_g = self.args.fix_delta_hat
      else:
         self.delta_hat_g = np.int16(np.argmax(abs_Ry))
      self.Ry_max  = abs_Ry[int(self.delta_hat_g)]
      self.Ry_min  = abs_Ry[int(np.argmin(abs_Ry))]
      sig_det  = self.Ry_max > self.TSIG
      sine_det = self.Ry_max / (self.Ry_min + 1e-12) < self.TSIN
      rho = np.clip(np.max(np.abs(self.Ry_smooth)), 0.0, 1.0 - 1e-6)
      snr_raw = 10.0 * np.log10(rho / (1.0 - rho) + 1e-12) - self.snr_offset_dB
      self.snr_est_dB = self.snr_corr_a * snr_raw + self.snr_corr_b
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
         self.eoo_smooth      = 0.0
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

      # Check for end of over
      if self._detect_eoo():
         if self.args.verbose:
            print("EOO detected", file=sys.stderr)
         self.count      = 0
         self.count1     = 0
         self.eoo_count  = 0
         self.eoo_smooth = 0.0
         # reset smoothed autocorrelation to prevent instant re-sync
         self.Ry_smooth = 0
         return "idle", None

      features_hat = self._update_frame_sync(az_hat, sig_det, sine_det)

      return next_state, features_hat

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
      self.rx_sym_td = (self.rx_phase_vec * self.rx_buf[st:en])[self.Ncp:]
      return self.model.receiver(self.rx_i, run_decoder=False)

   def _detect_eoo(self):
      """Detect EOO using channel time-domain sparsity."""
      pend_fd = np.fft.fft(self.model.pend.numpy())
      rx_fd   = np.fft.fft(self.rx_sym_td)
      active  = np.abs(pend_fd) > np.max(np.abs(pend_fd)) * 1e-3
      H_est   = np.zeros(self.M, dtype=np.complex64)
      H_est[active] = rx_fd[active] / pend_fd[active]
      h_est   = np.fft.ifft(H_est)
      e_total = np.sum(np.abs(h_est)**2) + 1e-12
      e_cp    = np.sum(np.abs(h_est[:self.Ncp])**2) + np.sum(np.abs(h_est[-self.Ncp:])**2)
      self._eoo_corr = e_cp / e_total
      self.eoo_smooth = self.ALPHA_EOO * self.eoo_smooth + (1.0 - self.ALPHA_EOO) * self._eoo_corr
      return self.eoo_smooth > self.TEOO

   def _update_frame_sync(self, az_hat, sig_det, sine_det):
      """Update odd/even metrics. Returns decoded features_hat if winning frame, else None."""
      metric = float(self.frame_sync_nn(az_hat)[0, 0, 0].detach())
      gamma  = self.BETA
      winning = False
      if self.s % 2:
         self.frame_sync_odd = gamma * self.frame_sync_odd + (1 - gamma) * metric
         winning = self.frame_sync_odd > self.frame_sync_even
      else:
         self.frame_sync_even = gamma * self.frame_sync_even + (1 - gamma) * metric
         winning = self.frame_sync_even > self.frame_sync_odd
      if winning:
         self.az_hat = az_hat
         features = self.model.core_decoder_statefull(torch.reshape(az_hat, (1, 1, self.model.latent_dim)))
         if self.args.limit_pitch:
            features[:, :, 18].clamp_(min=-1.4)
         if self.args.mute and (not sig_det or sine_det):
            features[:, :, 0] = -5.
         return features
      return None

   def _adjust_timing(self, nin):
      """Shift delta_hat and Ry_smooth to keep timing away from buffer boundaries."""
      if not self.timing_adj or self.args.fix_delta_hat:
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

   def _print_status(self, sig_det, sine_det, nin):
      print(
         f"{self.s:4d} {self.i:4d} {self.state:5s} nin: {nin:3d} "
         f"sig: {int(sig_det):1d} sine: {int(sine_det):1d} "
         f"c: {self.count:2d} nsd: {int(self.new_sig_delta_hat):1d} "
         f"nsf: {int(self.new_sig_f_hat):1d} c1: {self.count1:2d} "
         f"fs: {int(self.frame_sync_odd > self.frame_sync_even):d} "
         f"delta_hat: {self.delta_hat:3.0f} delta_hat_g: {self.delta_hat_g:3.0f} "
         f"f_off: {self.freq_offset:5.2f} f_off_g: {self.freq_offset_g:5.2f} "
         f"Ry_max: {self.Ry_max:5.2f} Ry_min: {self.Ry_min:5.2f} "
         f"snr_est: {self.snr_est_dB:5.1f} dB "
         f"eoo: {self.eoo_smooth:.3f} corr: {self._eoo_corr:.3f}",
         file=sys.stderr
      )
