#!/usr/bin/env python3
"""
Check that the mean snr_est_dB from rx2.py matches a target SNR3k within tolerance.

Usage: check_snr_est.py <snr3k_target_dB> <snr_est.f32> [tol_dB]

snr3k_target_dB : SNR3k in dB from the "Measured:" line of inference.py
snr_est.f32     : file written by rx2.py --write_snr_est
tol_dB          : pass/fail tolerance in dB (default 2.0)
"""
import sys
import numpy as np

snr3k_target = float(sys.argv[1])
snr_est_file = sys.argv[2]
tol_dB       = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

d = np.fromfile(snr_est_file, dtype=np.float32)

# Keep only samples where the receiver was in sync (snr_est > target - 6 dB)
d = d[d > snr3k_target - 6.0]

if len(d) == 0:
    print(f"FAIL: no valid snr_est samples above {snr3k_target - 6:.1f} dB")
    sys.exit(1)

snr_est_mean = float(np.mean(d))
err = snr_est_mean - snr3k_target
ok  = abs(err) <= tol_dB

print(f"SNR3k target: {snr3k_target:.2f} dB  snr_est mean: {snr_est_mean:.2f} dB  "
      f"error: {err:+.2f} dB  tol: ±{tol_dB:.1f} dB  {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
