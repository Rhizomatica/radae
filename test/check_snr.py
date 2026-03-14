"""
Read radae_rxe.py verbose output from stdin, extract SNRdB values from
lines where state==sync, average the last --tail samples, and check the
result is within --tol dB of --snr3k.  Prints PASS or FAIL.

Usage:
  cat rx.f32 | python3 radae_rxe.py 2>&1 >/dev/null | python3 test/check_snr.py --snr3k 8.24 --tol 3.0
"""

import sys, argparse, re

parser = argparse.ArgumentParser()
parser.add_argument('--snr3k', type=float, required=True, help='expected SNR3k in dB')
parser.add_argument('--tol',   type=float, default=3.0,   help='tolerance in dB (default 3.0)')
parser.add_argument('--tail',  type=int,   default=10,    help='number of trailing sync frames to average (default 10)')
args = parser.parse_args()

snr_values = []
for line in sys.stdin:
    if re.search(r'state:\s+sync', line):
        m = re.search(r'SNRdB:\s*([-0-9.]+)', line)
        if m:
            snr_values.append(float(m.group(1)))

if len(snr_values) < args.tail:
    print(f"FAIL: only {len(snr_values)} sync frames found, need at least {args.tail}")
    sys.exit(1)

avg = sum(snr_values[-args.tail:]) / args.tail
err = abs(avg - args.snr3k)
ok  = err < args.tol

print(f"SNR3k expected: {args.snr3k:.2f} dB  measured (avg last {args.tail}): {avg:.2f} dB  err: {err:.2f} dB  tol: {args.tol:.1f} dB")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
