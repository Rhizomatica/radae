# Pure-C RADE V2 RX migration

## Why

Production RX on sBitx currently runs `radae_rxe2.py` as a Python
subprocess fed modem IQ on stdin.  Python + PyTorch is the bottleneck:
the Pi 4 runs the Python path at ~0.9× realtime, forcing us to use a
Pi 5 (estacao2) for the receive side of every DV test.

Benchmarking on estacao3 (Pi 4, single Cortex-A72 core) shows the C
`rade_core_decoder` in `librade.so` runs at **~38× realtime** on the
250725 weights (see `memory/pi4_c_decoder_bench.md`).  FARGAN alone is
~12× RT on the same core.  The arithmetic says a pure-C RX stacks to
roughly 9× RT end-to-end — easily realtime on Pi 4.

The remaining block is the DSP + metadata plumbing that `rx2.py` does
in Python: OFDM demod, pilot tracking, timing sync, frame-sync neural
net, EOO detect, AGC, band-pass filter.

## What the Python RX does (what we're replacing)

`sbitx_radae.c` forks a subprocess:

```
python3 -u radae_rxe2.py --model_name 250725/.../epoch_200.pth \
                        --frame_sync_model_name 250725a_ml_sync
```

IQ flows in on stdin, decoded FARGAN features flow out on stdout.
Internally `radae_rxe2.py` instantiates `RADEv2Receiver` (from
`rx2.py`, ~550 LoC) which does:

- complex band-pass filter (`complex_bpf`)
- per-block AGC
- coarse + fine timing search on pilot symbols
- OFDM symbol demod + cyclic-prefix strip
- pilot-tracked phase/frequency correction
- frame alignment via `FrameSyncNet` (3-layer MLP, 56→64→64→1)
- EOO correlation detector with IIR smoothing
- latent-vector (`z_hat`) handoff to `rade_core_decoder` (already in C)
- comfort noise injection when signal is absent (`ComfortNoiseGenerator`)

All of that except the final `rade_core_decoder` currently runs in
Python.

## Target

`sbitx_radae.c` forks a new C binary (or calls into `librade.so`
directly):

```
radae_rx_v2 [--model path] [--fsync path]
```

Same IQ-in / features-out contract, no Python, no PyTorch import.
Eventually the subprocess model goes away too — `sbitx_controller`
links `librade.so` and calls `rade_rx_v2()` in-process, which also
removes the 13 s service-startup warmup.

## Staging

Each step is independently verifiable and leaves the production
Python path untouched.

### Step 1 — Port `FrameSyncNet` to C (in progress)

Smallest standalone piece.  Input is one 56-float latent vector, output
is a single sigmoid in [0, 1].  Weights export from the
`250725a_ml_sync` PyTorch checkpoint.

Deliverables:
- `export_frame_sync_weights.py` — dumps weights into a generated C
  header.
- `src/frame_sync.c` / `src/frame_sync.h` — `float frame_sync_forward(const float z_hat[])`.
- `src/frame_sync_data.c` — generated weight blob.
- `src/test_frame_sync.c` — reads (input, expected_output) pairs from
  stdin and asserts `|C_out - Python_out| < 1e-5`.

Acceptance: bit-accurate match against PyTorch on 1000 random inputs,
built into the existing CMake target list.

### Step 2 — Port `ComfortNoiseGenerator` to C

Small stateful generator.  Not on the hot path but needed for parity
with `radae_rxe2.py`.  ~80 LoC.

### Step 3 — Port `RADEv2Receiver` DSP to C

The bulk of the work.  Split further into:

3a. `complex_bpf` (Kaiser FIR, steady-state tap memory) — ~150 LoC C.
3b. AGC + coarse timing correlator (`Ry_norm`, `Ry_smooth`) — ~100 LoC.
3c. OFDM demod + CP strip + FFT per symbol — ~200 LoC.
3d. Pilot-tracked phase/freq correction — ~150 LoC.
3e. Frame-sync integration (uses Step 1 net) — ~100 LoC.
3f. EOO correlation detector — ~100 LoC.
3g. Resync state machine + `nin` adjust (`_adjust_timing`) — ~150 LoC.

Each sub-piece gets a unit test feeding a frozen Python-generated
reference trace.  `test_vectors/*.bin` is committed alongside.

### Step 4 — Glue via `rade_api.c`

Add `struct rade *rade_rx_v2_pure_c_open(const char *model_path,
const char *fsync_path, int flags)` that skips the Python import
entirely.  Keep the existing `rade_open` for backwards compatibility
with scripts that still want Python setup.

Add matching `rade_rx_v2_pure_c()` dispatch.  `rade_n_eoo_bits`,
`rade_nin`, `rade_snrdB_3k_est`, etc. already work off the `struct
rade` — just point them at the new state.

### Step 5 — `radae_rx_v2` standalone binary

Mirror of existing `radae_rx` but calling the pure-C entry point.
Unit-testable on a laptop (no hardware), bench on Pi 4.  Acceptance:
> 1.5× realtime on Pi 4 single core with WAV input that decodes cleanly.

### Step 6 — Wire into `sbitx_radae.c`

First deploy: replace the `python3 -u radae_rxe2.py ...` command line
with `radae_rx_v2 ...`.  Same stdin/stdout contract, subprocess model
unchanged — this is the minimum-risk rollout.  Revert is a one-line
change.

Later: link `librade.so` into `sbitx_controller` and call
`rade_rx_v2_pure_c()` in-process.  Removes IPC and Python warmup
entirely.

## Non-goals

- No changes to the **TX** path in this migration.  RADAE TX is already
  persistent (see commit `f8dc833`) and the 6 s Python warmup only
  fires once at `dsp_init`, not per-PTT.  A TX C port is a separate
  project.
- No retraining or model changes.  We're using the 250725 model as-is.
- No backwards compat with model19_check3 in the new C path — the
  production service has been on 250725 for months, and keeping two
  sets of weights in `librade.so` just bloats it.
