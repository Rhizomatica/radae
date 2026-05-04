# Pure-C RADE V2 TX migration

## Why

RX is now pure-C on Hermes.  TX still launches `radae_txe2.py` as a
Python subprocess.  The TX warmup (~6 s) is amortised by the persistent-
pipeline work (commit `f8dc833`), so TX is not the realtime blocker RX
was — but:

- a pure-C TX removes Python + PyTorch from the runtime TX path entirely
- it lets Hermes link `librade.so` and drop the subprocess model on both
  sides (eliminates IPC, simplifies shutdown)
- it makes the TX install footprint much smaller (no `torch`, `numpy`,
  `matplotlib` deps on the radio)
- it gives us symmetry with the RX rollout for ops familiarity

## What's already in C

The neural encoder — the numerically heaviest piece — is already ported:

- `src/rade_enc.c:rade_core_encoder` implements the stateful V2 encoder
  and is exercised by the legacy TX path via `RADE_USE_C_ENCODER` in
  `rade_api.c`.
- `src/rade_enc_data.c` contains 250725 weights (regenerated alongside
  the RX port in commit `861d06a`, `quantize=False`, bit-exact with
  PyTorch).
- `src/complex_bpf.{c,h}` already ports `radae.dsp.complex_bpf` and is
  used by the C RX BPF.

So the migration is **not** "port the encoder and DSP stack".  It's
"port the thin wrapper that turns latents into IQ".

## What `radae_txe2.py` actually does around the encoder

Reference: `radae_txe2.py:do_radae_tx` (~50 LoC) plus `do_eoo`:

1. pack 36-dim features -> 20 used features, append auxdata=-1 beacon
   pattern (`symb_repeat=4`) when auxdata is enabled
2. stateful encoder: `(1, Nzmf*enc_stride, feat) -> (1, Nzmf, latent_dim)`
3. QPSK map: `tx_sym = z[::2] + 1j*z[1::2]`
4. reshape to `(Nzmf*Ns, Nc)`
5. IDFT to time domain: `tx = tx_sym @ Winv`  (Nc -> M per symbol)
6. prepend cyclic prefix (`Ncp` samples copied from symbol tail)
7. bottleneck=3 hard clipper: `tx = exp(1j * angle(tx))`  (constant envelope)
8. optional `complex_bpf` (off in Hermes prod; used only from CLI)
9. EOO frame: precomputed constant `model.eoo_v2` = 6 copies of
   `pend_cp` scaled by `pilot_gain_eoo_v2`, optionally BPF-filtered

Every step above is O(Nmf) or smaller.  Total: ~150 LoC of C wrapper +
a small exported constants blob.

## Production invariants we can hard-code

From `radae_txe2.py` defaults and the Hermes command line
(`sbitx_radae.c:454`):

- `latent_dim=56`, `bottleneck=3`, `cp=0.004`, `time_offset=-16`,
  `correct_time_offset=-8`, `w1_dec=128`, `peak=True`, `auxdata=True`,
  `Nzmf=1`, `txbpf=False`, `send_eoo=True`

These match the 250725 model compiled into `librade.so`.  Non-production
configurations (bottleneck=1/2, auxdata=False) are **non-goals** for
this port.  `txbpf` is a CLI convenience we'll still expose but leave
off by default.

## The SIGUSR1 + pidfile contract

`sbitx_radae.c:247 radae_tx_emit_eoo` delivers `SIGUSR1` to the PID
stored in `/tmp/radae_tx.pid` (written by `radae_txe2.py --pid_file`).
The new C binary **must** preserve this contract in subprocess mode:

- accept `--pid_file <path>`; on startup write own PID there as text
- install a SIGUSR1 handler that flags "emit EOO at next safe point"
- between each input frame, if the flag is set, write `Neoo` IQ samples
  of `eoo_v2` to stdout, flush, clear the flag
- on stdin EOF (parent closed the pipe) emit one final EOO frame (same
  as Python's `finally: _emit_eoo()`), unlink the pid file, exit 0

The in-process V2 backend (`rade_tx_eoo(r, ...)`) bypasses all of the
above and just writes `Neoo` samples of `eoo_v2` into the caller's
buffer.

## Staging

Each step leaves the Python TX path intact as the oracle.

### Step 1 — Export TX-side constants

Ship `export_tx2_model_data.py` (mirror of `export_rx2_model_data.py`)
that emits `src/tx2_model_data.{c,h}`:

- `TX2_MODEL_M`, `_NCP`, `_NS`, `_NC`, `_LATENT_DIM`, `_FS`
- `TX2_MODEL_NMF = Nzmf * Ns * (M + Ncp)` and `_NEOO = 6 * (M + Ncp)`
- `tx2_model_Winv[NC][M]` — carrier IDFT matrix
- `tx2_model_pilot_gain_eoo_v2` — scalar
- `tx2_model_eoo_v2[NEOO]` — precomputed EOO waveform (constant)

Blob size ~12 KB.  The receiver's `rx2_model_w[]` and `rx2_model_pend[]`
are related but not directly reusable: RX has `Wfwd` (M×Nc), TX needs
`Winv` (Nc×M), and RX ships `pend` without the CP scaling.  Re-export
rather than share to keep module ownership clean.

Deliverables: `export_tx2_model_data.py`, `src/tx2_model_data.{c,h}`.
Acceptance: `make` succeeds; generated constants match a one-off
PyTorch dump within single-ULP tolerance.

### Step 2 — Port the TX wrapper math

New `src/tx2_encode.{c,h}` exposes:

```c
struct tx2_encode {
    RADEEncState enc_state;
    int auxdata;
    int txbpf_en;
    struct complex_bpf bpf;   /* only initialised when txbpf_en */
};

int tx2_encode_init(struct tx2_encode *tx, int auxdata, int txbpf_en);
void tx2_encode_reset(struct tx2_encode *tx);
void tx2_encode_destroy(struct tx2_encode *tx);

/* One modem frame: NB_TOTAL_FEATURES * Nzmf * enc_stride floats ->
 * TX2_MODEL_NMF complex samples. */
int tx2_encode_frame(struct tx2_encode *tx,
                     const float features_in[],
                     COMP tx_out[]);

/* Emit the constant EOO waveform, BPF-filtered if enabled. */
int tx2_encode_eoo(struct tx2_encode *tx, COMP eoo_out[]);
```

Implementation walks `radae_txe2.py:do_radae_tx` step by step, reusing
`rade_core_encoder` (unchanged) and the exported constants.  BPF uses
`complex_bpf_process` as the RX path does.

Deliverables: `tx2_encode.{c,h}`, `generate_tx2_encode_vectors.py` that
dumps Python-reference (feature_in, expected_iq_out) pairs via
`radae_tx_v2` in-process, `src/test_tx2_encode.c` consuming those
pairs.  Per-module tolerance ≈ 1e-5 (matches encoder test).

Acceptance: `test_tx2_encode` passes on Pi 4 + x86_64 on normal frames,
EOO output, and txbpf=on/off.

### Step 3 — Add V2 TX backend in `rade_api.c`

- Extend `enum rade_backend` with `RADE_BACKEND_TX_V2_PURE_C`.
- Add `rade_tx_v2_pure_c_open(const char model_file[], int flags)`
  mirroring `rade_rx_v2_pure_c_open`; it validates `model_file` against
  a compiled-in identifier (`RADE_TX_V2_COMPILED_MODEL_NAME`), calls
  `tx2_encode_init`, and fills the `struct rade` metadata
  (`n_features_in`, `n_floats_in`, `Nmf`, `Neoo`, `n_eoo_bits=0`).
- Make `rade_tx`, `rade_tx_eoo`, `rade_tx_set_eoo_bits` backend-aware:
  legacy branch unchanged, new branch dispatches to `tx2_encode_frame`
  / `tx2_encode_eoo`.  `rade_tx_set_eoo_bits` on the V2 backend is a
  no-op with a warning (V2 EOO has no soft-bit payload).

Deliverables: API additions in `rade_api.{c,h}`, `test_rade_tx_v2.c`
(streaming parity test, same shape as `test_rade_rx_v2.c`).
Acceptance: V2 open succeeds without importing Python; legacy path
still asserts `RADE_BACKEND_LEGACY` cleanly.

### Step 4 — Standalone `radae_tx_v2` binary

Mirror `src/radae_rx_v2.c`.  Flags and behaviour to match
`radae_txe2.py` exactly:

- `--model_name PATH` (validated against compiled-in identifier)
- `--pid_file PATH` (write own PID, unlink on exit)
- `--txbpf` (off by default)
- `--no_eoo` (skip the shutdown EOO)
- `-v` / `--verbose`
- stdin: feature frames (float32 LE)
- stdout: IQ frames (complex64 LE)
- SIGUSR1: emit EOO at next safe boundary (drain flag between stdin
  reads; match Python's `_EOORequest` semantics)
- EOF on stdin: flush any in-flight frame, emit EOO (if `send_eoo`),
  clean up pid file, exit 0

Deliverables: `src/radae_tx_v2.c`, CMake target.
Acceptance: `bash -c 'lpcnet_demo -features test.wav - | radae_tx_v2 …'`
produces a byte-for-byte identical (or within 1e-5) IQ stream vs the
Python binary, and SIGUSR1 produces a correctly-placed EOO.

### Step 5 — Wire into Hermes

Smallest-surface change in `hermes-net/trx_v2-userland/sbitx_radae.c`:

- add `RADAE_TX_BINARY_PATH "build/src/radae_tx_v2"` to `sbitx_radae.h`
- replace the `python3 -u radae_txe2.py …` in the TX subprocess
  snprintf (`:454`) with `%s …` using that macro
- nothing else changes — `radae_tx_emit_eoo` already signals
  `tx_python_pid` read from `/tmp/radae_tx.pid`, and the C binary
  preserves that pidfile contract
- document the rollback in `trx_v2-userland/README.md` alongside the
  existing RX rollback note

Acceptance: restart sbitx, key PTT, confirm DV decodes at the other
station; revert is a one-line edit in `sbitx_radae.c`.

Later (non-blocking): link `librade.so` into `sbitx_controller` and
call `rade_tx_v2_pure_c`/`rade_tx_eoo` in-process.  Eliminates the TX
subprocess and saves one pipe's worth of scheduling latency.

## Non-goals

- No V1 compatibility.  Production has been on 250725 for months; the
  legacy backend stays for whoever still needs V1, untouched.
- No `bottleneck=1`/`bottleneck=2` clippers.  Prod is bottleneck=3.
- No `auxdata=False` code path unless explicitly requested for bench
  testing.  Prod is auxdata=True.
- No GUI or audio-backend abstractions from the old `radae_decoder`
  TX path.
- No in-process `librade.so` link in Hermes in this migration (that's
  a follow-up after the subprocess swap proves itself).
- No changes to the RX path.

## Test strategy

Mirrors the RX port.  Python vectors generated on estacao3 (Pi 4,
Python 3.13), C tests run on both Pi 4 and x86_64 for endian parity.
End-to-end parity test ships reference IQ and compares byte-for-byte.
No live-radio test until the vector parity tests pass.

`cmake/BuildOpus.cmake` builds opus with `-DHIGH_ACCURACY`, which
swaps opus's polynomial tanh/sigmoid for the libm `tanh()`/`exp()`
PyTorch uses; without it the encoder accumulates ~0.1 latent drift
across the 11-layer DenseNet stack and the clipper amplifies it into
~0.13 IQ-sample errors (still fine for over-the-air decode, but no
longer single-ULP against PyTorch).  With HIGH_ACCURACY enabled the
TX wrapper achieves single-ULP parity:
- `txbpf=False`: tx_max ~1.6e-5, eoo_max ~1.3e-7
- `txbpf=True` : tx_max ~3.9e-6, eoo_max ~9.0e-7

Tolerances in `test_tx2_encode.c` are sized accordingly.
