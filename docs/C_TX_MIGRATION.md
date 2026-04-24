# Pure-C RADE V2 TX migration

## Why

The current RADEv2 TX path in Hermes is still Python-backed via
`radae_txe2.py`. Unlike RX, this is not an immediate realtime blocker
because the TX subprocess is already persistent, but a native TX path is
still desirable:

- remove Python/Torch from the production TX path
- simplify installs and runtime dependencies
- make TX symmetric with the successful pure-C RX rollout
- enable a future in-process Hermes integration

## What the Python TX does today

Hermes currently launches:

```text
lpcnet_demo -features -> radae_txe2.py -> complex IQ
```

`radae_txe2.py` currently:

- consumes one V2 TX chunk of FARGAN features (`Nzmf=1`, 40 ms)
- drops unused speech features (`36 -> 20`)
- optionally appends the fixed auxdata beacon pattern
- runs the **stateful encoder**
- maps latent pairs to complex QPSK carriers
- reshapes to `(Ns, Nc)` and performs OFDM synthesis with `Winv`
- inserts cyclic prefix
- applies the V2 bottleneck-3 constant-envelope clipper
- optionally applies `complex_bpf`
- emits the V2 EOO waveform (`eoo_v2`) on shutdown / SIGUSR1

The native path must first match this contract before any Hermes wiring.

## Important current-state constraint

The existing TX side in `rade_api.c` is still **legacy/V1-shaped**:

- `rade_tx()` asserts `RADE_BACKEND_LEGACY`
- `RADE_USE_C_ENCODER` only bypasses the core encoder inside the old
  Python `radae_txe` path
- `rade_tx_eoo()` and `rade_tx_set_eoo_bits()` also assume the legacy
  backend

So the V2 TX port is not a small extension of the current V1 path. It
needs a **parallel V2 pure-C TX backend**, analogous to
`rade_rx_v2_pure_c_open()` on the RX side.

## Target

Deliver the TX port in two stages:

1. **Standalone native binary**
   - same stdin/stdout contract as `radae_txe2.py`
   - features in, complex IQ out
   - optional TX BPF
   - EOO support

2. **Pure-C library/backend path**
   - opened without importing Python
   - callable from Hermes directly once parity is proven

## Staging

Each step should be independently testable and should leave the current
Python TX path available as the oracle.

### Step 1 — Freeze the Python TX reference contract

Document the exact current TX contract from `radae_txe2.py` and create
reference vectors for:

- normal TX frame(s)
- auxdata-disabled TX
- TX with `txbpf` enabled
- EOO output

Deliverables:

- `generate_rade_tx_v2_vectors.py`
- committed reference fixtures

Acceptance:

- reproducible Python-generated fixtures consumable by C tests

### Step 2 — Export/port the V2 TX encoder weights and metadata

Use the RX-port playbook:

- export the exact V2 stateful-encoder weights used by `radae_txe2.py`
- keep model identifiers explicit
- avoid all V1 latent-dim/frame-step assumptions

Deliverables:

- exporter script(s) for V2 TX encoder data
- generated C headers / source blobs
- explicit V2 TX model metadata/constants

Acceptance:

- C-side data loads with no Python dependency and matches the intended
  250725 / latent-56 model family

### Step 3 — Port the V2 TX wrapper math to C

Port the wrapper logic around the encoder:

1. feature packing / auxdata insertion
2. stateful encoder invocation
3. latent-to-complex-carrier mapping
4. OFDM synthesis + cyclic prefix
5. constant-envelope clipper path
6. optional `complex_bpf`
7. V2 EOO waveform emission

Acceptance:

- numerical parity against the frozen Python vectors from Step 1

### Step 4 — Add a V2 pure-C TX backend in `rade_api`

Add a new open path, e.g.:

```c
struct rade *rade_tx_v2_pure_c_open(const char model_file[], int flags);
```

and make the TX dispatch backend-aware so that:

- `rade_tx()`
- `rade_tx_eoo()`
- `rade_tx_set_eoo_bits()`

work for the new V2 backend without importing Python.

Acceptance:

- no Python import for the V2 TX backend
- array sizing helpers still work for callers

### Step 5 — Add a standalone `radae_tx_v2` native binary

Mirror the RX rollout pattern:

- same basic contract as `radae_txe2.py`
- easy A/B comparison against Python TX
- laptop-testable without Hermes

Acceptance:

- drop-in shell-pipeline replacement for `radae_txe2.py`

### Step 6 — Wire into Hermes with minimum risk

First deployment:

- replace `python3 -u radae_txe2.py ...` with the native binary
- keep the subprocess/stdin/stdout model unchanged

Later deployment:

- link the pure-C TX backend into Hermes directly and remove the
  subprocess

Acceptance:

- Hermes can switch between Python and native TX with a small, reversible
  change

## Test strategy

- keep Python TX as the oracle until parity is established
- compare complex IQ output numerically before any on-air testing
- include EOO and BPF cases in automated tests
- do live-radio testing only after strong parity inside `radae`

## Reuse from the older `radae_decoder` TX path

Only reuse the **integration patterns**, not the V1 modem assumptions.

Potentially useful:

- processing-loop structure
- TX meters / spectrum / recorder hooks
- runtime TX BPF control plumbing
- EOO metadata/callsign plumbing pattern

Not useful as direct source:

- V1 framing/constants
- desktop audio backend abstraction
- old linear resampler implementation

## Non-goals

- no direct reuse of V1 framing/constants
- no GUI/audio-backend work as part of the core TX port
- no Hermes rollout before `radae` parity tests exist
- no license-sensitive code copy from `radae_decoder` without review
