---
title: Trace contract
summary: "kohakutt.capture(fn, example_inputs, warmup=3) returns a Captured step; step(inputs) replays it. torch.compile(..., backend='ktt_tt' | 'ktt_tt_aot') uses the..."
tags:
  - concepts
  - trace
---

# Trace contract

`kohakutt.capture(fn, *example_inputs, warmup=3)` returns a `Captured` step;
`step(*inputs)` replays it. `torch.compile(..., backend="ktt_tt" | "ktt_tt_aot")` uses
the same mechanism per compiled graph.

- **Warm-up.** fn runs eagerly `warmup` times first: every program compiles and lazy
  state (grads, optimizer buffers) exists before capture. Compiled graphs run eagerly
  for their first `EAGER_CALLS = 2` calls and capture on the third.
- **Capture** records the device work without executing it.
- **Inputs.** Parameters, marked static addresses and inputs the graph updates in place
  are used in place (a replaced one is copied in before the replay and back out after);
  other inputs are copied into stable contiguous buffers per call.
- **State.** Each replay reads and writes the live state of every tensor fn mutates
  (parameters, optimizer state).
- **Restrictions.** fn must not move data to or from the host and must not depend on
  Python values that change between calls (use `capturable=True` for Adam / AdamW).
- **Host inputs.** A CPU tensor input is part of the captured program by value: a call
  with a different value runs that call eagerly instead of replaying.
- **Outputs** are copies, valid until the next replay of any trace: read or consume
  them before replaying again.

## Known issues

- Every tensor created during a capture is held until the trace is released, so a
  captured graph needs device memory for all of its intermediates at once. Large
  forwards (e.g. a 42-block DiT at 512 px) do not fit; run them eagerly.

- Device tensors allocated after a capture can be overwritten by later replays of that
  trace (TT-NN reuses the trace's temporary buffers). Persistent state is protected: while
  any trace is alive, in-place updates write existing buffers instead of rebinding, and
  caches stop creating new device buffers. Long-lived tensors created between replays
  (e.g. a KV cache grown during generation) should be allocated before the capture.
