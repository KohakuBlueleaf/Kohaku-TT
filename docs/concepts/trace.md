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
  them before replaying again. `capture(..., copy_outputs=False)` returns the trace's own
  output tensors instead, with no device copy per call; the next replay of that trace
  overwrites them.
- **Cost.** A replay itself costs about what the same ops cost eagerly (a 4 MB conv:
  155 µs replayed, 145 µs eager). Each call also copies its non-static inputs in and its
  outputs out, about 110 µs per 4 MB tensor with TT-NN's copy.

## Device memory

A capture frees its intermediates as an eager call does. A replay still writes every
address the capture wrote, so a trace keeps, until `step.release()`:

- **Tensors alive when the capture ends.** The outputs, and tensors the capture stored on
  longer-lived objects (cached views and copies). The trace holds them itself, whoever
  else drops them later.
- **Tensors the capture copied into.** The stable input buffers and state written in
  place.
- **Fences.** The address ranges of every buffer freed during the capture (dead
  intermediates and TT-NN's op-internal temporaries), merged, less what is still
  allocated when the capture ends. Each contiguous range is reserved again at its own
  address, DRAM and L1 alike, so no later allocation can land where a replay writes. A
  range that cannot be reserved at its address fails the capture: the trace is released
  and the error raised.

A trace's footprint is therefore about the device peak of one call of the captured
function, not the sum of its intermediates (Gemma-4 E2B at 512 tokens through `ktt_tt`:
0.99 GB kept for a peak of 0.90 GB). Traces alive together add their footprints; each
allocates outside the others' fences.

`kohakutt.torch_backend.trace.footprint(step)` returns the tensors held and created, the
number of fences and the bytes fenced and left allocated. `KTT_TRACE_HOLD=1` makes a
trace hold every tensor created in its capture instead (diagnostic).

## L1 guard

A program's circular buffers sit at the bottom of each core's L1, and L1 tensors are
allocated from the top down. TT-Metal checks that the two do not meet when a program is
enqueued; a replay enqueues nothing, so an L1 tensor allocated after a capture could lie
under a traced program's circular buffers and be overwritten by the next replay.

While a trace is alive, Kohaku-TT keeps its region clear:

- **Region.** The highest L1 address any program's circular buffers reached before the
  capture ended (recorded as TT-Metal allocates them). It is an upper bound for the
  trace's own programs.
- **Placement.** New activations go to L1 only while they stay above that region; the rest
  go to DRAM.
- **Check.** Each replay first checks that no live L1 buffer lies inside the region and
  raises instead of replaying when one does (for example an L1 tensor allocated by a
  TT-NN op that does not use the bridge's placement).

`KTT_TRACE_L1_GUARD=0` turns both off (diagnostic). `torch.ops.ktt._debug_trace_l1(id)`
returns the region's end, the lowest live L1 buffer address and the L1 base.

## Known issues

- A trace does not hold tensors that existed before its capture. Inputs and parameters
  are kept by the step and the module; any other device tensor the captured function
  reads must stay alive as long as the trace does. Once it is freed, a later allocation
  can take its memory and the replays read that instead.

- While any trace is alive, in-place updates write existing buffers instead of
  rebinding, and caches stop creating new device buffers.
