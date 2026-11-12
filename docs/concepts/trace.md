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
  for their first `EAGER_CALLS = 2` calls and capture on the third, which runs eagerly
  too (the capture's prepare pass); replays begin with the fourth call.
- **Capture** records the device work without executing it.
- **Inputs.** Parameters, marked static addresses and inputs the graph updates in place
  are used in place. A replaced Parameter is copied in before the replay and back out
  after (into the Parameter the trace was captured with). Any other replaced one (a new
  static KV cache per HF `generate`) of the same shape, dtype and placement takes over
  the trace's buffer: the tensor on the buffer before moves to storage of its own with
  its values, the new one's values are copied in and it is rebound (`set_`); later
  replays copy nothing. The new tensor is copied in and out per call instead when it is
  a view, a Parameter, requires grad, is not one whole contiguous storage, or has a live
  view or alias. The call runs eagerly when the tensor on the buffer cannot be moved
  (it has a live view). Other inputs are copied into stable contiguous buffers per
  call.
- **State.** Each replay reads and writes the live state of every tensor fn mutates
  (parameters, optimizer state).
- **Restrictions.** fn must not move data to or from the host and must not depend on
  Python values that change between calls (use `capturable=True` for Adam / AdamW:
  capturing the step of an optimizer built with `capturable=False` raises). Not
  detected: optimizers whose step skips torch's capture check (Adagrad with
  `lr_decay`), a step run on another thread than the capture's (the autograd engine's,
  e.g. an optimizer stepped in a backward hook), and a scheduler's float learning rate,
  which the trace keeps as captured.
- **Host inputs.** A CPU tensor input is part of the captured program by value: a call
  with a different value runs that call eagerly instead of replaying. Non-tensor inputs
  (numbers, strings; a compiled graph's Python ints) are too, for every `Captured`.
- **Compiled state.** The call that captures a compiled graph is answered by its
  capture's eager pass, so every tensor the graph updates in place (counters, caches,
  parameters, optimizer state) is updated once per call, replayed or not.
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
  TT-NN op that does not use the bridge's placement). A trace's fences are exempt: they
  hold memory an op of the capture used and freed (TT-NN's conv2d works in L1), which
  only replays write and whose contents nothing reads between replays.

`KTT_TRACE_L1_GUARD=0` turns both off (diagnostic). `torch.ops.ktt._debug_trace_l1(id)`
returns the region's end, the lowest live L1 buffer address and the L1 base.

## Decode loops

A token loop that uploads each step's inputs and downloads its token makes the device
wait for the host every step. `kohakutt.DecodeLoop(step, inputs, feedback,
read_every=8)` runs `out = step(*inputs)` and `inputs = feedback(out, inputs)` with
device ops only, and reads the emitted values (`emit(out)`, default a clone of `out`) to
the host once per `read_every` steps, so the host enqueues ahead of the device. A stop
condition acts at a read: up to `read_every - 1` extra steps run and are dropped, and
the loop then needs `reset(inputs)` before it runs again (as after an error).

`kohakutt.greedy_decode(forward, input_ids, cache, max_new_tokens, eos_token_id=None,
pad_token_id=None, attention_mask=None)` is that loop for a causal LM with a static
cache, giving HF's greedy `generate` tokens. `forward` is the model or its
`torch.compile(..., backend="ktt_tt")` form. The next token is the previous step's
device argmax and the position advances on the device. A 0/1 `attention_mask` (left
padding, or inferred from pad tokens in the prompt as HF does) is passed as one
fixed-shape mask over the whole cache. Rows after their end token are filled with
`pad_token_id` (default: the model's `generation_config.pad_token_id`, else the first end
token). The cache must hold prompt + `max_new_tokens` - 1 positions (checked, since an
out-of-range row is not checked on the device). Reuse one cache across calls: it is
reset in place, and a trace keeps its tensors. HF's `model.generate` is unchanged (its
per-token work runs on the host).

HF `model.generate` with `cache_implementation="static"` and the forward compiled by
`ktt_tt` replays one trace per token. Its new cache each call takes over the trace's
buffers on the first replay (see Inputs). A cache of a new shape (a larger
`max_new_tokens` than any earlier call, or a new batch size) compiles and captures again
(seconds), as does a new prompt length for the prefill. Keep the shapes fixed: set
`model.generation_config.max_cache_len` (or pass one `StaticCache` of a fixed
`max_cache_len` as `past_key_values`, resetting it between calls) and keep
`max_new_tokens` at or below it.

## Host transfers in a capture

A host read or write inside a capture raises `CaptureHostTransfer` (a `RuntimeError`)
naming the aten op and the Python line that caused it, then the device's message. The
known causes and their workarounds:

| Pattern | Op named | Workaround |
| --- | --- | --- |
| `.item()`, `.tolist()`, `bool(t)`, `if t:` on a device tensor | `aten._local_scalar_dense` | Keep the value on the device (`torch.where`), or decide on the host before the call. |
| `.cpu()`, `.numpy()`, `print(t)` | `aten._to_copy` / `aten.copy_` | Return the tensor and read it after the call. |
| A conv whose weight is computed inside fn (`w * 2`) or is a slice of another tensor (`w[:, :16]`): the weight is prepared on the host each call (a whole reshape, such as `F.conv1d`'s or `w[:, :, None]`, is cached) | `aten.convolution` | Compute or slice the weight once outside fn (`.clone()` for a slice). |

Hugging Face's packed-sequence check in its attention masks (a host read, e.g. with
`use_cache=False`) is skipped inside a capture, as under a CUDA graph capture.

## Known issues

- A trace does not hold tensors that existed before its capture. Inputs and parameters
  are kept by the step and the module; any other device tensor the captured function
  reads must stay alive as long as the trace does. Once it is freed, a later allocation
  can take its memory and the replays read that instead.

- While any trace is alive, in-place updates write existing buffers instead of
  rebinding, and caches stop creating new device buffers. The constant-fill cache
  stops only inside a capture, and drops no entry while a trace is alive.
