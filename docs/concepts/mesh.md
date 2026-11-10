---
title: Mesh: several chips as one device
summary: "kohakutt.mesh([0, 1]) opens chips 0 and 1 as one torch device (a 1×2 TT-NN mesh over the fabric). Torch code runs unchanged; tensors are spread over the chip..."
tags:
  - concepts
  - mesh
---

# Mesh: several chips as one device

`kohakutt.mesh([0, 1])` opens chips 0 and 1 as one torch device (a 1×2 TT-NN mesh
over the fabric). Torch code runs unchanged; tensors are spread over the chips and ops
run per chip.

## Placements

Every tensor on a mesh has one placement per mesh axis:

| Placement | Meaning |
|---|---|
| `R` | replicated: each chip holds the whole tensor |
| `S(d)` | sharded on dim `d`: each chip holds a contiguous slice (equal ceil shards; the last may be short) |
| `P(sum)`, `P(max)`, `P(min)` | partial: each chip holds a value that the reduction over chips completes |

`kohakutt.placement(t)` reports it (`"Replicated"`, `"Sharded(dim=1)"`,
`"Partial(sum)"`). A tensor of at least `KTT_MESH_SHARD_MIN` bytes (1 MiB) is created
sharded on its leading dim when that dim divides, else on the largest dim that divides;
smaller tensors are replicated. In eager execution the same floor holds when a partial
sum is completed: a `P(sum)` below it becomes replicated by an all-reduce, never a
shard, because every conversion of a small sharded activation costs host time there; a
captured trace keeps the cost-model choice, where a collective costs no host time.
Shard padding is undefined, and ops that read across it mask it with the reduction's
identity.

## Ops and conversions

Each op family declares how its inputs' dims map to its output (its factors). For each
mesh axis, a rule engine picks the input and output placements with the lowest cost
under a measured collective-cost model (each input's conversion priced by its own size
in bytes), converts the inputs, and runs the unchanged single-chip kernel on each
shard. A K-split matmul produces `P(sum)`; an elementwise op
passes `P(sum)` through where it is linear. Conversions run on the device:

| Conversion | TT-NN op |
|---|---|
| `S(d)` → `R` | all-gather |
| `P(sum)` → `S(d)` | reduce-scatter (fp32 on the rows of a 1×2 mesh: Kohaku-TT's own) |
| `P(sum)` → `R` | all-reduce |
| `S(i)` → `S(j)` | all-to-all |
| `R` → `S(d)` | partition (a local slice) |

A conversion made for one op is kept on the tensor for the ops that follow, so several
consumers of one tensor convert it once. A kept copy is always made from the tensor
itself by the conversion in the table. A sharded copy is never cut from the replicated
one: in bf16 the reduce-scatter and the all-reduce round differently (41 of 8192
elements of a 128 × 64 partial sum). A kept copy holds until a tensor is written in
place (`copy_` while a trace is alive, or a trace replay), and a capture and its
prepare pass do not use the memory.

The rule engine prices a conversion to a kept copy at zero. An op that would pass a
partial sum through then takes the completed copy when that saves a collective: an RMS
norm written out in fp32 on the partial sum of a column → row pair runs one
reduce-scatter, where it ran two and an all-gather. A cast, negation or scaling of a
partial sum still passes it through, and so does an op that computes in another dtype
than the partial sum: `partial_bf16 * y_fp32` sums the partials in fp32. `KTT_MESH_CONV_CACHE=0` runs every conversion again at the same
placements, with the same bits in every result; `KTT_MESH_CONV_FREE=0` prices kept
copies at their full cost, which gives the placements chosen without the memory.

Every op is launched once: output memory placements and program buffer sizes are chosen
before the launch, and a failure raises instead of running the op again. A layout the
device conversions do not support (shards off tile boundaries on the two tiled dims)
converts through the host; inside a trace capture that case releases the trace.
`torch.tt.mesh_stats()` counts each kind of conversion.

An op without a placed form for an input gathers that input to `R` and runs
replicated: an op with no placement rule, a view the shards do not follow (a reshape
that mixes the sharded dim with another), or a write into part of a tensor. Each such
call is counted in `torch.tt.mesh_fallback_ops()`, and the first one per op and input
placement since the last `torch.tt.reset_mesh_stats()` warns, naming the op and the
placement (`KTT_FALLBACK_WARN=0` silences the warning). `KTT_MESH_STRICT=1` raises
instead; the message names the op, the placement and the way out:
`kohakutt.replicate(x)` on the input, or `kohakutt.replicate(module)` for a parameter.

A write of a whole tensor (`dst.copy_(src)`, and so `load_state_dict` into a module on
the mesh) keeps the placement of `dst`: the value is converted to it, and a host value
is uploaded as shards. A `dst` holding a partial sum, or not yet written since
`torch.empty` (as inside `clone()`), takes the written value's placement.

## Why an op was placed as it was

`kohakutt.placement_log()` lists the placement decisions made while recording is on.
Recording is off by default and costs one flag check per op then;
`with kohakutt.record_placements():` turns it on for a block, `KTT_MESH_EXPLAIN=1` from
load, and `kohakutt.reset_placement_log()` empties the list (it keeps the last 65536
records).

```python
with kohakutt.record_placements():
    y = model(x)
for r in kohakutt.placement_log():
    print(r["op"], r["inputs"], "->", r["output"], r["conversions"], r["reason"])
```

Each record is one op call: `op`, the placement of each input before any conversion
and of the output (`"[R, S(0)]"`: one entry per mesh axis, over device dims, where a
tensor below rank 2 has leading unit dims), the conversions made (`input`, `from`,
`to`, `kind` as in the table above, and `reused`: True when a kept copy served it), and
the reason. Recorded are the ops the rule engine places, gather fallbacks,
`partial + replicated`, `kohakutt.shard` / `kohakutt.replicate` on a tensor, and
lookups in a sharded table. The reasons:

| Reason | Meaning |
|---|---|
| `replicated` | every input replicated on that mesh axis was the cheapest |
| `shard` | inputs sharded along one dim that passes to the output; the output is sharded |
| `reduce` | inputs sharded along a dim the op sums over; the output is a partial sum |
| `partial` | a partial sum passed through an op linear in it |
| `... (size floor)` | a cheaper choice would have completed a partial sum below `KTT_MESH_SHARD_MIN` to a shard |
| `... (memory)` | a kept copy made this choice the cheapest |
| `fallback: ...` | no placed form; the input was gathered |
| `partial + replicated: ...` | the all-reduce that completes the sum also adds the other input |
| `mark` | `kohakutt.shard(x, d)` / `kohakutt.replicate(x)` |
| `rows of a sharded table`, `rows at sharded indices` | an embedding lookup per chip |

## Placing a model

Where the model is written, marks set the placement of a module's tensors. Each returns
the same module, so the model stays a plain `nn.Module`:

| Mark | Linear / Conv | Embedding |
|---|---|---|
| `kohakutt.colwise(m)` | weight `S(0)`, bias `S(0)` (output features split) | weight `S(1)` |
| `kohakutt.rowwise(m)` | weight `S(1)`, bias `R` (input features split) | weight `S(0)` |
| `kohakutt.replicate(m)` | every tensor of `m` and its children `R` | |
| `kohakutt.shard(m, d)` | every tensor of `m` and its children `S(d)` | |
| `kohakutt.place(m, name="S(d)" or "R")` | the named tensors of any module | |

A child's own mark wins over a parent's. Marks take effect when the module moves onto a
mesh of two or more chips (`Module.to()`), or at once when it is already there, and do
nothing on one chip. Inside `forward`, `kohakutt.shard(x, d)` and `kohakutt.replicate(x)`
give an activation its placement at that point (one conversion if it differs; the
gradient passes through unchanged). `d` is a dim of `x` as given: a view, transpose or
slice is sharded on its own dims (materialized first when its shards do not follow the
storage's). An empty tensor takes the placement; ops read it as replicated. Given a
tensor its module marks otherwise
(`kohakutt.shard(self.weight, 1)` on a `colwise` weight), they warn, or raise under
`KTT_MESH_STRICT=1`, and return a placed copy; the module's tensor keeps its mark.

Unmarked Linears of at least `KTT_MESH_SHARD_MIN` bytes are paired column → row by
default (`KTT_MESH_AUTO_PAIRS=0` turns this off): projections named like `q_proj`,
`k_proj`, `v_proj`, `up_proj`, `gate_proj`, `fc1` are column-split, `o_proj`,
`out_proj`, `down_proj`, `fc2` row-split; other Linears that widen the model's residual
width are column-split and those that narrow back to it row-split. A column-split
Linear followed by a row-split one produces `P(sum)`, and one reduce-scatter completes
the pair. The Linears of one parent module form one pair, decided at its first eager
call by that call's rows: split when the matmul time the split saves exceeds the eager
all-reduce that completes it, else replicated (`KTT_MESH_AUTO_PAIRS=always` splits
every pair at once). An MLP d → 4d → d splits from about 4,700 rows per call at
d = 1024, 770 at d = 2048 and 170 at d = 4096.

## Under torch.compile

The `ktt_tt`, `ktt_tt_aot` and `ktt_fuse` backends keep every mark: module marks are
already on the tensors, and `kohakutt.shard(x, d)` / `kohakutt.replicate(x)` in
`forward` trace into the graph as one `ktt::distribute` each (a meta kernel gives its
shape, its autograd kernel passes the gradient through), so the capture and every
replay run the same conversion. Before a graph is compiled, its Linears that neither
the code nor a plan marks are paired from the graph itself: Linears whose outputs reach
only other Linears through elementwise ops, head reshapes and attention are
column-split, the Linears they reach row-split, at any rows per call: a compiled graph
replays as a trace, where a collective costs no host time. This replaces the name
pairing for those weights (`KTT_MESH_AUTO_PAIRS=0` turns both off).

`options={"mesh_plan": {...}}` places tensors by module path for that compile:

```python
model = torch.compile(model, backend="ktt_tt_aot", options={"mesh_plan": {
    "blocks.*.attn.to_q": "colwise",      # fnmatch over module or tensor paths
    "blocks.*.attn.to_out": "rowwise",
    "blocks.*.norm*": "replicate",
    "head": {"weight": "S(1)"},           # per tensor: "R", "S(d)", "shard(d)"
}})
```

The first matching entry wins. `colwise` / `rowwise` take the rule of how the graph
uses the tensor (Linear, convolution or embedding). A plan entry overrides a mark set
in code and the default pairing; placement is a property of the tensor, so it stays
after the compile. A plan gives the same placements, programs and results as the same
marks written in code. `kohakutt.device.marks.apply_plan(module, plan)` applies the
plan format eagerly, as marks. An activation mark in the graph that asks a tensor
marked in code or by the plan for another placement is reported when the graph
compiles, as in eager.

## Traces

`kohakutt.capture` works on a mesh: the collectives are captured with the compute, and
every replay on new inputs equals the eager result bit for bit. A kept conversion holds
only until the next in-place write, and a capture records its own conversions.

The last eager call before a capture (the prepare pass) runs in the capture's state, on
one chip and on a mesh, so the capture compiles nothing and sets up no collective:
views and copies take the path they take while traces are alive, tensors allocated by
torch are replicated, storages the call rebinds are written back as the capture writes
them back (converted to the original placement), and a host tensor of up to 64 KiB
moved to the device is uploaded once and copied on the device from then on. Inside the
capture, a host tensor whose bytes differ from the prepare pass's raises.

## Measured

Two Blackhole P150A cards, bf16, four MLP blocks (Linear d → 4d → d, GELU, residual),
median of 10 synchronized warm forwards,
default placement (column → row) against weights replicated with the rows split:

| Rows × d | 1 chip | 2 chips, default | 2 chips, replicated weights |
|---|---|---|---|
| 256 × 1024, captured trace | 1.18 ms | 1.03 ms | |
| 1024 × 2048, captured trace | 2.80 ms | 2.15 ms | |
| 2048 × 4096 | 13.40 ms (164 TFLOPS) | 9.17 ms (240 TFLOPS) | 9.18 ms |
| 4096 × 4096 | 26.02 ms (169 TFLOPS) | 17.26 ms (255 TFLOPS) | 15.98 ms |
| 2048 × 8192 | 45.81 ms (192 TFLOPS) | 26.87 ms (327 TFLOPS) | 30.17 ms |

Small shapes are limited by host dispatch, so a captured trace is the faster mode there.
Eager, the same stack with the default pair decided by rows per call (median of 30
synchronized forwards, 3 rounds):

| Rows × d | 1 chip | 2 chips, default | 2 chips, replicated weights |
|---|---|---|---|
| 256 × 1024 | 1.33 ms | 1.55 ms (replicated by rows) | 1.54 ms |
| 1024 × 2048 | 2.35 ms | 2.15 ms (split) | 2.22 ms |
| 2048 × 4096 | 13.48 ms | 9.10 ms (split) | 9.81 ms |

At 256 × 1024 even replicated weights are 16 % slower on two chips than on one: every
program is enqueued to both chips, and the host time per op on the stream grows by
12 % (`kohakutt.counters()`, 151 → 170 µs per op with the counters on). Collectives run on the same command queue as compute and do not overlap it
yet. Qwen2.5-0.5B (seq 128) on 1×2: 104.6 ms with the default placement, 59.5 ms
replicated, logits cosine 0.999876 against float64 (CPU bf16 0.999937).

Compiled (`torch.compile`, backend `ktt_tt`, trace replay), bf16, median of 5 warm
forwards, cosine of the output against float64 on the CPU:

| Model | 1 chip | 2 chips | cosine 1 / 2 chips (CPU bf16) |
|---|---|---|---|
| DINOv3 ViT-B/16, 224 px | 21.35 ms | 16.51 ms | 0.999523 / 0.999572 (0.999838) |
| DINOv3 ViT-L/16, 224 px | 50.84 ms | 37.85 ms | 0.999359 / 0.999361 (0.999858) |
| Gemma 4 E2B, 128 tokens | 136.7 ms | 127.1 ms | 0.999096 / 0.998793 (0.999575) |
| Gemma 4 E4B, 128 tokens | 185.9 ms | 182.3 ms | 0.995956 / 0.997905 (0.998369) |

fp32 `P(sum)` from a column → row Linear pair, completed on the rows by Kohaku-TT's
reduce-scatter (`KTT_MESH_FP32_RS=0`: all-gather and a local sum), per call inside a
captured trace of 10 back-to-back completions. Every result equals the fp32 sum of the
two chips' partials bit for bit (the adds run on the vector unit in fp32, rounded to
nearest even):

| `P(sum)` shape | → rows sharded | all-gather + sum | → replicated | all-gather + sum |
|---|---|---|---|---|
| 512 × 1024 | 30 µs | 110 µs | 58 µs | 104 µs |
| 1 × 1 × 8192 × 392 | 121 µs | 470 µs | 262 µs | 432 µs |
| 4096 × 4096 | 519 µs | 2135 µs | 942 µs | 1967 µs |

Each chip sends half its partial over the four Ethernet links (one sender core per link
under its Ethernet tile) into the peer's scratch, and 16 drain cores add each received
chunk to the kept half. TT-NN's reduce-scatter hangs at 1 × 1 × 8192 × 392 on 1×2
(tt-metal#38538), and it adds fp32 in the matrix unit's reduced-precision source
registers.

A partial sum that has to become replicated (the residual stream after a column → row
Linear pair) is completed by Kohaku-TT's all-reduce, one program per chip pair
(`KTT_MESH_REDUCE=0`: TT-NN's collectives): each chip sends its whole partial over the four
links, two bf16 tiles per packet, into a scratch paged per packet; 48 drain cores add the
received and kept tiles in fp32 and write the sum on both chips, bit for bit equal.
`partial + x` with a replicated `x` (or a placed `x` whose replicated copy exists) is
added by the same drains, so a residual block ends in one program where it took a
reduce-scatter, an add and an all-gather. A row-split Linear's bias runs inside each
chip's matmul as half the bias. MLP stack (4 blocks, 4096 → 16384 → 4096, GELU,
residual, 2048 rows, bf16) on 1×2, device time per block, 2026-11-10:

| Program | Time |
|---|---|
| up matmul (column-split, bias inside) | 713 µs |
| GELU | 289 µs |
| down matmul (row-split, bias inside) | 649 µs |
| all-reduce + residual | 288 µs (reduce-scatter 141 + add 66 + all-gather 110 before) |

The forward is 8.1 ms on two chips against 13.4 ms on one. The collective is bound by the
links (16.8 MB each way per block): 16 drains per link instead of 12 give 286 - 292 µs.

### All-reduce inside the next matmul (`KTT_MESH_FUSE=1`)

Off by default. With it, `partial + x` is held back, and when the next op is a Linear
on the sum, both run as one program per chip: the exchange moves the tiles column by
column (4 drains per link in grid rows 0 and 1), the drains count finished K blocks on
semaphores, and Kohaku-TT's `ktt_gemm` on rows 2 and up takes its K blocks from the four
link ranges in turn, each A block waiting for its count. Any other next op runs the
all-reduce first, as without the switch. `torch.tt.mesh_stats()["fused_linears"]` counts
the fused pairs; the first forward of a model does not fuse (its weights' local views
are created then).

Measured on the block above without bias: the fused program takes 996 - 1020 µs against
712 + 288 µs for the two programs, and a 3-block forward 3.72 - 3.74 ms against 3.58 -
3.59 ms; values are equal to the unfused path's bound (rel 4.08e-3 against float64) and
both chips are bit for bit equal. The exchange is link-bound and the matmul bound by DRAM
and NoC traffic, so running them together takes about their sum. A matmul that splits
its rows into a part it gives away early and a part it keeps was also built: it reads
the weight twice and takes 845 - 887 µs against 629 µs.

On two chips the ViTs gather the attention output once per layer
(`transpose(1, 2).contiguous()` of the head-split output, a `copy_` fallback), and the
Gemma models gather q, k and v at every attention (one or two KV heads do not split
over two chips).

## Ending a process

A mesh keeps fabric routers running on the cards' ethernet cores until the device is
closed, so a process must close it before it ends. `kohakutt.close()` does; so does
the interpreter's exit (normal end, `sys.exit`, an uncaught exception, SIGINT, which
Python turns into `KeyboardInterrupt`) and SIGTERM.

- SIGTERM: opening a device installs a handler that closes it, restores the previous
  handler and sends the signal again, so the process still ends by SIGTERM. A SIGTERM
  handler of your own stays; it must end the process through `sys.exit` for the close
  to run. The handler is only installed from the main thread.
- Python runs a handler between two Python instructions of the main thread, never
  inside a kernel launch (those run on the backend's worker thread). A signal that
  arrives during a long native call (a hung kernel) therefore waits for it.
- SIGKILL, a power loss or a crash cannot be handled. They leave the ethernet cores
  running; the next open then fails with `kohakutt.BoardResetRequired` after
  TT-Metal's 20 s wait, and the cards need `tt-smi -r`.

## Limits

- Activations stay in DRAM by default (`KTT_L1_BUDGET` 0, on one chip and on a
  mesh): some TT-NN ops still place their output in their input's memory, and on a
  mesh L1 activations then fill a bank past the budget (Qwen2.5-0.5B ran out of L1 at
  the logits). Setting `KTT_L1_BUDGET` restores L1 activations.
- Small models gain nothing from sharding: Qwen2.5-0.5B is 104 ms sharded against
  60 ms replicated on 1×2, because each layer's collectives cost more than its small
  matmuls save. `KTT_MESH_AUTO=0` keeps such a model replicated.
- Known failures, not specific to the mesh (one chip, `torch.compile`, also failing
  before the mesh work at 6063f57): the CNN training step in
  `tests/hardware/compile/test_compile_tt.py` and the Whisper case in
  `test_compile_zoo.py`. Both stop on a TT-Metal check during trace capture: an enqueue
  that is not allowed while capturing (`mesh_workload.cpp:200`, `!is_capturing_trace`)
  and a command-queue call that is not allowed while capturing
  (`fd_mesh_command_queue.cpp:851`, `!trace_id_.has_value()`). Both models run
  convolutions, and the convolution path is being replaced by Kohaku-TT's own kernel.
- One mesh at a time; 1×2 measured (the placement and rule code is written for N-D).
- Collectives do not overlap compute.
- `P(max)` / `P(min)` complete by all-gather and a local max / min.
- fp32 partial sums reduce-scatter only on the rows (second-to-last dim, leading dims 1,
  rows a multiple of 64) of a 1×2 mesh; other fp32 completions all-gather and sum
  locally.
- `rand` / `randn` on a sharded tensor draw per chip unless `KTT_PHILOX=1`.
