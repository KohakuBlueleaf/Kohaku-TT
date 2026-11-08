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
smaller tensors are replicated. Shard padding is undefined, and ops that read across it
mask it with the reduction's identity.

## Ops and conversions

Each op family declares how its inputs' dims map to its output (its factors). For each
mesh axis, a rule engine picks the input and output placements with the lowest cost
under a measured collective-cost model, converts the inputs, and runs the unchanged
single-chip kernel on each shard. A K-split matmul produces `P(sum)`; an elementwise op
passes `P(sum)` through where it is linear. Conversions run on the device:

| Conversion | TT-NN op |
|---|---|
| `S(d)` → `R` | all-gather |
| `P(sum)` → `S(d)` | reduce-scatter |
| `P(sum)` → `R` | all-reduce |
| `S(i)` → `S(j)` | all-to-all |
| `R` → `S(d)` | partition (a local slice) |

Every op is launched once: output memory placements and program buffer sizes are chosen
before the launch, and a failure raises instead of running the op again. A layout the
device conversions do not support (shards off tile boundaries on the two tiled dims)
converts through the host; inside a trace capture that case releases the trace.
`torch.tt.mesh_stats()` counts each kind of conversion, and an op without a rule gathers
its inputs to `R`, is counted in `torch.tt.mesh_fallback_ops()`, and raises under
`KTT_MESH_STRICT=1`.

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
give an activation its placement at that point (one conversion if it differs).

Unmarked Linears of at least `KTT_MESH_SHARD_MIN` bytes are paired column → row by
default (`KTT_MESH_AUTO_PAIRS=0` turns this off): projections named like `q_proj`,
`k_proj`, `v_proj`, `up_proj`, `gate_proj`, `fc1` are column-split, `o_proj`,
`out_proj`, `down_proj`, `fc2` row-split; other Linears that widen the model's residual
width are column-split and those that narrow back to it row-split. A column-split
Linear followed by a row-split one produces `P(sum)`, and one reduce-scatter completes
the pair.

## Traces

`kohakutt.capture` works on a mesh: the collectives are captured with the compute, and
every replay on new inputs equals the eager result bit for bit. The gathered copy of a
sharded tensor is cached only within one write epoch, and a capture records its own
gathers.

## Measured

Two Blackhole P150A cards, bf16, four MLP blocks (Linear d → 4d → d, GELU, residual),
median of 10 synchronized warm forwards (`experiments/mesh/mesh_devprof.py --timed 10`),
default placement (column → row) against weights replicated with the rows split:

| Rows × d | 1 chip | 2 chips, default | 2 chips, replicated weights |
|---|---|---|---|
| 256 × 1024, captured trace | 1.18 ms | 1.03 ms | |
| 1024 × 2048, captured trace | 2.80 ms | 2.15 ms | |
| 2048 × 4096 | 13.40 ms (164 TFLOPS) | 9.17 ms (240 TFLOPS) | 9.18 ms |
| 4096 × 4096 | 26.02 ms (169 TFLOPS) | 17.26 ms (255 TFLOPS) | 15.98 ms |
| 2048 × 8192 | 45.81 ms (192 TFLOPS) | 26.87 ms (327 TFLOPS) | 30.17 ms |

Small shapes are limited by host dispatch: each program costs 50-180 µs of host time on
the 1×2 mesh, most of it in TT-NN's launch for two chips, so a captured trace is the
faster mode there (eager: 2.74 ms on 1 chip and 3.72 ms on 2 at 256 × 1024, 2.42 and
4.26 ms at 1024 × 2048). Collectives run on the same command queue as compute and do not
overlap it yet. Qwen2.5-0.5B (seq 128) on 1×2: 104.6 ms with the default placement,
59.5 ms replicated, logits cosine 0.999876 against float64 (CPU bf16 0.999937).

## Limits

- Activations stay in DRAM on a mesh by default (`KTT_L1_BUDGET` 0 there, 0.4 on one
  chip): some TT-NN ops still place their output in their input's memory, and on a
  mesh L1 activations then fill a bank past the budget (Qwen2.5-0.5B ran out of L1 at
  the logits). Setting `KTT_L1_BUDGET` restores L1 activations.
- Small models gain nothing from sharding: Qwen2.5-0.5B is 104 ms sharded against
  60 ms replicated on 1×2, because each layer's collectives cost more than its small
  matmuls save. `KTT_MESH_AUTO=0` keeps such a model replicated.
- Known regression on this build (one chip, `torch.compile`): the CNN training step in
  `tests/hardware/compile/test_compile_tt.py` and the Whisper case in
  `test_compile_zoo.py` fail or stall. Both run convolutions; the likely cause is that a
  convolution whose metadata no longer fits L1_SMALL now raises instead of being run
  again after the program cache is cleared. Under investigation.
- One mesh at a time; 1×2 measured (the placement and rule code is written for N-D).
- Collectives do not overlap compute.
- `P(max)` / `P(min)` complete by all-gather and a local max / min.
- fp32 partial sums complete by all-gather and a local sum (no fp32 reduce-scatter).
- `rand` / `randn` on a sharded tensor draw per chip unless `KTT_PHILOX=1`.
