---
title: Kernel DSL, tile level
summary: "A guide to kernels written per unit of work (load, compute, store): first kernel, rows and reductions, matmuls and sharing, attention; the kd.* reference and what is not built."
tags:
  - concepts
  - dsl
---

# Kernel DSL: the tile level

`import kohakutt.dsl as kd` is the tile level of the kernel DSL. A kernel says what each
unit of work loads, computes and stores; the cores, circular buffers, NoC calls, register
slots and vector-unit code are derived. The result is a metal-level program
([`kohakutt.dsl.metal`](dsl.md)) built as Python objects, so every tile-level kernel
passes the same verifier, emitter and launch path as a hand-written metal-level one.
The tile level is experimental: its syntax may change between releases. Simple kernels
(norms, activations, rotary) lower to their hand-written speed; a complex one such as
flash attention takes 2.7x to 3x its metal-level twin's time today, so write a kernel
whose speed matters at the [metal level](dsl.md).

The guide goes from a first kernel to rows and reductions, matmuls and sharing, and
attention. Each step has a working example in
[`examples/kernels/dsl_tile/`](../../examples/kernels/dsl_tile/) with its measured
times; the [reference](#reference) lists every `kd.*` name.

## 1. A first kernel

```python
import kohakutt.dsl as kd


@kd.kernel
def swiglu(x: kd.Tensor, g: kd.Tensor, out: kd.Out, block: kd.Static[int] = 8):
    for t in kd.grid(kd.num_tiles(out), block=block):
        a = kd.load(x.tiles[t])
        b = kd.load(g.tiles[t])
        kd.store(out.tiles[t], a * kd.sigmoid(a) * b)


y = swiglu(x, g)  # tt tensors in, a tt tensor out
```

The function runs once per kernel class with proxy values: `kd.grid` runs its body once
with a symbolic unit, and operators on blocks record operations. A Python `if` on a size
raises `kd.PlanError` at that line.

| Parameter annotation | Meaning | A new value costs |
|---|---|---|
| `kd.Tensor` | an input tensor (bf16 or fp32); its sizes are runtime values | nothing |
| `kd.Out` | an output tensor, with the sizes and dtype of the first input (`out_dtype=` at the call changes the dtype) | nothing |
| `kd.Runtime[float]`, `kd.Runtime[int]` | a value given at every call (a float travels as its fp32 bits) | nothing |
| `kd.Static[int]` | part of the kernel class; `kd.auto(8, 4, 2, 1)` tries candidates in order and keeps the first whose program verifies within the L1 budget | one compile per value |

The output's dtype is part of the kernel class too. A kernel is a torch op
(`kohakutt_tile::<name>`, with a fake implementation), so it runs inside
`kohakutt.capture` and `torch.compile(backend="ktt_tt")` like any other op, and a new
size inside a size class builds no program ([5. In PyTorch](#5-in-pytorch)).

The units are tiles split evenly over the cores (`block` tiles per NoC barrier, at least
`block` per core). The reader reads `block` tiles of every loaded tensor per barrier; the
compute unpacks the operands of as many tiles as the register half holds (a divisor of
`block`), runs the generated function on each and packs them; the writer writes the
block and flushes once. Example: `swiglu.py` (equal to the metal-level SwiGLU bit for
bit), `rope.py` (`rope_complex`: column pairs).

### What a kernel becomes

```text
@kd.kernel function --trace--> tile IR --design--> plan --lower--> km.Program
                                                                  verify, emit, launch
```

- **Tile IR** (`kernel.ir(...)` prints it): the tensors, runtime and static values, the
  grid, and every block operation with the line that wrote it.
- **Design**, per kernel class: the form (tiles, rows, runs, matmul, attention), which
  rows are held in L1, the passes over a row, and each chain of per-element operations
  as one generated vector-unit function (`km.sfpu_fn`) with its register slots (4 fp32
  tiles at most). A chain reading more blocks than that is split at its last binary
  operation into stages run in one register scope, each later stage taking the result
  of the one before in slot 0.
- **Plan**, per call: the core count (a size class), whether a row fits L1, and the cost
  line. `kernel.plan(...).report()` prints it.
- **Lowering**: the metal-level program as objects (`kernel.program(...)`;
  `kernel.metal(...)` prints its model, `kernel.source(..., role=)` one kernel's C++).

The arguments of `plan`, `metal`, `program`, `source` and `ir` are those of a call: tt
tensors, CPU tensors or `kd.like(shape, dtype)` stand-ins, so no device is needed.

```text
kernel    rms_norm (row form, fp32 registers)
grid      16 tile rows over 16 cores (1 per core), block 1
stream    x (bfloat16) x.tiles[unit, :]: row held in L1
stream    scale (bfloat16) scale.tiles[0, :]: resident, read once per core
store     out.tiles[unit, :] (bfloat16)
  vector  rms_norm_fn2: 3 operands, setup mul
pass 1    exact sum of squares (vector unit, fp32), statistic chain rms_norm_fn1
L1        324 KB of buffers per core, free unknown
moved     4.72 MB (3.15 read, 1.57 written), multiply work 0 (matrix unit), launches 1
bound     DRAM 9.2 us, per-core reads (estimate) 7.0 us, vector unit (estimate) 12.1 us
binds     vector unit (estimate)
```

The bounds: all bytes at the nominal DRAM rate of 512 GB/s (320-357 GB/s is measured on
a P150A, so this bound is optimistic); the busiest core's reads at 28 GB/s per reading
core; the busiest core's vector-unit instructions at one per cycle per 32 lanes.
Lines marked "estimate" are not measured.

### Numerics

The destination registers are fp32 when the kernel reduces, reads or writes an fp32
tensor, casts to `kd.f32`, or calls a function that needs them (`kd.tanh`); otherwise
bf16 (8 tiles per half instead of 4). `@kd.kernel(dest="fp32")` or `dest="bf16"` fixes
the choice. Every generated function computes in fp32; each `+`, `-` and `*` rounds on
its own, a product feeding a sum is rounded first (as torch does), and the value is
rounded once to the output's format: by the function (to nearest even) on bf16
registers, by the packer on fp32 registers. The packer's fp32 to bf16 rounding breaks
ties away from zero (torch's `.bfloat16()` to even), so about 0.4 % of the rotary
kernel's bf16 outputs differ from torch's by one unit in the last place; its fp32
results equal torch's CPU arithmetic bit for bit. Kernel numerics options come from the
`kohakutt.numerics` scope; a kernel with an exact reduction requests `exact` (no
approximate vector-unit functions).

## 2. Rows and reductions

```python
@kd.kernel
def rms_norm(x: kd.Tensor, scale: kd.Tensor, out: kd.Out, eps: kd.Runtime[float]):
    g = kd.resident(scale.tiles[0, :])
    for t in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        ms = kd.reduce_sum(row * row, dim=-1, exact=True) / kd.size(x, -1)
        r = kd.rsqrt(ms + eps)
        kd.store(out.tiles[t, :], (row * r.col() * g.row()).cast(out.dtype))
```

The units are tile rows; the row's width in tiles is a runtime value. A row read only at
its own tile positions is streamed in chunks of 8 tiles. A row that is reduced and then
used again, or read at another tile's place (a slice or `kd.cat`), is held whole in L1:
its buffer is a sized value of the width (classes 1, 2, 3, 4, 6, 8, 12, ...), so a new
width inside its class builds no program. A kernel class has up to three programs: rows
held two deep, held one deep, and (for a reduced row only) read twice from DRAM; a call
takes the first whose buffers fit the L1 free at the call.

With the row statistic the unit runs two passes. Pass 1 unpacks each tile transposed and
adds its squares into the vector unit's statistics registers (the metal level's
`c.stats_*`, the backend's two-pass statistics); the per-row result moves to column 0,
the chain on it runs (`rsqrt(s / D + eps)`), and the tile is packed to an fp32 buffer.
Runtime scalars (`eps`, `1 / D`) reach a generated function as fp32 arguments, not
register slots. Pass 2 unpacks each tile again with the statistic broadcast by column
and runs the store's function. Example: `rms_norm.py` (equal to the metal-level norm bit
for bit, 6x to 10x faster than Gemma's norm module run eagerly).

### Column values

A row kernel may compute several values with one number per row and use them in order
(`online_softmax.py`, `layer_norm.py`):

```python
for r in kd.grid(kd.num_tile_rows(s)):
    row = kd.load(s.tiles[r, :])
    m_old, den_old = kd.load(m.tiles[r]), kd.load(den.tiles[r])
    m_new = kd.maximum(m_old, kd.reduce_max(row))
    e = kd.exp((row - m_new.col()) * scale)
    alpha = kd.exp((m_old - m_new) * scale)
    kd.store(p.tiles[r, :], e)
    kd.store(m_out.tiles[r], m_new)
    kd.store(den_out.tiles[r], den_old * alpha + kd.reduce_sum(e, exact=True))
```

Every row is held whole in L1 for its unit. Per unit, each column value is computed in
the order the kernel uses it, into a one-page fp32 buffer: a reduction, or a chain on
columns that a row chain reads through `.col()`. Then the row stores run, then the
column stores.

A row store runs 2 (or 4) tiles per register scope where its operands fit the four
fp32 slots: per stage of its chain, the operands the same for every tile of the unit
(column values, a row's statistic) are unpacked once into shared slots, then each
tile's own operands into a scratch slot before its function runs; each tile keeps its
result in its own slot (LayerNorm's affine: `(x - mean) * r` for both tiles, then
`* g + b` per tile). The operations and their rounding are those of one tile per
scope. A store of one row tile with at most two column values and two resident rows
(LayerNorm's and RMSNorm's) instead runs as one vector function over two tiles that
reads each broadcast value once per band of rows: the column values joined into one
tile per unit, the resident rows' tiles merged pairwise into one tile per pair, once
per core (rows up to 128 tiles). Units go to the cores in turn (core c takes c, c +
cores, ...).

- `kd.reduce_max(row)`, `kd.reduce_sum(row)`: a loaded row on the matrix unit (with a
  tile of ones the compute packs once). Its inputs pass through the 19-bit source
  registers, and tile padding is not masked, so the row width is a multiple of 32
  (checked at the call).
- `kd.reduce_sum(chain, exact=True)`: any row chain, recomputed tile by tile, transposed
  in the registers and added in fp32 into the statistics registers. A ragged last tile
  counts only its columns. An exact sum of a stored row is taken by the packer while
  the row is stored (L1 accumulation) instead of a second walk.
- LayerNorm's `mean = reduce_sum(r, exact) / D` then `reduce_sum((r - mean.col())^2,
  exact)` is recognised and computed by one statistics scope (shifted sums, the mean,
  then M2 about it), the backend norms' calls.
- A column output has the shape and dtype of the kernel's first column input (else
  [..., R, 1] in fp32).

### A row split over cores

A kernel with a row statistic (an exact sum, LayerNorm's moments) may run each tile row
on `g` consecutive cores, each holding a contiguous slice of the row's tiles and of
resident rows such as a norm's weight. G groups take rows k, k + G, ... Each exact sum
becomes the slice's fp32 partial; the reader takes the page the compute packed
(`k.front`), meets its group at a sync point, fetches the `g` partials of its group and
syncs again before letting the page go; the compute adds them in fp32. LayerNorm's mean
and M2 are one exchange: each slice's mean and M2 about it, combined by Chan's formula
(the mean weighted by the slices' element counts, M2 as the slices' M2 plus n_k (mean_k
- mean)^2). Each core stores its slice, a chunk of 8 tiles per write flush.

- `kd.grid(n)`: the plan builds the splits 2, 4 and 8 and picks per call the one (or one
  core per row) whose busiest core's time model is lowest: per tile, the last row's
  tiles again (computed after their read), and the exchanged pages per row; by the
  sizes' classes, so every size of a class takes the same program. Splitting pays even
  with two rows per core (7040 x 2048: 168 against 200 us for RMSNorm): a core's last
  row is computed after its read, and a slice is shorter.
- `kd.grid(n, group=g)` pins g (`group=1`: one core per row, the sum order of the
  metal-level norm, bit for bit).

### Runs of tiles (GroupNorm, weight gradients)

A unit may be a contiguous run of tiles in flat order, `x.tiles[a:b]` (in an NCHW
tensor, the channels of one group: `x.tiles[u * run : (u + 1) * run]`), held in L1:

| Written | Meaning |
|---|---|
| `kd.reduce_sum(run_chain, exact=True)` | the fp32 sum of every element of the run (the last tile column counted to the width, rows past the height counted out) |
| `value.all()` | a per-unit value over a whole tile |
| `p.each(k)` | tile j of the run reads tile j // k of block p (per-channel tiles, one constant tile per channel made on the host); per-channel blocks get their own buffers |
| `x.tiles[c : n : wt]` | a strided run: a column of tiles (tile c of every tile row) |
| `kd.reduce_sum(run_chain, dim=-2, exact=True)` | the fp32 sums over the run's rows, one per column (a weight gradient) |
| `kd.store(out.tiles[u], value)` | a per-unit value stored as one tile; the output is [D] (the run tensor's last dim) |

The run's length is the same for every unit, and the tensor's height is at most 32 or a
multiple of 32 (checked at the call). Examples: `group_norm.py` (4x to 40x faster than
`F.group_norm` at the SDXL UNet's shapes), `rms_norm_backward.py` (dx and dweight).

## 3. Matmuls and sharing

```python
@kd.kernel
def linear_gelu(a: kd.Tensor, w: kd.Tensor, b: kd.Tensor, out: kd.Out, nt=2):
    g = kd.grid2d(rows=kd.num_tile_rows(a), cols=kd.cdiv(kd.num_tiles(w, -1), nt))
    for i, j in g.units():
        acc = kd.accumulator((1, nt))
        for k in kd.range(kd.num_tiles(a, -1)):
            acc += kd.load(a.tiles[i, k]) @ kd.load(w.tiles[k, j * nt : (j + 1) * nt])
        x = acc + kd.load(b.tiles[0, j * nt : (j + 1) * nt]).row()
        kd.store(out.tiles[i, j * nt : (j + 1) * nt], gelu_tanh(x))
```

The form is fixed: the left operand is `kd.load(a.tiles[i, k])` or `h.tile(k)` with `h`
a row block of unit i, the right one `kd.load(w.tiles[k, j * nt:(j + 1) * nt])` (w stored
[K, N]), the store `out.tiles[i, j * nt:(j + 1) * nt]`; the stored value is a vector
chain on the accumulator that may read a row of an [N] tensor through `.row()` (a bias)
and scalars. The output has the left operand's leading sizes and w's last size.

The lowering is the layout of the backend's matmul program (the metal-level
`matmul_km.py`): column groups of `nt` tiles, a core keeping its group's block of w,
reading a in chunks of up to 32 K tiles, summing in the registers and running the chain
on each accumulator slot. A chain-free kernel equals `matmul_km.py` bit for bit. With
`h.tile(k)` the row kernel computing `h` runs as the prologue of each output row
(`norm_linear`).

### A shared weight, two shared operands

```python
for k in kd.range(kd.num_tiles(a, -1)):
    ab = kd.load(a.tiles[i, k], share=g.row)
    wb = kd.load(w.tiles[k, j * nt : (j + 1) * nt], share=g.col)
    acc += ab @ wb
```

`share=g.col` on w: the units of one grid column use the same block of w. The kernel is
lowered in the layout of the backend's block matmul (`gemm_km.py`): each core owns a
range of output rows; the top core of each core column reads each block of w once and
multicasts it down (a metal-level lane); partial sums over K blocks stay in L1 by packer
accumulation; the chain runs on the last K block before the pack. With `share=g.row` on a
as well, the whole grid computes a 2-D split of the output: the left core of each row
multicasts a along the row, the top core of each column multicasts w down it, so a and w
are each read once. The plan is the backend's 2-D plan; where the backend runs that
program, the output equals `a @ w` bit for bit, and the time is at parity
(`shared_linear.py`).

## 4. Attention: carried values

```python
for u in kd.grid(heads * blocks):
    head, b = u // blocks, u % blocks
    first, n = head * qt + b * R, kd.minimum(R, qt - b * R)
    qb = kd.load(q.tiles[first : first + n, :])
    m, den, acc = kd.carry(-1e30), kd.carry(0.0), kd.carry(0.0)
    for j in kd.range(kd.cdiv(kt, B)):
        k0, nk = head * kt + j * B, kd.minimum(B, kt - j * B)
        kb, vb = kd.load(k.tiles[k0 : k0 + nk, :]), kd.load(v.tiles[k0 : k0 + nk, :])
        s = kd.mask_scores((qb @ kb.T) * scale, masks, b * R, j * B, kd.size(k, -2))
        m_new = kd.maximum(m, kd.reduce_max(s))
        p = kd.exp(s - m_new.col())
        alpha = kd.exp(m - m_new)
        den.next(den * alpha + kd.reduce_sum(p))
        acc.next(acc * alpha.col() + p @ vb)
        m.next(m_new)
    kd.store(out.tiles[first : first + n, :], (acc / den.col()).cast(out.dtype))
```

The form: per unit a block of query tile rows of one head; per iteration of `kd.range`
a block of key tiles; one `q @ k.T`, one `p @ v`, one `kd.reduce_max` of the scores and
one `kd.reduce_sum` of the probabilities. The probability chain, the column values (the
running maximum and sum, alpha) and the stored chain are generated vector-unit functions
as in the other forms; the scores are `(q @ k.T)` times scalars.

Lowering, per query row of the block and key block: the scores four tiles per register
scope (a mask tile copied into a slot, the products accumulated onto it on the matrix
unit in one block call per inner tile, the score chain), packed fp32 and bf16; the row
max on the matrix unit from the bf16 copy; consecutive column chains (m_new, alpha) in
one scope; p packed bf16, several tiles per scope with the column operand unpacked
once; the row sum on the matrix unit; then the accumulator tiles, two per scope,
rescaled by their chain and `p @ v` added in the same registers. Carried values live in
buffers that rotate per row: the old value at the front, the new one pushed behind it,
the old one popped at the row's end. q, k and v are bf16; key tiles past a block's end
score -1e30, and the reader repeats the block's last key tile there. Each query block
reads every key block from DRAM (no sharing between cores yet).

With at most 3 key tiles per block (`keys=3`) and the online-softmax roles above
(m_new = max(m, row max), alpha, den * alpha + row sum), a step is one register scope
for the scores, their row max (with m, on the vector unit) and p, then one for alpha,
then the accumulator and den: the row sum is `p @ ones` on the matrix unit, added onto
den * alpha like `p @ v` onto acc * alpha. Scores never leave the registers. Measured
slower than the two-phase form with 16 key tiles today (see the examples' README).

`kd.exp2(x)` is the backend flash kernel's 2^x under default numerics (exponent and a
degree-2 polynomial of the fraction, relative error up to 1.75e-3), exactly 1 at 0 so
that alpha = exp2(m - m) rescales nothing; the example writes its scores in base 2
(the scale times log2 e). Example: `flash_attention.py` (causal and not, grouped heads,
ragged lengths; error equal to the backend's flash kernel).

## 5. In PyTorch

- **Capture and compile.** A tile kernel is a torch op with a fake implementation: it
  replays inside `kohakutt.capture` and runs under `torch.compile(backend="ktt_tt")`;
  replays build nothing (`in_use.py`: replay and compiled calls equal eager bit for
  bit).
- **Autograd.** Pair kernels in a `torch.autograd.Function` (`in_use.py`:
  `RMSNormFunction` with `rms_norm` forward and `rms_norm_backward` +
  `rms_norm_dweight`; gradients within 5e-7 of float64 autograd; training losses equal
  to the torch-op model within 1e-3).

## Reference

| Name | Meaning |
|---|---|
| `@kd.kernel`, `@kd.kernel(name=, dest=)` | a kernel; `kernel(...)` calls it, `.plan` / `.ir` / `.metal` / `.program` / `.source` show it |
| `kd.Tensor`, `kd.Out`, `kd.Runtime[t]`, `kd.Static[int]`, `kd.auto(...)` | parameter annotations (above) |
| `kd.like(shape, dtype)` | a stand-in tensor for `plan` and friends |
| `kd.grid(n, block=1, group=None)` | units 0 .. n - 1 over the cores; `group=g`: a row on g cores (None: the plan picks, 1: one core per row) |
| `kd.grid2d(rows=, cols=)`, `g.units()`, `g.row`, `g.col` | matmul output blocks (i, j) and their sharing groups |
| `kd.range(n)` | a device loop inside the grid (a matmul's K, attention's key blocks) |
| `x.tiles[t]`, `x.tiles[r, :]`, `x.tiles[r, c]`, `x.tiles[r, a:b]` | a tile, a tile row, one tile or tiles of a row |
| `x.tiles[a:b]`, `x.tiles[a:b:s]`, `x.tiles[a : a + n, :]` | a run of tiles, a strided run, a block of tile rows |
| `kd.load(view, pad=, share=)`, `kd.resident(view)`, `kd.store(view, value)` | read per unit, read once per core, write |
| `kd.num_tiles(x, dim)`, `kd.num_tile_rows(x)`, `kd.size(x, d)`, `kd.cdiv(a, b)`, `kd.minimum(a, b)` on integers | sizes and integer arithmetic (host or device) |
| `+ - * /`, unary `-`, `abs`, `kd.maximum`, `kd.minimum` | per element on the vector unit; numbers, runtime floats and sizes are scalars |
| `kd.exp`, `kd.sigmoid`, `kd.silu`, `kd.tanh`, `kd.sqrt`, `kd.rsqrt`, `kd.reciprocal`, `kd.relu`, `kd.square` | TT-Metal's vector-unit functions |
| `kd.exp2` | 2^x as the backend flash kernel computes it (relative error up to 1.75e-3), exactly 1 at 0 |
| `kd.lagging_max(m, r, lag)` | `r` where it exceeds `m` by more than `lag`, else `m`: a softmax reference allowed to lag |
| `.cast(kd.f32)`, `.cast(kd.bf16)`, `kd.f32`, `kd.bf16` | formats: free at a load or store, rounding (to nearest even) inside a chain |
| `.row()`, `.col()`, `.all()`, `.each(k)` | broadcasts: first row, first column, a per-unit value, per-channel tiles |
| `blk[..., lo:hi]`, `kd.cat([a, b], dim=-1)`, `.even`, `.odd`, `kd.pairs(e, o)` | column slices, joins, column pairs |
| `kd.reduce_sum(x, dim=-1, exact=)`, `kd.reduce_max(x)` | row (or run, or `dim=-2` column) reductions |
| `kd.accumulator((1, nt))`, `acc += a @ b`, `.tile(k)` | the matmul form |
| `blk.T`, `a @ b` | a transposed right operand; products in the attention form |
| `kd.carry(init)`, `c.next(value)` | a value carried across `kd.range` |
| `kd.mask_scores(s, masks, q0, k0, keys, causal)`, `kd.score_masks(keys)` | attention masks (ragged keys, causal) and their tiles |
| `kd.PlanError` | what the tile level refuses, with the kernel's line |

## What is not built

- Sharing of attention's key and value blocks between cores (the backend's flash kernel
  multicasts them down core columns); the running maximum and sum kept per row in the
  vector unit's lanes (the backend's flash kernel's state); attention's backward.
- Matmul kernels beyond the forms above (w not held per core, several accumulators
  outside the attention form), the matrix unit's whole-tile operations.
- Channels, dedicated sender cores (`kd.on`), progress counters, pins and the route
  check; a split over cores for kernels other than row and column kernels (a GroupNorm
  group, a run).
- Reductions along other dims than those above, a matrix-unit reduction of a computed
  row, column values in tile kernels, `kd.where` and comparisons, `@kd.elementwise`,
  `kd.cpp`, `kd.when`.
- Registration as the backend's kernel for an aten op.
