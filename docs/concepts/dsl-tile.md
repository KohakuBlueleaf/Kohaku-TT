---
title: Kernel DSL, tile level
summary: "Kernels written per unit of work (load, compute, store), traced, planned and lowered to the metal level."
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

Working examples: [`examples/kernels/dsl_tile/`](../../examples/kernels/dsl_tile/)
(SiLU, SwiGLU, RMS norm, rotary embedding in the rotate-half and complex forms).

## A kernel

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

The output's dtype is part of the kernel class too.

## Views, blocks and operations

| Written | Meaning |
|---|---|
| `x.tiles[t]` | one 32 x 32 tile at flat tile index `t` (a "tile" kernel) |
| `x.tiles[r, :]` | tile row `r`: 32 rows of the full width, leading dims folded into the row count (a "row" kernel) |
| `kd.load(view, pad=0)` | a block read from DRAM per unit; `pad` declares the value past a ragged edge (a reduction needs it) |
| `kd.resident(view)` | a block read once per core and kept for the whole kernel (a weight row) |
| `kd.store(view, value)` | writes a block to an output |
| `kd.num_tiles(x)`, `kd.num_tiles(x, -1)`, `kd.num_tile_rows(x)`, `kd.size(x, d)`, `kd.cdiv(a, b)` | sizes, as runtime integers; `+ - * // %` on them and on the unit are computed on the device |
| `+ - * /`, unary `-`, `abs`, `kd.maximum`, `kd.minimum` | per element on the vector unit; a number, a `kd.Runtime[float]` or a size is a scalar operand |
| `kd.exp`, `kd.sigmoid`, `kd.silu`, `kd.tanh`, `kd.sqrt`, `kd.rsqrt`, `kd.reciprocal`, `kd.relu`, `kd.square` | TT-Metal's vector-unit code for that function |
| `.cast(kd.f32)`, `.cast(kd.bf16)` | at a load (unpack format) or at the store (pack format): free; a cast to bf16 inside a chain rounds there, to nearest even (`km.sfpu.bf16`) |
| `v.row()`, `v.col()` | the first row (column) of a block repeated over its tile, through the unpacker's broadcast |
| `kd.reduce_sum(row * row, dim=-1, exact=True)` | the fp32 sum of squares along the last dim, one value per row; `.col()` broadcasts it back |
| `kd.reduce_sum(chain, exact=True)`, `kd.reduce_sum(row)`, `kd.reduce_max(row)` | the fp32 sum of any row block on the vector unit; the sum or max of a loaded row on the matrix unit (see [Column values](#column-values)) |
| `x.tiles[r]` in a row kernel | a column: tile r of an [..., R, 1] tensor, one value per row (load or store) |
| `blk[..., lo:hi]`, `kd.cat([a, b], dim=-1)` | column slices and joins of row blocks at multiples of 32 columns (checked at the call) |
| `a.even`, `a.odd`, `kd.pairs(even, odd)` | the even and odd columns of a block (complex pairs), and a block built from them |

Division by a scalar is a multiplication by its fp32 reciprocal (computed on the host);
division by a block is a multiplication by the vector unit's reciprocal.

## What a kernel becomes

```text
@kd.kernel function --trace--> tile IR --design--> plan --lower--> km.Program
                                                                  verify, emit, launch
```

- **Tile IR** (`kernel.ir(...)` prints it): the tensors, runtime and static values, the
  grid, and every block operation with the line that wrote it.
- **Design**, per kernel class: the form (tiles or rows), which rows are held in L1, the
  passes over a row, and each chain of per-element operations as one generated
  vector-unit function (`km.sfpu_fn`) with its register slots (4 fp32 tiles at most).
  A chain reading more blocks than that is split at its last binary operation into
  stages, run one after the other in the same register scope, each later stage taking
  the result of the one before in slot 0 (LayerNorm's `d * r * g + b`: four blocks,
  then the bias). Stages run in row and tile kernels, not on a matmul's accumulator
  or inside an exact sum.
- **Plan**, per call: the core count (a size class), whether a row fits L1, and the cost
  line. `kernel.plan(...).report()` prints it.
- **Lowering**: the metal-level program as objects (`kernel.program(...)`,
  `kernel.metal(...)` prints its model, `kernel.source(..., role=)` one kernel's C++).

The arguments of `plan`, `metal`, `program`, `source` and `ir` are those of a call; tt
tensors, CPU tensors or `kd.like(shape, dtype)` stand-ins, so no device is needed.

### Tile kernels

The units are tiles split evenly over the cores (`block` tiles per NoC barrier, at least
`block` per core). The reader reads `block` tiles of every loaded tensor per barrier; the
compute unpacks the operands of as many tiles as the register half holds (a divisor of
`block`), runs the generated function on each and packs them; the writer writes the
block and flushes once.

### Row kernels

The units are tile rows; the row's width in tiles is a runtime value. A row read only at
its own tile positions is streamed in chunks of 8 tiles. A row that is reduced and then
used again, or read at another tile's place (a slice or `kd.cat`), is held whole in L1:
its buffer is a sized value of the width (`p.sized`, classes 1, 2, 3, 4, 6, 8, 12, ...),
so a new width inside its class builds no program. A kernel class has up to three
programs: rows held two deep, held one deep, and (for a reduced row only) read twice
from DRAM, once per pass; a call takes the first whose buffers fit the L1 free at the
call. A row read at another tile's place that does not fit raises `kd.PlanError`.

With a reduction the unit runs two passes. Pass 1 unpacks each tile of the row transposed
and adds its squares into the vector unit's statistics registers (the metal level's
`c.stats_*` statements, the backend's two-pass statistics, as the metal-level RMS norm
example uses them); the per-row result is moved to column 0 (`transpose_dest`), the
chain on the statistic runs (for RMS norm: `rsqrt(s / D + eps)`), and the tile is
packed to an fp32 buffer. Runtime scalars (`eps`, `1 / D`) reach a generated function
as fp32 arguments (`km.sfpu.scalar`), not as register slots.
Pass 2 unpacks each tile again with the statistic broadcast by column and runs the
store's function.

A `kd.cat` becomes one generated function per piece, chosen on the device by the tile
index (`if (j < end) ... else ...`).

### Numerics

The destination registers are fp32 when the kernel reduces, reads or writes an fp32
tensor, casts to `kd.f32`, or calls a function that needs them (`kd.tanh`); otherwise
bf16 (8 tiles per half instead of 4). `@kd.kernel(dest="fp32")` or `dest="bf16"` fixes
the choice. Every generated function computes in fp32; each `+`, `-` and `*` rounds on
its own, a product feeding a sum is rounded first (as torch does), and the value is
rounded once to the output's format: by the function (to nearest even) on bf16
registers, by the packer on fp32 registers.
Measured on a Blackhole p150: the fp32 results of the rotary kernel equal torch's fp32
CPU arithmetic bit for bit; the packer's fp32 to bf16 rounding is to nearest with ties
away from zero (torch's `.bfloat16()` breaks ties to even), so about 0.4 % of the
rotary kernel's bf16 outputs (products of bf16 values land on ties often) differ from
torch's by one unit in the last place. Kernel
numerics options come from the `kohakutt.numerics` scope; a kernel with an exact
reduction requests `exact` (no approximate vector-unit functions).

## The plan report

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

The bounds: all bytes at the nominal DRAM rate of 512 GB/s (320-357 GB/s
is measured on a P150A, so this bound is optimistic); the busiest core's reads at 28 GB/s
per reading core (the upper end of the per-core rate in [hardware](hardware.md)); the
busiest core's vector-unit instructions at one per cycle per 32 lanes (instruction
counts per function are estimates). Lines marked "estimate" are not measured.

## Column values

A row kernel may compute several values with one number per row and use them in order:

```python
@kd.kernel
def softmax_step(s: kd.Tensor, m: kd.Tensor, den: kd.Tensor,
                 p: kd.Out, m_out: kd.Out, den_out: kd.Out, scale: kd.Runtime[float]):
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

Every row is held whole in L1 for its unit (rows that do not fit raise
`kd.PlanError`). Per unit, each column value is computed in the order the kernel uses
it, into a one-page fp32 buffer: a reduction, or a chain on columns that a row chain
reads through `.col()`. Then the row stores run tile by tile, then the column stores.

- `kd.reduce_max(row)`, `kd.reduce_sum(row)`: a loaded row on the matrix unit
  (`c.reduce_init` / `c.reduce` with a tile of ones the compute packs once). Its
  inputs pass through the 19-bit source registers, so an fp32 row's max or sum is of
  rounded values; tile padding is not masked, so the row width must be a multiple of 32
  (checked at the call).
- `kd.reduce_sum(chain, exact=True)`: any row chain of at most two blocks, recomputed
  tile by tile, transposed in the registers and added in fp32 into the statistics
  registers, which wait in slots 2 and 3 (`c.stats_save` / `c.stats_restore`) while the
  chain runs. A ragged last tile counts only its columns.
- A column output has the shape and dtype of the kernel's first column input (else
  [..., R, 1] in fp32).

The example's results and times are in `examples/kernels/dsl_tile/online_softmax.py`
and its card tests: the max equals torch's, p is the fp32 exp rounded once, and the
sum is within 2e-6 of the float64 sum.

## Matmul kernels

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

| Written | Meaning |
|---|---|
| `kd.grid2d(rows=R, cols=C)`, `g.units()` | output blocks (i, j): tile row i, column group j |
| `kd.accumulator((1, nt))` | `nt` <= 4 fp32 output tiles summed in the destination registers |
| `kd.range(n)` | a device loop; its variable is symbolic while tracing |
| `acc += a_block @ w_block` | a product on the matrix unit, added to the accumulator |
| `x.tiles[r, c]`, `x.tiles[r, c0:c1]` | one tile, or tiles c0 .. c1 - 1 of tile row r |
| `row_block.tile(k)` | tile k of a row block computed in the kernel (a matmul's left operand) |

The form is fixed: the left operand is `kd.load(a.tiles[i, k])` or `h.tile(k)` with
`h` a row block of unit i, the right one `kd.load(w.tiles[k, j * nt:(j + 1) * nt])`
(w stored [K, N]), the store `out.tiles[i, j * nt:(j + 1) * nt]`; the stored value is a
vector chain on the accumulator that may read a row of an [N] tensor through `.row()`
(a bias) and scalars. Anything else raises `kd.PlanError` at its line. The output has
the left operand's leading sizes and w's last size.

The lowering is the layout of the backend's matmul program (the metal-level
`matmul_km.py`): column groups of `nt` tiles, each with a share of the cores; a core
reads its group's block of w (K x `nt` tiles, at most 256 pages) once and keeps it, reads
a in chunks of up to 32 K tiles per output row tile, sums the row's products in the
registers, runs the chain on each accumulator slot (an extra operand such as the bias
takes a slot past the accumulator's) and packs. A chain-free kernel equals
`matmul_km.py` bit for bit.

With `h.tile(k)` the left operand is computed in the kernel: the row kernel that
computes `h` (any row kernel of the sections above, its rows held whole in L1) runs as
the prologue of each output row, packs `h` into an L1 buffer of the operand's format
(`h` ends in `.cast(w.dtype)`), and the products read that buffer. Each column group
recomputes its rows' `h`, and the buffer is sized exactly by K (one program per K).

### A shared weight

```python
for k in kd.range(kd.num_tiles(a, -1)):
    wb = kd.load(w.tiles[k, j * nt : (j + 1) * nt], share=g.col)
    acc += kd.load(a.tiles[i, k]) @ wb
```

`share=g.col` says that the units of one grid column use the same block of w. The
kernel is then lowered in the layout of the backend's block matmul (the metal-level
`gemm_km.py`): each core owns a range of output row tiles and reads only its own rows of
a; the top core of each core column reads each block of w once and multicasts it to the
cores below it (a metal-level lane, `p.lane`), so w is read once per core column
instead of once per core. The plan (block of `bm` x `bn` output tiles, K blocks of
`kb`, the output subblock) is the backend's search, computed per call; partial sums
over K blocks are kept in L1 by packer accumulation. The kernel's chain runs on each
output subblock of the last K block before its pack, with its extra operands (a bias)
in the register slots the subblock leaves free (a subblock of at most 4 minus their
count). A chain-free kernel equals `gemm_km.py` bit for bit. The left operand is a
load (`row.tile(k)` with a shared w is not built). It needs a grid of two core rows.

## What is not built

- Matmul kernels beyond the form above (a transposed right operand, w not held per
  core, several accumulators), loop state (`kd.state`), the matrix unit's whole-tile ops.
- Sharing between cores beyond a shared w (`share=g.row` for a, channels, core
  roles, progress), pins and the route check.
- Reductions along other dims, a matrix-unit reduction of a computed row, column values
  in tile kernels, `kd.where` and comparisons, `@kd.elementwise`, `kd.cpp`, `kd.when`.
- Registration as the backend's kernel for an aten op; gradients.
