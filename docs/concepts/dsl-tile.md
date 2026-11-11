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
| `.cast(kd.f32)`, `.cast(kd.bf16)` | at a load (unpack format) or at the store (pack format); free |
| `v.row()`, `v.col()` | the first row (column) of a block repeated over its tile, through the unpacker's broadcast |
| `kd.reduce_sum(row * row, dim=-1, exact=True)` | the fp32 sum of squares along the last dim, one value per row; `.col()` broadcasts it back |
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
and adds its squares into the vector unit's statistics registers (the backend's two-pass
statistics calls, `two_pass_stats_*`, as the metal-level RMS norm example uses them);
the per-row result is moved to column 0 (`transpose_dest`), the chain on the statistic
runs (for RMS norm: `rsqrt(s / D + eps)`), and the tile is packed to an fp32 buffer.
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

The bounds: all bytes at the DRAM rate (512 GB/s); the busiest core's reads at 28 GB/s
per reading core (the upper end of the per-core rate in [hardware](hardware.md)); the
busiest core's vector-unit instructions at one per cycle per 32 lanes (instruction
counts per function are estimates). Lines marked "estimate" are not measured.

## What is not built

- The matrix unit: matmul, accumulators, loop state, the fast (matrix-unit) reductions.
- Sharing between cores (`share=`, channels, core roles), pins and the route check.
- Exact sums other than the sum of squares of a loaded row, more than one reduction per
  kernel, a cast to bf16 inside a chain, `kd.where` and comparisons, `@kd.elementwise`,
  `kd.cpp`, `kd.range`, `kd.when`.
- Registration as the backend's kernel for an aten op; gradients.
