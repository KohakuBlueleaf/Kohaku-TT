---
title: ktt_norm
summary: Layer norm and RMS norm over the last dim, forward (with saved mean / rstd) and backward, fp32 statistics for bf16 and fp32 inputs - what it computes, routing, coverage, grid, NoC plan, L1, compute, bound, performance against TT-NN and limits.
tags:
  - reference
  - kernels
  - normalization
---

# ktt_norm

`y = (x - mean) * rstd * gamma + beta` (layer norm) or `y = x * rstd * gamma` (RMS norm,
plus `beta` for `ktt.rms_norm`) over the last dim W of `x [..., M, W]`, with
`rstd = 1 / sqrt(var + eps)`, and the gradients `dx`, `dgamma`, `dbeta`. `x` bf16 or fp32,
`gamma` / `beta` bf16 or fp32 (each optional). Statistics and all arithmetic are fp32 for
every input dtype; the saved `mean` / `rstd` are fp32.

Source: `src/kohakutt_native/csrc/aten/nn/ktt_norm/` (`fwd_*.cpp` forward, `bwd_*.cpp`
backward, `colsum_*.cpp` the dgamma / dbeta column sum, `dm.hpp` the statistics exchange,
`program.cpp` plan and programs, `args.hpp` the argument sentinel). Routing:
`aten/nn/aten_norm.cpp`, `aten/nn/aten_fused.cpp`, `aten/nn/device_norm.cpp`.

## Routing

| Setting | Effect |
|---|---|
| default | `native_layer_norm` / `native_layer_norm_backward` over the last dim, `rms_norm` (with or without gradients) and `ktt.rms_norm` run on `ktt_norm` |
| `KTT_CUSTOM_KERNELS=0` | TT-NN: `moreh_layer_norm` (+ backward) for bf16 (bf16 statistics; over one dim they come from `ttnn::mean` and a Welford `ttnn::var`, and the backward gets x as rows `[M, W]`, because moreh's own last-dim statistics are wrong past the first row of every 32 and misread for odd rows on Blackhole), the composite for fp32; `ttnn::rms_norm` for bf16 without gradients |
| `KTT_NORM_TRACE=1` | prints each new plan to stderr |
| `KTT_NORM_GROUP`, `KTT_NORM_DEPTH` | force the group size / resident rows (sweeps) |

## Coverage

- x: TILE, interleaved, bf16 or fp32, rank >= 2; any W and any M (ragged last tiles).
- gamma / beta: absent, or 1-D of length W, bf16 or fp32, independently of x's dtype.
- `native_layer_norm` returns fp32 `mean` / `rstd` of shape `[..., M, 1]` (as on CUDA);
  the backward takes them. A layer norm over more than one trailing dim takes TT-NN's
  `moreh_layer_norm` (bf16) or the composite. Mesh-placed operands take the composite.

## How it works

Sizes in tiles: Wt = ceil(W / 32) column tiles, R tile rows (leading dims flattened, each
slice's M padded to tiles).

### Grid and groups

- A unit of work is one tile row (32 rows x W). A group of g consecutive cores shares a
  tile row: core i holds column tiles `[i seg, (i + 1) seg)`, seg = ceil(Wt / g) (the last
  core fewer). Groups take tile rows group-strided.
- g (1 to 16) is the group size whose segment fits L1 with the fewest tiles on the busiest
  core (rows per group times seg, plus two tiles' worth per row exchange when g > 1). It
  is g = 1 for many rows of moderate width; g grows when there are fewer tile rows than
  cores or a row does not fit one core.

### NoC plan

| Stream | Kernel (RISC) | NoC |
|---|---|---|
| x (and dy) segment per tile row, starting at a per-group rotation (bank spread) | reader (NCRISC) | NoC0 |
| gamma / beta segment, once | reader | NoC0 |
| mean / rstd tiles per row (backward) | reader | NoC0 |
| y / dx tiles, flushed every 4 | writer (BRISC) | NoC1 |
| statistics exchange (g > 1): row 0 of two fp32 tiles to row `index` of every peer's gather tiles, then a semaphore increment | writer | NoC1 |
| mean / rstd tiles (first core of the group), dgamma / dbeta partials | writer | NoC1 |

Both kernels run in `DM_DEDICATED_NOC` mode on different NoCs (writer = 1 - reader,
asserted in the host). No tile data is moved by RISC-V stores: receive tiles are zeroed
by NoC reads of the zero region, partial statistics leave straight from the packed tile.

### Compute: forward

- Statistics on transposed x tiles (the unpacker transposes) with TT-Metal's SFPU
  two-pass helpers (`welford.h`): each SFPU lane accumulates one row, pass one the shifted
  sum, pass two the centred sum of squares; a ragged last tile is a row count, so padding
  never enters a statistic and no mask is built. RMS norm: one pass of squares.
- Groups (g > 1): each core's (mean, M2) rows go to every peer; each core merges them on
  the SFPU about core 0's mean m0 (Chan): d_c = mean_c - m0, D = sum n_c d_c / W,
  mean = m0 + D, M2 = sum M2_c + n_c (d_c - D)^2. Merging the differences keeps the mean
  within half an fp32 ulp at large offsets (a merge of n_c mean_c products loses about half
  an ulp more).
- y: per row, one statistics tile S (mean in faces 0 / 2, rstd in faces 1 / 3, broadcast
  by the unpacker); per pair of column tiles a parameter tile P (gamma / beta rows, built
  once per launch); one SFPU traversal per pair computes `(x - m) r g + b` for both tiles.
  The last tile's padding columns are zeroed (lane-id mask).
- fp32 DST in half-sync mode; fp32 circular buffers unpack to DST unrounded; no FPU
  multiply runs, so the result does not depend on the math fidelity. Fidelity and SFPU
  approximations come from `kohakutt.numerics` with the kernel requiring exact SFPU
  functions (approximations off unless numerics is strict).

### Compute: backward

With `xh = (x - m) r` and `g = r dy gamma`, per tile row:

1. Per tile: xh and g to fp32 rows in L1; `dy xh` and `dy` accumulated over the core's
   rows by the packer (L1 accumulation) into the dgamma / dbeta partials.
2. Row sums A = sum g, B = sum g xh in DST over the segment, reduced per row; added over
   the group (same exchange as the forward).
3. Per tile: `dx = g - A / W - xh B / W` (RMS norm: without A).

Padding columns and a slice's padding rows are zeroed on the SFPU in xh, g and dy, so
non-finite padding in x or the statistics never reaches dgamma / dbeta. After the last row,
the partials `[groups * 32, W]` are summed per column by a second program.

### L1 (per core)

Forward:

| CB | Contents | Format | Tiles |
|---|---|---|---|
| 0 | x segment, resident rows | x | depth x seg |
| 1, 2 | gamma, beta segment | parameter | seg each |
| 3 | P: gamma / beta rows of each column-tile pair | fp32 | ceil(seg / 2) |
| 5 | S: the row's mean / rstd broadcast | fp32 | 1 |
| 6, 19 | mean / rstd columns, statistics out | fp32 | 2 each |
| 16 | y | x | 8 |
| 4, 7, 17, 18 | g > 1: gather (two rows), counts, partials, own count | fp32 | 4, 1, 2, 1 |

Backward:

| CB | Contents | Format | Tiles |
|---|---|---|---|
| 0, 9 | x, dy segments, resident rows | x | depth x seg each |
| 1 | gamma segment | parameter | seg |
| 8 | mean / rstd columns | fp32 | 2 x depth |
| 5, 13 | S (mean, rstd), S2 (A / W, B / W) broadcast | fp32 | 1 each |
| 10, 11 | g, xh of the row | fp32 | seg each |
| 12 | A, B columns | fp32 | 2 |
| 16 | dx | x | 8 |
| 21, 22 | dgamma, dbeta partial sums | fp32 | seg each |
| 4, 17 | g > 1: gather, partials | fp32 | 4, 2 |

depth = 2 (the reader fills the next tile row while the compute works on this one), 1 when
only that fits. The total is checked against `program::cb_budget()` before launch.

## Bound

Memory-bound. The forward moves x in and y out (2 + 2 bytes per element for bf16), the
backward x and dy in and dx out (6 bytes); statistics add two fp32 tiles per tile row.
The data-movement pattern alone (the same reader / writer split, rotation and flushes on
the 110-core grid, no compute) copies at 441-451 GB/s read + write for bf16 and 419 GB/s
for fp32 on one p150a. At 445 GB/s a core has 1367 cycles (1.35 GHz) per bf16 tile in the
forward and 2050 in the backward.

Compute per tile on one core (compute-only loop, W 1024, bf16, affine): forward 918 cycles
(statistics 375, normalisation 543), RMS norm 771. The forward is DRAM-bound with a third
of the time to spare.

## Performance

Per-call times against TT-NN are not measured in this tree yet. Compute per tile on one
core is under Bound.

## Limits

- Normalisation over more than one trailing dim is not covered (TT-NN's `moreh_layer_norm`
  for bf16, with bf16 statistics, or the composite).
- Group norm uses its own kernels (`ktt_group_norm`), not these.
- The backward's dgamma / dbeta need a second program (the column sum over the groups'
  partials).
- Each kernel checks an argument sentinel (`args.hpp`) after its last common runtime
  argument and returns at once when it is missing, so a kernel and a host built from
  different argument layouts cannot hang the card.
