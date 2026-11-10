---
title: ktt_softmax
summary: Softmax over the last dim for bf16 and fp32, any rank and width, fp32 statistics on the SFPU, the tile padding of a ragged width ignored inside the kernel and -inf treated as a mask - what it computes, routing, coverage, grid, NoC plan, L1, compute, bound, performance against TT-NN and limits.
tags:
  - reference
  - kernels
  - normalization
---

# ktt_softmax

`y = exp(x - m) / sum(exp(x - m))` over the last dim W of `x [..., M, W]`, with `m` the
row max. `x` bf16 or fp32, `y` in x's dtype. The row max, the exponentials, the row sum
and the scale are fp32 for every input dtype; `y` is rounded once.

Source: `src/kohakutt_native/csrc/aten/nn/ktt_softmax/` (`reader.cpp`, `compute.cpp`,
`writer.cpp`, `program.cpp` plan and program, `args.hpp` the argument sentinel; the
statistics exchange is `ktt_norm`'s `dm.hpp`). Routing: `bridge::softmax` in
`aten/reduce/device_reduction.cpp`.

## Routing

| Setting | Effect |
|---|---|
| default (`KTT_OWN_SOFTMAX=auto`) | `_softmax` over a last dim that is not a multiple of 32 (`torch.softmax`, `F.softmax`), bf16 / fp32, runs on `ktt_softmax`; an aligned last dim runs TT-NN's softmax unchanged; on one card and per chip of a mesh |
| `KTT_OWN_SOFTMAX=1` | `ktt_softmax` on every bf16 / fp32 call over the last dim (tests) |
| `KTT_OWN_SOFTMAX=0` or `KTT_CUSTOM_KERNELS=0` | TT-NN's softmax only; a ragged last dim first gets its tile padding zeroed (`ttnn::fill_implicit_tile_padding`, one more program), since TT-NN's softmax reads it |
| `KTT_SOFTMAX_TRACE=1` | prints each new plan to stderr |

`KTT_OWN_SOFTMAX` is read per call. On a ragged last dim the kernel replaces TT-NN's
softmax plus the padding fill; on an aligned one TT-NN's softmax reads no padding and
stays (the Performance table also measures `ktt_softmax` there).

`_safe_softmax` (SDPA's masking) takes its probabilities over the last dim from it too.
Softmax over another dim and the backward (`_softmax_backward_data`, a composite of a
product, a row sum and two binaries) stay on TT-NN. `log_softmax` is a composite of
TT-NN reductions and binaries (see Limits).

## Coverage

- x: TILE, interleaved, bf16 or fp32, any rank (a 1-D input is one row), any W and any
  M (ragged last tiles in either dim).
- Mesh: the bridge op runs on whatever the placement rules hand it. A replicated input
  runs on both chips; an input sharded on a dim other than the softmax dim runs per chip
  on its rows (`local_softmax` in `aten/nn/aten_softmax.cpp`); an input sharded on the
  softmax dim keeps its existing composite (max and sum as statistics across the
  chips), which does not call this kernel. The 1x2 mesh suites
  (`tests/hardware/mesh/test_mesh_unary.py`, `test_mesh_reduce.py`, with ragged
  softmax cases such as `63 x 65`) pass on this route.

## How it works

Sizes in tiles: Wt = ceil(W / 32) column tiles, R tile rows (leading dims flattened,
each slice's M padded to tiles).

### Grid and groups

The plan is `ktt_norm`'s: a unit of work is one tile row; a group of g consecutive cores
shares a tile row, core i holding column tiles `[i seg, (i + 1) seg)`; groups take tile
rows group-strided, or (g = 1 with more tile rows than cores) the next row from a
counter shared by all cores. g (1 to 16) is the group size whose segment fits L1 with the
fewest tiles on the busiest core (plus two tiles' worth per exchange when g > 1): g = 1
for many rows of moderate width, larger g for few rows or rows wider than one core's L1.

### NoC plan

| Stream | Kernel (RISC) | NoC |
|---|---|---|
| x segment per tile row, starting at the row's rotation (bank spread) | reader (NCRISC) | NoC0 |
| row index to the compute and the writer (16-byte pages) | reader | local |
| shared row counter (dynamic rows) | reader | NoC0 atomic |
| (max, sum) rows to every peer of the group, then a semaphore increment | writer (BRISC) | NoC1 |
| y tiles from the row's rotation, flushed every 4 | writer | NoC1 |

### Compute

All arithmetic is SFPU work on fp32 DST (half-sync, four tiles per half); no FPU
multiply runs, so the math fidelity does not matter. Per tile row of the core's segment:

1. **Max** (one DST acquire): the segment's tiles are maxed elementwise into one tile,
   then each row of 32 columns is folded inside the SFPU lanes (pair and face maxima,
   then three lane rotations) and written back as a column broadcast. The broadcast row
   max m is packed with `m~ = (m == -inf ? 0 : m)` in faces 0 / 2 and m in faces 1 / 3.
2. **Exponentials** (three tiles per acquire): `e = exp(x - m~)` (TT-Metal's guarded
   fp32 exp: -inf and underflow to 0, NaN of either sign to NaN), packed to an fp32 ring
   of the segment.
3. **Sum** (one acquire): the e tiles added elementwise and folded as in 1; `r = 1 / s`
   (two Newton steps), NaN where m or s is not finite.
4. **Output** (three tiles per acquire): `y = e r`, rounded once to x's dtype.

A group (g > 1) exchanges each core's m and s as rows after step 3 and merges them on the
transposed gather (column c: core c): `M = max m_c`, `S = sum exp(m_c - M) s_c`, and
each core's scale `r_c = exp(m_c - M) / S` (NaN where M or S is not finite).

Padding and masks:

- Columns: only the core holding a ragged last column tile masks, and only that tile:
  its padded columns are replaced by -inf before the max and before the exponential (so
  e is exactly 0 there), and y is written 0 there. Nothing is added to a padded value.
- Rows: rows never mix (every step is per row), so a slice's padded rows cost nothing
  extra; on the slice's last tile row y's padded rows are written 0.
- -inf: an element of -inf gives `exp(-inf - m~) = 0`; a row (or a group core's
  segment) of all -inf has `m~ = 0`, `e = 0`, `s = 0` and no inf - inf. A row of all
  -inf gives NaN, as torch.
- +inf or NaN (either sign) in a row: NaN for the whole row, as torch. A +inf or a
  positive NaN becomes the row max; a NaN with the sign bit set (x86's default NaN) is
  not taken by the max but gives a NaN e and so a NaN sum. On bf16 output a NaN may read
  back as inf (see [limitations](../limitations.md)).

### L1 (per core)

| CB | Contents | Format | Tiles |
|---|---|---|---|
| 0 | x segment, resident rows | x | depth x seg |
| 5 | m~ / m of the row (broadcast) | fp32 | 1 |
| 6 | r of the row (broadcast) | fp32 | 1 |
| 7 | e of the row's segment | fp32 | seg |
| 16 | y | x | 8 |
| 24, 25 | row indices to compute / writer | 16-byte pages | 4 each |
| 4, 17 | g > 1: gather (two rounds of two rows), partials | fp32 | 4, 2 |

depth = 2 (the reader fills the next tile row while the compute works on this one), 1
when only that fits. The total is checked against `program::cb_budget()` before launch.

## Bound

Memory traffic is x in and y out (2 + 2 bytes per element for bf16, 4 + 4 for fp32);
at about 445 GB/s a core of the full grid has about 1370 cycles per bf16 tile. The
kernel is compute-bound instead: `[8, 12, 224, 224]` bf16 (4704 tiles, 43 per core)
takes 96 us, about 3000 cycles per tile, against a DRAM bound of 43 us. Replacing the
exponential by the identity (measurement only) gave 62 us, so the exponential costs
about 1150 cycles per tile (36 per SFPU row) and the remaining passes (the elementwise
max and sum, the fp32 e ring written once and read twice, the per-acquire statistics
tiles) about 1900.

## Performance

Device time per call (trace of 20 calls, median replay / 20), one p150a, warm, us.
"TT-NN" is the route `KTT_OWN_SOFTMAX=0` takes (with the padding fill on a ragged last
dim); "TT-NN aligned" is TT-NN's softmax on the tile-aligned shape of the same tile
count, i.e. TT-NN without the fill.

Ragged last dim (the default route is `ktt_softmax`):

| Shape | bf16 own | bf16 TT-NN | bf16 TT-NN aligned | fp32 own | fp32 TT-NN | fp32 TT-NN aligned |
|---|---|---|---|---|---|---|
| [8, 197, 197] | 22.2 | 31.5 | 18.6 | 25.9 | 34.4 | 22.7 |
| [16, 197, 197] | 37.4 | 42.2 | 29.0 | 41.5 | 47.5 | 34.0 |
| [32, 197, 197] | 50.8 | 54.5 | 38.5 | 55.5 | 65.1 | 43.5 |
| [8, 12, 197, 197] | 108.0 | 104.7 | 77.5 | 116.0 | 162.0 | 125.2 |
| [8, 197, 100] | 16.0 | 27.6 | 13.5 | 19.1 | 31.5 | 15.4 |
| [96, 197, 250] | 119.6 | 145.4 | 86.4 | 129.8 | 243.4 | 140.2 |
| [96, 197, 500] | 213.2 | 240.6 | 164.8 | 249.6 | 393.8 | 279.0 |
| [1, 1000] | 17.4 | 61.2 | 51.4 | 18.2 | 64.8 | 52.6 |
| [64, 1000] | 17.4 | 59.5 | 51.3 | 18.6 | 61.3 | 54.1 |
| [1, 32, 4000] | 26.1 | 183.9 | 184.1 | 28.6 | 190.1 | 190.0 |
| [4096, 4000] | 309.5 | 435.1 | 434.6 | 403.7 | 543.2 | 545.5 |
| [2, 33, 70] | 13.2 | 22.8 | 13.3 | 16.3 | 25.0 | 16.0 |
| [4, 40, 33] | 10.9 | 25.7 | 9.1 | 13.9 | 26.0 | 11.6 |

`ktt_softmax` is faster than TT-NN plus the fill in every cell but bf16
`[8, 12, 197, 197]` (3 % slower), and at or below TT-NN alone on fp32 with many rows,
on rows of 1000 and wider, and on the small cells.

Aligned last dim (the default route stays TT-NN; `ktt_softmax` forced with
`KTT_OWN_SOFTMAX=1`):

| Shape | bf16 own | bf16 TT-NN | fp32 own | fp32 TT-NN |
|---|---|---|---|---|
| [8, 224, 224] | 20.4 | 18.5 | 25.6 | 23.2 |
| [8, 12, 224, 224] | 96.7 | 77.0 | 122.3 | 125.1 |
| [1, 1024] | 17.9 | 51.4 | 18.5 | 53.3 |
| [64, 1024] | 16.9 | 51.4 | 17.6 | 53.1 |
| [1, 32, 4096] | 27.0 | 188.7 | 36.4 | 196.7 |
| [4096, 4096] | 321.3 | 445.3 | 408.8 | 558.7 |

More aligned bf16 cells (own / TT-NN): `[8, 224, 768]` 48.4 / 53.6, `[8, 224, 1024]`
61.1 / 66.4, `[96, 224, 512]` 198 / 164, `[96, 224, 2048]` 783 / 706, `[4096, 2560]`
217 / 284. TT-NN splits only tile rows over the cores; `ktt_softmax` also splits a row
over a group of cores, so it wins on few wide rows (fewer tile rows than cores, or a
remainder round) and loses about 20 % on many narrow bf16 rows. Errors against float64
(std normal times 4): bf16 equal to CPU bf16 in every cell (max relative 3.9e-3, mean
1.4e-3); fp32 max relative 6e-7 to 9e-7 against CPU's 6e-7 to 2.8e-6.

## Limits

- Only the last dim. Softmax over dim -2 would need a variant whose unit is a strip of
  column tiles (elementwise statistics across tile rows, then a fold over the 32 rows of
  a tile); it is not built.
- `log_softmax` stays the composite (six device programs, correct under dirty padding).
  The kernel could give it as `y = x - m - log s` from the same statistics (x kept
  resident to the last pass, `log` of the sum per row); not built.
- The backward is TT-NN's composite.
- Each kernel checks an argument sentinel (`args.hpp`) after its last common runtime
  argument and returns at once when it is missing.
