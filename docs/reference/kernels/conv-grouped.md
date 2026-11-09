---
title: ktt_conv2 (grouped convolution)
summary: Grouped, depthwise and strided 1-D / 2-D convolution with input and weight gradients - what it computes, routing, coverage, grid, NoC plan, L1, compute, performance against TT-NN conv2d and limits.
tags:
  - reference
  - kernels
  - convolution
---

# ktt_conv2: grouped, depthwise and strided convolution

Forward, input gradient and weight gradient of 1-D and 2-D convolutions with
`groups > 1` (grouped, depthwise, channel multipliers), and of one-group convolutions
over a single row with a stride. Any stride, dilation and padding; kernels up to
16 x 16; bf16 activations; bf16, bfp8 or bfp4 weights; fp32 accumulation; bf16 output.

Source: `src/kohakutt_native/csrc/aten/nn/ktt_conv2/` (gather: `reader.cpp`,
`compute.cpp`, `writer.cpp`, shared `taps.hpp`; weights: `prep.cpp`,
`prep_packed.cpp`; weight gradient: `wgrad_*.cpp`, `reduce_*.cpp`), layout kernels in
`csrc/aten/nn/ktt_conv/` (`rows_in*.cpp`, `rows_out*.cpp`), routing in
`aten/nn/device_conv.cpp` and `aten/nn/aten_conv.cpp`. Sweep:
`tests/hardware/ops/test_conv_sweep.py`.

## Enabling it

On by default. `KTT_GROUPED_CONV` (read per call) selects the route for the
convolutions in the coverage below:

| Value | Route |
|---|---|
| `own` (default) | `ktt_conv2`, forward and both gradients |
| `ttnn` | TT-NN's conv2d for the forward (used as the sweep's baseline) |
| `composite` | the per-tap composite of device ops |

`KTT_CUSTOM_KERNELS=0` turns every Kohaku-TT kernel off. Test-only overrides:
`KTT_CONV2_BLOCK=n` caps the taps per K block; `KTT_CONV2_SPLIT=n` forces the tap-row
split into n kernel-row ranges.

## Coverage

Decided in `device_conv.cpp` from shapes and dtypes before launch (`own_grouped_conv()`):

- bf16 tt input and weight (weights stored bf16, bfp8 or bfp4);
- `groups > 1`, or one group over one input row with a stride > 1;
- KH, KW <= 16; at most 8 phases per axis;
- `plan()` finds a configuration within the L1 budget (one column tile's weights,
  taps x Kmax tiles, next to one tap of rows).
- Input gradient (`own_grouped_input_grad()`, any stride, dilation and padding) and
  weight gradient (`own_grouped_weight_grad()`, bf16 input and grad_out) for every
  forward it takes.
- 1-D: PyTorch lifts 1-D convolutions to `[N, C, 1, L]` views; when every operand is
  such a view of a whole `[N, C, L]` storage, the rank-3 storages run on the NCL layout
  kernels with no reshape op.

Outside the coverage: TT-NN's conv2d (one group, 2-D) or the per-tap composite. Groups
of 32 or more channels with kernels whose weights do not fit (for example groups 2,
C = 256, 11 x 11: 968 KB of weights per column tile) go to the dense implicit-GEMM path.

## How it works

A convolution is a gather over channels-last rows. `rows_in.cpp` (NCHW) or
`rows_in_1d.cpp` (NCL) lays the input out once as zero-padded pixel rows `[N Hp Wp, Cp]`
(Cp: C rounded up to 32). Output pixel row r = (image, y, x) of phase f reads, per tap
(a, b) of f, input row `image Hp Wp + (y sh + roff[a]) Wp + x sw + coff[b]`. Output
column tile j (32 output channels) sums, over the taps and only the input channel tiles
of j's groups, `act[t, k] @ w[j, t, k]`, where w is a sparse weight tile (a group's
32 x 32 block, a block-diagonal tile for groups under 32 channels, a diagonal tile for
depthwise). `rows_out.cpp` / `rows_out_1d.cpp` write the output rows back as NCHW / NCL.

- **Forward:** one phase; tap offsets (i dh, j dw); strides (sh, sw).
- **Input gradient, any stride:** sh x sw phases of stride-1 taps. Gradient pixel
  u = s y' + f takes the forward taps i with (f + p - i d) mod s = 0, at grad_out offset
  (f + p - i d) / s plus the grad rows' front padding. The weight is transposed per
  group and the tap index carried, so nothing is flipped. A phase without taps (kernel
  smaller than stride) is written as zeros.
- **Weight gradient:** per weight tile (j, t, k), the sum over output pixels of
  `G^T[m, j] @ A_t[m, k]` (G: grad_out rows over the forward's output pitch, zero in its
  pad columns; A_t: the forward gather of tap t), split into pixel chunks. Chunk sums
  run in fp32 on the SFPU and are scattered into the torch layout with
  round-to-nearest-even.
- **Packed weights** (one input channel per group: depthwise, channel multipliers):
  `prep_packed.cpp` writes per output column tile one row `[taps][32][Cg]`; the writer
  core zeroes the item's weight tiles from a zero tile and writes the entries while the
  reader streams rows. Groups of 2-31 channels use sparse tiles.

### Grid and roles

- Gather (forward, input gradient): output rows in tiles of 32 pixel rows (mt = phases
  x phase tiles), output channels in column tiles (Ot). A work item is (row tiles
  [m0, m1), column tiles [n0, n0 + nt)); the Ot / nt column groups split the 110 cores
  evenly, each core a contiguous range of row tiles; items beyond the core count are
  dealt round-robin (at most 48 per core).
- nt = min(4, Ot) when the taps fit, else 2, else 1; else nt = 1 with K blocks of fewer
  taps. One-phase gathers with Ot <= 4 that are strided or need K blocks read whole
  rows (nt = Ot), which turns many per-tap reads into a few runs per kernel row.
- Small outputs: when items keep at most half the cores busy, column tiles whose input
  channel tiles are disjoint (depthwise, groups up to 32 channels) take one column tile
  per core.
- Output rows span ceil(Wp / Sw) columns per image row; columns past OW are computed
  and dropped by `rows_out.cpp`. Every size is a runtime argument; the descriptor key
  holds only the plan (nt, taps per block, window, taps per read barrier, split).
- Weight gradient: a unit is (column group of nt tiles, chunk of row tiles); chunks =
  cores / column groups, at least 4 row tiles each unless packed. Chunk sum: one
  1024-value block per core; scatter: one torch weight tile per core.
- Layout kernels: a unit is (image, row tile, column tile, 32 channels, R tile rows)
  with R = 8, 4 or 2, the most rows that still give every core a unit.

### NoC plan

| Stream | Mover | From | To | NoC |
|---|---|---|---|---|
| tap windows: (32 + spread / Sw) Sw input rows per kernel row and row tile | reader (NCRISC) | DRAM rows | CB 3 (compute tilizes from it) | NoC0 |
| tap rows (32 rows of the item's channel tiles; no window) | reader | DRAM rows | CB 0 | NoC0 |
| sparse weight tiles, the item's kernel rows | reader | DRAM | CB 1, once per item | NoC0 |
| packed weight rows (depthwise, multipliers) | writer (BRISC) | DRAM row, zero tile | CB 6, CB 1 | NoC1 |
| output tiles | writer | CB 16 | DRAM | NoC1 |
| layout kernels | one data-movement kernel | DRAM tiles / rows | L1, then DRAM rows / tiles | NoC0 |

- Input rows are read as runs of consecutive rows within a 32-row page. Window mode
  reads, per row tile and kernel row, the Sw input rows of each of 32 + spread / Sw
  output positions once into a CB 3 page; a tap's rows are every Sw-th row of the page
  from its column offset on, and the compute tilizes them straight from the page, so no
  tap is copied. Window mode is used at column stride 1 when the windows fit, and at
  stride Sw > 1 when the item reads whole rows or the window has fewer rows than the
  taps would read.
- No division in data-movement loops: row pages hold 32 rows (shift and mask); output
  positions, phases and groups advance by comparison.
- Weights are read once per work item (its kernel rows only), not per row tile. Packed
  weights cost 64 bytes per tap and column tile from DRAM instead of a 2 KB tile.

### L1 (per core)

| CB | Contents | Format | Pages | Bytes |
|---|---|---|---|---|
| 0 | tap rows, K = the item's channel tiles (G taps per read barrier; not used in window mode) | bf16 | 2 G K | 4096 G K |
| 1 | the item's weights, nt x taps x Kmax tiles | bf16 / bfp8 | nt taps Kmax | 2048 (1088) each |
| 2 | tilized taps of one K block (B taps) | bf16 | B K | 2048 B K |
| 3 | tap windows, one page per kernel row (window mode) | bf16 | 2 KH | 128 KH (32 + spread / Sw) Sw K |
| 4 | zero output tile (writer) | bf16 (fp32 split) | 1 | 2048 (4096) |
| 5, 6 | packed weights: zero tile, one packed row | bf16 | 1 + 1 | 2048 + 64 taps Cg |
| 16 | output tiles | bf16 (fp32 split) | 2 nt | 4096 nt (8192 nt) |
| 24 | K-block sums (more than one block) | fp32 | nt | 4096 nt |

`cb_bytes()` mirrors this table; `plan()` checks it against `program::cb_budget()`
before launch. Weight gradient per unit: CB 0 (2 K), CB 1 (2 nt grad rows), CB 2
(per_chunk K tilized input), CB 3 (per_chunk windows of one kernel row, window mode),
CB 4 (nt), CB 5 (per_chunk nt G^T tiles, resident for all taps), CB 6 (one packed row
of partials), CB 16 (2 fp32); per_chunk halves until the total fits.

### Compute

- `compute_kernel_hw_startup<SrcOrder::Reverse>`, fp32 DST, HiFi4 (bf16 x bf16
  products exact; the FPU does not bind these kernels).
- Per row tile, each tap's 32 x K row-major block is tilized into CB 2 (window mode:
  straight from the CB 3 page, the unpacker set to a row pitch of Sw rows); then one
  DST acquire holds the item's nt <= 4 output tiles while `matmul_tiles` runs over the
  block's taps and each column tile's channel tiles.
- K blocks (a phase's taps do not fit CB 2): each block's sums pack into CB 24 with
  packer L1 accumulation, then CB 24 (unpacked to DST as fp32) is packed as bf16.
- Weight gradient: the grad block is tilized and transposed (`transpose_tile`) into
  G^T, resident for all taps; per tap the chunk's input blocks are tilized; per DST
  group of up to 4 weight tiles, `matmul_tiles(G^T, A)` sums in DST and packs fp32.
  Chunk sums: `copy_tile` with fp32 unpack and `add_binary_tile` on the SFPU. The bf16
  rounding of the scatter is integer arithmetic on the data-movement core.
- Block-float weights: bfp8 sparse tiles are unpacked as stored; bfp4 and packed
  weights hold the stored values as bf16 (exact). Groups under 32 channels keep their
  weights as bf16, since their block-diagonal tiles would be requantized across groups.

## Bound

Per forward (`roofline()` in `test_conv_sweep.py`):

- FPU: one 32 x 32 x 32 tile product per (row tile, tap, input channel tile of each
  output column tile), 64 cycles at HiFi4 on 110 cores. Depthwise and grouped weights
  are mostly zeros, so this term is small: ConvNeXt-T stage 1 (7 x 7, 96 channels,
  56 x 56) is 14 406 tile products, 6.2 us.
- DRAM at 512 GB/s: the layout pass (input read, padded rows written), the gather's
  windows or tap rows, the output rows written, then read and written back as NCHW.
  ConvNeXt-T stage 1: 8.2 MB, 15.9 us.

Every sweep cell is DRAM-bound by this model. Measured forwards reach 1-17 % of it; the
time goes to:

| Part | Cost | Why |
|---|---|---|
| fixed cost | 3 programs per call: about 25 us device dispatch and 30-60 us host enqueue each | dominates every call under about 0.2 ms |
| `rows_in.cpp` | ConvNeXt-T stage 1: about 60 us; batch 8: 230-450 us | transposition on the RISC-V (about 5 cycles per element) and one 64-byte NoC write per pixel and channel tile |
| gather | ConvNeXt-T stage 1: 79 us (46 us matmul, 25 us tilize, the rest waiting for packed weights) | about 700 cycles per tap and column tile (tilize through DST one tile per section, then `matmul_tiles`); packed weight expansion at about 17 cycles per entry |
| `rows_out.cpp` | ConvNeXt-T stage 1: 64-70 us; batch 8: 230-345 us | scatter into face rows on the RISC-V |

## Performance

One p150a, bf16, warm device-synchronized ms per call (fastest of 3 batches of 10
calls, host time included). TT-NN: conv2d's forward timed the same way in the same
process. Error: cosine against CPU float64 (CPU bf16's in parentheses). Every cell ran
forward, input gradient and weight gradient on Kohaku-TT programs only, with no
descriptor-cache miss in warm calls.

Model shapes:

| Shape | fwd ms | bound us | of bound | TT-NN ms | TT-NN / ours | dgrad ms | wgrad ms | fwd cos (CPU bf16) |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-T s1 dw 7x7 | 0.212 | 15.9 | 7 % | 0.541 | 2.55 | 0.214 | 0.456 | 0.999996 (0.999996) |
| ConvNeXt-T s2 dw 7x7 | 0.149 | 8.2 | 6 % | 0.386 | 2.59 | 0.150 | 0.420 | 0.999996 (0.999996) |
| ConvNeXt-T s3 dw 7x7 | 0.192 | 4.7 | 2 % | 0.438 | 2.28 | 0.189 | 0.435 | 0.999996 (0.999996) |
| ConvNeXt-T s4 dw 7x7 | 0.182 | 2.8 | 2 % | 0.408 | 2.25 | 0.179 | 0.468 | 0.999996 (0.999996) |
| ConvNeXt-T s1 dw 7x7, batch 8 | 0.741 | 127.4 | 17 % | 1.109 | 1.50 | 0.743 | 0.837 | 0.999996 (0.999996) |
| MobileNetV2 dw 3x3 s1 | 0.242 | 15.8 | 7 % | 0.541 | 2.23 | 0.245 | 0.338 | 0.999996 (0.999996) |
| MobileNetV2 dw 3x3 s2 | 0.199 | 9.8 | 5 % | 0.384 | 1.93 | 0.159 | 0.222 | 0.999996 (0.999996) |
| MobileNetV2 dw 3x3 s2, batch 8 | 1.111 | 189.5 | 17 % | 2.857 | 2.57 | 1.254 | 1.541 | 0.999996 (0.999996) |
| MobileNetV3 dw 5x5 s2 | 0.151 | 8.0 | 5 % | 0.347 | 2.30 | 0.142 | 0.271 | 0.999996 (0.999996) |
| ResNeXt-50 g32 c128 | 0.208 | 13.0 | 6 % | 0.906 | 4.36 | 0.210 | 0.371 | 0.999996 (0.999996) |
| ResNeXt-50 g32 c256 s2 | 0.307 | 16.1 | 5 % | 0.386 | 1.26 | 0.259 | 0.360 | 0.999996 (0.999996) |
| ResNeXt-50 g32 c512 s2 | 0.217 | 8.7 | 4 % | 0.456 | 2.10 | 0.200 | 0.506 | 0.999996 (0.999996) |
| ResNeXt-50 g32 c1024 | 0.193 | 7.2 | 4 % | 0.516 | 2.68 | 0.193 | 1.151 | 0.999996 (0.999996) |
| strided 1-D (audio) | 0.403 | 26.6 | 7 % | 219.873 | 545 | 0.416 | 0.905 | 0.999996 (0.999996) |
| depthwise 1-D | 0.459 | 25.4 | 6 % | 11.238 | 24.5 | 0.483 | 0.563 | 0.999996 (0.999996) |

Shape grid: kernel 1, 2, 3, 5, 7, 11 x stride 1-4 x dilation 1, 2 x groups 2, 16 and
depthwise; 2-D on `[1, 64, 56, 56]`, 1-D on `[1, 128, 2048]`.

| Grid | Cells | fwd ms | TT-NN / ours | of bound | numerics at least CPU bf16's |
|---|---|---|---|---|---|
| 2-D | 132 | 0.099-0.465 | 1.18-6.14 (all 132 faster) | 1-13 % | 132 |
| 1-D | 132 | 0.111-0.293 | 18.7-293 (130 cells; TT-NN gave no result on 2 depthwise cells) | 2-7 % | 132 |

## Limits

- Kernels larger than 16 x 16, more than 8 phases per axis, non-bf16 activations, and
  weights too large for one column tile are outside the coverage.
- Every call is far from its DRAM bound: the layout kernels transpose on the RISC-V
  cores and the three programs per call carry a fixed dispatch cost.
- Every item of a column group reads the same weights from DRAM (sparse tiles: up to
  110 times the weight bytes); there is no per-group read-once and multicast.
- The tap-row split (`KTT_CONV2_SPLIT`) is correct but slower per call than the
  default plan (its host time per call is 300-430 us higher), so it is off unless
  forced.
- Every kernel checks the argument-layout version (`ktt_conv/args.hpp`, last
  compile-time argument and common argument 0) before reading any argument and returns
  on a mismatch.
