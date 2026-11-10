---
title: ktt_conv_nhwc (channels-last convolution)
summary: Single-group "same" stride-1 convolution on Nhwc storage - what it computes, routing, coverage, grid and NoC plan, L1 by block, plan choice, weights, the weight gradient, performance against TT-NN conv2d and limits.
tags:
  - reference
  - kernels
  - convolution
---

# ktt_conv_nhwc: channels-last convolution

Forward, input gradient and weight gradient of single-group 2-D convolutions with stride 1,
dilation 1, an odd K x K kernel and padding (K - 1) / 2, on channels-last (Nhwc) storage.

- **Forward**: bf16 activations and weights, fp32 accumulation, bf16 output stored as Nhwc
  with zero tile padding. A 1x1 conv runs as the bridge's matmul of the Nhwc rows
  `[N H W, C]` with the `[C, O]` weight.
- **Input gradient**: the same conv from O to C channels with the weight's taps reversed and
  its channels swapped, prepared once per weight version like the forward W'.
- **Weight gradient** (K >= 3): its own program, `ktt_conv_nhwc_wgrad` (below).

Source: `src/kohakutt_native/csrc/aten/nn/ktt_conv_nhwc/` (`reader.cpp`, `writer.cpp`,
`feeder.cpp`, `compute.cpp`, host side in `program.cpp`) and
`aten/nn/ktt_conv_nhwc_wgrad/`; routing in `aten/nn/device_conv.cpp` (`conv2d_nhwc`,
`conv2d_nhwc_input_grad`, `conv2d_nhwc_weight_grad`) and `aten/nn/aten_conv.cpp`. Tests:
`tests/hardware/ops/test_conv_nhwc.py`.

## Enabling it

The route runs when the input is stored as Nhwc, or when `KTT_NHWC=1` (read per call)
and the input is Nchw. An Nchw input is then converted inside the op, into DRAM. With
`KTT_NHWC` unset nothing produces Nhwc activations, so default behaviour is unchanged.
`KTT_CUSTOM_KERNELS=0` turns it off.

## What it computes

Output pixel tile m (32 consecutive pixels of the flattened `N H W` rows) times Cout tile
n is the sum over Cin tiles and taps of A(m, tap) x W'(tap, Cin, n). A(m, tap) is the 32
input pixels shifted by the tap's offset `(dy - p) W + (dx - p)`, with rows outside the
image taken as zeros.

- **A tiles** are built on each core by NoC copies from channel planes in its L1: one copy
  per run of valid rows per face, from a copy table built once per geometry and plan on the
  host (one page per core and pixel block). Rows outside the image come from the core's
  zero region. No RISC-V store moves tile data.
- **W'** is the weight reordered on the host to `[Ct K K 32, Nt 32]`, built once per
  weight version.

## Grid and NoC plan

- **Cores.** A pixel block is `go` adjacent columns inside one panel. Column j of a block
  owns Cout group j (`bn = Nt / go` tiles); each row of a block is a pixel group of `bm`
  output tiles.
- **Blocks.** A core walks its share as `npb` pixel blocks of `bm_b` output tiles, each
  over all Cin tiles, and per Cin step as `nnb` sub-blocks of `bn_b` Cout tiles. Planes
  and A tiles are made once per (pixel block, Cin step) and reused by every Cout
  sub-block; W' is streamed again for every pixel block.
- **W'.** The column top (row 0) reads each W' block from DRAM on NoC0 and multicasts it
  down its column. A 4-slot ring runs the transport one block ahead of the compute.
- **Planes.** With `go = 1` each core reads its own input range (halo included) on NoC1.
  With `go > 1` a feeder core right of the block, in the same panel, reads the group's
  input range and multicasts it along the row on NoC1.
- **Flow control.** A receiver counts in at its sender for block v + 1 only after block v
  arrived and the slot of block v + 1 is free. A core's count therefore never exceeds the
  blocks sent plus one, and the sender's test (count >= receivers x blocks) holds for every
  receiver.
- **A builds** are split between the reader (RISCV_1) and the writer (RISCV_0) by a
  per-core count. The column top builds fewer, since it also sends W'.
- Reader and writer use different NoCs on every core.

## L1

Per core, set by the block sizes and not by the problem size: 4 W' slots of `K K bn_b`
tiles, 2 blocks of `bm_b K K / 2` A tiles each for the reader's and the writer's builds, 4
planes slots of `bm_b + ceil(2 (p W + p) / 32) + 1` tiles, two table pages per mover, two
output subblocks and the fp32 partials of one pixel block (`bm_b bn` tiles).

## Plan choice

Per call, on the stream worker, the first plan whose circular buffers fit the free L1 span
(`program::cb_budget()`):

1. the fewest blocks per core (`npb x nnb`);
2. then the fewest pixel blocks (each one streams W' again);
3. then the grid's modeled cost: the maximum of FPU time, A builds and DRAM bytes per core.

A core's whole share as one block comes first, so a shape that fits unblocked runs the
same plan as before blocking existed. Shapes whose share does not fit run blocked instead
of falling back. `ktt._debug_nhwc_conv_plan(go, bm_b, npb, bn_b, iters)` forces one plan
(tests, timing); `ktt._debug_nhwc_conv_last_plan()` reports the plan that ran.

## Weight gradient

`dW'(cb, t, n) = sum over pixel tiles m of X^T(cb, m) x G(m, T - 1 - t, n)`: X^T is the
input's Nhwc tile (m, cb) transposed (32 channels x 32 pixels), G the output gradient's
tiles of Cout tile n shifted by the mirrored tap, built from dY planes by the same NoC
copies and copy tables as the conv's A tiles. The two operands are stored in perpendicular
directions, so the matmul needs no transpose. The result has W' orientation,
`[Ct K K 32, Nt 32]`.

- **X^T.** A forward in training mode (gradients enabled and the weight requires grad)
  also writes X^T: compute turns each center-tap A tile, which is the input tile itself,
  with the unpacker's transpose and the writer stores it. The bridge keeps it per input
  handle and write count. Without a kept X^T (forward without gradients, input written
  since) a small program turns the input's tiles first.
- **Grid.** The forward's grid with the roles renamed: column j of a slot owns a group of
  Cin tiles and receives X^T blocks from its column top (NoC0 multicast); a (slot, row)
  group owns Cout tiles and receives dY planes from its feeder (NoC1). Every core streams
  all pixel tiles in blocks of `kb`.
- **Blocks.** Per (Cout tile, Cin block of `bc` tiles): per pixel block and tap, `bc`
  products per built tile into `bc` fp32 Dst tiles (the built tile is unpacked once), packed
  with L1 accumulation across pixel blocks. L1: 4 X^T slots of `bc kb` tiles, 2 blocks each
  of `kb K K / 2` built tiles, 4 planes slots, `bc K K` fp32 partials.
- **To the weight's layout.** In the same bridge op dW' is viewed as
  `[Ct, taps, 32, Nt 32]`, the taps moved inside the channels (one permute), cut to
  `[C K K, O]` and transposed. That is a Flat weight's stored rows `[O, C K K]`; a weight in
  tile form gets them reshaped to `[O, C, K, K]`.
- `KTT_NHWC_WGRAD=0` (read per call) keeps the per-tap route.
  `ktt._debug_nhwc_wgrad_plan(go, bc, kb, bo, iters)` forces a plan;
  `ktt._debug_nhwc_wgrad_runs()` and `ktt._debug_nhwc_xt_rotates()` count launches.

Microseconds per call, HiFi2, eager, activations in DRAM:

| Shape (K 3) | Plan | Kernel | X^T program + program | Gradient in the weight's layout | Per-tap route |
|---|---|---|---|---|---|
| ResNet 1x64x56x56 -> 64 | go 2, bc 1, kb 8 | 100.1 | 150.6 | 305.4 | 6391.3 |
| VAE 1x512x64x64 -> 512 | go 4, bc 4, kb 12 | 147.0 | 173.1 | 498.3 | 21714.6 |
| SD1.5 2x320x64x64 -> 320 | go 5, bc 2, kb 16 | 250.3 | 272.4 | 402.3 | 15565.8 |
| SD1.5 2x640x32x32 -> 640 | go 4, bc 3, kb 16 | 170.6 | 200.3 | 703.4 | 17543.6 |
| SD1.5 2x1280x16x16 -> 1280 | go 4, bc 4, kb 8 | 216.0 | 286.7 | 2280.1 | 43119.4 |

"Kernel" is one pass of the kernels' loops; the next column adds the X^T program (what a
gradient without a kept X^T pays); the gradient column is the whole op for Flat weights,
reorder included, with X^T made by its own program. With few channels and many pixels few
cores are busy (ResNet: 4); a split over pixels with a sum of fp32 partials is designed
but not built.

## Performance

Microseconds per call, bf16, HiFi2, fp32 accumulation, activations in DRAM, one P150A.
"Kernel" is one pass of the kernels' loops (in-kernel loop); "per call" a bridge trace of
four convs. TT-NN is `ttnn.conv2d` on the same Nhwc input and numerics, through a trace.

| Shape (N x C x H x W -> O, K 3) | Plan | Kernel | Per call | TT-NN conv2d |
|---|---|---|---|---|
| ResNet 1x64x56x56 -> 64 | go 1, bm 3, 33 cores | 9.8 | 19.9 | 50.1 |
| VAE 1x512x64x64 -> 512 | go 4, bm 7, 80 cores | 134.2 | 147.2 | 215.9 |
| SD1.5 2x320x64x64 -> 320 | go 1, bm 3, 88 cores | 107.4 | 166.7 | 216.4 |
| SD1.5 2x640x32x32 -> 640 | go 4, bm 4, 64 cores | 130.1 | 139.8 | 162.4 |
| SD1.5 2x1280x16x16 -> 1280 | go 4, bm 2, 32 cores | 347.5 | 358.4 | 268.9 |
| SDXL 2x1280x32x32 -> 1280 | go 4, bm 4, 64 cores | 439.1 | 456.1 | 425.0 |
| SDXL 2x640x64x64 -> 640 | go 4, bm 13, 80 cores | 393.8 | 413.6 | 494.3 |
| SDXL 2x320x128x128 -> 320 | go 1, bm 10, 2 Cout sub-blocks | 323.9 | 427.7 | 810.8 |
| VAE 1x512x128x128 -> 512 | go 1, bm 5, 2 Cout sub-blocks | 468.0 | 530.9 | 925.7 |
| VAE 1x256x256x256 -> 256 | go 1, 3 pixel blocks of 7 | 948.5 | 996.2 | 1837.4 |
| VAE 1x128x512x512 -> 128 | go 1, 7 pixel blocks of 12 | 1485.5 | 1544.5 | 1120.1 |

The last four rows fell back to TT-NN's `conv2d` before blocking.

- At 1280 channels the W' stream bounds the call (the column top reads and multicasts
  about 110 cycles per W' tile).
- On wide images the planes halo (`ceil(2 (W + 1) / 32) + 1` tiles: 18 at W 256, 34 at
  W 512) is read again for every pixel block and Cin step and outweighs the block itself:
  1x128x512x512 reads 261 MB for a 33 MB input, whichever cores read it (a plan with
  feeders measures the same 1.5 ms).
- The 1x1 VAE shortcut `1x512x64x64 -> 256` runs as a matmul: 42.7 us eager against 47.3
  for TT-NN.

## Limits

- Stride 1, dilation 1, odd K, "same" padding, one group, bf16.
- The planes ring grows with `W K`: when no plan's circular buffers fit the free L1 span,
  the conv runs on TT-NN's `conv2d` of the Nhwc rows (counted by
  `ktt._debug_nhwc_conv_fallbacks()`).
- Input and output are placed in DRAM.
- The bias is added to the bf16 conv output (two roundings), as on the other conv routes.
- W' is built on the host once per weight version: about 1 s (512 x 512 x 3 x 3) to 7 s
  (1280 x 1280 x 3 x 3) for a weight stored as `[O, C, K, K]` tiles (not re-measured for
  Flat weights). In a trace, a weight written between replays is re-prepared before the
  next replay.
- The weight gradient of a 1x1 conv takes the per-tap route.
- The weight gradient has no TT-NN route to fall back to: when no block fits the free L1
  span it raises. The kept X^T doubles the conv input held during a training step.
