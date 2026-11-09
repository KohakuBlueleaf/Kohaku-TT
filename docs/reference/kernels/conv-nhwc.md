---
title: ktt_conv_nhwc (channels-last convolution)
summary: Single-group "same" stride-1 convolution on Nhwc storage - what it computes, routing, coverage, grid and NoC plan, L1, weights, performance against TT-NN conv2d and limits.
tags:
  - reference
  - kernels
  - convolution
---

# ktt_conv_nhwc: channels-last convolution

Forward and input gradient of single-group 2-D convolutions with stride 1, dilation 1,
an odd K x K kernel and padding (K - 1) / 2, on channels-last (Nhwc) storage. The input
gradient is the same conv from O to C channels with the weight's taps reversed and its
channels swapped, prepared once per weight version like the forward W'. bf16 activations and weights,
fp32 accumulation, bf16 output stored as Nhwc with zero tile padding. A 1x1 conv runs as
the bridge's matmul of the Nhwc rows `[N H W, C]` with the `[C, O]` weight.

Source: `src/kohakutt_native/csrc/aten/nn/ktt_conv_nhwc/` (`reader.cpp`, `writer.cpp`,
`feeder.cpp`, `compute.cpp`, host side in `program.cpp`), routing in
`aten/nn/device_conv.cpp` (`conv2d_nhwc`) and `aten/nn/aten_conv.cpp`. Tests:
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
  per run of valid rows per face, from a per-core copy table built once per geometry and
  plan on the host. Rows outside the image come from the core's zero region. No RISC-V
  store moves tile data.
- **W'** is the weight reordered on the host to `[Ct K K 32, Nt 32]`, built once per
  weight version.

## Grid and NoC plan

- **Cores.** A pixel block is `go` adjacent columns inside one panel. Column j of a block
  owns Cout group j (`bn = Nt / go` tiles); each row of a block is a pixel group of `bm`
  output tiles.
- **W'.** The column top (row 0) reads each W' block from DRAM on NoC0 and multicasts it
  down its column. A 4-slot ring runs the transport one block ahead of the compute.
- **Planes.** With `go = 1` each core reads its own input range (halo included) on NoC1.
  With `go > 1` a feeder core right of the block, in the same panel, reads the group's
  input range and multicasts it along the row on NoC1.
- **A builds** are split between the reader (RISCV_1) and the writer (RISCV_0) by a
  per-core count. The column top builds fewer, since it also sends W'.
- Reader and writer use different NoCs on every core.

## L1

Per core: 4 W' slots of `K K bn` tiles, 2 blocks of A tiles each for the reader's and
the writer's builds, 4 planes slots of the widest input range, two table pages, two output
subblocks and an fp32 partials block of `bm bn` tiles. The plan is chosen per call as the
cheapest one whose circular buffers fit the free L1 span (`program::cb_budget()`).

## Performance

Microseconds per call, bf16, HiFi2, fp32 accumulation, activations in DRAM, one P150A.
TT-NN is `ttnn.conv2d` on the same Nhwc input and numerics, through a trace.

| Shape (N x C x H x W -> O, K) | ktt_conv_nhwc, eager | TT-NN conv2d |
|---|---|---|
| ResNet 1x64x56x56 -> 64, 3 | 41.2 | 48.3 |
| VAE 1x512x64x64 -> 512, 3 | 150.9 | 215.8 |
| VAE 1x512x64x64 -> 256, 1 | 42.7 | 47.3 |
| SD1.5 2x320x64x64 -> 320, 3 | 164.3 | 216.5 |
| SD1.5 2x640x32x32 -> 640, 3 | 140.0 | 161.8 |
| SD1.5 2x1280x16x16 -> 1280, 3 | 369.7 | 267.3 |
| SDXL 2x1280x32x32 -> 1280, 3 | 456.8 | 425.6 |

At 1280 channels each core holds one output tile and the W' stream bounds the call; the
blocked kernel (pixel blocks x Cout sub-blocks) is the next step. Inside a bridge trace,
replays currently cost about 80 us more per call than eager, for this program and for
TT-NN's matmul alike.

## Limits

- The weight gradient takes the existing route (per-tap products summed in fp32).
- Stride 1, dilation 1, odd K, "same" padding, one group, bf16.
- When no plan's circular buffers fit the free L1 span, the conv runs on TT-NN's
  `conv2d` of the Nhwc rows (counted by `ktt._debug_nhwc_conv_fallbacks()`).
- Input and output are placed in DRAM.
- The bias is added to the bf16 conv output (two roundings), as on the other conv routes.
- W' is built on the host, about 1 s (512 x 512 x 3 x 3) to 7 s (1280 x 1280 x 3 x 3) per
  weight version. In a trace, a weight written between replays is re-prepared before the
  next replay.
