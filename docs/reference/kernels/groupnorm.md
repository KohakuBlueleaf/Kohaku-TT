---
title: ktt_group_norm (row form)
summary: Group norm on storage with the channels last (Nhwc tiles, and [N, C, L] through its transpose), bf16 / fp32, forward and backward - what it computes, routing, coverage, the two programs, L1, bound, measured times and limits.
tags:
  - reference
  - kernels
  - normalization
---

# ktt_group_norm (row form)

`y = (x - mean) * rstd * weight + bias` with `mean` and `rstd = 1 / sqrt(var + eps)` per
(image, group of C / G channels) over the image's pixels. `x` is stored with its channels
last: one TILE tensor whose tiles are 32 pixels by 32 channels. The result has the same
storage, so a conv, group norm, SiLU, conv chain never leaves channels-last.

Source: `src/kohakutt_native/csrc/aten/nn/ktt_group_norm/` (`rows_moments_*.cpp`,
`rows_affine_*.cpp`, `rows.hpp`, `rows_dm.hpp`; `program.cpp` plan and programs). Routing:
`aten/nn/aten_norm.cpp`, `aten/nn/device_group_norm.cpp`. The same directory holds the
kernels for channels-first storage (`moments_*`, `stats`, `affine_*`), which reduce
per-channel planes.

## Routing

| Input | Route |
|---|---|
| `[N, C, H, W]` stored Nhwc, `KTT_NHWC=1`, covered shape | row form, result Nhwc |
| `[N, C, L]`, L and C multiples of 32, covered shape, no gradient | row form on the `[N, L, C]` transpose; the result is returned as the transposed view of an `[N, L, C]` tensor |
| anything else (bf16 / fp32, 4-D) | channels-first kernels (an Nhwc input is converted once) |
| `KTT_OWN_GROUP_NORM=0` | the composite of TT-NN reductions and binaries |

"No gradient" means grad mode is off or neither the input nor weight / bias requires a
gradient. `native_group_norm` also returns `mean` and `rstd` `[N, G]`, which only its
backward reads. On the row-form route without a gradient they are zeros (not the
statistics). With one, program 2 also writes each channel's group mean and rstd as two
zero-filled fp32 `[N, C]` tensors (row n by the core that starts each segment); the
returned mean and rstd are fp32 views of every (C / G)-th column of those, and the backward
reads the `[N, C]` tensors directly.

## Coverage

- x: bf16 or fp32; any C / G whose groups reach at most 8 channel tiles each (up to 192
  channels per group; groups may cross channel tiles, as 320 channels in 32 groups do);
  one image, or several of at least 32 pixels each (images may share pixel tiles).
- C and the pixel count need not be multiples of 32: the tile padding of
  the input is zero and stays zero in the result.
- weight / bias: absent, or 1-D of length C, bf16 or fp32, independently of x's dtype.

## How it works

Sizes in tiles: Ct = ceil(C / 32) channel tiles. A row tile belongs to the image of its
first row; a segment is (image, channel tile): the image's row tiles in one column of the
tensor. The tiles in segment order are cut into equal ranges, one per core (at least 4
tiles each, at most 32 cores on one segment); the part of a segment inside a range is a
piece. With several images whose pixel count is not a multiple of 32, an image's last row
tile also holds the first rows of the next image (a shared tile).

### Program 1: per-channel sums

Per piece, each tile is unpacked to DST and one SFPU pass adds x and x^2 elementwise into
two fp32 accumulator tiles; at the end of the piece both are summed over their rows. Row 0
of each (32 per-channel sums) goes to row k of the segment's two partial pages in DRAM,
k being the core's index among the segment's cores. Padding rows and channels add zero.
A shared tile is accumulated twice by its core, each time with the other image's rows
zeroed by NoC reads of the zero region; the second pair of sums goes to the next image's
partial pages, in the row after that segment's own cores.

### Program 2: statistics and affine

The cores and ranges are the same. Per piece:

1. For every channel tile the piece's groups reach (its own, and neighbours when groups
   cross tiles), that segment's two partial pages are read whole; the rows past its core
   count are zeroed by NoC reads of the zero region; a column reduce gives the per-channel
   sums.
2. Each channel's sum is replaced by its group's sum with mask tiles M (1 at (channel,
   index of its group among the piece's groups)): the sums as a column times M, column
   reduce (one value per group), added over the reached tiles by the packer in L1; that
   row times the piece's own M, transposed, column reduce.
3. `mean = S1 / n`, `var = max(S2 / n - mean^2, 0)`, `rstd = rsqrt(var + eps)`,
   `a = rstd * weight`, `b = bias - mean * a`, all on the SFPU in fp32; a and b are zeroed
   at padded channels.
4. `y = x * a + b` for the piece's tiles, a and b broadcast down the rows, two tiles per
   DST acquire.
5. A piece that ends in a shared tile repeats steps 1 to 3 for the next image's segment
   and builds whole a and b tiles for that one tile: each image's rows from its own a and
   b, selected by two row-mask tiles.

Every core of a segment repeats steps 1 to 3 for itself, so no core waits for another:
the programs use no semaphores and no core-to-core writes. The writer zeroes the padding
rows of the last row tile by NoC reads of the zero region before writing it.

The mask, valid-channel and row-mask tiles hold only the constants 0 and 1; the reader
writes them (at most 64 ones per piece for the first two, a tile of ones per shared tile
for the row masks, cut by zero-region reads).

### Backward

`native_group_norm_backward` on an Nhwc-stored input of a covered shape runs two programs of
the same shape (`rows_bwd_*.cpp`; the writers are the forward's) and returns the input
gradient Nhwc. With per channel `Sg = sum dy` and `Sgx = sum dy x` over the image's pixels,
D elements per group, mean m and rstd r:

- per group `A = sum w Sg` and `B = sum w r (Sgx - m Sg)` over its channels;
- per channel `P = r w`, `Q = -r^2 B / D`, `R = -r A / D - m Q`;
- `dx = dy P + x Q + R`, `dweight = sum over images of r (Sgx - m Sg)`, `dbias = sum over
  images of Sg`.

Program 1 reads x and dy together and accumulates dy and dy x (the two accumulators plus x
and dy fill the four fp32 DST tiles); its partial pages are the forward's. Program 2 reads
the image's row of the per-channel mean and rstd (fp32 `[N, C]`, expanded from the saved
`[N, G]` by tensor ops; the other rows are zeroed by zero-region reads), forms A and B with
the forward's group step, and computes dx one tile per acquire. The piece that starts a
segment writes the image's dweight and dbias terms as row n of two fp32 `[N, C]` tensors;
their sums over the images are the gradients. An upstream gradient that is not stored Nhwc
is converted first. Shared tiles take P, Q and R mixed by rows, as a and b in the forward.

### NoC plan

| Stream | Kernel | NoC |
|---|---|---|
| x tiles, 8 per read barrier; partial pages, weight / bias tile | reader | NoC0 |
| partial rows (4 writes of 64 B per piece); y tiles, 8 per write barrier | writer | NoC1 |

### L1 (per core)

| CB | Contents | Format | Tiles |
|---|---|---|---|
| 0 | x block ring | x | 16 |
| 16 | program 1: the two sum tiles; program 2: y block ring | fp32; x | 4; 16 |
| 1, 2 | weight, bias tile | parameter | 2 each |
| 3, 4, 8 | partial pages; group mask; the piece's own mask and valid-channel tile | fp32 | 2 each |
| 5, 6, 7 | per-channel sums, group sums, a / b | fp32 | 2 each |
| 9, 10, 11 | shared tile: the next image's a / b, the row masks, the mixed a / b tiles | fp32 | 2 each |

The sizes depend only on the block size (8 tiles). Every push is a whole block.

### Compute

fp32 DST; fp32 circular buffers unpack to DST unrounded; no FPU multiply runs, so the
result does not depend on the math fidelity. The variance is `E[x^2] - mean^2` from fp32
sums, as in the channels-first kernels.

## Bound

Memory-bound: x is read twice and y written once, three times the tensor's bytes. At the
445 GB/s copy rate of [ktt_norm](layernorm.md) that is 0.45 ms for `[1, 128, 512, 512]`
bf16 (67 MB) and 0.03 ms for `[1, 512, 64, 64]`.

## Performance

One P150A, bf16, 32 groups with weight and bias, eager, each call synchronized, mean of 8
calls after warm-up, 2026-11-10. "Channels-first" is the same call on an Nchw input (three
programs, result Nchw).

| x | row form | channels-first |
|---|---|---|
| `[1, 128, 512, 512]` | 0.96 ms | 1.17 ms |
| `[1, 256, 256, 256]` | 0.52 ms | 0.99 ms |
| `[1, 512, 128, 128]` | 0.27 ms | 1.17 ms |
| `[1, 512, 64, 64]` | 0.36 ms | 1.16 ms |

A conv that follows also saves the conversion of its input to Nhwc. Not compared with
TT-NN's `group_norm` yet.

## Limits

- Several images of under 32 pixels each (unless the pixel count is a multiple of 32) and
  groups of more than 192 channels take the channels-first kernels.
- `[N, C, L]` calls that need gradients take the composite forward and backward.
- The row-form forward returns mean and rstd as fp32 for every input dtype (the other
  routes return x's dtype). Given bf16 statistics from another route, the backward's
  gradients are within about 3e-3 of float64, as the composite's are.
- The weight and bias gradients are the sum over the images of the per-image terms (one
  small reduction each for more than one image) and a cast to the parameter dtype.
- `[1, 128, 512, 512]` takes 2.1x its DRAM bound. Most of the gap is not in the kernels:
  a one-program elementwise op at this shape (SiLU, 134 MB moved) takes 0.55 ms, of which
  0.20 ms is the per-op floor, and at that data rate two programs moving 201 MB take
  0.79 ms. The row form is 15 to 20 % above that; the moments run one tile per SFPU call and
  the affine broadcasts a and b again for every pair of tiles, and which of the two
  accounts for the rest is not measured.
- The two smallest rows of the table are within timing noise of each other.
