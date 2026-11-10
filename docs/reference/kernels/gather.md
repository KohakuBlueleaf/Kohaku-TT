---
title: ktt_gather
summary: torch.gather and take_along_dim along the last dim as one program whose work is linear in the output elements - routing, coverage, the program, its bound, measured times against TT-NN's gather and limits.
tags:
  - reference
  - kernels
---

# ktt_gather

`out[..., h, j] = x[..., h, idx[..., h, j]]`: a gather along the last dim of a TILE
tensor, bf16, fp32 or int32, with int32 indices.

Source: `src/kohakutt_native/csrc/aten/index/ktt_gather/` (`program.cpp`, `reader.cpp`),
routed from `bridge::gather` in `aten/tensor/device_view.cpp`.

## Routing

| Call | Route |
|---|---|
| `torch.gather(x, -1, idx)`, `take_along_dim(x, idx, -1)`, gathers on 1-D tensors, `take_along_dim` without a dim, view reads that go through a gather | `ktt_gather` when idx has x's leading dims and at most x's rows |
| other dims, other leading dims, block-float x | TT-NN's gather |
| the same, with x above 262,144 elements | an error naming `KTT_TTNN_GATHER_LARGE=1`, which runs TT-NN's gather anyway (it takes minutes at that size) |
| `KTT_OWN_GATHER=0` or `KTT_CUSTOM_KERNELS=0` | TT-NN's gather |

index_select and embedding keep their row gather ([aten support](../aten_support.md)).

## The program

- One unit is one output tile. Units are dealt over both data-movement processors of
  every core in use (reader on NoC0, writer on NoC1); each writes its own output tiles.
- Per unit:
  - Read the index tile: whole, or only its valid rows when fewer than 8.
  - Zero the output tile when it is ragged.
  - Read one 64-byte DRAM segment per source element. Consecutive elements in the same
    segment share one read, so sorted or reversed indices cost one read per 16 fp32
    elements.
  - After one barrier, copy each element from its segment into the output tile, then
    write the tile.
- Indices are clamped to `[0, W)`.
- Every size is a common runtime argument, so a new shape compiles nothing.

## Bound

Random small DRAM reads are limited by the DRAM channels: 0.85 G reads/s for the whole
chip with both processors of every core issuing, 0.12 G/s when all reads hit one
channel. So a random gather costs about 1.2 ns per element. A 1-D tensor is stored as
`[1, N]` tiles with 32 valid elements per tile, so its output costs one 4 KiB tile write
per 32 elements: 12.9 ns at 317 GB/s.

## Performance

One P150A, flat fp32 `torch.gather(x, 0, idx)` with N elements, warm, mean of 10 calls
then one synchronize (host time included), 2026-11-11. TT-NN's gather was timed with
`KTT_OWN_GATHER=0` in the same run.

| N | reversed index | random index | TT-NN, reversed |
|---|---|---|---|
| 131,072 | 0.10 ms | 0.24 ms | 279 ms |
| 262,144 | 0.17 ms | 0.46 ms | 1111 ms |
| 270,336 | 0.18 ms | 0.50 ms | (14.3 s with its first call) |
| 524,288 | 0.33 ms | 1.30 ms | |
| 1,048,576 | 0.63 ms | 1.79 ms | did not finish in 100 s |

All outputs equal PyTorch's bit for bit.

## Limits

- Last dim only; other dims run TT-NN's gather (see Routing).
- Indices outside `[0, W)` are clamped, not reported.
- Elements are read one 64-byte segment at a time even when a whole row would fit in L1.
  Dense random gathers on short rows are therefore bound by the DRAM segment rate, not by
  bandwidth.
