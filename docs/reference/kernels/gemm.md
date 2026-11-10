---
title: ktt_gemm
summary: Block matmul with 2-D / 1-D multicast layouts for bf16 activations and bf16 / bfp8 / bfp4 2-D weights - what it computes, routing, coverage, layouts, NoC plan, L1, compute, config choice, performance against TT-NN and limits.
tags:
  - reference
  - kernels
  - matmul
---

# ktt_gemm

`out[..., M, N] = a[..., M, K] @ w`: `a` bf16, `w` a 2-D weight in bf16, bfp8 or bfp4
stored `[K, N]` or `[N, K]`, fp32 accumulation, bf16 output. Batch dimensions of `a`
are flattened into M.

Source: `src/kohakutt_native/csrc/aten/matmul/ktt_gemm/` (`a_writer.cpp` A reads and
multicasts plus output writes, `w_reader.cpp` W reads and multicasts, `compute.cpp`, `program.cpp` plan and program,
`args.hpp` the argument sentinel). Routing: `aten/matmul/device_matmul.cpp`.

## Routing

| Setting | Effect |
|---|---|
| `KTT_GEMM=1` | `mm` and `linear` calls in its coverage run on `ktt_gemm` (read once per process) |
| unset or `KTT_GEMM=0` | those calls run TT-NN's matmul with Kohaku-TT's program configs |
| `KTT_CUSTOM_KERNELS=0` | every Kohaku-TT kernel is off, this one included |
| `KTT_GEMM_TRACE=1` | prints each new plan to stderr |

With `KTT_GEMM=1`, a call outside the coverage, or one whose smallest plan does not fit
the free L1 span, takes TT-NN's matmul. The kernel stays opt-in until it passes the
[gate](../../concepts/kernel-design.md#the-gate) (see Performance).

## Coverage

- `mm` and `linear` with bf16 output, `a` not transposed, a 2-D weight, both TILE.
- Weights bf16, bfp8 or bfp4 (see [block floats](../../concepts/block-float.md)); `[N, K]`
  weights are transposed within their tiles by the unpacker, no transpose op runs.
- Any M, N, K. When K is not a multiple of 32, the tile padding of both operands is
  zeroed first (`ttnn::fill_implicit_tile_padding`); ragged M and N are computed on pad
  tiles and dropped by the writer.
- Bias: added after the matmul (`ttnn::add`).
- Not covered (other variants of the matmul family): batched weights (`bmm`), transposed
  A, K-split with fp32 partial outputs, fp32 operands.

## How it works

Sizes in tiles: Mt (all rows of `a`), Nt, Kt. Grid: gx x gy cores (11 x 10 on p150a).

### Layouts

- **2D.** Core (x, y) owns output rows `[y pcM, (y + 1) pcM)` and columns
  `[x pcN, (x + 1) pcN)`, pcM = ceil(Mt / gy), pcN = ceil(Nt / gx). The panel's right
  column reads A from DRAM and multicasts it leftwards along its row inside the panel; the top row
  reads W and multicasts it down its column. Per core, each received A tile is used bn
  times and each W tile bm times.
- **1D.** Core i (column-major) owns output rows `[i pcM, (i + 1) pcM)`, pcM =
  ceil(Mt / (gx gy)), and every column. Each core reads its own A rows; each column's top
  core reads W and multicasts it down the column. Each W tile is used pcM times per core.
- Choice: 1D when pcM(1D) >= 3 and pcM(1D) > min(pcM, pcN) of 2D (more W reuse than 2D's
  reuse of either operand); otherwise 2D. Tall-M shapes (M large, N and K small) take 1D.
- Panels: multicast rectangles stay inside one physically contiguous column span
  (`program::panels()`, from each chip's SoC descriptor).
- Each core walks its range in nbh x nbw blocks of bm x bn tiles; per block, nk K blocks
  of kb tiles (the last may be shorter). Every core runs the same block sequence; tiles
  past the core's range or the tensor are computed and dropped by the writer.

### NoC plan

| Stream | Kernel (RISC) | Cores | NoC |
|---|---|---|---|
| A DRAM reads, multicasts and handshakes; output tiles | `a_writer.cpp` (BRISC) | every core | NoC1 |
| W DRAM reads, multicasts and handshakes | `w_reader.cpp` (NCRISC) | every core | NoC0 |

- Each operand moves on its own RISC and its own NoC, so the A and W streams of a K block
  run at the same time. With both on one RISC and one NoC the kernel was bound by data
  movement: 762 us per call at 2048 x 8192 x 4096, 736 us with the products removed, 647 us
  with the streams split.
- A NoC1 multicast runs from high to low coordinates: the A sender is the panel's last
  column and its rectangle is given end first.
- An output block is written by the A kernel after it has delivered the first A block of
  the next output block (the last one after the loop).
- Both kernels run in `DM_DEDICATED_NOC` mode on different NoCs: in dedicated mode both
  RISCs program the same command buffers and share the NIU counters of a NoC, so two
  kernels on one NoC break each other's barriers.
- Multicast handshake per block: each receiver reserves its CB slot, sets its flag
  INVALID and increments the sender's semaphore; the sender waits for all receivers,
  multicasts the block to the same L1 address, flushes, then multicasts VALID.

### L1 (per core)

| CB | Contents | Format | Tiles | Depth |
|---|---|---|---|---|
| 0 | A block | bf16 (2048 B) | bm kb | 2 |
| 1 | W block | bf16 / bfp8 (1088 B) / bfp4 (576 B) | kb bn | 2 |
| 16 | output subblocks | bf16 | sh sw | 2 |
| 24 | partial sums (nk > 1) | fp32 (4096 B) | bm bn | 1 |

The plan fits `program::cb_budget()`: the free L1 span below the lowest live L1 buffer at
the call (after the output is allocated), rounded down to 32 KiB (the budget class).
Plans are cached per (Mt, Kt, Nt, W tile size, grid, budget class), and the program key
holds every plan field, so a cached program always fits the L1 it runs in. The same shape
can take a different plan when other L1 buffers are live.

### Compute

- `compute_kernel_hw_startup<SrcOrder::Reverse>` (matmul operand order), fp32 DST in
  half-sync mode.
- Subblock sh x sw with sh sw <= 4 (one half of fp32 DST): the first of 1x4, 4x1, 2x2,
  1x3, 3x1, 1x2, 2x1, 1x1 that divides the block exactly.
- Per subblock and K block, `matmul_block` over the block's K tiles. K blocks 0 to
  nk - 2 pack into CB 24 with packer L1 accumulation (off for block 0); the last K block
  reloads the subblock from CB 24 (unpacked to DST as fp32, `UnpackToDestFp32`), adds its
  products and packs bf16 to CB 16.
- Math fidelity comes from the numerics scope ([numerics](../../concepts/numerics.md)),
  HiFi2 by default.

### Config choice

- A table of configs measured fastest per call on the card (MAMF-style: many calls
  averaged) for the benchmark shapes, used when it fits the budget.
- Otherwise a rule: the layout above; the fewest output blocks that fit (ties: the least
  DRAM re-reading, nbw Mt + nbh Nt; 1D splits N into at least two blocks when Nt >= 8, so
  a block's writes overlap the next block's compute); then the largest kb of 16, 8, 4, 2,
  1 that fits; then the subblock above.
- No runtime model picks a config.

## Performance

One p150a, 110-core grid, bf16 activations and weights, HiFi2, fp32 accumulation. Device
time per call from a trace of 20 back-to-back calls replayed 5 times. `linear`: weight
`[N, K]`; `mm`: weight `[K, N]`. TT-NN route: the same calls with `KTT_GEMM=0
KTT_CUSTOM_KERNELS=0` (`minimal_matmul` for `mm`, `ttnn::linear` with Kohaku-TT's 2-D
multicast config for `linear`).

| M x K x N | ktt_gemm linear / mm (us) | TT-NN route linear / mm (us) |
|---|---|---|
| 1024 x 1024 x 1024 | 68.2 / 37.2 | 29.9 / 35.8 |
| 2048 x 2048 x 2048 | 121 / 122 | 111 / 121 |
| 4096 x 4096 x 4096 | 685 / 682 | 704 / 812 |
| 2560 x 3072 x 4096 | 333 / 334 | 333 / 379 |
| 16384 x 1024 x 1024 | 371 / 373 | 327 / 296 |
| 16384 x 384 x 384 | 97 / 97 | 107 / 98 |
| 1024 x 8192 x 4096 | 366 / 364 | 365 / 381 |
| 4096 x 1024 x 8192 | 512 / 519 | 717 / 572 |
| 2048 x 8192 x 4096 | 629 / 628 | 649 / 735 |
| 2048 x 4096 x 8192 | 693 / 692 | 690 / 692 |

Measured 2026-11-10 with each operand on its own RISC and NoC. ktt_gemm is level with or
faster than the TT-NN route from 2560 x 3072 x 4096 up, slower on the two smallest square
shapes (`linear` at 1024 x 1024 x 1024 by 2.3x) and on the 1D layout at 16384 x 1024 x
1024, where every core's A kernel both reads A and writes the output. The measured configs
(`measured()`) date from the one-reader kernel and have not been re-tuned.

## Limits

- Slower than the TT-NN route on small square shapes and on 16384 x 1024 x 1024 (table
  above); it stays opt-in.
- Batched weights, transposed A, K-split and fp32 operands are not covered; those calls
  take TT-NN's matmul.
- Each kernel checks an argument sentinel (`args.hpp`) after its last common and
  per-core runtime argument and returns at once when it is missing, so a kernel and a
  host built from different argument layouts cannot hang the card.
