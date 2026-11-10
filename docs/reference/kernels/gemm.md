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
| unset or `KTT_GEMM=auto` | covered calls with a bf16 weight run on `ktt_gemm` where the rule below holds, the rest TT-NN's matmul with Kohaku-TT's program configs |
| `KTT_GEMM=1` | every call in its coverage runs on `ktt_gemm` (bf16 / bfp8 / bfp4 weights) |
| `KTT_GEMM=0` | every call runs TT-NN's matmul with Kohaku-TT's program configs |
| `KTT_CUSTOM_KERNELS=0` | every Kohaku-TT kernel is off, this one included |
| `KTT_GEMM_TRACE=1` | prints each new plan to stderr |

`KTT_GEMM` is read once per process. The `auto` rule (`gemm_faster()` in
`device_matmul.cpp`), with M the rows of `a` (batch dimensions folded), K and N the
weight's sizes, from the sweep under Performance: `ktt_gemm` when the weight is bf16 and

- M <= 256, or
- N > K, or
- N = K >= 4096 and 1024 <= M <= 4096.

Block-float weights take TT-NN's matmul under `auto` (not swept). A call outside the
coverage, or one whose smallest plan does not fit the free L1 span, takes TT-NN's matmul
under every setting.

### Compiles

`ktt_gemm` compiles its kernels once per process: every shape value (M, N, K, the plan,
`[N, K]` or `[K, N]`) is a runtime argument; the compile-time arguments are the tensor
accessors of A, W and the output (memory placement only) and the sender / receiver
define. Measured with `kohakutt.counters()["kernel_compiles"]`, one process, `linear`
then `mm` per shape: the first call compiled 7 kernels, every later call 0 (rows 160,
224, 288, 2048, 2080, 4096, 16384, K and N 1024 / 4096). The TT-NN route compiled 4 to 9
kernels for each new row count (its factories take the row count as compile-time
arguments). Under `auto`, calls the rule sends to TT-NN's matmul keep that per-shape
compile; the ones on `ktt_gemm` (every call up to 256 rows, decode included) do not.

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

Measured 2026-11-10 with each operand on its own RISC and NoC. The measured configs
(`measured()`) date from the one-reader kernel and have not been re-tuned.

### Crossover sweep (the `auto` rule)

Measured 2026-11-11 on one p150a shared with other jobs, the same method (20 calls in a
trace, replayed 5 times), `KTT_GEMM=1` against `KTT_GEMM=0` (today's TT-NN route, which
includes Kohaku-TT's `ktt_matmul` at 16384 rows). Microseconds per call, `linear` / `mm`;
**bold**: `ktt_gemm` slower than the TT-NN route by more than 5% in that mode. Relative
error against CPU float64 (64 sampled rows) was identical on both routes in every cell:
4.3e-3 to 4.4e-3 (norm), 4.5e-3 to 6.6e-3 (largest element over the largest output).

| M | K x N | ktt_gemm | TT-NN route | `auto` |
|---|---|---|---|---|
| 32 | 384 x 384 | 4.8 / 4.5 | 8.6 / 22.8 | ktt_gemm |
| 32 | 1024 x 1024 | 9.8 / 9.4 | 26.7 / 21.0 | ktt_gemm |
| 32 | 4096 x 4096 | 100 / 95 | 116 / 149 | ktt_gemm |
| 32 | 4096 x 1024 | 26.0 / 38.5 | 72.6 / 46.9 | ktt_gemm |
| 128 | 1024 x 1024 | 13.6 / 13.4 | 39.4 / 49.4 | ktt_gemm |
| 128 | 4096 x 4096 | 121 / 123 | 144 / 160 | ktt_gemm |
| 128 | 4096 x 1024 | 37.9 / 38.0 | 120 / 47.4 | ktt_gemm |
| 512 | 384 x 384 | 8.1 / 7.9 | 17.4 / 14.8 | TT-NN |
| 512 | 1024 x 1024 | **44.8** / 21.9 | 25.7 / 49.9 | TT-NN |
| 512 | 4096 x 4096 | **162** / 143 | 136 / 160 | TT-NN |
| 512 | 1024 x 4096 | 53.2 / 53.2 | 52.2 / 78.1 | ktt_gemm |
| 512 | 4096 x 1024 | **53.6** / 53.1 | 44.8 / 53.2 | TT-NN |
| 1024 | 1024 x 1024 | **35.6** / 35.6 | 29.0 / 45.2 | TT-NN |
| 1024 | 4096 x 4096 | 194 / 195 | 202 / 227 | ktt_gemm |
| 1024 | 1024 x 4096 | 73.0 / 71.8 | 93.6 / 96.5 | ktt_gemm |
| 1024 | 4096 x 1024 | **137** / **104** | 95.4 / 77.4 | TT-NN |
| 2048 | 384 x 384 | 21.2 / 18.9 | 20.7 / 28.0 | TT-NN |
| 2048 | 1024 x 1024 | **57.3** / **57.4** | 49.6 / 53.4 | TT-NN |
| 2048 | 4096 x 4096 | 346 / 346 | 356 / 435 | ktt_gemm |
| 2048 | 1024 x 4096 | 134 / 133 | 199 / 184 | ktt_gemm |
| 2048 | 4096 x 1024 | **169** / **170** | 106 / 134 | TT-NN |
| 4096 | 384 x 384 | 33.9 / 33.9 | 33.3 / 44.3 | TT-NN |
| 4096 | 1024 x 1024 | **109** / **109** | 87.3 / 94.8 | TT-NN |
| 4096 | 4096 x 4096 | 684 / 681 | 703 / 813 | ktt_gemm |
| 4096 | 1024 x 4096 | 261 / 261 | 398 / 330 | ktt_gemm |
| 4096 | 4096 x 1024 | **319** / **330** | 190 / 214 | TT-NN |
| 16384 | 384 x 384 | 96.0 / 95.4 | 116 / 110 | TT-NN |
| 16384 | 1024 x 1024 | 370 / 371 | 783 / 780 | TT-NN |
| 16384 | 4096 x 4096 | 2721 / **2800** | 2761 / 2472 | TT-NN |
| 16384 | 1024 x 4096 | 1006 / 993 | 1187 / 1122 | ktt_gemm |
| 16384 | 4096 x 1024 | **1239** / **1242** | 750 / 700 | TT-NN |

Also measured, all `ktt_gemm` faster: M 32 and 128 at 1024 x 4096 and 128 at 384 x 384.
`ktt_gemm` loses where K > N from 512 rows up and at N = K = 1024 from 512 to 4096 rows.
The rule keeps a few cells where `ktt_gemm` is faster on TT-NN's matmul (N = K = 384
from 512 rows, 16384 x 1024 x 1024) to stay a three-clause rule.

## Limits

- Slower than the TT-NN route where K > N from 512 rows and on mid-size square shapes
  (table above); `auto` routes those to TT-NN's matmul.
- Batched weights, transposed A, K-split and fp32 operands are not covered; those calls
  take TT-NN's matmul.
- Each kernel checks an argument sentinel (`args.hpp`) after its last common and
  per-core runtime argument and returns at once when it is missing, so a kernel and a
  host built from different argument layouts cannot hang the card.
