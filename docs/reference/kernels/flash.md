---
title: ktt_flash
summary: Single-pass flash attention for bf16 - what it computes, coverage, how it is routed, the lane plan and K / V multicast, the compute step, the row state and exponentials, masks, numerics, performance against TT-NN SDPA and limits.
tags:
  - reference
  - kernels
  - attention
---

# ktt_flash

`out[B, H, Sq, Dv] = softmax(q k^T scale + mask) v` with q `[B, H, Sq, D]` and
k / v `[B, Hk, Sk, D]` / `[B, Hk, Sk, Dv]` (H a multiple of Hk: grouped heads), bf16
TILE tensors, fp32 accumulation.

Source: `src/kohakutt_native/csrc/aten/attention/ktt_flash/` (`reader.cpp`,
`compute.cpp`, `writer.cpp`, the vector-unit steps in `sfpu.hpp`, host `program.cpp`,
constants shared by host and kernels in `flash.hpp`). Routing:
`aten/attention/aten_attention.cpp` and `device_attention.cpp`.

## Routing

`scaled_dot_product_attention` on bf16 tensors runs on `ktt_flash` whenever it is in
coverage; `KTT_FLASH=0` (read per call) sends those calls to the previous routes
(`ktt_attention`, TT-NN SDPA). The `CenteredBfp8` attention numerics mode keeps TT-NN.

## Coverage

- bf16 q / k / v, D and Dv multiples of 32 up to 256, any Sq and Sk (a partial last key
  tile is masked in the kernel).
- MHA, GQA and MQA (H a multiple of Hk); every (batch, head) pair computed separately.
- Causal (Sq == Sk).
- Additive and bool masks `[B or 1, H or 1, Sq or 1, Sk]` (bool masks become additive in
  the aten layer). Not combined with causal.
- Accurate and approximate exponentials (numerics' `approx`), fidelity from numerics.
- Calls whose jobs would overflow a core's runtime arguments (at most 46 jobs per core:
  very many (batch, head) pairs with short queries, over about 4500 pairs of one query tile
  on the 110-core grid) and calls whose circular buffers do not fit L1 keep today's route;
  both are decided on the stream worker.
- Block-causal masking (each query block sees keys up to the end of its block) is in the
  kernel (compile-time block size) but not routed: PyTorch SDPA has no block-causal flag,
  and recognizing the mask form would mean reading the mask.
- Not covered: dropout, variable-length batches, the backward pass. Calls that need
  gradients keep working: the existing backward recomputes from q, k, v and the output.

## How it works

Sizes in tiles: Sqt = ceil(Sq / 32), Skt, DT = D / 32, DVT = Dv / 32.

### Jobs, lanes and K / V multicast

The work is (batch, head, query tile row). Each grid column is cut into lanes of L cores
(L divides the grid height). A lane job is up to L x R query rows of one head (R = 13 query
rows per core when L1 allows); the lane's top core (sender) reads the head's K and V from
DRAM in blocks of KB key tiles (12 when L1 allows) and multicasts each block down the lane,
then a VALID flag on the same NoC; every receiver first frees its slot and signals the
sender. Each core of the lane takes its share of the lane job's rows: contiguous rows, or
for causal calls rows L apart so the triangle's work spreads evenly.

Non-causal lanes take even ranges of the (batch, head, row) order; causal lanes take query
chunks largest work first onto the least loaded lane. The host picks L as the length with
the fewest score tiles on the busiest core, then the longest (fewest K / V reads).

NoC plan: every reader runs on NoC0 (senders multicast down their column, NoC0 runs +y) and
every writer on NoC1. Multicast stays inside one grid column. One reader, one writer and one
compute kernel cover the grid (the sender role is a runtime argument), which keeps the host
time per call at TT-NN SDPA's.

### Compute step

A step is one query tile row against a sub-chunk of up to 3 key tiles: S = Q_r K^T into
Dst tiles 0-2 (fp32 Dst, half-sync), with the row's state tile in Dst 3. Mask terms are
added on the matrix unit as identity x mask tile (the causal diagonal, a partial last key
tile, a per-row user mask) or column-0 ones x mask row (a broadcast user mask). The pack
thread then updates the row reference, writes P = exp(S scale - reference) as bf16 and packs
the state. The output section accumulates O_r += P V and the row sum l_r = P x ones in L1
(fp32, packer L1 accumulation); when a row's reference moved, O_r and l_r are first scaled
by alpha. Steps run in pairs of different rows (two score sections, then two output
sections) so one step's vector work overlaps the next step's matmuls. The last step of a
row finalizes O_r / l_r (reciprocal plus two Newton steps) to bf16.

### Row reference

Each row keeps a reference v (bf16, per row): P = 2^(A x + v - U0) with A = scale log2 e.
The reference only moves when a score would make P exceed 2^16 (tau 16); then v takes the
new row maximum and the row's O and l are rescaled by alpha = 2^(v_new - v_old). Rows whose
maximum grows slowly never pay the rescale. The state tile also carries one flag per tile
(any row moved), read by the output section to pick the plain or the rescaling path.

### Exponentials

The vector unit computes P per numerics' `approx` setting:

- Accurate (default): 2^u split into exponent and fraction, a degree-2 polynomial for the
  fraction (max rel err 1.75e-3), arguments clamped below so masked scores give 0.
- Approximate: P's bits are the integer A x + v placed as exponent and mantissa (shift by
  15); the integer comes out of the multiply-add with a 1.5 x 2^23 bias, is clamped at 0
  and shifted, four macro sub-units per group, one group per two instructions.

### Constant tiles

The ones tile, identity, causal diagonal mask, column-0 ones and the partial-tile key mask
are built by the vector unit from lane coordinates (no RISC-V stores) and packed once per
launch while the readers' first DRAM reads are in flight.

## Numerics

- fp32 Dst accumulation for every matmul and the row state; fidelity from Kohaku-TT's
  numerics (HiFi2 by default). At HiFi2 the matmul keeps 7 of Q's 8 significand bits.
- Masks: logits clamped at -1e4 (bool false and -inf become -1e4) and divided by the scale
  on the device before the call. The mask term is exact at HiFi2 (the mask is the operand
  whose bits all take part); at LoFi a finite bias keeps 5 significant bits.
- Scores (logits) must stay within about +-1e4: beyond that the bf16 row reference rounds
  by more than 32 doublings.

Rel RMS against float64 (accurate exp, HiFi2): about 6e-3 (TT-NN SDPA with fp32 Dst: 1.1e-2;
CPU bf16 1.7e-3). Approximate exp: about 1.9e-2 (TT-NN: 2.4e-2).

## Performance

Per call (trace of 10 back-to-back calls), Blackhole p150, HiFi2, us. TT-NN SDPA runs at
the same settings (fp32 Dst, the same exponential); the last column is TT-NN's bf16-Dst
streaming path, which forces the approximate exponential (rel RMS 2.4e-2).

| Case [B x H x S x D] | ktt_flash | TT-NN fp32 Dst | TT-NN bf16 Dst |
|---|---|---|---|
| 1x32x4096x64 | 2694 | 7756 | 1239 |
| 1x24x4096x128 | 2323 | 6319 | 1272 |
| 1x32x4096x64, approx | 1808 | 1901 | 1235 |
| 8x16x256x64, approx | 68 | 65 | 58 |
| 1x8x1024x64 causal | 65 | 465 | 95 |
| 1x8x1024x64 causal, approx | 52 | 129 | 94 |
| 1x16x2048x64, mask [1, 1, S, S] | 537 | 1677 | 878 |
| 2x8x1024x128 (GQA 4), mask [2, 1, S, S] | 217 | 514 | 293 |
| 2x8x1024x128 (GQA 4), mask [2, 1, 1, S] | 179 | 514 | 295 |
| 1x32x4096x64, LoFi | 2600 | 7506 | 1138 |

The accurate exponential is vector-unit bound (about 300 of the 450 cycles per score tile);
the approximate one is bound by the pack thread's per-step vector work.

## Limits

- Short sequences in approximate mode are DRAM bound; ktt_flash there is within a few
  percent of TT-NN's time.
- Per-row masks are read once per head: a [B, 1, Sq, Sk] mask costs Sq x Sk x 2 bytes per
  head of DRAM traffic.
- Each new (Sq, Sk, D) compiles new kernels (the tile counts are compile-time arguments).
