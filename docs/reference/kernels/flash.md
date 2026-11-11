---
title: ktt_flash
summary: Single-pass flash attention for bf16 and its training backward - what it computes, coverage, how it is routed, the lane plan and K / V multicast, the compute step, the row state and exponentials, masks, the backward's dQ and dK / dV programs, numerics, performance against TT-NN SDPA and tt-train, and limits.
tags:
  - reference
  - kernels
  - attention
---

# ktt_flash

`out[B, H, Sq, Dv] = softmax(q k^T scale + mask) v` with q `[B, H, Sq, D]` and
k / v `[B, Hk, Sk, D]` / `[B, Hk, Sk, Dv]` (H a multiple of Hk: grouped heads), bf16
TILE tensors, fp32 accumulation.

Training calls also get the gradients dq, dk, dv from two backward programs.

Source: `src/kohakutt_native/csrc/aten/attention/ktt_flash/` (forward `reader.cpp`,
`compute.cpp`, `writer.cpp`; backward `bw_dq_*.cpp` and `bw_dkv_*.cpp`; the vector-unit
steps in `sfpu.hpp`, host `program.cpp`, constants shared by host and kernels in `flash.hpp`
and `flash_bw.hpp`). Routing: `aten/attention/aten_attention.cpp` and
`device_attention.cpp`.

## Routing

`scaled_dot_product_attention` on bf16 tensors runs on `ktt_flash` whenever it is in
coverage; `KTT_FLASH=0` (read per call) sends those calls to the previous routes
(`ktt_attention`, TT-NN SDPA). The `CenteredBfp8` attention numerics mode keeps TT-NN.
`kohakutt.sdpa_kernel("flash")` runs a scope on `ktt_flash` only and raises on a call
it does not take ([numerics](../../concepts/numerics.md#attention-kernels)).

Calls with gradients, no mask, the default scale and Sq == Sk take the
training route: the forward also writes each query row's log-sum-exp, and the backward
reads it. When `ktt_flash` covers the backward (below), both run on `ktt_flash`; otherwise,
or with `KTT_FLASH=0`, both run on tt-train's fused kernels (`sdpa_fw` / `sdpa_bw`) when S
is a multiple of 32, and at other lengths the call leaves the training route (the forward
without statistics, the backward recomputed from q, k, v and the output). For such a
length the aten layer asks the stream once per shape whether `ktt_flash`'s training
programs fit and keeps the answer; a later call that no longer fits raises. A
backward runs on `ktt_flash` only with statistics `ktt_flash` wrote; the statistics have
tt-train's shape and meaning, so tt-train's backward also runs on them.

`KTT_FLASH_BWD_ORIENT=kq` (read per call) selects the key-row orientation of the backward's
dK / dV program for S a multiple of 32 (other lengths run the default orientation); see
[Backward orientation](#backward-orientation) for its limit.

## Coverage

- bf16 q / k / v, D and Dv multiples of 32 up to 256, any Sq and Sk (a partial last key
  tile is zeroed past the last key and masked in the kernel; see
  [Tile padding](#tile-padding)).
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
- Backward (training route): dense and causal, MHA / GQA / MQA, D and Dv up to 256; dk and
  dv sum over each K / V head's query heads. Any S with equal heads (a ragged last tile is
  masked in the kernels: gradients of the padded rows are exactly 0 and the padding's
  contents do not reach any gradient); grouped heads need S a multiple of 32. The backward
  always uses the accurate exponential. Calls outside this (or whose backward circular
  buffers or jobs do not fit) train on tt-train's kernels as before when S is a multiple of
  32.
- Not covered: dropout, the backward with masks. Masked calls that need gradients keep
  working: their backward recomputes from q, k, v and the output.
- Packed variable-length batches: [flash-varlen.md](flash-varlen.md).

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
copied into the score tiles first and the score matmul accumulates onto them: the causal
diagonal, a partial last key tile or both combined (constant tiles), a per-row user mask's
tiles, or a broadcast user mask row (unpacker row broadcast). The pack
thread then updates the row reference, writes P = exp(S scale - reference) as bf16 and packs
the state. The output section accumulates O_r += P V and the row sum l_r = P x ones in L1
(fp32, packer L1 accumulation); when a row's reference moved, O_r and l_r are first scaled
by alpha. Steps run in pairs of different rows (two score sections, then two output
sections) so one step's vector work overlaps the next step's matmuls. The last step of a
row finalizes O_r / l_r (reciprocal plus two Newton steps) to bf16.

### Row reference

Each row keeps a reference v (fp32, per row, in an fp32 state tile unpacked straight to Dst):
P = 2^(A x + v - U0) with A = scale log2 e. v = U0 - A m is kept unrounded, so A x + v
stays exact to fp32 for scores of any size (a bf16 v would be off by up to half an ulp of
A m, which passes the exponent's 128-doubling headroom once |A m| >= 2^15).
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

The ones tile, the causal diagonal mask, the partial-tile key mask and the two combined are
built by the vector unit from lane coordinates (no RISC-V stores) and packed once per launch
while the readers' first DRAM reads are in flight.

The kernels reprogram the vector unit's constant registers 11-14; every compute kernel ends
by setting register 11 back to -1, the core-wide constant other kernels read (tt-llk's
`_restore_lconst_neg1_`).

## Backward

The training forward is a second program variant (its own cache key; the forward without
gradients is unchanged). Per step the pack thread also sums the step's fp32 P exactly into
lane partials (8 per row: each lane adds 4 columns of every score tile), packed fp32 and
accumulated per row like O (scaled by alpha when the reference moves). At the row's end an
8-lane reduction gives the row sum l, and lse = ln sum_j exp(scale s_j) = ln 2 ((U0 - v) /
kScale + log2 l) (exponent plus a degree-5 log2 polynomial, max error 2.9e-5 doublings) goes
to the fp32 statistics `[B, H, Sq, 32]`: row r's lse in every column of row r, the layout of
tt-train's statistics. The training call returns column 0 as its log-sum-exp.

The backward is FlashAttention-2's: with D_i = rowsum(dO_i O_i), P = exp(scale S - lse),
dP = dO V^T and dS = P (dP - D), it accumulates dQ = scale dS K, dK = scale dS^T Q and
dV = P^T dO (the scale applied once per output tile). Two programs run back to back, both
with readers on NoC0 and writers on NoC1, constants (ones, causal diagonal) built by the
vector unit:

- dQ program: jobs, lanes and K / V multicast exactly as the forward; each reader issues its
  rows' reads and the first key block's under one barrier. Per query row: D = rowsum(dO O)
  (elementwise product, then x ones on the matrix unit, fp32) and w = 639 - lse log2 e go into
  one fp32 row statistics tile (w at the even columns, D at the odd ones), kept for the row's
  steps and written to DRAM for the second program. Per (row, key tile): S = Q K^T and
  dP = dO V^T into Dst 0 and 1, the row statistics tile into Dst 3; one vector pass forms
  P = 2^(A s + w - 639) (accurate exponential) and dS = P (dP - D), packed bf16. Per chunk of
  up to 6 key tiles, dQ_r += dS K accumulates in L1 (fp32), rows paired as in the forward.
- dK / dV program: a job is up to R key tile rows of one K / V head (K and V resident); the
  lane's top core streams the query blocks (QB tiles of Q and dO and their row statistics
  tiles) of the head's query heads and multicasts them down the lane. Per (query tile, key
  row): S = Q K^T and dP = dO V^T with the forward's operands in the forward's roles, then
  the same vector pass as in the dQ program; P and dS are packed bf16. The matrix unit
  computes SrcB x SrcA, and the unpacker can transpose a tile only on its way into SrcA, so
  P and dS (the left operands of P^T dO and dS^T Q) are not turned. The kernel turns the
  streamed Q and dO tiles instead, once per chunk of up to 6 query tiles (unpacker transpose
  into SrcA, copied to Dst, packed bf16), and accumulates the transposed outputs
  dV_j^T += dO^T P and dK_j^T += Q^T dS as plain matmuls in L1 (fp32), in one Dst section
  when D and Dv are at most 64. A finished row is scaled (dK), packed bf16, turned by the
  same unpacker move and written. Causal rows skip the query tiles above the diagonal; lanes
  take key chunks largest work first.

A tile the pack thread writes into a slot of a reserved buffer (the row statistics, the fp32
accumulators) is read back by the unpacker only after a token the pack thread pushes behind
those packs.

Ragged lengths (S not a multiple of 32) use two masks, both exact:

- Keys past the last key: the forward's constant tile (-1e30 in those columns; on the causal
  diagonal tile combined with the diagonal mask) is copied into the score tile of the last
  key tile before the score matmul, in both backward programs. P and dS are exactly 0
  there, so those columns add nothing to dQ and their dK / dV rows are 0.
- Queries past the last query: in the dQ program, the last query tile's w gets a row mask
  (-1e30 in those rows) before the statistics tile is formed. P and dS are exactly 0 for
  those rows against every key in both programs (the dK / dV program reads the same
  statistics tile), so they add nothing to dK / dV and their dQ rows are 0.

The key mask alone would let a padded key of inf or NaN through (inf - 1e30 is inf, and
0 x inf is NaN in P V): the readers zero those rows first ([Tile padding](#tile-padding)).

### Tile padding

The rows of a tile past the logical length hold whatever the tile held before (a slice of
a larger tensor keeps them), so the kernels never use their values. In the forward
(inference and training) and dQ readers the lane sender, and in the dK / dV reader every
core for its resident key rows, overwrite the key rows past the last key of the last key
tile of each head in K and in V with zeros in L1, before the block is multicast or pushed:
one NoC read per padded face span from the core's zero region (`MEM_ZEROS_BASE`), then a
read barrier. The keys in the last key tile come as a runtime argument after the jobs (the
dK / dV reader: key tiles per head x 64 + that count), so no kernel compiles per length,
and a tile-aligned call issues no extra read. Padded keys then score exactly 0 before the
key mask, and padded V rows add 0 x 0 to O, dQ and dV. Padded query rows only reach
padded output rows (forward) and are masked through the row statistics (backward). The
packed variable-length readers zero Q, K, V, dO and O rows past the last token the same
way ([flash-varlen.md](flash-varlen.md)). No `fill_implicit_tile_padding` program runs on
any ktt_flash route; the TT-NN SDPA fallback keeps it on K and V.

### Backward orientation

Default: every program computes S = Q K^T as the forward does (Q through the matrix unit's
row operand, K through its column operand, the same accumulation order), so the backward
rebuilds the forward's P from the same score bits at every fidelity. Measured on dinov3
layer-0 tensors (scale x score up to 7.7e4): the forward's and the backward's P agree within
1.3e-3 at HiFi2 and 1.3e-2 at LoFi, and P's rows sum to 1 within 3e-4 / 1.4e-2.

`KTT_FLASH_BWD_ORIENT=kq` runs the dK / dV program in the key-row orientation: S^T = K Q^T
and dP^T = V dO^T, then dV_j += P^T dO and dK_j += dS^T Q directly, with no turned tiles
(the dQ program then writes the statistics in column layout, built from the row tile by
vector-unit lane moves: a 4x4 lane transpose and stores predicated on the lane index). Its
scores are not the forward's, because the operands swap roles in the matrix unit:

- below HiFi4 the matrix unit drops low bits of its row operand, here K instead of Q;
- at every fidelity it rounds each product to 10 fractional bits of the largest product
  among the 8 it is summed with, and that rounding depends on the operand roles.

S^T differs from S by up to about 2^-10 of that largest product. With ordinary activations
this is far below one and both orientations give the same gradients (measured: rel 8e-3
apart at HiFi2, 2.5e-3 at HiFi4, each within the bound of float64). When
scale x 2^-10 x the largest q_d k_d product approaches 1 (dinov3: products near 1e5), P^T no
longer matches the forward's log-sum-exp: at HiFi4 the gradients are finite and wrong (P's
rows sum to 3e4 on the dinov3 tensors), below HiFi4 they overflow. Use the default there.

## Numerics

- fp32 Dst accumulation for every matmul and the row state; fidelity from Kohaku-TT's
  numerics (HiFi2 by default). At HiFi2 the matmul keeps 7 of Q's 8 significand bits.
- Masks: logits clamped at -5e5 (bool false and -inf become -5e5, the packed kernel's
  masked logit) and divided by the scale on the device before the call; the key columns
  past the last key (tile padding) are set to the masked value. A masked key stays out of
  the softmax while its logit is below a kept key's plus about 5e5 (the earlier -1e4 let
  masked keys with logits 1e4 above the kept ones through). A row whose first key tiles
  are all masked starts its reference at the masked value, which fp32 carries exactly.
  Mask terms are copied into the score tiles exactly, at any fidelity.
- No data moves through the matrix unit's multiplier: mask terms are tile copies (a row
  broadcast for a broadcast mask) and the backward's turned tiles are unpacker transposes,
  all exact at any fidelity.
- Scores of any fp32 size: the row reference and the backward's w are fp32, so the
  exponent's argument is exact to fp32 (tested with raw scores near 1e6, as in dinov3's
  massive-activation channels). The scores themselves carry the matrix unit's rounding
  (10 fractional bits below the largest of each 8 products): at dinov3's logits the forward
  is within 2e-2 of float64 and gradients of near-tied rows follow the device's scores.

Rel RMS against float64 (accurate exp, HiFi2): about 6e-3 (TT-NN SDPA with fp32 Dst: 1.1e-2;
CPU bf16 1.7e-3). Approximate exp: about 1.9e-2 (TT-NN: 2.4e-2).

Backward, rel RMS of each gradient against float64 autograd: at HiFi2 dq and dk 9.9e-3 -
1.1e-2, dv 7.3e-3 - 8.2e-3; at HiFi4 2.7e-3 - 3.0e-3 (CPU bf16: 2.4e-3 - 4.6e-3). At HiFi2
S and dP keep 7 of Q's and dO's 8 significand bits and P, dS are bf16 operands; a CPU
emulation of exactly these operand precisions gives the same numbers. The log-sum-exp is the exact fp32 one of the device's own
scores (its row sum is an fp32 vector-unit sum, not P x ones on the matrix unit, whose 7-bit
operand truncation would bias every row's statistic). Against float64 it differs by about
3e-3 RMS (Q's truncation in S). After an approximate-exp forward the gradients are within
1.3e-2 - 1.9e-2 (the backward's exponential is accurate, the statistics and O are the
forward's). Gradients of keys only masked scores see are exactly 0.

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

### Training

Training forward (with statistics) and backward against tt-train's `sdpa_fw` / `sdpa_bw`
(`KTT_FLASH=0`; fixed HiFi4) and the aten layer's recomputing backward (masked calls:
matmuls, softmax and its backward as separate ops), fp32 Dst, accurate exponential, at the
`kohakutt.numerics` fidelity. Per call, us: device time of the programs (device profiler,
first kernel start to last kernel end, summed over the call's programs), then through
Kohaku-TT's op in a trace of 10 calls (the backward as a forward-plus-backward step minus the
forward), then eager. "kq" is the backward with `KTT_FLASH_BWD_ORIENT=kq`.

| Case [B x H x S x D], fidelity | | ktt_flash fwd | tt-train fwd | ktt_flash bwd | ktt_flash bwd, kq | tt-train bwd | recompute bwd |
|---|---|---|---|---|---|---|---|
| 8x16x256x64, HiFi2 | device | 100 | 229 | 229 (dQ 119, dK / dV 112) | 246 (138, 120) | 680 | |
| | trace | 115 | 243 | 246 | 260 | 692 | |
| | eager | 124 | 259 | 266 | 377 | 723 | 6209 |
| 8x16x256x64, LoFi | device | 98 | 229 | 231 (dQ 118, dK / dV 114) | 255 (140, 121) | 668 | |
| | trace | 114 | 241 | 240 | 259 | 694 | |
| 1x8x1024x64 causal, HiFi2 | device | 68 | 154 | 124 (dQ 52, dK / dV 71) | 128 (60, 68) | 453 | |
| | trace | 80 | 163 | 131 | 134 | 463 | |
| | eager | 195 | 189 | 348 | 300 | 500 | 5409 |
| 1x8x1024x64 causal, LoFi | device | 67 | 153 | 124 (dQ 52, dK / dV 71) | 128 (60, 68) | | |
| | trace | 81 | 164 | 128 | 134 | 463 | |

The backward takes 2.3x (8x16x256x64) and 1.8x (causal) the training forward's device time,
and a third of tt-train's backward. The default orientation is the faster one: its turned
tiles cost less than the key-row orientation's second statistics tile per query tile row
and the lane moves that build it. At 8x16x256x64 both programs are bound by their DRAM reads
(each core's Q, dO, O and statistics rows, K / V per lane); the lane senders, which also read
K / V, finish last. Eager calls at these sizes are host bound and vary from run to run.

## Limits

- Short sequences in approximate mode are DRAM bound; ktt_flash there is within a few
  percent of TT-NN's time.
- Per-row masks are read once per head: a [B, 1, Sq, Sk] mask costs Sq x Sk x 2 bytes per
  head of DRAM traffic.
- Each new (Sq, Sk, D) compiles new kernels (the tile counts are compile-time arguments).
- Backward with inputs far from zero mean: dS = P (dP - D) subtracts two terms of the size
  of dO V^T, and at HiFi2 each carries 7 significand bits. With q, k, v and dO all offset
  by +2 (unit-variance noise, 1 x 1 x 64 x 32 to 1 x 12 x 512 x 64) dq is 0.37 - 0.48 and
  dk 0.06 - 0.10 rel RMS from float64 (offset by +5: 1.8 - 3.2 and 0.22 - 0.49) while dv
  stays at 6e-3; zero-mean inputs are within the bounds above.
- `KTT_ATTENTION_CENTER=1` (off by default, read per call) removes the common offset
  from a training call (q, k or v requires grad, no mask): the per-head mean over the
  keys is subtracted from k before the forward, and from v and the saved output before
  the backward. Softmax, the output and the gradients do not depend on these means. With
  it: offset +2: dq 1.2e-2 - 1.6e-2, dk 6.1e-3 - 8.6e-3; offset +5: dq 2.2e-2 - 4.2e-2,
  dk 7.1e-3 - 1.2e-2; dv 4.4e-3 - 7.2e-3; zero-mean inputs unchanged. What it changes
  besides, and why it is a switch:
  - k less its mean is rounded to bf16. A mean that comes from a few very large keys,
    not from a common offset, coarsens every other key (keys of 251 on 8 channels at 4
    of 512 positions: dk 2.2e-2 against 2e-2 without the switch).
  - The log-sum-exp the training forward returns is that of the centred scores (lower
    by scale x q . mean of k), and a training forward differs from the same call without
    gradients in rounding.
  - What is left in dq follows q's own offset and the bf16 rounding of the saved output.
  - Cost per training call: the forward is 4 device ops instead of 1, the backward 5
    instead of 1 (6 with grouped heads); the centred keys stay on the device with the
    statistics.
- The backward recomputes S and dP in both programs (7 matmuls per score tile against the
  forward's 2) and reads Q, dO and the statistics twice.
- The dK / dV program turns DT + DVT tiles per streamed query tile on every core of a lane:
  a core holding one or two key rows per job spends as long on them as on its scores.
- The key-row backward orientation (`KTT_FLASH_BWD_ORIENT=kq`) is only for ordinary logits
  (see Backward orientation).
