# Performance

How Kohaku-TT measures and fixes performance, measurements on the MOSS-Image pipeline,
and the optimizations they led to. TT-NN behaviour behind several of them is in
[TT-NN issues](../reference/ttnn-issues.md). Numbers are from one Blackhole P150A.

## Principles

**Bench ops, not models.** A model is a sequence of aten calls. Each op is measured on
its own, cold and warm, at representative shapes, and slow ops are fixed there; the
remaining cost (dispatch, kernel seams, memory placement) is general and measured once.
A per-model profile checks the totals.

**Cold equals warm.** A first call at a new shape costs what a repeat call costs. In
Kohaku-TT's own kernels every size is a runtime argument and compile-time arguments are
only what changes the code (dtype, op body, layout class, kernel size): one binary serves
every shape, and a new shape needs no host work beyond runtime arguments.

**Host and device are measured apart.** Host CPU time against wall time per call
separates host work (CPU ≈ wall), waiting on the device (wall ≫ CPU) and JIT compiles
(child processes). Device time comes from the device profiler; host time from Tracy's
host zones and the gaps between device ops.

**Own kernels where TT-NN falls short.** Kohaku-TT replaces a TT-NN op when measurement
shows it wrong or slow on real workloads (convolution, attention configuration,
pointwise chains, some vector ops) and uses TT-NN elsewhere. Every feature is on by
default with an off switch. Each own kernel is checked against a CPU reference from
small to large shapes.

**Precision is a knob, correctness is not.** Error is reported as statistics against
float64 and against the same model in bf16 on the CPU, which separates native low
precision (bf16 activations, block-float weights, HiFi2) from wrong math. For a whole
model, comparing each block's output against the same block on the CPU locates any wrong
kernel.

**Memory placement adapts.** Activations live in L1 when they fit; when a program's
buffers clash with them, they spill to DRAM and the L1 budget halves for the rest of the
process.

## MOSS-Image measurements

### DiT

One forward at 1360x768 with 1615 text tokens (5695 tokens), bfp8 weights, HiFi2:

| configuration | forward |
|---|---|
| fp32 elementwise, attention with padding mask | 2.452 s |
| attention without the padding mask | 1.966 s |
| fused block, minimal matmul, streaming attention | 0.652 s |

One CFG step (cond and uncond forward, one chip, text tokens refined once):

| image | text tokens | CFG step |
|---|---|---|
| 1024x1024 | 300 | 0.935 s |
| 1360x768 | 1615 | 1.104 s |
| 1024x1024 | 1615 | 1.129 s |

In end-to-end sampling, steps after the first run at 1.04-1.21 s; the first step at a
new size takes 1.6-3.4 s.

Error against fp32 CPU at 256x256: tt with bfp8 weights at HiFi2, rel 1.36e-1, cosine
0.991; tt with bf16 weights at HiFi4, rel 9.9e-2; the same model in bf16 on the CPU, rel
5.0e-2. Per block, tt is within 1-2.5x of the CPU bf16 block's error.

### DiT ops

| op (DiT shape) | before | after |
|---|---|---|
| modulated RMSNorm (fp32 composite → `ktt.rms_norm`) | ~3.5 ms | 0.36 ms |
| q / k RMSNorm | ~1.5 ms | 0.15 ms |
| RoPE (7-op composite → `ktt.rotary`) | ~2 ms | 0.60 ms |
| SiTU (fp32 eager → fused chain) | ~11 ms | 1.93 ms |
| gated residual | ~1.3 ms | 0.47 ms |
| linear 3072 → 9216 (`ttnn.linear` → minimal matmul) | 2.52 ms | 1.69 ms |
| linear 3072 → 16384 | 4.41 ms | 2.83 ms |
| linear 8192 → 3072 | 2.02 ms | 1.42 ms |
| attention, 5695 tokens | 11.5-15.6 ms | 2.70 ms |

### VAE decode (1312x800)

| configuration | warm | cold (first call at a size) |
|---|---|---|
| composite group norm, convolution as per-tap matmuls | 5.2 s | 137 s |
| TT-NN conv2d, native group norm | 1.1 s | 61-69 s |

Cold time is host setup inside TT-NN ops on new shapes: tiled reshape and padded slice
take 54 of 61 s ([TT-NN issues](../reference/ttnn-issues.md)).

### DiT noise floor and CFG

The change of the DiT velocity under an input perturbation far below bf16 resolution
(rel 1e-6), 512x512 with 96 text tokens:

| implementation | output change |
|---|---|
| fp32, host CPU | 5.7e-6 (proportional to the perturbation) |
| bf16, host CPU | 4.5% |
| tt, bfp8 weights, attention with fp32 dest | 7.3% |
| tt, bfp8 weights, streaming attention (bf16 dest) | 12% |

A bf16 implementation of this DiT turns any rounding difference into a few percent of
output change; streaming attention adds most of tt's excess, a residual stream in fp32
barely changes it (12% → 11.6%). Classifier-free guidance scales the noise with the
guidance: at one mid step of a 1312x800 sample, |cond - uncond| is 8-10% of |cond|, and
the guided velocity moves by 10% at cfg 1.5, 21-23% at cfg 3 and 32-35% at cfg 4.5.
Batched sampling (different accumulation order than batch 1) gives a different
realization of the same noise; every op is batch-exact on its own.

### Text encoder (Qwen3.5, 5 prompts, up to 1615 tokens)

| implementation | time |
|---|---|
| tt, gated delta rule one chunk at a time | 98 s (warm) |
| host CPU, fp32 | 158 s |

The gated delta rule with the chunk-parallel work out of the recurrence (2 matmuls and 2
elementwise ops per chunk in the loop) takes 115 ms against 3239 ms for the
one-chunk-at-a-time form on a GDN layer's batch (5 x 32 heads x 1664 tokens), same
error. Final hidden states on tt (bfp8 weights) against fp32 on the CPU: rel 2.9-3.7%,
cosine 0.9994.

## First call at a new size

MOSS VAE decode, a 1120x928 image after a decode at 1024x1024 in the same process
(kernels already built for the first size):

| op | TT-NN path | Kohaku-TT kernels |
|---|---|---|
| casts bf16 <-> fp32 (168 calls) | 13.2 s | 1.8 s |
| group norm in fp32 (30 calls) | 10.5 s | 0.14 s |
| attention-block linears (4 calls) | 1.5 s | 1.3 s |
| convolution (36 calls) | 1.1 s | 1.1 s |
| decode, first call | 29.8 s | 5.8 s |
| decode, repeat call | 1.61 s | 1.73 s |

With Kohaku-TT's matmul for the attention-block linears, the convolution reader windows
and 32-row pages, and the group statistics kernel, the same 1120x928 decode after the
1024x1024 one takes 2.40 s on its first call and 0.80 s from its third (1.54 s on the
second, after one L1 clash halves the activation budget).

The convolution's layout kernels, nearest upsampling, pixel shuffle, group norm (sums,
group statistics and affine in fp32) and casts take every size as a runtime argument; the
remaining first-call cost is TT-NN matmul and attention programs compiled per size.

## Kohaku-TT kernels against TT-NN's ops

Warm time; results against float64 on the CPU next to the CPU's at the same dtype.

| op | shape | Kohaku-TT | TT-NN | result |
|---|---|---|---|---|
| argmax (last dim) | 1 x 4096 x 3072 | 0.6-0.8 ms | 35-37 ms | indices equal to PyTorch's (ties: first) |
| argmax (last dim) | 3 x 517 x 1000 | 0.4-0.7 ms | 7-14 ms | equal |
| cumsum (last dim, bf16) | 1 x 4096 x 3072 | 3.7 ms | 57 ms | statistics equal to the CPU's |
| index_select (dim 1, fp32) | 3 x 1043 x 1000, 300 indices | 0.16 ms | 4.2 ms (gather) | exact |
| index_select (dim 1, bf16) | 3 x 1043 x 1000, 300 indices | 0.23 ms | 2.1 ms (gather) | exact |
| embedding (bf16) | 1000 x 77 table, 15 indices | 0.16 ms | 0.36 ms (TT-NN embedding) | exact |
| flip(-1) (fp32), index_select(-1) | 3 x 1043 x 1000 | 0.22 ms, 0.43 ms (transposes around the row gather) | 3.98 ms, 1.31 ms (gather) | exact |
| pixel shuffle, r 2 | 2 x 512 x 50 x 82 | 2.3 ms | 9.8 ms (matmul path) | exact |
| nearest upsample x2 (fp32) | 1 x 192 x 136 x 120 | 3.8 ms | 11.4 ms (matmul path) | exact |
| group norm (fp32) | 1 x 192 x 136 x 120 | 1.03 ms | 2.3 ms (composite) | statistics of the CPU's fp32 result (output, mean, rstd) |
| group norm (bf16) | 2 x 128 x 100 x 164 | 1.39 ms | | statistics equal to the CPU's bf16 result |
| conv2d 1x1 (bf16) | 2 x 256 x 50 x 82 -> 256 | 0.77 ms | 0.87 ms (permute / pad layout) | equal statistics |
| conv2d 3x3 (bf16) | 2 x 128 x 100 x 164 -> 128 | 1.78 ms | 2.57 ms (permute / pad layout) | equal statistics |
| cast fp32 -> bf16 | 1 x 192 x 136 x 120 | 0.3 ms | | 1-ulp differences on ties |
| embedding (fp32) | 32000 x 1024 table, 4 x 517 indices | 0.40 ms | 42.7 ms (gather) | exact |
| embedding (fp32) | 151936 x 2560 table, 1615 indices | 0.74 ms | | exact |
| prod (last dim, fp32) | 3 x 517 x 64 | 2.17 ms (first call 0.11 s) | 0.37 ms (first call 1.8 s) | Kohaku-TT: rel error median 1.4e-7 (CPU fp32: 1.4e-7); TT-NN: 6.4e-2 |
| linalg.det (fp32) | 4 x 64 x 64 | 237 ms (LU, 64 elimination steps) | 522 ms (Kohaku-TT's LU with TT-NN's prod) | rel error median 1.2e-6 (CPU fp32: 5.9e-7); with TT-NN's prod: 6.3e-2 |

Linear algebra runs as fp32 elementwise steps and fp32 sums (no matmul):
partial-pivot LU, Gauss-Jordan inverse, substitution, Cholesky and Householder QR take
one step per column; eigh and the SVD take parallel Jacobi rounds of n / 2 rotations
(row / column gathers), then one Newton-Schulz step on V and the eigenvalues /
singular values from A. Warm time for 4 matrices, error against float64 next to the
CPU's fp32 result (rel error median):

| op | shape | Kohaku-TT | rel error median (CPU fp32) |
|---|---|---|---|
| linalg.cholesky | 4 x 64 x 64 | 110 ms | 1.3e-7 (1.3e-7) |
| linalg.qr (R) | 4 x 80 x 64 | 190 ms | 2.6e-7 (2.3e-7) |
| linalg.det | 4 x 64 x 64 | 237 ms | 1.2e-6 (5.9e-7) |
| linalg.inv | 4 x 64 x 64 | 238 ms | 3.8e-7 (2.5e-7) |
| linalg.solve (16 columns) | 4 x 64 x 64 | 398 ms | 3.6e-7 (2.0e-7) |
| linalg.svdvals | 4 x 80 x 64 | 1.79 s | 4.7e-8 (1.0e-7) |
| linalg.eigvalsh | 4 x 64 x 64 | 2.08 s | 7.5e-8 (3.1e-7) |

Kohaku-TT's matmul (`aten/linalg/ktt_matmul`: each core's weight block resident in L1,
M a runtime argument, the same cores for every M) takes bf16 matmuls with K a multiple
of 32 and K, N up to 1024. The first call at a new M takes 0.3-0.7 ms against
30-1570 ms for TT-NN's matmul (programs compiled per M). Warm, it runs where it is the
faster of the two: from 16384 rows with an output of 12.5M to 30M elements TT-NN's
matmul takes 3-7x longer (16384 x 1024 @ 1024 x 1024: 0.97 ms against 5.65 ms); elsewhere
TT-NN's is 1.2-3.4x faster (16384 x 384 @ 384 x 384: 0.50 ms against 0.32 ms). Outputs
have the statistics of TT-NN's.

With K = N = 1024 (bf16 `[M, 1024] @ [1024, 1024]`), first call at a new M and warm
time:

| M | Kohaku-TT's matmul: first / warm | TT-NN's matmul: first / warm |
|---|---|---|
| 512 | 0.4 ms / 0.34 ms | 106 ms / 0.11 ms |
| 2086 | 0.5 ms / 0.37 ms | 33 ms / 0.12 ms |
| 3230 | 0.6 ms / 0.42 ms | 42-691 ms / 0.14 ms |
| 4096 | 0.6 ms / 0.46 ms | 749 ms / 0.20 ms |
| 6000 | 0.6 ms / 0.54 ms | 744 ms / 0.25 ms |

`KTT_OWN_MATMUL_MIN_ROWS` runs Kohaku-TT's matmul from that row count on (first call at
a new M about 0.5 ms).

MOSS VAE decode convolutions at 1120x928 (36 calls, warm): 803 ms with one input row read
per tap and row page, 526 ms with one read of each kernel row's window shared by its
taps, 469 ms with 32 pixel rows per page of the convolution's input rows (outputs equal
in all three). The 96-channel 3x3 convolutions run at 6.8 TFLOP/s, the 384-channel ones
at 17-22 TFLOP/s.

argmax over a few very long rows (fewer than 8 units of 32 x 32 rows scanning more
than 64 tiles) stays on TT-NN's argmax.

## Attention

24 heads x 128 unless noted, bf16 inputs; error against float64 on the CPU (the CPU's
bf16 result: cosine 0.999997-0.999998).

| case | Kohaku-TT warm | TT-NN warm | Kohaku-TT cosine | TT-NN cosine |
|---|---|---|---|---|
| 1043 tokens | 0.97 ms | 0.31 ms | 0.999971 | 0.999714 |
| 4110 tokens | 8.35 ms | 1.60 ms | 0.999953 | 0.999697 |
| 4100 tokens, 1 head x 512 | 1.99 ms | 0.92 ms | 0.999924 | 0.999553 |
| causal, 4379 tokens | 5.48 ms | 9.66 ms | 0.999976 | 0.999950 |
| 24 query / 8 key heads, batch 2, 1043 tokens | 1.48 ms | 9.27 ms (math path) | 0.999971 | |

A first call at a new length takes 4-12 ms in Kohaku-TT's kernel and 2.3-2.5 s in
TT-NN's (kernel compilation). Batch rows match the row computed alone exactly at any
batch and head count.

## Host time per own program

Each of Kohaku-TT's own programs (and each user program, e.g. a fused pointwise chain)
is a TT-NN program descriptor: circular buffers, kernels, runtime arguments. The
descriptor is built once per program, sizes and memory placement, and reused with only
the buffer addresses rewritten (`KTT_DESC_CACHE=0` builds own programs' descriptors on
every call; user programs stay cached). Values equal
on every core are common runtime arguments; per-core arguments carry only the core's
share (its first tile or unit, or its block). Outputs are bit-identical with and without
either.

Host µs per program on the stream worker, warm, best of 3-5 runs (`kohakutt.counters()`),
built on every call and per-core arguments against now, next to device µs per call from a
replayed trace (`kohakutt/host_overhead.py` in the experiments repository):

| program | size | host µs / program, built per call | now | device µs / call |
|---|---|---|---|---|
| unary (exp) | 512 tiles | 95.5 | 40.2 | 12.0 |
| unary (exp) | 4096 tiles | 97.8 | 44.3 | 44.9 |
| cast bf16 -> fp32 | 4096 tiles | 95.2 | 40.8 | 73.3 |
| binary, row broadcast | 4096 tiles | 51.2 | 26.1 | 54.1 |
| binary, head broadcast | 4096 tiles | 97.7 | 44.3 | 51.5 |
| causal attention | 8 heads x 1024 x 64 | 94.9 | 36.1 | 292 |
| matmul | 4096 x 1024 @ 1024 x 1024 | 111.1 | 45.4 | 264 |
| conv 3x3 (3 programs) | 64 x 64 x 64 | 69.7 | 29.7 | 218 |
| group norm (3 programs) | 128 x 64 x 64 | 135.4 | 50.5 | 239 |
| nearest upsample x2 | 64 x 64 x 64 | 68.4 | 25.8 | 294 |
| pixel shuffle r 2 | 256 x 64 x 64 | 80.6 | 27.2 | 382 |
| embedding (fp32) | 512 indices | 65.7 | 27.0 | 71.5 |
| cumsum (bf16) | 1024 x 1024 | 50.5 | 26.5 | 646 |

Building the descriptor took 15-75 µs per program (per-core argument vectors, kernel and
buffer descriptors); a reuse takes 1-3 µs. What remains is TT-NN's `generic_op`
(program-cache lookup, runtime-argument copy, command assembly and enqueue: 13-41 µs per
program, lower with fewer per-core arguments) and the output allocation (5-20 µs). Small
elementwise programs, a fused chain at 512 tiles and embedding with few indices stay
host-bound; the other rows are device-bound.

## Optimizations

- Unmasked non-causal attention with equal heads runs TT-NN's streaming kernel (which
  masks a ragged last K tile), q / k chunks up to 256, at most 48 (batch, head) pairs
  per call.
- Unmasked causal and grouped-head attention run Kohaku-TT's flash attention
  (`aten/attention/ktt_attention`): every size a runtime argument, K tiles past the
  causal diagonal never read, the diagonal tile masked, fp32 DST and intermediates.
- Weights stored `[K, N]` (`block_float_linears(..., kn=True)`) run TT-NN's minimal
  matmul.
- `torch.ops.ktt.rms_norm` (weight and bias) and `torch.ops.ktt.rotary`: one program each.
- Fused pointwise chains (`kohakutt.fusion`) for SiTU and gated residuals.
- Group norm forward in three programs: per-plane sums, group statistics with the
  per-plane scale and shift (one core), the affine.
- Kohaku-TT's convolution (`aten/nn/ktt_conv`): an implicit GEMM, one binary for every
  size, weights prepared once per weight (`KTT_OWN_CONV=0`: TT-NN's conv2d, with
  prepared weights cached per size).
- Adaptive L1 budget; spill and retry on L1 clashes and L1 OOM; program cache cleared on
  L1_SMALL exhaustion.
- Own programs' descriptors built once per program, sizes and memory placement
  (`KTT_DESC_CACHE`), uniform runtime arguments as common runtime arguments.
- Triangular inverse on whole tiles.
- Gated delta rule with the chunk-parallel work out of the recurrence (MOSS text
  encoder).

## Open work

- Convolution: stride 2 and 1x1 in the same reader; channels-last activations through the
  VAE.
- First-call cost of tiled reshape, padded slice and permute.
- Gated delta rule: an own recurrence kernel (state resident in L1 per head and value
  block); variable-length text batches.
- Kohaku-TT's attention for plain non-causal calls: 8.0 ms at 4110 tokens against 1.6
  ms for TT-NN's. Chunks of 8 q x 4 k tiles (half the K / V reads of 4 x 8) take 9.0 ms
  and 16 x 2 take 11.4 ms: the time goes to per-chunk compute steps, not K / V reads.
- Matmuls with K or N above 1024 and plain non-causal attention: programs compiled per
  size in TT-NN (`minimal_matmul` carries M as a compile-time argument); Kohaku-TT's
  matmul is 2-5x slower warm at K = 3072.
- Matmul input layouts.
- Batched DiT with L1-resident activations fails in a biased linear (TT_FATAL "Broadcast
  error"); `KTT_L1_BUDGET=0` runs.
