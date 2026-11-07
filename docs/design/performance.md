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

## Optimizations

- Unmasked attention runs TT-NN's streaming kernel (which masks a ragged last K tile),
  q / k chunks up to 256.
- Weights stored `[K, N]` (`block_float_linears(..., kn=True)`) run TT-NN's minimal
  matmul.
- `torch.ops.ktt.rms_norm` (weight and bias) and `torch.ops.ktt.rotary`: one program each.
- Fused pointwise chains (`kohakutt.fusion`) for SiTU and gated residuals.
- Group norm forward from per-channel sums and one broadcast affine.
- Kohaku-TT's convolution (`aten/nn/ktt_conv`): an implicit GEMM, one binary for every
  size, weights prepared once per weight (`KTT_OWN_CONV=0`: TT-NN's conv2d, with
  prepared weights cached per size).
- Adaptive L1 budget; spill and retry on L1 clashes and L1 OOM; program cache cleared on
  L1_SMALL exhaustion.
- Triangular inverse on whole tiles.
- Gated delta rule with the chunk-parallel work out of the recurrence (MOSS text
  encoder).

## Open work

- Convolution: stride 2 and 1x1 in the same reader; channels-last activations through the
  VAE.
- First-call cost of tiled reshape, padded slice and permute.
- Gated delta rule: an own recurrence kernel (state resident in L1 per head and value
  block); variable-length text batches.
- A per-op cold / warm benchmark over every registered op.
- Attention precision: a kernel as fast as the streaming path with fp32 accumulation
  (`KTT_SDPA_FP32_DEST=1` selects the fp32-dest path; it reads a ragged last key tile's
  padding).
- Matmul input layouts.
- Batched DiT with L1-resident activations fails in a biased linear (TT_FATAL "Broadcast
  error"); `KTT_L1_BUDGET=0` runs.
