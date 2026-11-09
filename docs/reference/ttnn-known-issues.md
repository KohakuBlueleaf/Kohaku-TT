---
title: TT-NN known issues
summary: TT-NN and TT-Metal defects that affect PyTorch programs on Tenstorrent cards, and how Kohaku-TT avoids each one.
tags:
  - reference
  - ttnn
  - correctness
---

# TT-NN known issues

Kohaku-TT runs most ops through TT-NN. The defects below change results, cost time or
fail a call in TT-NN itself; each row says how Kohaku-TT avoids it, so a PyTorch program
on Kohaku-TT does not see it. Rows marked **open** still reach results in the cases
named. Measured on a Blackhole p150a.

## SDPA mixes outputs across (batch, head) pairs

With enough (batch, head) pairs in one `ttnn.transformer.scaled_dot_product_attention`
call, some pairs' outputs differ from the same pair computed alone. Identical inputs at
every pair, bf16, head dim 128; worst pair against the pair alone:

| Layout | 1600 tokens | 1615 tokens | 4110 tokens |
|---|---|---|---|
| B 1, H 24 / 32 / 48 / 49 / 56 | exact | exact | exact |
| B 2, H 12 / 24; B 3, H 8 | exact | exact | |
| B 1, H 63 | exact | exact | abs max 3.6e-2, cos 0.997 |
| B 1, H 64 | abs max 9.2e-2, cos 0.970 | abs max 1.2e-1, cos 0.974 | abs max 4.7e-2, cos 0.997 |
| B 3, H 24 | abs max 2.2e-1, cos 0.989 | abs max 2.0e-1, cos 0.990 | abs max 1.3e-1, cos 0.997 |
| B 1, H 96 | exact | exact | abs max 9.8e-2, cos 0.977 |
| B 1, H 128 | abs max 2.1e-1, cos 0.934 | abs max 2.4e-1, cos 0.940 | abs max 1.0e-1, cos 0.969 |

No q / k chunk size avoids it; the error sits in a subset of positions (median absolute
error 0). B 1 with 24-48 heads is exact at every length tried (279 to 8192 tokens).

How Kohaku-TT avoids it: bf16 attention that [ktt_flash](kernels/flash.md) supports
(dense, causal, additive / bool masks, grouped heads) runs there by default; it shares no
state across (batch, head) pairs. With `KTT_FLASH=0`, and for calls ktt_flash does not take,
causal and grouped-head attention run Kohaku-TT's own attention kernel, and plain calls go
to TT-NN's SDPA in pieces of at most 48 pairs (whole batch rows while a row has at most 48
heads, else head ranges that keep whole K / V head groups), with the outputs concatenated.
`KTT_SDPA_MAX_UNITS` changes the pair limit. Reproduce with
[`experiments/attention_isolation.py`](../../experiments/attention_isolation.py).

## Wrong results

| TT-NN behaviour | Effect | How Kohaku-TT avoids it |
|---|---|---|
| SDPA with fp32 dest reads a ragged last K tile's padding as keys | wrong outputs at lengths not a multiple of 32 | unmasked attention uses the streaming (bf16 dest) path, which masks the tile |
| `ttnn.matmul` / `ttnn.linear` / `minimal_matmul` sum over the inner dimension's tile padding, which other ops (reshape) leave non-zero | K not a multiple of 32: nondeterministic wrong results | both operands' tile padding is zeroed when K % 32 != 0 |
| bf16 conv2d with one input and one output channel over one input row and a kernel wider than 1 | cosine 0.03-0.16 against the CPU | that case runs Kohaku-TT's convolution |
| conv2d weights prepared under the auto config depend on the input size | wrong values when reused at another size | Kohaku-TT's convolution; the TT-NN path caches prepared weights per size |
| `ttnn.embedding` fed a TILE int32 index computed on device | 450 of 1000 rows wrong | TILE indices go to Kohaku-TT's row gather |
| `ttnn.mesh_partition` of a TILE tensor at shard boundaries off the 32 grid | wrong values or a fatal error | such boundaries are partitioned on the host |
| `noc_async_read_barrier_with_trid` does not wait for plain reads issued after `noc_async_read_set_trid` (Blackhole) | data read before it lands | Kohaku-TT's kernels use global read barriers |
| `generic_op`'s cache-hit fast path rewrites only buffer addresses, while the program hash covers the argument count, not the values | a hit at another size keeps the first call's sizes | Kohaku-TT's programs pass plain values and cache their own descriptors |

## Precision

| TT-NN behaviour | Effect | How Kohaku-TT avoids it |
|---|---|---|
| bf16 compute does not carry NaN (`x * 1`, `x + y`, `neg`, `exp`, `where` give +-inf; `relu` gives 0); casts between bf16 and fp32 lose NaN | NaN becomes inf or 0 | NaN-sensitive reductions (`nanmedian`, argmax / argmin) test NaN on the input's own dtype. **Open** for ops that fill NaN and compute on it in bf16 (masked median / argmin through `where`) |
| fp32 matmul operands keep fewer mantissa bits than fp32, at every fidelity | relative error about 1e-4 instead of 1e-7 | selections and nearest resizes split fp32 into three bf16 parts; linear algebra runs elementwise in fp32. **Open** for linear-mode resizes, fp32 / grouped convolutions on TT-NN and the FFT |
| `ttnn.bitcast` int32 -> fp32 drops low mantissa bits | wrong frexp / nextafter for fp32 | those ops run on tt for bf16 only |
| `ttnn.prod` of fp32 rounds to bf16 | relative error about 1e-3 | products are computed as fp32 elementwise halving (`KTT_TREE_PROD`) |
| `ttnn.all_reduce` of fp32 sums at reduced precision (1x2 mesh) | relative error up to 1e-3 | fp32 partial sums are gathered and added on each chip in fp32 |
| `moreh_layer_norm` keeps mean and rstd in the input dtype | bf16 statistics | **open** on the TT-NN layer-norm path |

## Speed and first-call cost

| TT-NN behaviour | Effect | How Kohaku-TT avoids it |
|---|---|---|
| Tiled reshape, padded slice, permute, tilize and untilize build host tables for each new shape | seconds of host time on the first call at a size (a MOSS-Image VAE decode: 61 s first call, 1.1 s repeated) | kept off hot paths; Kohaku-TT's kernels take shapes as runtime arguments |
| Many kernels take sizes as compile-time arguments | 1-4 s of kernel compilation for each new shape | Kohaku-TT's attention, convolution and elementwise kernels take sizes at run time. **Open** for permute, pad and TT-NN matmul |
| `ttnn.linear` with `transpose_b` and no program config | 5-6 TFLOPS on 2048-row shapes | explicit 2-D multicast configs: 170-218 TFLOPS |
| `ttnn.linear` is 1.4-1.6x slower than `minimal_matmul` | slower projections | `[K, N]`-stored weights run `minimal_matmul` |
| Causal SDPA is slower than non-causal | 9.7 ms against 1.6 ms (24 heads, about 4k tokens) | causal attention runs Kohaku-TT's kernel (5.4 ms) |
| `ttnn.embedding` accepts bf16 tables only | fp32 rows through a gather: 42.7 ms | Kohaku-TT's row gather: 0.40 ms |

## Memory and API limits

| TT-NN behaviour | Effect | How Kohaku-TT avoids it |
|---|---|---|
| Ops place their output in their input's memory, collectives included | an all-gather output (mesh size x input) placed in full L1: out of memory | every op and collective gets an output memory config sized before launch |
| conv2d sliding-window metadata accumulates in L1_SMALL per shape | out of memory after a few resolutions | L1_SMALL is 96 KiB. **Open** |
| conv2d DRAM slicing finds no height-sharded config for some shapes | fatal error | auto config for those shapes |
| A collective's first run writes setup from the host, which trace capture refuses | a capture stalls | the capture's prepare pass runs every collective it records |
| `ttnn.linear` without a program config misbroadcasts a bias over rank > 2 input | fatal "Broadcast error" | bias added separately |
| `nlp_concat_heads` circular buffers grow with the head count | 53 MB per core at 1615 heads | used up to 128 heads |
| Permute of a 5-D tensor with small dims pads them to tiles | 13-17 GB allocations | no 5-D permutes in convolution |
