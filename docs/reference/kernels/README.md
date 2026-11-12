---
title: Kernels
summary: Reference pages for Kohaku-TT's own Metalium kernels - what each computes, how to enable it, coverage, design, performance against TT-NN and limits.
tags:
  - reference
  - kernels
---

# Kernels

Kohaku-TT ships its own Metalium programs under
`src/kohakutt_native/csrc/aten/<family>/ktt_<name>/`. Each page below follows the
design-sheet sections of [kernel design](../../concepts/kernel-design.md): what the op
computes, how it is routed and enabled, what it covers, how it works (grid and roles,
NoC plan per stream, L1, compute), its bound, its current performance against TT-NN
with the conditions, and its limits.

| Kernel | Op | Enabled | Page |
|---|---|---|---|
| `ktt_gemm` | `mm`, `linear` with a 2-D bf16 / bfp8 / bfp4 weight | default by size (`KTT_GEMM=auto`); `KTT_GEMM=1` every covered call | [gemm.md](gemm.md) |
| `ktt_flash` | `scaled_dot_product_attention`, bf16: dense, causal, additive / bool masks, MHA / GQA / MQA, any D and Dv whose buffers fit L1 (about 1216 for one head; not whole tiles: zero-padded in L1); training forward and backward (no mask, Sq == Sk; grouped heads: S % 32 == 0) | default (`KTT_FLASH=0` disables) | [flash.md](flash.md) |
| `ktt_flash`, packed variable length | `kohakutt.sdpa_varlen` and torch's `varlen_attn` on tt: sequences packed back to back with offsets, dense and causal, MHA / GQA, forward and backward | called directly | [flash-varlen.md](flash-varlen.md) |
| `ktt_conv2` | grouped, depthwise and strided 1-D / 2-D convolution, forward and gradients | default (`KTT_GROUPED_CONV=ttnn` disables) | [conv-grouped.md](conv-grouped.md) |
| `ktt_conv_nhwc` | single-group 2-D convolution, stride 1, odd K, "same" padding, on Nhwc storage (forward, input gradient, weight gradient for K >= 3) | Nhwc input, or `KTT_NHWC=1` | [conv-nhwc.md](conv-nhwc.md) |
| `ktt_norm` | layer norm and RMS norm over the last dim, bf16 / fp32, forward with fp32 statistics and backward | default | [layernorm.md](layernorm.md) |
| `ktt_group_norm` (row form) | group norm on channels-last storage and on `[N, C, L]`, bf16 / fp32, forward and (channels-last) backward | Nhwc input under `KTT_NHWC=1`; `[N, C, L]` by default | [groupnorm.md](groupnorm.md) |
| `ktt_softmax` | `softmax` over the last dim, bf16 / fp32, any rank and width (the forward of `_softmax`; its backward stays the composite) | default on a last dim not a multiple of 32 (`KTT_OWN_SOFTMAX=auto`); 1 every call, 0 disables | [softmax.md](softmax.md) |
| `ktt_gather` | `gather` / `take_along_dim` along the last dim, bf16 / fp32 / int32 | default (`KTT_OWN_GATHER=0` disables) | [gather.md](gather.md) |
| `ktt_adam` | one Adam / AdamW step of one parameter (parameter and both moments), bf16 / fp32 | `torch.optim.Adam` / `AdamW` with `fused=True` | [adam.md](adam.md) |

`KTT_CUSTOM_KERNELS=0` turns every Kohaku-TT kernel off and routes those ops to TT-NN.
Other `ktt_*` programs in the tree (elementwise, reductions, embedding, group norm on
channel planes, upsample, the RNG, `ktt_attention`, `ktt_matmul`, `ktt_conv`) are covered by the
[aten support](../aten_support.md) table and their source comments.
