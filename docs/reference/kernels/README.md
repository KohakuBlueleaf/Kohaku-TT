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
| `ktt_gemm` | `mm`, `linear` with a 2-D bf16 / bfp8 / bfp4 weight | `KTT_GEMM=1` | [gemm.md](gemm.md) |
| `ktt_flash` | `scaled_dot_product_attention`, bf16: dense, causal, additive / bool masks, MHA / GQA / MQA, D and Dv multiples of 32 up to 256 | default (`KTT_FLASH=0` disables) | [flash.md](flash.md) |
| `ktt_conv2` | grouped, depthwise and strided 1-D / 2-D convolution, forward and gradients | default (`KTT_GROUPED_CONV=ttnn` disables) | [conv-grouped.md](conv-grouped.md) |
| `ktt_conv_nhwc` | single-group 2-D convolution, stride 1, odd K, "same" padding, on Nhwc storage (forward) | Nhwc input, or `KTT_NHWC=1` | [conv-nhwc.md](conv-nhwc.md) |
| `ktt_norm` | layer norm and RMS norm over the last dim, bf16 / fp32, forward with fp32 statistics and backward | default | [layernorm.md](layernorm.md) |

`KTT_CUSTOM_KERNELS=0` turns every Kohaku-TT kernel off and routes those ops to TT-NN.
Other `ktt_*` programs in the tree (elementwise, reductions, embedding, group norm,
upsample, the RNG, `ktt_attention`, `ktt_matmul`, `ktt_conv`) are covered by the
[aten support](../aten_support.md) table and their source comments.
