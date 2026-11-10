---
title: Measured results
summary: Kohaku-TT against tt-xla and against TT-NN's kernels on one Blackhole P150A - whole models, MOSS-Image, user kernels and own kernels, with their conditions.
tags:
  - reference
  - performance
---

# Measured results

One Blackhole P150A, bf16 unless named. How these are measured, and more per-op numbers:
[performance](../concepts/performance.md); per-kernel numbers are on each
[kernel page](kernels/README.md); two-card numbers are in [mesh](../concepts/mesh.md#measured).

## Against tt-xla

Kohaku-TT eager and compiled against the same PyTorch workloads on tt-xla (torch-xla with
the TT PJRT plugin), one stack per process, median ms per step after warm-up
(`src/kohakutt/bench/perf/xla_compare.py`, `workloads.py`):

| Workload | Kohaku-TT eager | Kohaku-TT compiled | tt-xla |
|---|---|---|---|
| ViT-B/16 inference, batch 1 | 77 img/s | 114 img/s | 8 img/s |
| ViT-B/16 inference, batch 8-64 | 181-189 img/s | 181-188 img/s | up to 97 img/s |
| ViT-B/16 training step, batch 8 | 244 ms | | 100.9 s |
| Qwen2.5-0.5B prefill, 128 tokens | 129 ms | 45 ms | 207 ms |
| Qwen2.5-0.5B prefill, 512 tokens | 142 ms | 106 ms | 199 ms |

## MOSS-Image

`examples/06_moss_image`: bfp8 DiT weights, bf16 activations with fp32 math inside each
fused op, HiFi2, eager, one chip. One DiT forward at 1360 x 768 with 1615 text tokens
(5695 tokens) takes 0.652 s. A CFG step is the cond and the uncond forward:

| Image | Text tokens (cond / uncond) | CFG step |
|---|---|---|
| 1024 x 1024 | 300 / 8 | 0.935 s |
| 1360 x 768 | 1615 / 8 | 1.104 s |
| 1024 x 1024 | 1615 / 8 | 1.129 s |

The VAE decodes a 1312 x 800 image in 1.1 s; the first decode at a new size takes about
a minute (host setup in TT-NN ops, see [TT-NN known issues](ttnn-known-issues.md)).
Details and per-op numbers: [performance](../concepts/performance.md#moss-image-measurements).

## Kernels

Device-synchronized time per call over 20 async calls:

| Kernel | Kohaku-TT | Reference |
|---|---|---|
| DiT projections at M = 4115, explicit 2-D multicast configs | 140-161 TFLOPS | |
| SwiGLU as a user program, 1024 x 4096 bf16 | 0.132 ms | `F.silu(a) * b`: 0.283 ms |
| SwiGLU as a user program, 4096 x 4096 bf16 | 0.238 ms | `F.silu(a) * b`: 0.429 ms |
| RoPE as a user program, 9 x 16 x 576 x 64 bf16 | 0.113 ms | cast, complex multiply, cast: 0.325 ms |

## Own kernels against TT-NN

Per call, same precision settings for both, HiFi2:

| Op | Shape | Kohaku-TT | TT-NN |
|---|---|---|---|
| SDPA (`ktt_flash`) | 1x32x4096x64 | 2694 µs | 7756 µs |
| SDPA, causal | 1x8x1024x64 | 65 µs | 465 µs |
| SDPA, additive mask | 1x16x2048x64 | 537 µs | 1677 µs |
| conv 3x3 (`ktt_conv_nhwc`) | VAE 1x512x64x64 | 150.9 µs | 215.8 µs |
| conv 3x3 | SD1.5 2x320x64x64 | 164.3 µs | 216.5 µs |
| conv 3x3 | ResNet 1x64x56x56 | 41.2 µs | 48.3 µs |

## Real models against the same models on TT-NN's kernels

`KTT_CUSTOM_KERNELS=0` for the TT-NN column; bf16, eager, warm; error is the median
relative error against a CPU float64 run (CPU bf16 in brackets):

| Model | Kohaku-TT | TT-NN kernels | Error, Kohaku-TT / TT-NN |
|---|---|---|---|
| DINOv3 ConvNeXt-tiny, forward 224 px | 14.1 ms | 365.8 ms | 4.1e-2 / 8.3e-2 (3.4e-2) |
| DINOv3 ConvNeXt-tiny, training step | 199.8 ms | 1427.9 ms | gradients 0.067 / 0.127 (0.061) |
| DINOv3 ViT-B/16, forward 224 px | 35.6 ms | 45.3 ms | 2.7e-2 / 3.5e-2 (2.0e-2) |
| DINOv3 ViT-B/16, training step | 103.2 ms | 112.2 ms | loss 0.007 / 0.019 (0.011) |

TT-NN spends roughly 35-65 µs of host time per program, so models made of many small
ops are bound by program count rather than by the card; Kohaku-TT's fused ops and
pointwise chains (`torch.ops.ktt.rms_norm`, `ktt.rotary`, `kohakutt.fusion`) collapse
such chains into one program each.
