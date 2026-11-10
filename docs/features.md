---
title: Features
summary: What Kohaku-TT provides, feature by feature, with the measured results behind each claim.
tags:
  - overview
---

# Features

- **Faster than tt-xla on the same card, eager or compiled.** ViT-B/16 at batch 1: 77
  images/s eager and 114 compiled against tt-xla's 8; a ViT-B/16 training step at batch
  8: 244 ms against 100.9 s; Qwen2.5-0.5B prefill of 128 tokens: 129 ms eager and 45 ms
  compiled against 207 ms ([measured](reference/measured.md#against-tt-xla)).
- **Every Core ATen op runs on tt.** 193 of 193 Core ATen overloads have a tt kernel or
  a composite of tt kernels; across all of aten, 559 kernels and 2026 composites of 3120
  overloads ([support table](reference/aten_support.md)). An op without one raises
  `KohakuTTUnsupportedOperator` naming the overload; the CPU runs it only when asked.
- **A real device, not a tensor wrapper.** Storage-backed tensors with PyTorch's views
  and aliasing (in-place writes through views, version counters), `torch.tt` for
  `synchronize`, memory statistics and RNG state, `torch.save` / `load`, `state_dict`,
  printing, `module.to("tt")` with buffers.
- **Kernels where TT-NN falls short.** Kohaku-TT's own Metalium programs replace TT-NN's
  where they measure faster or where TT-NN is wrong:
  - flash attention (`ktt_flash`, default): single-pass online softmax with fp32 row
    statistics, masks, causal, MHA / GQA / MQA; 2.4-7.2x TT-NN's SDPA at the same precision
    with the accurate exponential, about equal with the approximate one
    ([reference](reference/kernels/flash.md)); sequences of different lengths packed
    back to back with no padding (`kohakutt.sdpa_varlen`, torch's `varlen_attn`), forward
    and backward ([reference](reference/kernels/flash-varlen.md));
  - layer norm / RMS norm (`ktt_norm`, default) with fp32 statistics, forward and backward,
    where TT-NN's bf16-statistics kernels give wrong training results
    ([reference](reference/kernels/layernorm.md));
  - grouped, depthwise and strided conv with gradients (`ktt_conv2`), and a channels-last
    dense conv (`ktt_conv_nhwc`: forward, input gradient and weight gradient, opt-in),
    1.2-2.5x TT-NN's conv2d on VAE / SD / ResNet layers, slower at 1280 channels and on
    512-pixel-wide images
    ([reference](reference/kernels/conv-nhwc.md));
  - blocked unary SFPU chains (up to 1.7x TT-NN's eltwise at mid sizes), broadcast
    binaries that read the broadcast tile once (1.1-1.17x), per-shape 2-D multicast matmul
    configs (140-161 TFLOPS on the MOSS DiT's projections).
- **Faster and more accurate than TT-NN on real models.** ConvNeXt-tiny: forward 26x and a
  training step 7.1x faster than the same model on TT-NN's kernels, with gradients as
  close to float64 as CPU bf16's; DINOv3 ViT-B: forward 1.27x faster with lower error
  ([measured](reference/measured.md#real-models-against-the-same-models-on-tt-nns-kernels)).
- **Your kernels as torch ops.** A reader / compute / writer Metalium program described
  in Python or built in C++ becomes `torch.ops.<ns>.<op>` on tt tensors, in order with
  every other op; a SwiGLU written this way runs in 0.132 ms against 0.283 ms for
  `F.silu(a) * b` at 1024 x 4096, and a RoPE rotation written in SFPU instructions in
  0.113 ms against 0.325 ms for the cast, complex multiply and cast it replaces
  ([custom kernels](reference/custom-kernels.md)).
- **Block-float weights, as stored.** `kohakutt.block_float_linears(model, "bfp8")`
  keeps every linear weight in BFLOAT8_B (or BFLOAT4_B), stored as its (in, out)
  transpose; `F.linear` multiplies them with bf16 activations and fp32 accumulation
  through TT-NN's minimal matmul (1.4-1.6x `ttnn.linear` at the MOSS DiT's
  projections), without a dequantized copy ([block floats](concepts/block-float.md)).
- **Fused transformer ops.** `torch.ops.ktt.rms_norm(x, weight, bias, eps)` (affine
  RMSNorm, e.g. a modulated norm) and `torch.ops.ktt.rotary(x, cos, sin)` (rotate-half
  RoPE) run one program each on tt, fp32 math on bf16 tensors; elsewhere their fp32
  composites.
- **Precision as a scope.** `kohakutt.numerics(fidelity=, approx=)` sets the math of
  every op queued under it, per thread, per module (`numerics_for`) or per function;
  HiFi2 by default, with floors where kernels need precision or get it for free
  ([numerics](concepts/numerics.md)).
- **Numerics you can check.** `KernelAudit` runs every aten call of a block against a
  CPU float64 ruler and a same-precision CPU simulation and reports the error statistics
  of both, per call, per module and per model ([tools](reference/knobs.md#python-tools)).
- **Two cards as one device.** `kohakutt.mesh([0, 1])` shards large tensors over a 1 x 2
  mesh and inserts the collectives an op needs ([mesh](concepts/mesh.md)).
