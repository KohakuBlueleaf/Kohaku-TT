---
title: Limitations
summary: What Kohaku-TT does not do, or does slowly, today - the mesh, traces, fp32 matmuls, small tensors, data movement, attention, L1-resident activations and first calls.
tags:
  - reference
  - limitations
---

# Limitations

- **One device or mesh per process**, and the mesh is built and tested as 1 x 2. An op
  without a placed form gathers its input to both chips (a warning names the op), and
  a bf16 partial sum is rounded on each chip before the chips' sums are added, so ops on
  it carry a larger rounding error than on one card ([mesh](../concepts/mesh.md)).
  Known and not yet explained: attention on three pending-sum inputs with 33 rows
  (`[2, 3, 33, 32]`) returned inf once in about 17 runs of the mesh tests and passed 14
  repeats in a row afterwards.
- **A trace holds every intermediate** of the captured graph, so very large forwards do
  not fit as one trace; run them eagerly.
- **fp32 matmuls** are accurate to about 1e-3 relative (bf16-level operand precision),
  not to CPU fp32.
- **1-D tensors** are one padded tile row, which costs memory for large vectors.
- **bf16 NaN** may come back as inf after device ops (a hardware property; fp32 is
  unaffected).
- **Data movement** (permutes, gathers of arbitrary strided views) runs as TT-NN op
  chains, not yet near memory bandwidth. A view that no TT-NN op expresses (e.g. a 6-D
  permute that moves size-2 dims last) is gathered through a per-element index built on
  the host: correct, but slow and memory-heavy at image sizes. `torch.gather` /
  `take_along_dim` have their own kernel along the last dim
  ([gather](kernels/gather.md)); it clamps out-of-range indices instead of raising,
  and any other gather above 262,144 input elements raises unless
  `KTT_TTNN_GATHER_LARGE=1`. `F.pixel_shuffle` /
  `F.pixel_unshuffle` have their own kernels; prefer them over the reshape + permute
  spelling. Tensors whose last two dims are small (e.g. `[..., 2, 128]` from
  `repeat_interleave` over heads before the sequence dim) pad each trailing matrix to
  32 x 32 tiles: up to 16x the memory (a 5 x 1615-token GDN head repeat asked for 8.4
  GB); repeat or expand after moving the sequence dim next to the last.
- **Attention** runs fused for 4-D bf16 q / k / v with one head dim, a multiple of 32;
  K / V heads that divide the Q heads (MHA, GQA, MQA); no dropout; causal only with
  Sq == Sk and no mask; and a mask that broadcasts to `[B, H, Sq, Sk]`. With gradients,
  grouped heads also need no mask, Sq == Sk, S a multiple of 32 and the default scale.
  Every other call (fp32, dropout, other head dims, causal with a mask) runs PyTorch's
  math composite on tt ops. A fused call runs `ktt_flash` where it covers the shape (D
  up to 256, at most 46 jobs per core, buffers that fit L1; see
  [flash](kernels/flash.md#coverage)), otherwise Kohaku-TT's earlier attention program or
  TT-NN's SDPA; there an unmasked non-causal call with equal heads runs TT-NN's streaming
  kernel with bf16 (not fp32) dest registers. The backward of a masked call recomputes
  from q, k, v and the output.
- **L1-resident activations** (opt-in, `KTT_L1_BUDGET`, the fraction of each core's L1
  they may fill, default 0): TT-NN's matmul sizes its blocks from the L1 free at the call,
  so with activations in L1 one shape keeps getting new programs (ConvNeXt training steps
  of 0.1-2.7 s against a steady 83 ms). With a budget set, when a program's circular
  buffers would not fit beside them, every L1 activation moves to DRAM, the program runs
  again, and the budget halves for the rest of the process (models whose kernels take
  most of L1, such as the MOSS DiT's attention and matmuls, settle near all-DRAM). A view
  and its base that share an L1 buffer each get their own DRAM copy then, so an in-place
  write through one is no longer seen through the other. The default keeps every
  activation in DRAM and aliasing exact.
- **First call at a new shape** of TT-NN's tiled reshape, padded slice, permute and conv2d
  costs host setup proportional to the tensor (seconds at image sizes; a MOSS VAE
  decode at a new resolution takes about a minute, then about 1 s). Repeat calls at the
  same shapes are warm ([TT-NN known issues](ttnn-known-issues.md)).
- **Selection matmuls** (interpolation, pixel shuffle, `index_copy`) copy values
  exactly, fp32 included, at three products per fp32 operand.
