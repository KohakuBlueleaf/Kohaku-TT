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
- **A trace holds every intermediate** of the captured graph, so very large forwards do
  not fit as one trace; run them eagerly.
- **Compiled graphs with Python-int inputs** replay only for the ints of the captured
  call; other calls run eagerly. A decode step whose ints change every token (Gemma's
  sliding-window positions) therefore gains no speed from `ktt_tt`.
- **fp32 matmuls** are accurate to about 1e-3 relative (bf16-level operand precision),
  not to CPU fp32.
- **Integer dtypes** are stored as int32 on the card: int64 keeps PyTorch's dtype but
  only the int32 range (an upload beyond it raises; a result beyond it is wrong).
  uint8, int8 and int16 keep their dtype and are wrapped into their range after every
  op, as PyTorch's integer arithmetic wraps. Sums, products and cumulative sums of bool
  and integer tensors are int64, as in PyTorch. bool is stored as bf16 0 / 1.
- **1-D tensors** are one padded tile row, which costs memory for large vectors.
- **bf16 NaN** may come back as inf after device ops (a hardware property; fp32 is
  unaffected).
- **Data movement** (permutes, gathers of arbitrary strided views) runs as TT-NN op
  chains, not yet near memory bandwidth. A view that no slice, permute or reshape
  expresses (e.g. a diagonal `as_strided`) is gathered through a per-element index
  built on the host: correct, but slow and memory-heavy at image sizes. `torch.gather` /
  `take_along_dim` have their own kernel along the last dim
  ([gather](kernels/gather.md)), other dims too when the index has x's sizes elsewhere;
  it clamps out-of-range indices instead of raising, and any other gather above 262,144
  input elements raises unless
  `KTT_TTNN_GATHER_LARGE=1`. `F.pixel_shuffle` /
  `F.pixel_unshuffle` have their own kernels; prefer them over the reshape + permute
  spelling. Tensors whose last two dims are small (e.g. `[..., 2, 128]` from
  `repeat_interleave` over heads before the sequence dim) pad each trailing matrix to
  32 x 32 tiles: up to 16x the memory (a 5 x 1615-token GDN head repeat asked for 8.4
  GB); repeat or expand after moving the sequence dim next to the last.
- **Attention** runs fused for 4-D bf16 q / k / v (fp32 without gradients: cast to bf16
  at entry) with one head dim (flash zero-pads one that is not whole tiles); K / V heads that divide the Q heads (MHA, GQA, MQA); no dropout;
  causal without a mask (Sq != Sk: PyTorch's top-left form, one flash call where it
  fits, else Sq == Sk pieces);
  and a mask that broadcasts to `[B, H, Sq, Sk]`. With gradients, grouped heads also
  need no mask, Sq == Sk, S a multiple of 32 and the default scale. Every other call
  (fp32 with gradients, dropout, head dims no kernel takes) runs PyTorch's
  math composite on tt ops. A fused call runs `ktt_flash` where it covers the shape (D
  whose buffers fit L1: up to about 1216 for one head, at most 46 jobs per core; see
  [flash](kernels/flash.md#coverage)), otherwise Kohaku-TT's earlier attention program or
  TT-NN's SDPA; there an unmasked non-causal call with equal heads runs TT-NN's streaming
  kernel with bf16 (not fp32) dest registers. The backward of a masked call recomputes
  from q, k, v and the output. `ktt_flash` and the packed kernel apply a bool or `-inf`
  mask as -5e5 added to the score, so a masked key is removed while its score is less
  than about 5e5 above the kept keys'.
- **Attention output strides**: when q, k and v are head views of `[B, T, C]`
  projections (`view(B, T, H, D).transpose(1, 2)`, or a fused qkv) and `ktt_flash` reads
  them in place, the output is a `[B, H, T, D]` view of a `[B, T, H D]` tensor, with the
  strides PyTorch's meta (and CUDA's flash) give it. It is not contiguous:
  `out.view(B * H, T, D)` raises where it worked on the other routes; use `.reshape` or
  `.contiguous()` (the merge back, `out.transpose(1, 2).reshape(B, T, H * D)`, is free).
  `KTT_FLASH_PITCHED_OUT=0` restores a dense output.
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
