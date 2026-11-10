---
title: Layout experiments: TILE vs ROW_MAJOR, conversions and views
summary: "Measured on one P150A (fast dispatch, 1x1) with plain TT-NN calls from Python, bf16 n x n, device-synced mean of 20 calls after warm-up."
tags:
  - concepts
  - layout
---

# Layout experiments: TILE vs ROW_MAJOR, conversions and views

Measured on one P150A (fast dispatch, 1x1) with plain TT-NN calls from Python, bf16
n x n, device-synced mean of 20 calls after warm-up.

| n | tilize | untilize | mm (TILE) | tilize x2 + mm | transpose TILE / RM | slice TILE / RM | reshape TILE / RM |
|---|---|---|---|---|---|---|---|
| 256 | 52.6 | 49.5 | 62.4 | 155.1 | 62.1 / 60.9 | 59.6 / 58.1 | 61.0 / 48.4 |
| 1024 | 48.0 | 40.3 | 72.3 | 158.5 | 62.9 / 76.6 | 68.3 / 58.5 | 60.0 / 48.8 |
| 4096 | 223.1 | 181.8 | 3032.6 | 3456.8 | 179.6 / 1194.4 | 117.1 / 103.0 | 718.0 / 184.3 |

All times in µs. Reshape is (n, n) -> (2n, n/2); slice keeps the first n/2 rows.

## Reading

- **Up to 1024 every op costs one dispatch (40-75 µs).** The layout itself is invisible;
  what costs is the number of ops. Matmul from ROW_MAJOR is 2.2x matmul on TILE only
  because it is three dispatches.
- **At 4096 the layouts pull in opposite directions.** Transpose is 6.6x cheaper on
  TILE (in-tile transpose plus tile reorder); reshape that changes the last dim is 3.9x
  cheaper on ROW_MAJOR (TILE must retile). Row slices cost the same.
- **Conversion is bandwidth-cheap.** Tilize of 32 MB: 223 µs (~290 GB/s read+write).
  Tilizing both matmul inputs adds 14% at n=4096 (425 µs on 3.03 ms).
- **Tilize-on-read matmul bound.** A reader that tilizes while loading would save at
  most the two tilize ops: ~14% at 4096, and two dispatches (~100 µs) at small sizes;
  it pays mainly through the dispatch count.

## Direction for tt tensor views

Neither layout wins outright, so storing one layout and materializing every view eagerly
is the wrong default. Direction:

1. tt tensors carry torch strides as metadata over a TILE base; views are lazy.
2. Consumers fold what they can: transpose into matmul (`transpose_a/b` cost nothing),
   row slices into reader offsets.
3. Anything else materializes once, choosing the layout by the op (reshape via
   ROW_MAJOR, transpose via TILE), and the result is cached on the view.

This matches how CUDA keeps strided tensors and lets kernels (cuBLAS `op(A)`) absorb
transposes, with the tile constraint handled at materialization.

## Channels-last storage (Nhwc)

A device handle has a physical format. The default, Nchw, is one TILE tensor of the
handle's device shape. A rank-4 `[N, C, H, W]` bf16 or fp32 handle can instead be stored
as Nhwc: one TILE tensor `[1, 1, N*H*W, C]`, each tile 32 pixels by 32 channels, with
zeros in the tile padding (rows up to a multiple of 32, channels up to C rounded up to
32). This is TT-NN's conv activation layout. As with torch's `channels_last`, the tensor
keeps its logical NCHW sizes; only the device storage differs. Nhwc handles are never
mesh-placed.

**When conversions happen.** Ops read inputs through `dense()`, which always returns
Nchw:

- **Reads.** An Nhwc storage is converted once (TT-NN permute and reshape programs on
  the stream; the values are copied exactly for bf16 and fp32), and the result is cached
  on the handle for the current write epoch. Later reads reuse it.
- **Views.** Slicing, permute, reshape and `as_strided` resolve against the Nchw copy.
- **Cache invalidation.** In-place device writes (trace inputs, replays) start a new
  epoch, so the next read converts again. A trace capture records its own conversion and
  neither reads nor fills the cache.
- **Writes.** A whole-tensor write stores the new value, so the storage becomes Nchw
  (an Nhwc value of the same sizes is stored as it is). A partial write splices into the
  Nchw copy. While traces are alive, writes are copied back into the existing Nhwc
  buffer instead.
- **Copies.** `.cpu()`, `copy_` and `_to_copy` read the Nchw copy. `clone` and empty-like
  buffers keep the format; `copy_into` converts its source to the destination's format;
  a mesh `distribute` converts to Nchw first.
- **Merged-heads copies.** Some copies move a head-major tensor's head dim next to its
  last dim: `x.movedim(1, -2)` for `x = [B, H, S..., D]`. One example is an attention
  output read as `.view(B, H, h, w, D).permute(0, 2, 3, 1, 4).reshape(B, h, w, H*D)`.
  - A contiguous copy like this is stored as `[B, S..., H*D]`. It costs one concat-heads
    program, plus a row reshape when the S dims cut tile rows.
  - A following projection reads it without a reshape.
  - This applies to bf16 / fp32 with D a multiple of 32 and at most 128 heads, in eager
    mode. Inside a trace capture the copy keeps its own layout.

The cached copy holds a second buffer of the tensor while its source lives.

**Opt-in accessor.** An op that can read Nhwc calls `dense_any_layout(op, t)`. It returns
the storage's handle as stored when `t` covers the whole storage at the storage's own
sizes, and `dense()` otherwise. The op branches on `bridge::format(h)`.
`bridge::to_nhwc(h)` and `bridge::to_nchw(h)` give either format of any handle, cached
the same way. A bridge op producing Nhwc passes `Format::Nhwc` to `detail::timed_op`.

### Propagation (`KTT_NHWC=1`)

With `KTT_NHWC=1` (read on every call; off by default), activations stay Nhwc between
the ops that can read it. These ops read their inputs through `dense_propagate()`:

- **Conv.** A single-group conv with stride 1, dilation 1, an odd K x K kernel and
  padding (K - 1) / 2 runs on Kohaku-TT's `ktt_conv_nhwc`
  ([reference](../reference/kernels/conv-nhwc.md)). It reads Nhwc input as stored and
  produces Nhwc output; a 1x1 conv runs as a matmul on the Nhwc rows. This route also
  takes an Nhwc input when `KTT_NHWC` is unset. Other single-group 2-D convs run on
  TT-NN's `conv2d`, which reads an Nhwc input as stored and returns its
  `[1, 1, N*OH*OW, O]` output as an Nhwc tensor, with no permutes. Grouped and depthwise
  convs still run on Kohaku-TT's kernels, which read Nchw.
- **Elementwise.** Unary, scalar and binary ops, and dtype casts, keep Nhwc.
  - A binary op stays Nhwc when an Nhwc operand has the result's sizes. A same-sized
    Nchw operand is converted (cached per write epoch). A `[1, C, 1, 1]` or one-element
    operand, such as a conv bias, is read as the `[1, 1, 1, C]` row it is already
    stored as, so it needs no conversion.
  - When the larger operand is Nchw, the Nhwc one is converted to Nchw.
  - Ops that map 0 to 0 leave the tile padding at zero. After other ops, the padding is
    filled with zeros again; only shapes with tile padding pay for that.
- **Nearest upsampling.** Integer factors on bf16 tensors with C a multiple of 32 run
  on TT-NN's `upsample` and stay Nhwc.

- **Group norm.** bf16 or fp32, with one image or at least 32 pixels
  per image: it reads Nhwc as stored
  and returns Nhwc ([reference](../reference/kernels/groupnorm.md)). Otherwise its
  kernels reduce per-channel planes and its input is converted once.

Every other op reads Nchw through `dense()`.

### Limits

Measured on one P150A, bf16, eager, back-to-back calls:

| Conversion | 1x512x64x64 | 1x64x56x56 |
|---|---|---|
| to Nhwc | 0.40 ms | 0.08 ms |
| to Nchw | 0.06 ms | 0.06 ms |

- **to Nhwc** is TT-NN's tiled permute `(0, 2, 3, 1)`. It moves both tile dims, so it
  tilizes, transposes and untilizes every tile.
- **to Nchw** is the permute `(0, 3, 1, 2)`. It moves one tile dim and is about 6x
  cheaper at the larger size.

Propagation stays off by default. Converting an Nchw activation to Nhwc costs about as
much as the conv itself at 512 channels, and the backward of the channels-last conv
route is not ours yet.

## Flat storage (conv weights)

A TILE tensor pads each of its last two dims to 32, so a conv weight `[O, C, 3, 3]` in the
default format holds 114x its values: a 512 x 512 x 3 x 3 bf16 weight takes 537 MB. Such
tensors are stored Flat instead, as one TILE tensor `[d0, d1 * ... * dk]` (`[O, C*KH*KW]`
for a conv weight, 4.7 MB here). The elements are in their logical order.

- **When.** Tensors that torch allocates (parameters, host copies, `zeros_like` optimizer
  state) are stored Flat when they are bf16 or fp32 and replicated, have rank 3 or more,
  `d0 >= 32`, `d1 > 1`, and last two dims below 32, and when the tile form would be over
  4x Flat's. `d1 == 1` is a depthwise conv weight: it stays in the tile form, which is
  small and is what `ktt_conv2` reads. `KTT_FLAT=0` turns Flat off.
- **Kept Flat by:** uploads and `.cpu()` (no conversion), casts, fills, random draws,
  elementwise ops including `addcmul`, `addcdiv`, `lerp` and `where`, and reductions
  over all dims. AdamW and `clip_grad_norm_` therefore run on Flat weights, gradients
  and state.
- **Convs:**
  - Kohaku-TT's `ktt_conv` builds its weight matrix from the Flat rows on device.
  - The channels-last conv (`ktt_conv_nhwc`) prepares its weights on the host from the
    Flat values, for the forward and the input gradient.
  - TT-NN's conv takes a prepared weight. Where that is the tap-major matrix
    `[1, 1, KH*KW*C, O]`, it is built on device from the Flat rows after every weight
    write, with no host transfer. This holds for the convs TT-NN folds into a matmul
    (stride equal to the kernel, as in ConvNeXt's 2x2 downsamples) and, measured, for a
    64 -> 128 3x3 stride-2 conv. It is not assumed: at the first conv of each geometry
    a probe weight whose values identify their positions is prepared by TT-NN and by
    the device build, and the two are compared, tile padding included (one more host
    preparation and one conv on an empty input per geometry). A geometry that differs
    (a C = 3 stem) keeps the host preparation, which reshapes the Flat rows on the host.
  - The per-tap matmul routes (the gradients of these convs, fp32 and transposed convs)
    read each tap as a row range of the tap-major matrix `[KH*KW*C, O]`, built from the
    Flat rows on device and cast to fp32 afterwards. They write the weight gradient as
    Flat rows. For a 768 x 384 x 2 x 2 weight that is about 110 MB of reordering per
    backward, where reading the tile form took 1.7 GB.
  - Grouped convs (`ktt_conv2`) read and write the tile form. A Flat weight (more than
    one input channel per group) is converted on its first read (one more op, cached
    per write epoch; inside a trace, on every replay), and its weight gradient is
    reshaped to Flat rows (one more op per backward). Depthwise weights are never Flat
    and cost nothing extra.
  - Weight gradients come out Flat.
  - `weight.reshape(O, -1)` is the stored tensor itself.
- **Expansions.** Other reads (slices, partial reductions, `ktt_conv2`) convert to the
  tile form, once per write epoch. They are counted, and each one over 64 MB warns once
  per shape. A trace capture records its own conversion, as for Nhwc.
- **Block floats.** `block_float_` converts a Flat weight to the tile form first:
  block-float storage is never Flat.
