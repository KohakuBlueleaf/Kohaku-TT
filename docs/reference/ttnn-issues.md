# TT-NN issues

TT-NN / TT-Metal behaviour that affects Kohaku-TT, with evidence, the workaround in
Kohaku-TT and its status. Numbers are from one Blackhole P150A.

| Issue | Effect | Workaround | Status |
|---|---|---|---|
| Tiled reshape builds a host page map for each new shape | seconds of host time on the first call at large sizes | avoided on hot paths; own kernels take shapes as runtime args | open |
| Padded slice: host setup for each new shape | as above, smaller | as above | open |
| Permute, tilize, untilize: host setup for each new shape | up to 2-4 s on the first call | own kernels where it matters | open |
| SDPA with fp32 dest reads a ragged last K tile's padding as keys | wrong outputs at lengths not a multiple of 32 | unmasked attention runs the streaming (bf16 dest) path, which masks the tile | worked around |
| SDPA with many (batch, head) pairs mixes outputs across pairs | wrong outputs for some pairs (cosine 0.93-0.997 against the pair computed alone) | at most 48 pairs per call; larger calls split over batch rows and head ranges | worked around |
| SDPA with fp32 dest and exact exp is 4x slower than the streaming path | 11.5 ms vs 2.7 ms at 5695 tokens | streaming path, q / k chunks 256 | worked around |
| `ttnn.linear` without a program config misbroadcasts a bias over rank > 2 input | TT_FATAL "Broadcast error" | bias added separately | worked around |
| `ttnn.linear` is 1.4-1.6x slower than `experimental.minimal_matmul` | DiT projections | `[K, N]`-stored weights run minimal matmul | worked around |
| conv2d prepared weights depend on the input size under the auto config | wrong values when reused at another size | Kohaku-TT's convolution; TT-NN path caches weights per size | worked around |
| conv2d sliding-window metadata accumulates in L1_SMALL for each shape | OOM after a few resolutions | L1_SMALL 96 KiB; on L1_SMALL OOM the program cache is cleared and the op retried | worked around |
| conv2d DRAM slicing finds no height-sharded config for some shapes | TT_FATAL | auto config for those shapes | worked around |
| Ops place their output in their input's memory | L1 OOM when L1 is full | explicit placement; spill and retry on L1 OOM | worked around |
| `nlp_concat_heads` circular buffers grow with the head count | 53 MB per core at 1615 heads | used up to 128 heads | worked around |
| Permute of a 5-D tensor with small dims pads them to tiles | 12.9-17 GB allocations | no 5-D permutes in convolution | worked around |

## Host setup for each new shape

A MOSS VAE decode at 1312x800 takes 61 s on the first call and 1.1 s on a repeat call.
All 638 kernels come from the JIT cache: nothing compiles. A Tracy host capture puts
61.1 s of the 61.2 s span in host gaps between device ops, grouped by the op that
follows:

| next device op | host gap | calls | largest |
|---|---|---|---|
| `ReshapeViewDeviceOperation` | 43.7 s | 486 | 6.07 s |
| `PaddedSliceDeviceOperation` | 10.8 s | 94 | 1.77 s |
| others | 6.6 s | | |

The tiled reshape computes an input-to-output page map on the host
(`compute_reshape_mapping_host_tensor` in `reshape_tiled_program_factory.cpp`) and caches
it with the program, so the first call at a shape costs work proportional to the
tensor's pages. Most of these reshapes and slices come from TT-NN conv2d's DRAM slicing.
Replacing a tiled reshape by a row-major one (a view) does not help: the untilize, tilize
and permute around it have the same kind of first-call cost (up to 4.2, 2.4 and 2.6 s),
and the decode took 82 s.

## SDPA

TT-NN's SDPA masks the keys of a partial last K tile only on its streaming compute path,
which runs when `fp32_dest_acc_en` is false (`k_partial_col` in
`sdpa_program_factory.cpp`); with fp32 dest the last tile's padding is read as keys. The
streaming path at q / k chunks 256 is also the fastest configuration: 2.70 ms at 5695
tokens and 1.55 ms at 4097 (24 heads x 128), against 11.5-12.4 ms with fp32 dest and
exact exp.

### Outputs mix across (batch, head) pairs

With enough (batch, head) pairs in one call, some pairs' outputs differ from the same
pair computed alone. Identical inputs at every pair, bf16, head dim 128, streaming
path; worst pair against the pair alone:

| layout | 1600 tokens | 1615 tokens | 4110 tokens |
|---|---|---|---|
| B 1, H 24 / 32 / 48 / 49 / 56 | exact | exact | exact |
| B 2, H 12 / 24; B 3, H 8 | exact | exact | |
| B 1, H 63 | exact | exact | abs max 3.6e-02, cos 0.997 |
| B 1, H 64 | abs max 9.2e-02, cos 0.970 | abs max 1.2e-01, cos 0.974 | abs max 4.7e-02, cos 0.997 |
| B 3, H 24 | abs max 2.2e-01, cos 0.989 | abs max 2.0e-01, cos 0.990 | abs max 1.3e-01, cos 0.997 |
| B 1, H 96 | exact | exact | abs max 9.8e-02, cos 0.977 |
| B 1, H 128 | abs max 2.1e-01, cos 0.934 | abs max 2.4e-01, cos 0.940 | abs max 1.0e-01, cos 0.969 |

B 1 with 24, 32, 40 or 48 heads is exact at 279, 517, 1043, 1169, 1448, 1615, 2048,
2900, 4110, 4379, 5695 and 8192 tokens. No q / k chunk size avoids it (chunks 32-256 tried at 1448-5695 tokens). The error is
in a subset of positions (median abs error 0). Kohaku-TT calls SDPA with at most 48
pairs: whole batch rows while a row has at most 48 heads, else head ranges that keep
whole K / V head groups; the outputs are concatenated.

## conv2d weight preparation

Weights prepared at 50x82 and reused at larger sizes match bit for bit up to 200x328 at 192 channels and 100x164 at 384 channels, and differ
by 40-460 (max |diff|) beyond: the auto config changes the sharding and the prepared
layout with it. With height sharding pinned, prepared weights are exact at every size
and the warm conv is faster (8 ms vs 31 ms at 192 channels, 400x656), but no
height-sharded config exists at 384 channels from 200x328 up.
