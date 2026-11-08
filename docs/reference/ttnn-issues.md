# TT-NN issues

TT-NN / TT-Metal behaviour that affects Kohaku-TT, with evidence, the workaround in
Kohaku-TT and its status. Numbers are from one Blackhole P150A.

| Issue | Effect | Workaround | Status |
|---|---|---|---|
| Tiled reshape builds a host page map for each new shape | seconds of host time on the first call at large sizes | avoided on hot paths; own kernels take shapes as runtime args | open |
| Padded slice: host setup for each new shape | as above, smaller | as above | open |
| Permute, tilize, untilize: host setup for each new shape | up to 2-4 s on the first call | own kernels where it matters | open |
| SDPA with fp32 dest reads a ragged last K tile's padding as keys | wrong outputs at lengths not a multiple of 32 | unmasked attention runs the streaming (bf16 dest) path, which masks the tile | worked around |
| SDPA with many (batch, head) pairs mixes outputs across pairs | wrong outputs for some pairs (cosine 0.93-0.997 against the pair computed alone) | causal and grouped-head calls run Kohaku-TT's attention; plain calls are split to at most 48 pairs | worked around |
| Kernels take sizes as compile-time arguments | every new shape compiles kernels: 1-4 s on the first call | Kohaku-TT's own kernels take sizes as runtime arguments (attention, convolution, elementwise) | open for permute, pad, matmul |
| bf16 compute does not carry NaN | NaN becomes +-inf (`x * 1`, `x + y`, `neg`, `exp`, `where`) or 0 (`relu`); fp32 keeps NaN; copies keep it | `nanmedian` tests NaN before computing; ops that fill NaN and compute on it in bf16 (`torch.masked.median`, `torch.masked.argmin` through `where`) differ from PyTorch | open |
| `ttnn.embedding` accepts bf16 tables only | fp32 rows through gather: 42.7 ms for 2068 rows of a 32000 x 1024 table | Kohaku-TT's row gather for fp32 tables: 0.40 ms | worked around |
| `noc_async_read_barrier_with_trid` after `noc_async_read_set_trid` and plain `noc_async_read` / `noc_async_read_page` does not wait for those reads (Blackhole) | data read before it lands (Kohaku-TT's conv reader: cosine 0.996 against 0.999995) | global read barriers | worked around |
| Casts between bf16 and fp32 do not keep NaN (`ttnn.typecast` and Kohaku-TT's cast, unpacked to DST or not) | bf16 -> fp32 gives -inf, fp32 -> bf16 gives +inf | argmax / argmin test NaN on the input's own dtype | open |
| `ttnn.embedding` fed a TILE int32 index (typecast to uint32, to row-major, reshape to 1-D) picks wrong rows | 450 of 1000 bf16 rows wrong for an index computed on device (an uploaded row-major index is exact) | TILE indices go to Kohaku-TT's row gather | worked around |
| fp32 matmul operands keep fewer mantissa bits than fp32, also at HiFi4 | bicubic resize of fp32 (interpolation matrices): rel error median 1.9e-4 against the CPU's 4.9e-8 | nearest resizes and selections split fp32 into three bf16 parts (`exact_select`); linear algebra runs as elementwise fp32 steps and fp32 sums | open |
| `ttnn.bitcast` int32 -> fp32 runs as a unary compute op and drops the low mantissa bits | the bits of 1.0 plus one read back as 1.0; frexp mantissas of fp32 values off from the 11th bit | frexp / nextafter on tt for bf16 only (bf16 values fit the kept bits) | open |
| `ttnn.prod` of an fp32 tensor rounds to bf16 | rel error ~1e-3 even over one element (CPU fp32: 1e-7); a determinant as the product of U's diagonal: rel 2e-2 at n 16 | Kohaku-TT multiplies halves elementwise in fp32 (`KTT_TREE_PROD`) | worked around |
| Causal SDPA is slower than non-causal | 9.7 ms against 1.6 ms (24 heads, 4379 / 4110 tokens) | Kohaku-TT's attention: 5.4 ms | worked around |
| SDPA with fp32 dest and exact exp is 4x slower than the streaming path | 11.5 ms vs 2.7 ms at 5695 tokens | streaming path, q / k chunks 256 | worked around |
| `ttnn.linear` without a program config misbroadcasts a bias over rank > 2 input | TT_FATAL "Broadcast error" | bias added separately | worked around |
| `ttnn.matmul` / `ttnn.linear` / `minimal_matmul` sum over the inner dim's tile padding, which other device ops (reshape) leave non-zero | K not a multiple of 32: nondeterministic wrong results (bf16 [250, 10] @ [10, 1]: abs max 27-41 across runs against a CPU bf16 baseline of 0.95) | both operands' tile padding is zeroed (`fill_implicit_tile_padding`) when K % 32 != 0 | worked around |
| `ttnn.all_reduce` of an fp32 tensor sums at reduced precision (1x2 mesh) | x on one chip plus 0 on the other: rel error max 9.7e-4 (64 x 96, 128 x 256; exact at 50 x 70); bf16 exact | fp32 partial sums are gathered and added on each chip in fp32 | worked around |
| `ttnn.mesh_partition` of a TILE tensor at shard boundaries that are not multiples of 32 | wrong values along a split row dim (50 rows in halves); TT_FATAL "Can only slice tilized tensor with width begin index aligned to tiles" along a split width (96 columns in halves) | shard boundaries off the tile grid are partitioned on the host | worked around |
| `ttnn.linear` is 1.4-1.6x slower than `experimental.minimal_matmul` | DiT projections | `[K, N]`-stored weights run minimal matmul | worked around |
| `ttnn.linear` with transpose_b and no program config (TT-NN's default) is 5-6 TFLOPS on 2048-row shapes | 2048x4096 @ 4096x4096: 12.3 ms; @ 4096x11008: 37.6 ms; 2048x11008 @ 11008x4096: 33.3 ms; 2048x16384 @ 16384x4096: 47.9 ms (`KTT_MATMUL_EXPLICIT=0`) | Kohaku-TT's explicit 2-D multicast configs, with output blocks smaller than `per_core_N` when L1 is short: 0.40-1.26 ms (170-218 TFLOPS) | worked around |
| bf16 conv2d with one input and one output channel over a single input row and a kernel wider than 1 | wrong outputs: cosine 0.03-0.16 against the CPU (1 x 10 and 1 x 33 inputs, 1 x 3 and 1 x 4 kernels); 1 x 1 kernels, two rows, two channels and 10 x 1 inputs are right | that case runs Kohaku-TT's convolution | worked around |
| conv2d prepared weights depend on the input size under the auto config | wrong values when reused at another size | Kohaku-TT's convolution; TT-NN path caches weights per size | worked around |
| conv2d sliding-window metadata accumulates in L1_SMALL for each shape | OOM after a few resolutions | L1_SMALL 96 KiB | open |
| conv2d DRAM slicing finds no height-sharded config for some shapes | TT_FATAL | auto config for those shapes | worked around |
| Ops place their output in their input's memory, collectives included (an all-gather's output is the mesh size times its input) | L1 OOM when L1 is full: a 128 MiB all-gather output was placed in L1 after its 64 MiB input | every op and collective gets an output memory config sized before launch | worked around |
| A collective's first run sets up its semaphores and buffers with host writes, which a trace capture refuses (`Writes are not supported during trace capture`) | a capture whose collective had not run eagerly stalled | the capture's prepare pass runs every collective the capture records | worked around |
| `nlp_concat_heads` circular buffers grow with the head count | 53 MB per core at 1615 heads | used up to 128 heads | worked around |
| Permute of a 5-D tensor with small dims pads them to tiles | 12.9-17 GB allocations | no 5-D permutes in convolution | worked around |
| `generic_op`'s cache-hit fast path for `Buffer*` runtime-argument bindings (`emplace_runtime_args`) rewrites only the bound addresses, while the program hash covers the runtime-argument count, not the values (`generic_op_device_operation.cpp`, `mesh_device_operation_adapter.hpp`) | a hit at another size with the same core and argument count keeps the first call's sizes | own programs pass plain values (the hit copies every value) and cache their descriptors themselves | worked around |

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

## Kernel compilation for each new shape

The first call of an op at a shape never seen before compiles the kernels whose
compile-time arguments carry the shape. Kernel-cache entries created by one first call
(bf16, a fresh shape each time):

| op | first call | kernels compiled |
|---|---|---|
| conv2d 3x3, 128 channels | 3.1-4.2 s | permute (4), pad (1-2), tilize / untilize; the implicit-GEMM convolution itself: none |
| conv2d 1x1, 256 channels | 2.8-3.0 s | permute (4), pad (1) |
| SDPA, 24 heads | 2.3-2.5 s | reader, compute, writer |
| linear, M new | 1.0-1.3 s | 7 matmul kernels |
| group_norm | 1.3-3.7 s | reduce, fill_pad |
| softmax, add (Kohaku-TT kernels) | 30-50 ms | none |

With the on-disk kernel cache warm, a fresh process still spends 70-1000 ms in the
first call (program construction). `experimental.minimal_matmul` carries `M_tiles` and
`M_blocks_per_core` as compile-time arguments. The 2-D multicast matmul recompiles its
reader / writer kernels for every new M even at an unchanged `per_core_M` (M of 36, 37
and 40 tiles at 3072 x 3072: 4 new kernels each, 0.9-1.2 s); its compute kernel is
reused.

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
