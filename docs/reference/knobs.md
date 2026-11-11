---
title: Knobs, commands and tools
summary: Every environment variable Kohaku-TT reads, the command-line tools, and the Python diagnostic tools.
tags:
  - reference
  - configuration
---

# Knobs, commands and tools

## Environment variables

| Variable | Default | |
|---|---|---|
| `KTT_FIDELITY` | `HiFi2` | default fidelity of the FPU's multiplies: `LoFi`, `HiFi2`, `HiFi3`, `HiFi4` (`kohakutt.numerics` scopes override it; `KTT_MATMUL_FIDELITY` is an alias) |
| `KTT_APPROX` | 0 | 1 turns SFPU approximations on by default |
| `TT_LOGGER_LEVEL` | `ERROR` | TT-Metal's own log level; Kohaku-TT sets `ERROR` at import unless the variable is already set. `WARN` shows the RISC-V compiler's warnings of a first kernel build (wheel installs; see [install](../install.md#from-a-clone)); `INFO` also brings back the device open / close, UMD and kernel-cache lines. The names are lower or upper case `trace`, `debug`, `info`, `warn`, `error`, `off` (`WARNING` is not one and means `INFO`) |
| `KTT_L1_BUDGET` | 0 | the fraction of each core's L1 that activations may fill (L1 interleaved); 0 keeps them in DRAM (see [limitations](limitations.md)) |
| `KTT_MATMUL_EXPLICIT` | 1 | 0 uses TT-NN's default matmul program configs instead of Kohaku-TT's per-shape 2-D multicast configs |
| `KTT_NATIVE_CONV` | 1 | 0 runs bf16 2-D convs as per-tap matmuls instead of TT-NN's conv2d |
| `KTT_OWN_CONV` | 1 | 0 runs stride-1 single-group convs (C a multiple of 32) on TT-NN's conv2d instead of Kohaku-TT's implicit-GEMM conv |
| `KTT_CONV_WINDOW` | 1 | 0 reads each convolution tap's input rows separately instead of one window per kernel row |
| `KTT_CONV_LAYOUT` | 1 | 0 runs TT-NN's permute / pad / tilize around Kohaku-TT's conv instead of its own layout kernels (sizes as runtime arguments) |
| `KTT_OWN_GROUP_NORM` | 1 | 0 runs bf16 group norm as the composite of reductions and binaries instead of Kohaku-TT's moments / affine kernels (sizes as runtime arguments) |
| `KTT_OWN_GATHER` | 1 | 0 runs every `gather` on TT-NN's gather instead of Kohaku-TT's last-dim gather |
| `KTT_OWN_SOFTMAX` | auto | `softmax` over the last dim: `auto` runs Kohaku-TT's `ktt_softmax` when that dim is not a multiple of 32 and TT-NN's softmax otherwise, 1 `ktt_softmax` on every bf16 / fp32 call, 0 never (TT-NN's softmax, after zeroing a ragged last dim's tile padding); read per call ([softmax.md](kernels/softmax.md#routing)) |
| `KTT_TTNN_GATHER_LARGE` | unset | 1 lets a gather Kohaku-TT's kernel does not cover run TT-NN's gather on inputs above 262,144 elements (otherwise an error: it takes minutes) |
| `KTT_OWN_UPSAMPLE` | 1 | 0 runs nearest upsampling (integer factors dividing 32) and pixel shuffle as interpolation matmuls instead of Kohaku-TT's gather kernels |
| `KTT_OWN_ATTENTION` | 1 | unmasked causal and grouped-head bf16 attention run Kohaku-TT's flash attention; `all` also plain non-causal attention, 0 none (TT-NN's SDPA) |
| `KTT_ATTENTION_FP32` | 1 | 0 runs Kohaku-TT's attention with bf16 DST and intermediates (faster; the running sums lose precision at long sequences) |
| `KTT_SDPA_FP32_DEST` | 0 | 1 runs TT-NN's unmasked attention with fp32 dest registers (lower noise; reads a ragged last key tile's padding as keys) |
| `KTT_L1_SMALL` | 98304 | bytes per core reserved for TT-NN's small L1 buffers (conv metadata) |
| `KTT_OWN_MATMUL` | 1 | 0 sends every bf16 matmul to TT-NN's matmul; 1 runs Kohaku-TT's matmul (every size a runtime argument) where it is faster warm: K, N up to 1024, M from 16384 rows and an output of 12.5M to 30M elements |
| `KTT_OWN_MATMUL_MIN_ROWS` | unset | set: Kohaku-TT's matmul for every bf16 matmul with K, N up to 1024 and at least this many rows (first call at a new M about 0.5 ms instead of TT-NN's 30-1570 ms) |
| `KTT_TREE_PROD` | 1 | 0 sends `prod` to TT-NN's prod (rounds fp32 to bf16) instead of fp32 elementwise multiplies of halves |
| `KTT_MINIMAL_MATMUL` | 1 | 0 sends `[K, N]`-stored weights and plain bf16 `a @ b` to `ttnn.matmul` instead of TT-NN's minimal matmul |
| `KTT_CUSTOM_KERNELS` | 1 | 0 runs TT-NN's implementation for every op that has an own program |
| `KTT_KM_PAGE_CHECK` | `warn` | the launch check of metal-level page indices and declared ranges (PAGE-1, [DSL](../concepts/dsl.md)): `warn` refuses a proven out-of-range access and warns once about one it cannot prove, `strict` also refuses those, 0 skips the check |
| `KTT_TRACE_L1_GUARD` | 1 | 0 lets activations allocated after a capture share L1 with the trace's circular buffers, and skips the per-replay check ([trace](../concepts/trace.md)) |
| `KTT_TRACE_HOLD` | 0 | 1 makes a trace hold every tensor created in its capture instead of only those alive at its end (diagnostic; the footprint becomes the sum of the intermediates; [trace](../concepts/trace.md)) |
| `KTT_FLASH` | 1 | 0 runs bf16 attention on the previous routes instead of `ktt_flash` |
| `KTT_FLASH_BWD_ORIENT` | unset | `kq`: the dK / dV program of `ktt_flash`'s backward in the key-row orientation (ordinary logits only; [flash](kernels/flash.md)) |
| `KTT_ATTENTION_CENTER` | 0 | 1 subtracts the per-head mean of the keys and values in attention calls that need gradients, for inputs with a common offset (dq 0.37-0.48 → 0.012-0.016 at an offset of 2 standard deviations); more device ops per training call, see [flash](kernels/flash.md) for what else it changes |
| `KTT_NHWC` | 0 | 1 keeps activations channels-last between the ops that read it; single-group "same" stride-1 convs then run `ktt_conv_nhwc` |
| `KTT_NHWC_WGRAD` | 1 | 0 sends the channels-last conv's weight gradient through the per-tap route instead of its own program (read per call) |
| `KTT_FLAT` | 1 | 0 stores conv-weight-like tensors (rank 3 or more, `d0 >= 32`, `d1 > 1`, last two dims below 32) in the tile form of their own shape instead of as `[d0, rest]` rows (see [tensor layout](../concepts/layout.md)) |
| `KTT_HEAD_MAJOR` | 1 | 0 keeps per-head tensors `[B, T, H, D]` with few heads (views of a `[B, T, H*D]` tensor read by elementwise ops, and their results; bf16 / fp32, `T >= 32`, `D >= 32`, at most half the tile bytes) in the tile form of their own shape instead of as `[B, H, T, D]` (see [tensor layout](../concepts/layout.md)) |
| `KTT_GEMM` | auto | `mm` and `linear` in `ktt_gemm`'s coverage: `auto` runs Kohaku-TT's `ktt_gemm` on the sizes where it is level with or faster than TT-NN's matmul (the rule in [gemm.md](kernels/gemm.md#routing)) and TT-NN's matmul elsewhere; 1 runs `ktt_gemm` on every covered call (bf16 / block-float weights); 0 runs TT-NN's matmul on every call |
| `KTT_DESC_CACHE` | 1 | 0 builds the program descriptor of every own program on each call, instead of reusing one per program, sizes and memory placement with only the buffer addresses rewritten (user programs stay cached); `verify` reuses and also rebuilds, raising when the two differ |
| `KTT_PHILOX` | 0 | 1 draws `rand` / `randn` (and the distributions built on them) from Philox4x32-10 at each element's global index, so a mesh-sharded draw equals the 1-card draw bit for bit; about 100x slower than TT-NN's per-core generators (`torch.ops.ktt.philox_uniform` draws it directly) |
| `KTT_CPU_FALLBACK` | 0 | 1 runs ops without a tt kernel on the CPU instead of raising, with a warning and a per-op count (`kohakutt.fallback_counts()`) |
| `KTT_FALLBACK_WARN` | 1 | 0 silences the fallback warning; counts are still kept |
| `KTT_MESH_AUTO` | 1 | 0 creates every tensor on a mesh replicated |
| `KTT_MESH_SHARD_MIN` | 1048576 | bytes from which a new tensor (and a Linear in the default pairing) is sharded |
| `KTT_MESH_AUTO_PAIRS` | 1 | 0 leaves unmarked Linears at the size-based placement instead of the column → row pairing |
| `KTT_MESH_STRICT` | 0 | 1 raises when an op has no placed form for an input instead of gathering it, naming the op, the placement and what to do; at 0 the op warns once per op and placement (`KTT_FALLBACK_WARN=0` silences it) and is counted in `torch.tt.mesh_fallback_ops()` |
| `KTT_MESH_MIGRATE` | 0 | 1 moves persistent tensors to the placement their consumers ask for |
| `KTT_MESH_CONV_CACHE` | 1 | 0 runs a placement conversion again for every op that needs it instead of keeping the converted copy on the tensor; the placements chosen and every result's bits are the same |
| `KTT_MESH_CONV_FREE` | 1 | 0 prices a conversion to a kept copy at its full cost: the placements chosen without the conversion memory |
| `KTT_MESH_EXPLAIN` | 0 | 1 records every placement decision from load for `kohakutt.placement_log()` (`with kohakutt.record_placements():` records a block) |
| `KTT_MESH_ALL_TO_ALL` | 1 | 0 converts S(i) → S(j) by all-gather and partition instead of one all-to-all |
| `KTT_MESH_CCL` | 1 | 0 converts placements through the host instead of TT-NN collectives |
| `KTT_MESH_REDUCE` | 1 | 0 completes partial sums with TT-NN's collectives instead of Kohaku-TT's all-reduce program (which also adds a residual operand) |
| `KTT_MESH_FUSE` | 0 | 1 runs an all-reduce inside the program of the Linear that consumes its sum (`torch.tt.mesh_stats()["fused_linears"]`); slower than the two programs today ([mesh](../concepts/mesh.md)) |
| `KTT_FABRIC` | 1 | 0 opens a mesh without the fabric (no collectives) |
| `TT_VISIBLE_DEVICES` | every chip | chips this process may open (TT-Metal) |
| `KOHAKUTT_CACHE` | `~/.cache/kohakutt` | where caches go: the kernel compiler, generated kernel sources |

Test-only variables (`KTT_HW`, `KTT_TEST_CHIP`) are in [development](../development.md#checks).

## Commands

| Command | Does |
|---|---|
| `ktt doctor` | read-only report of the host and every card: firmware, link, telemetry |
| `ktt sfpi` | fetches the RISC-V kernel compiler into the cache (once, after a wheel install) |
| `python -m kohakutt.tools.support` | regenerates the aten support table from the registered kernels |
| `python -m kohakutt.bench.ops [--ops ...] [--json ...]` | per-op cold / warm time, error statistics against float64 next to the CPU same-dtype result, batch rows against each row alone |
| `python -m kohakutt.tools.cxx_format [--check]` | formats the C / C++ sources |

`ktt doctor` is a read-only report of the host and every card on it: identity, firmware,
PCIe link against the expected one, clocks, power and temperature, with a warning per
problem.

## Python tools

| Tool | Does |
|---|---|
| `kohakutt.tools.kernel_audit.KernelAudit` | every aten call of a block against a float64 ruler and a same-precision CPU simulation (error statistics of both), or each call's device-synchronized time |
| `kohakutt.tools.profile.OpProfile` | time, programs and host transfers per aten overload of a block |
| `kohakutt.diff_stats(result, reference)` / `kohakutt.format_stats(stats)` | max / median / mean of the absolute and relative error (rel over nonzero reference positions), cosine similarity and non-finite count, compared in float64; `format_stats` prints them on one line |
