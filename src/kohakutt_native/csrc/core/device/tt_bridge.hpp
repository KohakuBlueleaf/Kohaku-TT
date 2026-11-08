// The TT-NN side of the native backend behind a plain C++ interface: no TT-NN or torch
// types cross it, so each side compiles against its own headers.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/mesh/placement.hpp"

namespace ktt::bridge {

    // Physical device dtypes. BFloat8B / BFloat4B (TT block floats) store bf16 tensors
    // for matmul operands; other ops read them through a typecast to bf16.
    enum class DType { BFloat16, Float32, Int32, BFloat8B, BFloat4B };

    inline bool is_block(DType dtype) {
        return dtype == DType::BFloat8B || dtype == DType::BFloat4B;
    }

    // Elementwise op ids; each maps to one TT-NN op in aten/eltwise/device_eltwise.cpp.
    enum class UnaryOp {
        Neg,
        Abs,
        Exp,
        Log,
        Sqrt,
        Rsqrt,
        Reciprocal,
        Tanh,
        Sigmoid,
        Relu,
        Gelu,
        Silu,
        Sin,
        Cos,
        Erf,
        Acos,
        Asin,
        Asinh,
        Atan,
        Atanh,
        Acosh,
        Cosh,
        Sinh,
        Tan,
        Exp2,
        Expm1,
        Log2,
        Log10,
        Log1p,
        Erfc,
        Erfinv,
        Lgamma,
        Digamma,
        Sign,
        Floor,
        Ceil,
        Trunc,
        Frac,
        Hardsigmoid,
        Hardswish,
        Mish,
        IsNan,
        IsInf,
        IsFinite,
        I0
    };
    enum class BinaryOp {
        Add,
        Sub,
        Mul,
        Div,
        Maximum,
        Minimum,
        Pow,
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        BitAnd,
        BitOr,
        BitXor,
        Atan2,
        Shl,
        Shr,
        Gcd,
        Lcm
    };

    // A device tensor; defined in core/device/handle.hpp.
    struct Handle;
    using HandlePtr = std::shared_ptr<Handle>;

    // Host-visible work counters, for checking that ops stay on the device; op_ns is
    // host time spent inside TT-NN op calls.
    struct Counters {
        int64_t uploads = 0;
        int64_t downloads = 0;
        int64_t ops = 0;
        int64_t op_ns = 0;
        // Own programs: launches, descriptor-cache hits, host ns per phase.
        int64_t programs = 0;
        int64_t desc_hits = 0;
        int64_t alloc_ns = 0;
        int64_t args_ns = 0;
        int64_t desc_ns = 0;
        int64_t enqueue_ns = 0;
    };

    // ---- device (core/device/device.cpp)
    // ---------------------------------------------------------

    // Opens the process's mesh (rows x cols over physical `device_ids`, fast dispatch)
    // as the one tt device; tensors are replicated across its chips. close() releases
    // it.
    void open(const std::vector<int>& device_ids, int rows, int cols);
    void close();
    bool is_open();

    // From process exit on, handles are leaked instead of freed (TT-Metal's statics may
    // be gone); called by the Python atexit hook.
    void mark_exiting();
    int mesh_size();

    // The compute (Tensix) grid of one chip: {columns, rows}.
    std::vector<int64_t> compute_grid();

    // Device bytes held by live tensors (padded, per chip) and their peak; drains the
    // stream first.
    struct MemoryStats {
        int64_t allocated = 0;
        int64_t peak = 0;
    };

    MemoryStats memory_stats();
    void reset_peak_memory();

    // Waits until all queued device work is done.
    void synchronize();

    // Drains the device profiler's buffers into its log (TT_METAL_DEVICE_PROFILER=1
    // runs).
    void read_device_profiler();
    // Drops TT-NN's compiled programs and the device buffers they hold (e.g. conv
    // sliding-window metadata in L1_SMALL); programs rebuild from the JIT cache.
    void clear_program_cache();

    Counters counters();

    // ---- trace (core/device/trace.cpp)
    // ------------------------------------------------------

    // Starts / ends capturing every enqueued op into a TT-NN trace; every handle
    // created while capturing is held until trace_release. trace_end returns the held
    // count.
    int64_t trace_begin();
    int64_t trace_end();
    bool capturing();
    // New activations in DRAM only (`dram`) for the ops queued after this call: trace
    // preparation and capture place every buffer alike.
    void trace_placement(bool dram);

    // True while a capture is active or any captured trace is alive.
    bool traces_alive();

    // Counts in-place device writes (copy_into, trace replays); caches keyed by a
    // handle compare it to detect contents changed under an unchanged handle.
    int64_t write_epoch();

    // Device random draws (seeded per call): U(from, to) shaped like `like`, N(0, 1) of
    // `shape`, Bernoulli(p) per element of `p` (as 0 / 1 in p's dtype).
    HandlePtr uniform(const HandlePtr& like, double from, double to, uint32_t seed);
    HandlePtr normal(const std::vector<int64_t>& shape, DType dtype, uint32_t seed);
    HandlePtr bernoulli(const HandlePtr& p, uint32_t seed);
    // Replays a trace (non-blocking) / releases it and its held handles.
    void trace_execute(int64_t id);
    void trace_release(int64_t id);
    // Live buffers allocated while trace `id` was active (TT-NN tracker; needs
    // TT_METAL_TRACE_ALLOC_TRACKING=1, else 0): buffers a replay may overwrite.
    int64_t unsafe_buffers(int64_t id);
    // Fences of trace `id` that landed away from their temporary's address
    // (diagnostic).
    int64_t trace_misplaced_fences(int64_t id);
    // Copies src's contents into dst's buffer in place (trace inputs and state
    // write-back).
    void copy_into(const HandlePtr& dst, const HandlePtr& src);
    // A new buffer with a's contents. Handles held by a trace are overwritten by
    // replays, so tensor copies must copy data, never share a handle.
    HandlePtr clone(const HandlePtr& a);

    // ---- memory (aten/tensor/device_memory.cpp)
    // -----------------------------------------------------

    // Row-major host data <-> a TILE-layout device tensor of `shape` and `dtype`; a 1-D
    // shape (n) is stored as (1, n). download reads the replica on mesh chip `chip`.
    HandlePtr upload(const void* data, const std::vector<int64_t>& shape, DType dtype);
    void download(const HandlePtr& handle, void* dst, int chip = 0);
    HandlePtr empty(const std::vector<int64_t>& shape, DType dtype);
    DType dtype(const HandlePtr& handle);
    // The device (logical TT-NN) shape of a handle / of logical torch sizes.
    std::vector<int64_t> device_sizes(const HandlePtr& handle);
    std::vector<int64_t> device_sizes_of(const std::vector<int64_t>& logical);

    // ---- mesh placements (core/mesh/device_mesh.cpp)
    // ------------------------------------------------

    // Mesh sizes per axis: {rows, cols}.
    std::vector<int> mesh_shape();
    // The placement of a handle (replicated unless made by upload_placed / distribute).
    Placement placement(const HandlePtr& handle);
    // upload() of host data placed by `p`; P(sum) axes hold the data on coordinate 0
    // and zeros elsewhere.
    HandlePtr upload_placed(
        const void* data,
        const std::vector<int64_t>& shape,
        DType dtype,
        const Placement& p
    );
    // a with the same values at placement `p`.
    HandlePtr distribute(const HandlePtr& a, const Placement& p);
    // The replicated copy of a: a itself when replicated, else a cached gather fallback;
    // raises under KTT_MESH_STRICT=1.
    HandlePtr replicated(const HandlePtr& a);
    // The placement of a new tensor of device shape `dims` (R under KTT_MESH_AUTO=0):
    // tensors of at least KTT_MESH_SHARD_MIN bytes (default 1 MiB) shard on the largest
    // mesh axis (leading dim if it divides, else the largest dim that divides); else R.
    Placement auto_placement(const std::vector<int64_t>& dims, DType dtype);
    // empty() of device shape `dims` at placement `p`.
    HandlePtr empty_placed(
        const std::vector<int64_t>& dims, DType dtype, const Placement& p
    );
    // full() of device shape `dims` at placement `p` (R / S axes): every chip's local
    // tensor holds `value`.
    HandlePtr full_placed(
        const std::vector<int64_t>& dims, double value, DType dtype, const Placement& p
    );
    // DRAM bytes free / allocatable, summed over the mesh's chips.
    struct MemoryInfo {
        int64_t free = 0;
        int64_t total = 0;
    };
    MemoryInfo mem_info();
    // The per-chip (local) device shape of a handle, and chip `chip`'s local values
    // (shard padding included) read into `dst`.
    std::vector<int64_t> local_sizes(const HandlePtr& handle);
    void download_local(const HandlePtr& handle, void* dst, int chip);
    // Gather fallbacks (count, global bytes) since load or the last reset.
    struct MeshCounters {
        int64_t gathers = 0;
        int64_t gather_bytes = 0;
        int64_t device_gathers = 0;  // redistributions to R through TT-NN collectives
        int64_t ccl_failures = 0;    // collectives that raised (host path instead)
    };
    MeshCounters mesh_counters();
    void reset_mesh_counters();

    // ---- ops (aten/<family>/device_*.cpp)
    // -------------------------------------------------------------

    HandlePtr unary(UnaryOp op, const HandlePtr& a);
    // Same-shape, same-dtype operands; comparisons return bf16 0/1.
    HandlePtr binary(BinaryOp op, const HandlePtr& a, const HandlePtr& b);
    HandlePtr binary_scalar(BinaryOp op, const HandlePtr& a, double scalar);
    HandlePtr typecast(const HandlePtr& a, DType dtype);
    // a's bits reinterpreted as `dtype` (same element size: fp32 <-> int32).
    HandlePtr bitcast(const HandlePtr& a, DType dtype);
    // a clamped to [lo, hi] (either bound optional).
    HandlePtr clamp(
        const HandlePtr& a, std::optional<double> lo, std::optional<double> hi
    );
    // cond (0/1) ? a : b.
    HandlePtr where(const HandlePtr& cond, const HandlePtr& a, const HandlePtr& b);
    // a + v * b * c, a + v * b / c, a + weight * (end - a).
    HandlePtr addcmul(
        const HandlePtr& a, const HandlePtr& b, const HandlePtr& c, double v
    );
    HandlePtr addcdiv(
        const HandlePtr& a, const HandlePtr& b, const HandlePtr& c, double v
    );
    HandlePtr lerp(const HandlePtr& a, const HandlePtr& end, double weight);

    // A device tensor of `shape` filled with `value`.
    HandlePtr full(const std::vector<int64_t>& shape, double value, DType dtype);

    // A tensor like `like` (shape, dtype) filled with `value` on device (trace-safe).
    HandlePtr full_like(const HandlePtr& like, double value);

    // [0, n) as a (1, n) device tensor, cached per (n, dtype).
    HandlePtr arange(int64_t n, DType dtype);

    // Max over device dims `dims`; result in the device shape of logical sizes `out`.
    HandlePtr amax(
        const HandlePtr& a,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& out
    );

    // Numerically stable softmax along device dim `dim`.
    HandlePtr softmax(const HandlePtr& a, int64_t dim);

    // Index-like ops (aten/index/device_index.cpp): argmax along the last device dim
    // (int32, device shape of logical `out`), lower triangle, cumulative sum along a
    // device dim, [start, end) by step as a 1-D tensor.
    HandlePtr argmax_last(const HandlePtr& a, const std::vector<int64_t>& out);
    HandlePtr tril(const HandlePtr& a, int64_t diagonal);
    HandlePtr cumsum(const HandlePtr& a, int64_t dim);
    HandlePtr cumprod(const HandlePtr& a, int64_t dim);
    HandlePtr arange_range(int64_t start, int64_t end, int64_t step, DType dtype);

    // (values, int32 indices) of a sorted along device dim `dim` / of its top k.
    std::pair<HandlePtr, HandlePtr> sort(
        const HandlePtr& a, int64_t dim, bool descending, bool stable
    );
    std::pair<HandlePtr, HandlePtr> topk(
        const HandlePtr& a, int64_t k, int64_t dim, bool largest, bool sorted
    );

    // Rows of 2-D bf16 `table`, or of each [V, D] slice of fp32 / bf16 `table` [..., V,
    // D] when row_gather(), at int32 `index` (device [1, count]); `version` names the
    // 2-D bf16 table's contents (a row-major copy is cached per version).
    bool row_gather();
    HandlePtr embedding_rows(
        const HandlePtr& table, int64_t version, const HandlePtr& index, int64_t count
    );

    // a gathered along device dim `dim` at `index` (same rank as a).
    HandlePtr gather(const HandlePtr& a, int64_t dim, const HandlePtr& index);
    // a with src written along device dim `dim` at `index` (same rank); reduce ""
    // (replace), "add" or "multiply".
    HandlePtr scatter(
        const HandlePtr& a,
        int64_t dim,
        const HandlePtr& index,
        const HandlePtr& src,
        const std::string& reduce
    );

    // Sum over device dims `dims`; the result has the device shape of logical sizes
    // `out`.
    HandlePtr sum(
        const HandlePtr& a,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& out
    );

    // Product over device dim `dim` (all dims when empty); device shape of logical
    // `out`.
    HandlePtr prod(
        const HandlePtr& a, std::optional<int64_t> dim, const std::vector<int64_t>& out
    );

    // View materialization (aten/tensor/device_view.cpp): dense results in TILE
    // layout.
    HandlePtr repeat(const HandlePtr& a, const std::vector<int64_t>& repeats);
    // Concatenation of `parts` along device dim `dim`.
    HandlePtr concat(const std::vector<HandlePtr>& parts, int64_t dim);
    HandlePtr reshape(const HandlePtr& a, const std::vector<int64_t>& sizes);
    HandlePtr permute(const HandlePtr& a, const std::vector<int64_t>& dims);
    HandlePtr slice(
        const HandlePtr& a,
        const std::vector<int64_t>& begins,
        const std::vector<int64_t>& ends,
        const std::vector<int64_t>& steps
    );

    // x / rms(x) * gamma + beta over the last device dim, one TT-NN program (fp32
    // accumulation; gamma / beta null or [1, W] rows of x's dtype).
    HandlePtr rms_norm(
        const HandlePtr& x,
        double eps,
        const HandlePtr& gamma,
        const HandlePtr& beta = nullptr
    );

    // 2-D convolution of x [N, C, H, W] (bf16) with w [O, C / groups, KH, KW]: TT-NN's
    // sliding-window conv2d on channels-last x, -> [N, O, OH, OW] (aten/nn/
    // device_conv.cpp; prepared weights cached per weight handle and geometry).
    HandlePtr conv2d(
        const HandlePtr& x,
        const HandlePtr& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups
    );

    // Group norm pieces (aten/nn/device_group_norm.cpp, ktt_group_norm) for x [N, C, H,
    // W] (bf16 / fp32): (sum x, sum x^2) per (n, c) as fp32 [N, C, 1, 1]; y = x a + b
    // with a, b fp32 [N, C, 1, 1], in x's dtype.
    std::pair<HandlePtr, HandlePtr> group_norm_moments(const HandlePtr& x);
    // From the moments, weight / bias [C] (null when absent), `count` elements per
    // group: {a, b} per (n, c) and {mean, rstd} as `out` [N, groups].
    std::array<HandlePtr, 4> group_norm_stats(
        const HandlePtr& s1,
        const HandlePtr& s2,
        const HandlePtr& weight,
        const HandlePtr& bias,
        int64_t groups,
        double count,
        double eps,
        DType out
    );
    HandlePtr group_norm_affine(
        const HandlePtr& x, const HandlePtr& a, const HandlePtr& b
    );

    // Nearest upsampling of x [..., H, W] (bf16 / fp32) by integer factors kh, kw
    // dividing 32 -> [..., H kh, W kw] (aten/nn/device_upsample.cpp, ktt_upsample).
    HandlePtr upsample_nearest(const HandlePtr& x, int64_t kh, int64_t kw);
    // Pixel shuffle of x [..., C r^2, H, W] (bf16 / fp32, r dividing 32) -> [..., C,
    // r H, r W].
    HandlePtr pixel_shuffle(const HandlePtr& x, int64_t r);

    // Rotate-half RoPE: x * cos + cat(-x2, x1) * sin for x [B, H, S, D] (D 32 or a
    // multiple of 64) and cos / sin [1, 1, S', D], S' >= S, of x's dtype.
    HandlePtr rotary(const HandlePtr& x, const HandlePtr& cos, const HandlePtr& sin);

    // Normalization and losses (aten/nn/device_norm.cpp; bf16 TT-NN moreh kernels).
    // Layer norm over the trailing `normalized` device dims (gamma / beta may be null):
    // y, and mean / rstd in the device shape of x's leading dims.
    struct LayerNorm {
        HandlePtr y, mean, rstd;
    };

    LayerNorm layer_norm(
        const HandlePtr& x,
        int64_t normalized,
        double eps,
        const HandlePtr& gamma,
        const HandlePtr& beta
    );

    // Requested gradients of layer_norm (null when not requested).
    struct LayerNormGrads {
        HandlePtr input, gamma, beta;
    };

    LayerNormGrads layer_norm_backward(
        const HandlePtr& grad,
        const HandlePtr& x,
        const HandlePtr& mean,
        const HandlePtr& rstd,
        int64_t normalized,
        const HandlePtr& gamma,
        bool want_input,
        bool want_gamma,
        bool want_beta
    );

    // nll of log-probabilities logp [N, C] at int32 targets [N] (mean or sum) with the
    // mean's divisor (count of non-ignored targets; null for sum), and its gradient for
    // upstream `grad`.
    struct NllLoss {
        HandlePtr loss, total;
    };

    NllLoss nll_loss(
        const HandlePtr& logp, const HandlePtr& target, bool mean, int64_t ignore_index
    );
    HandlePtr nll_loss_backward(
        const HandlePtr& logp,
        const HandlePtr& target,
        const HandlePtr& grad,
        const HandlePtr& divisor,
        int64_t ignore_index
    );
    // d gelu(x) * grad (exact GELU).
    HandlePtr gelu_backward(const HandlePtr& grad, const HandlePtr& x);

    // softmax(q k^T * scale + mask) v for [B, H, S, D] q / k / v
    // (aten/attention/device_attention.cpp; TT-NN flash attention); mask null or
    // additive [B, 1 or H, Sq, Sk].
    HandlePtr sdpa(
        const HandlePtr& q,
        const HandlePtr& k,
        const HandlePtr& v,
        const HandlePtr& mask,
        bool is_causal,
        std::optional<double> scale
    );

    // Training attention (tt-train's fused kernels;
    // aten/attention/device_attention.cpp): [B, H, S, D] q / k / v with S % 32 == 0,
    // scale 1 / sqrt(D), no mask or causal. The forward also returns its row statistics
    // [B, H, S, 32] for the backward, which returns dq, dk, dv.
    struct TrainAttention {
        HandlePtr out, stats;
    };

    TrainAttention sdpa_train(
        const HandlePtr& q, const HandlePtr& k, const HandlePtr& v, bool is_causal
    );
    std::vector<HandlePtr> sdpa_train_backward(
        const HandlePtr& grad,
        const HandlePtr& out,
        const HandlePtr& q,
        const HandlePtr& k,
        const HandlePtr& v,
        const HandlePtr& stats,
        bool is_causal
    );

    // q, k, v as [B, heads, T, D] from a fused [B, T, 3 heads D] projection (one TT-NN
    // program; D tile-aligned).
    std::vector<HandlePtr> split_qkv_heads(const HandlePtr& qkv, int64_t heads);

    // [B, heads, T, C / heads] from a [B, T, C] tensor (one TT-NN program; C / heads
    // tile-aligned).
    HandlePtr split_heads(const HandlePtr& x, int64_t heads);

    // [B, H, T, D] heads merged into [B, T, H D] (one TT-NN program; D tile-aligned).
    HandlePtr concat_heads(const HandlePtr& x);

    // Matmuls accumulate in fp32 at KTT_MATMUL_FIDELITY (LoFi when `fast`); operands
    // may be block floats, mixed with bf16 / fp32. a @ b with either operand optionally
    // transposed (no data movement for the transpose), in dtype `out` (default: a's,
    // bf16 for a block-float a).
    HandlePtr mm(
        const HandlePtr& a,
        const HandlePtr& b,
        bool transpose_a,
        bool transpose_b,
        std::optional<DType> out = std::nullopt,
        bool fast = false
    );
    // x @ weight^T (+ bias): torch.nn.functional.linear with weight (out, in), or with
    // `kn` x @ weight for weight stored (in, out) (TT-NN's minimal matmul).
    HandlePtr linear(
        const HandlePtr& x,
        const HandlePtr& weight,
        const HandlePtr& bias,
        bool kn = false
    );

    // ---- numerics (core/device/numerics.cpp) ----------------------------------------

    // The calling thread's numerics scope: fidelity rank 0..3 (LoFi, HiFi2, HiFi3,
    // HiFi4), SFPU approximations, strict (kernel requirements ignored), attention
    // mode 0..2 (exact, K centered, K centered + Q / K bfp8).
    struct NumericsState {
        int fidelity;
        bool approx;
        bool strict;
        int attention;
    };

    void numerics_push(int fidelity, bool approx, bool strict, int attention);
    void numerics_pop();
    NumericsState numerics_current();

    // ---- user programs (core/device/run_program.cpp, extension.cpp) ----------------

    // Runs the Metalium program described by `spec` (JSON from kohakutt.kernels) on the
    // default stream. Tensor references in it index `inputs`, then the outputs, which
    // are allocated as TILE tensors of logical sizes `out_shapes` and `out_dtypes`.
    std::vector<HandlePtr> run_program(
        const std::string& spec,
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes
    );

    // run_program() for the program registered as `name` by an extension library
    // (ktt::extension::register_program).
    std::vector<HandlePtr> run_registered(
        const std::string& name,
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes
    );

}  // namespace ktt::bridge
