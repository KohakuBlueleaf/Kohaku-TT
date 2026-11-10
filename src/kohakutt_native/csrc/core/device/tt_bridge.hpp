// The TT-NN side of the native backend behind a plain C++ interface: no TT-NN or torch
// types cross it, so each side compiles against its own headers.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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

    // A handle's storage (core/device/layout.cpp): Nchw, a TILE tensor of its device
    // shape; Nhwc ([N, C, H, W] only), TILE [1, 1, N H W, C] with zero tile padding;
    // Flat (rank >= 3), TILE [d0, d1 ... dk]; HeadMajor (rank >= 3), TILE [.., H, T,
    // D].
    enum class Format { Nchw, Nhwc, Flat, HeadMajor };

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
        GeluTanh,
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
        // TT-Metal's program-cache entries after the last own program launch, and
        // kernel sources its JIT build compiled in this process.
        int64_t program_entries = 0;
        int64_t kernel_compiles = 0;
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
    // be gone); called by the Python atexit hook. `close_device` also closes a
    // device without fabric (for a caller that ends the process without teardown).
    void mark_exiting(bool close_device = false);

    // True when `h` was made under an earlier open of the device than the current one.
    bool stale(const HandlePtr& h);
    int mesh_size();

    // The compute (Tensix) grid of one chip: {columns, rows}.
    std::vector<int64_t> compute_grid();

    // Circular-buffer bytes per core a program launched now may use (below the
    // lowest live L1 buffer, within KTT_L1_BUDGET); waits for queued work first.
    int64_t cb_budget();

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

    // Starts / ends capturing every enqueued op into a TT-NN trace; trace_end returns
    // the number of handles the trace holds (docs/concepts/trace.md).
    int64_t trace_begin();
    int64_t trace_end();
    bool capturing();
    // New activations in DRAM only (`dram`) for the ops queued after this call: trace
    // preparation and capture place every buffer alike.
    void trace_placement(bool dram);
    // Moves `a`'s buffer to DRAM when the activation policy put it in L1 (a value that
    // becomes a storage's base: parameters and buffers written in place).
    void keep_in_dram(const HandlePtr& a);

    // True while a capture is active, a capture's prepare pass runs, or any captured
    // trace is alive.
    bool traces_alive();

    // Counts in-place device writes (copy_into, trace replays); caches keyed by a
    // handle compare it to detect contents changed under an unchanged handle.
    int64_t write_epoch();

    // Device random draws (seeded per call): U(from, to) shaped like `like`, N(0, 1) of
    // `shape`, Bernoulli(p) per element of `p` (as 0 / 1 in p's dtype).
    HandlePtr uniform(const HandlePtr& like, double from, double to, uint32_t seed);
    HandlePtr normal(const std::vector<int64_t>& shape, DType dtype, uint32_t seed);
    HandlePtr bernoulli(const HandlePtr& p, uint32_t seed);
    // Counter-based uniforms in [0, 1) of logical `shape` (24 bits; bf16 truncated):
    // element i is word `word` of Philox4x32-10 at counter i under `key`, so a result
    // placed at `p` holds the same values as a replicated one. Null past rank 8.
    HandlePtr philox_uniform(
        const std::vector<int64_t>& shape,
        DType dtype,
        uint64_t key,
        uint32_t word,
        const Placement& p
    );
    // True under KTT_PHILOX=1: random ops draw Philox on the global element index (else
    // TT-NN's per-core generators, about 100x faster).
    bool philox_enabled();
    // Replays a trace (non-blocking) / releases it and its held handles.
    void trace_execute(int64_t id);
    void trace_release(int64_t id);
    // Live buffers allocated while trace `id` was active (TT-NN tracker; needs
    // TT_METAL_TRACE_ALLOC_TRACKING=1, else 0): buffers a replay may overwrite.
    int64_t unsafe_buffers(int64_t id);
    // Trace `id`: {handles held, handles created in its capture, fences, device bytes
    // per chip fenced, those of the capture's buffers left allocated}; empty: no trace.
    std::vector<int64_t> trace_footprint(int64_t id);
    // The per-bank address range [begin, end) of `h`'s device buffer (test hook).
    std::vector<int64_t> buffer_range(const HandlePtr& h);
    // Reserves DRAM [begin, end) per bank as a fence does and frees it again; returns
    // the plugs it needed, throws when the range is not free (test hook).
    int64_t try_fence(int64_t begin, int64_t end);
    // {the highest L1 address trace `id`'s circular buffers reach (0: none; -1: no such
    // trace), the lowest live L1 buffer address (-1: none), the L1 base address}.
    std::vector<int64_t> trace_l1_state(int64_t id);
    // Copies src's contents into dst's buffer in place (trace inputs and state
    // write-back), converted to dst's format first when the two differ.
    void copy_into(const HandlePtr& dst, const HandlePtr& src);
    // A new buffer with a's contents, in a's format. Handles held by a trace are
    // overwritten by replays, so tensor copies must copy data, never share a handle.
    HandlePtr clone(const HandlePtr& a);

    // ---- memory (aten/tensor/device_memory.cpp)
    // -----------------------------------------------------

    // Row-major host data <-> a TILE-layout device tensor of `shape` and `dtype`
    // stored as `format` (Nchw or Flat); a 1-D shape (n) is stored as (1, n). download
    // reads the replica on mesh chip `chip`.
    HandlePtr upload(
        const void* data,
        const std::vector<int64_t>& shape,
        DType dtype,
        Format format = Format::Nchw
    );
    void download(const HandlePtr& handle, void* dst, int chip = 0);
    HandlePtr empty(
        const std::vector<int64_t>& shape, DType dtype, Format format = Format::Nchw
    );
    DType dtype(const HandlePtr& handle);
    // The device (logical TT-NN) shape of a handle / of logical torch sizes.
    std::vector<int64_t> device_sizes(const HandlePtr& handle);
    std::vector<int64_t> device_sizes_of(const std::vector<int64_t>& logical);

    // ---- physical format (core/device/layout.cpp)
    // ---------------------------------------------------

    Format format(const HandlePtr& handle);
    // `a` in the format: a, or a converted copy cached on `a` per write_epoch() (not
    // while tracing). Nhwc / Flat: non-empty, replicated, bf16 / fp32, rank 4 / >= 3.
    HandlePtr to_nchw(const HandlePtr& a);
    HandlePtr to_nhwc(const HandlePtr& a);
    HandlePtr to_flat(const HandlePtr& a);
    HandlePtr to_head_major(const HandlePtr& a);
    // The head-major rule for device shape `dims` [.., T, H, D]: bf16 / fp32, T and D
    // >= 32, head-major tile bytes at most half the shape's (KTT_HEAD_MAJOR=0: never).
    bool head_major_rule(const std::vector<int64_t>& dims, DType dtype);
    // Names the op reading tensors on the calling thread, for head_major_reads().
    void layout_reader(std::string_view op);
    // Counts one read of a head-major tensor in its own shape's tile form by the
    // calling thread's reader; head_major_reads(): "op=count" since the last reset.
    void count_head_major_read();
    std::vector<std::string> head_major_reads();
    void reset_head_major_reads();
    HandlePtr to_format(const HandlePtr& a, Format format);
    // `a` stored as `format`, converted now (not cached; a itself when stored so).
    HandlePtr convert_layout(const HandlePtr& a, Format format);
    // Conversions run by to_nchw() / to_nhwc() / to_flat() / convert_layout() since
    // load, and those of them from Flat to Nchw (the tile form padded per plane).
    int64_t layout_conversions();
    int64_t layout_expansions();
    // The logical shape of the tensor storing device shape `dims` as `format`.
    std::vector<int64_t> stored_sizes_of(
        const std::vector<int64_t>& dims, Format format
    );
    // `a`'s stored tensor as an Nchw handle of its stored shape (no copy).
    HandlePtr as_stored(const HandlePtr& a);
    // True when h is `base` or an as_stored() view of it (one buffer).
    bool aliases(const HandlePtr& h, const HandlePtr& base);
    // Conv weight w [O, C, KH, KW] (Flat or Nchw) as the [rows, O] matrix
    // detail::tap_major() (`blocked`: detail::block_tap_major()) builds.
    HandlePtr tap_major(const HandlePtr& w, bool blocked);
    // m [O, taps C] (Nchw; column t C + c holding w[o, c, t]) as the Flat handle of
    // device shape `shape` [O, C, ...]: the per-tap matrices of a weight, side by side.
    HandlePtr flat_from_taps(const HandlePtr& m, const std::vector<int64_t>& shape);
    // True under KTT_NHWC=1 (read per call): TT-NN's conv2d returns Nhwc, and
    // elementwise ops and nearest upsampling keep Nhwc operands Nhwc.
    bool nhwc_propagation();
    // `stored` (device shape stored_dims(shape, format)) relabeled, no copy, as device
    // shape `shape` stored as `format`: e.g. a [1, 1, 1, C] row as Nhwc [1, C, 1, 1].
    HandlePtr view_as(
        const HandlePtr& stored, const std::vector<int64_t>& shape, Format format
    );
    // The largest |value| in the tile padding of Nhwc `a` (diagnostic; waits).
    double padding_max(const HandlePtr& a);

    // ---- mesh placements (core/mesh/device_mesh.cpp)
    // ------------------------------------------------

    // x @ w (+ bias) as linear(x, w, bias, kn) for an x that is the latest reduce_add's
    // result and still held back on the stream: the all-reduce and the matmul in one
    // program (mesh_rs::all_reduce_matmul) when it takes them, else one after the
    // other. Null when x is not such a result (nothing is run).
    HandlePtr reduce_linear(
        const HandlePtr& x, const HandlePtr& w, const HandlePtr& bias, bool kn
    );

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
    // distribute(a, p) kept on `a` for the current write epoch and returned again
    // (`reused` set); not kept in a capture, its prepare pass or KTT_MESH_CONV_CACHE=0.
    HandlePtr converted(const HandlePtr& a, const Placement& p, bool* reused = nullptr);
    // The placements converted() has made of `a` in the current write epoch.
    std::vector<Placement> conversions_of(const HandlePtr& a);
    // True (once) when a conversion inside the active trace capture had no device path.
    bool take_capture_host_conversion();
    // Marks a capture's prepare pass and the capture (runtime/capture.cpp): placed
    // tensors' gathers then bypass the replicated() cache.
    void set_tracing(bool on);
    // The replicated copy of a: a itself when replicated, else converted(a, R); as a
    // `fallback` it is counted as one and raises under KTT_MESH_STRICT=1.
    HandlePtr replicated(const HandlePtr& a, bool fallback = true);
    // The replicated sum of `a` (P(sum) over a 1x2 mesh, R elsewhere) plus `addend` of
    // the same shape and dtype (replicated, or placed and gathered in this epoch),
    // in one collective program; null when the operands do not qualify.
    HandlePtr reduce_add(const HandlePtr& a, const HandlePtr& addend);

    // One Adam / AdamW step (aten/optim/ktt_adam):
    //   g' = (negate ? -g : g) + l2 p;  m' = beta1 m + (1 - beta1) g'
    //   v' = beta2 v + (1 - beta2) g'^2;  p' = decay p - step m' / (sqrt(v') scale +
    //   eps)
    struct AdamStep {
        double beta1 = 0, beta2 = 0, scale = 1, eps = 0, step = 0, decay = 1, l2 = 0;
        bool negate = false;
    };

    // {p', m', v'} of one program over p, g, m, v (bf16 / fp32; one shape, dtype,
    // format and placement without partial axes); empty when they do not qualify.
    std::vector<HandlePtr> adam_step(
        const HandlePtr& p,
        const HandlePtr& g,
        const HandlePtr& m,
        const HandlePtr& v,
        const AdamStep& s
    );

    // a * scale, cached on `a` until it is written (not inside a trace capture).
    HandlePtr scaled_copy(const HandlePtr& a, double scale);
    // KTT_MESH_SHARD_MIN (default 1 MiB), read per call: tensors below it stay R.
    int64_t shard_min_bytes();
    // The placement of a new tensor of device shape `dims` (R under KTT_MESH_AUTO=0):
    // tensors of at least KTT_MESH_SHARD_MIN bytes (default 1 MiB) shard on the largest
    // mesh axis (leading dim if it divides, else the largest dim that divides); else R.
    Placement auto_placement(const std::vector<int64_t>& dims, DType dtype);
    // upload() / empty() of a tensor allocated by torch (factory ops, host copies): at
    // auto_placement(). Bridge-internal buffers use upload() / empty() and stay R.
    HandlePtr upload_auto(
        const void* data, const std::vector<int64_t>& shape, DType dtype
    );
    HandlePtr empty_auto(const std::vector<int64_t>& shape, DType dtype);
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
    // A handle viewing placed `a`'s shards as one replicated-shaped tensor of the local
    // shape (a itself when replicated): ops run on it per chip. It owns no bytes.
    HandlePtr local_view(const HandlePtr& a);
    // The result `local` of an op run on local views, as a handle of device shape
    // `shape` placed by `p` (local itself when p is replicated).
    HandlePtr adopt_placed(
        const HandlePtr& local, const std::vector<int64_t>& shape, const Placement& p
    );
    // The per-chip (local) device shape of a handle, and chip `chip`'s local values
    // (shard padding included) read into `dst`.
    std::vector<int64_t> local_sizes(const HandlePtr& handle);
    void download_local(const HandlePtr& handle, void* dst, int chip);

    // Gather fallbacks (count, global bytes) since load or the last reset.
    struct MeshCounters {
        int64_t gathers = 0;
        int64_t gather_bytes = 0;
        // Device-side conversions by kind, conversions through the host, and
        // collectives that raised (the host path ran instead).
        int64_t all_gathers = 0;
        int64_t all_reduces = 0;
        int64_t reduce_scatters = 0;
        int64_t partitions = 0;
        int64_t host_conversions = 0;
        int64_t ccl_failures = 0;
        int64_t all_to_alls = 0;
        // All-reduces that ran in the program of the linear consuming their result.
        int64_t fused_linears = 0;
    };

    MeshCounters mesh_counters();
    void reset_mesh_counters();

    // ---- ops (aten/<family>/device_*.cpp)
    // -------------------------------------------------------------

    // Nhwc operands give Nhwc results (zero tile padding): two of one shape, or one an
    // Nhwc [1, C or 1, 1, 1] vector; binary() runs other mixes on Nchw copies.
    HandlePtr unary(UnaryOp op, const HandlePtr& a);
    // Same-shape, same-dtype operands; comparisons return bf16 0/1.
    HandlePtr binary(BinaryOp op, const HandlePtr& a, const HandlePtr& b);
    HandlePtr binary_scalar(BinaryOp op, const HandlePtr& a, double scalar);
    // a * b as complex numbers on interleaved fp32 pairs [..., 2 N], b of a's shape or
    // broadcast over a's leading dims; conj_a / conj_b conjugate. Null without a
    // program.
    HandlePtr complex_mul(
        const HandlePtr& a, const HandlePtr& b, bool conj_a, bool conj_b
    );
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
    // full() without the copy: one cached tensor per (shape, value, dtype) for float
    // dtypes, shared by every caller; for results nothing writes in place.
    HandlePtr shared_full(const std::vector<int64_t>& shape, double value, DType dtype);

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

    // 2-D convolution of x [N, C, H, W] (bf16) with w [O, C / groups, KH, KW] (bf16;
    // `weight_format` BFloat8B / BFloat4B: the weights as stored block floats) ->
    // [N, O, OH, OW]: Kohaku-TT's implicit-GEMM convs or TT-NN's sliding-window conv2d
    // (aten/nn/device_conv.cpp; prepared weights cached per weight handle and
    // geometry).
    HandlePtr conv2d(
        const HandlePtr& x,
        const HandlePtr& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        DType weight_format = DType::BFloat16
    );
    // True when conv2d() runs this grouped, depthwise or strided one-row conv (sizes x,
    // w) on Kohaku-TT's ktt_conv2 (KTT_GROUPED_CONV, read per call: "own", the
    // default).
    bool own_grouped_conv(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        DType weight_format = DType::BFloat16
    );
    // True when such a conv goes to TT-NN's conv2d instead (KTT_GROUPED_CONV=ttnn).
    bool ttnn_grouped_conv(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        int64_t groups
    );
    // Kohaku-TT's channels-last conv (aten/nn/ktt_conv_nhwc): true when conv2d_nhwc()
    // may take a conv of these sizes (one group, stride 1, dilation 1, an odd K x K
    // kernel with padding (K - 1) / 2; own kernels on).
    bool nhwc_conv(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups
    );
    // The conv of x (bf16 [N, C, H, W] stored as Nhwc, or as Nchw under
    // nhwc_propagation()) with w (bf16 [O, C, K, K]) plus bias (bf16 [O], or null):
    // [N, O, H, W] stored as Nhwc; null when the route does not take it. K = 1 runs as
    // the bridge's matmul of the Nhwc rows [N H W, C] with [C, O]. When no plan fits
    // the circular-buffer budget on the stream, TT-NN's conv2d of the Nhwc rows runs
    // instead. keep_input_t (K >= 3): the conv also writes X^T (x's Nhwc tiles
    // transposed) and keeps it for conv2d_nhwc_weight_grad() of the same x.
    HandlePtr conv2d_nhwc(
        const HandlePtr& x,
        const HandlePtr& w,
        const HandlePtr& bias,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        bool keep_input_t = false
    );
    // The input gradient [N, C, H, W] of a conv2d_nhwc() conv with w (bf16 [O, C, K,
    // K]) from go (bf16 [N, O, H, W], stored as Nhwc or as Nchw under
    // nhwc_propagation()): the same channels-last conv with w's taps reversed and its
    // channels swapped, stored as Nhwc; null when the route does not take it.
    HandlePtr conv2d_nhwc_input_grad(const HandlePtr& go, const HandlePtr& w);
    // Channels-last convs run since load: own-program launches (counted on the stream)
    // and 1x1 matmuls.
    int64_t nhwc_conv_runs();
    // Convs conv2d_nhwc() sent to conv2d()'s route since load: in a capture without a
    // current prepared weight, or (counted on the stream) with no plan that fits.
    int64_t nhwc_conv_fallbacks();
    // Inputs the channels-last conv's own program moved from L1 to DRAM since load
    // (counted on the stream).
    int64_t nhwc_conv_input_moves();
    // TT-NN conv route, since load (counted on the stream): weight preparations through
    // the host, weight preparations built on device, geometries whose device build
    // equalled TT-NN's prepared weight, and geometries where it did not.
    std::vector<int64_t> conv_weight_preps();
    // The weight gradient of a conv2d_nhwc() conv (w: its weight sizes [O, C, K, K], K
    // >= 3) from go (bf16 [N, O, H, W]) and its input x (bf16 [N, C, H, W]), each
    // stored as Nhwc or converted in the op, on aten/nn/ktt_conv_nhwc_wgrad; null when
    // the route does not take it. weight_rows: bf16 [O, C K K], a Flat weight's stored
    // rows; else the program's own rows, bf16 [Ct K K 32, Nt 32], row (cb K K + t) 32 +
    // c' and column o holding dW[o, 32 cb + c', t]. X^T is the one conv2d_nhwc() kept
    // for x, else made by its own program.
    HandlePtr conv2d_nhwc_weight_grad(
        const HandlePtr& go,
        const HandlePtr& x,
        const std::vector<int64_t>& w,
        bool weight_rows = true
    );
    // Launches of the channels-last conv's weight-gradient program, and of the X^T
    // program, since load (counted on the stream).
    int64_t nhwc_wgrad_runs();
    int64_t nhwc_xt_rotates();
    // Test hooks (ktt_conv_nhwc_wgrad::debug_plan / last_plan).
    void nhwc_wgrad_debug_plan(
        int64_t go, int64_t bc, int64_t kb, int64_t bo, int64_t iters
    );
    std::string nhwc_wgrad_last_plan();
    // Test hook (ktt_conv_nhwc::debug_plan): with go > 0 the channels-last conv's own
    // program takes only that plan; `iters` passes of its kernels' loops per launch.
    void nhwc_conv_debug_plan(
        int64_t go, int64_t bm_b, int64_t npb, int64_t bn_b, int64_t iters
    );
    // The plan of the last own-program launch of the channels-last conv.
    std::string nhwc_conv_last_plan();
    // The input gradient [N, C, H, W] (sizes x) of a conv2d() of x with w from grad_out
    // go [N, O, OH, OW] on ktt_conv2 (any stride: stride x stride phases);
    // own_grouped_input_grad() is true when it takes these sizes.
    bool own_grouped_input_grad(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        DType weight_format = DType::BFloat16
    );
    HandlePtr conv2d_input_grad(
        const HandlePtr& go,
        const HandlePtr& w,
        const std::vector<int64_t>& x,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        DType weight_format = DType::BFloat16
    );
    // The weight gradient [O, C / groups, KH, KW] (sizes w, bf16) of a conv2d() of x
    // (bf16) from grad_out go on ktt_conv2; own_grouped_weight_grad() is true when it
    // takes these sizes.
    bool own_grouped_weight_grad(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups
    );
    HandlePtr conv2d_weight_grad(
        const HandlePtr& go,
        const HandlePtr& x,
        const std::vector<int64_t>& w,
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
    // Group norm on storage with the channels last (ktt_group_norm's row-form
    // programs): x [N, C, H, W] stored as Nhwc, or x [N, L, C] with L and C multiples
    // of 32 normalised over (L, C / groups); the result in x's shape, dtype and format.
    // Supported for bf16 / fp32 with N = 1 or at least 32 rows per image, groups of up
    // to 192 channels.
    bool group_norm_rows_supported(const HandlePtr& x, int64_t groups);
    HandlePtr group_norm_rows(
        const HandlePtr& x,
        const HandlePtr& weight,
        const HandlePtr& bias,
        int64_t groups,
        double eps
    );
    // The input gradient of group_norm_rows() for x and dy [N, C, H, W] stored as Nhwc
    // (same dtype), mean and rstd per channel (fp32 [N, C]), weight [C] or null: {dx
    // (Nhwc), the weight and the bias gradient terms per (image, channel) as fp32 [N,
    // C]}.
    std::array<HandlePtr, 3> group_norm_rows_backward(
        const HandlePtr& x,
        const HandlePtr& dy,
        const HandlePtr& mean,
        const HandlePtr& rstd,
        const HandlePtr& weight,
        int64_t groups
    );
    // group_norm_rows() with the mean and rstd of each channel's group as fp32 [N, C].
    std::array<HandlePtr, 3> group_norm_rows_sums(
        const HandlePtr& x,
        const HandlePtr& weight,
        const HandlePtr& bias,
        int64_t groups,
        double eps
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

    // Normalization and losses (aten/nn/device_norm.cpp): Kohaku-TT's norm kernels
    // (ktt_norm) over the last dim for bf16 / fp32, TT-NN's moreh kernels otherwise.
    // True when Kohaku-TT's norm kernels run (KTT_CUSTOM_KERNELS is not 0).
    bool own_norm_kernels();

    // Layer norm over the trailing `normalized` device dims (gamma / beta may be null):
    // y, and mean / rstd: fp32 of x's device shape with the last dim 1 (Kohaku-TT's
    // kernels), else in x's dtype and the device shape of x's leading dims.
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

    // RMS norm over the last device dim on Kohaku-TT's kernels: y, and rstd fp32 of x's
    // device shape with the last dim 1 (`mean` null).
    LayerNorm rms_norm_train(const HandlePtr& x, double eps, const HandlePtr& gamma);

    // Requested gradients of rms_norm_train (`beta` null).
    LayerNormGrads rms_norm_backward(
        const HandlePtr& grad,
        const HandlePtr& x,
        const HandlePtr& rstd,
        const HandlePtr& gamma,
        bool want_input,
        bool want_gamma
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
    // True when sdpa_train takes q / k / v with a sequence length that is not a
    // multiple of 32 (ktt_flash's training programs fit). Asked on the stream once per
    // shapes and `is_causal`, then cached; sdpa_train raises when such a call no longer
    // fits.
    bool sdpa_train_ragged(
        const HandlePtr& q, const HandlePtr& k, const HandlePtr& v, bool is_causal
    );
    // Packed variable-length attention (ktt_flash): q [Tq, H D], k [Tk, Hk D], v [Tk,
    // Hk Dv] bf16 hold the tokens of every sequence back to back, a token's `heads` /
    // `kv_heads` heads side by side; `cu_q` / `cu_k` are the cumulative sequence
    // offsets [0, ..., total]. Each query attends to the keys of its sequence
    // (`is_causal`: to its first min(Lk, r + 1 + Lk - Lq) keys). `plan` / `masks`: the
    // offsets' row records (int32 [pages x 32, 32]) and mask tiles (bf16 [tiles x 32,
    // 32]) on the device. Returns [Tq, H Dv] and with `stats` the rows' log-sum-exp
    // [Tq, H 32] (else a null handle). `key_mask`: bf16 [1, Tk] added to the scores of
    // each key (additive, already divided by `scale`, finite), or null. Raises when
    // ktt_flash cannot take the call.
    TrainAttention sdpa_varlen(
        const HandlePtr& q,
        const HandlePtr& k,
        const HandlePtr& v,
        int64_t heads,
        int64_t kv_heads,
        const HandlePtr& plan,
        const HandlePtr& masks,
        const std::vector<int64_t>& cu_q,
        const std::vector<int64_t>& cu_k,
        bool is_causal,
        double scale,
        bool stats,
        const HandlePtr& key_mask = nullptr
    );
    // Gradients {dq, dk, dv} (packed as q, k, v) of sdpa_varlen from its output `out`,
    // its `stats` and the output gradient `grad` [Tq, H Dv]. `key_plan` / `key_masks`:
    // the offsets' key-side records and mask tiles (varlen::key_plan) in the forms of
    // `plan` / `masks`; `key_mask`: sdpa_varlen's. Raises when ktt_flash cannot take
    // the call.
    std::vector<HandlePtr> sdpa_varlen_backward(
        const HandlePtr& grad,
        const HandlePtr& out,
        const HandlePtr& q,
        const HandlePtr& k,
        const HandlePtr& v,
        const HandlePtr& stats,
        int64_t heads,
        int64_t kv_heads,
        const HandlePtr& plan,
        const HandlePtr& masks,
        const HandlePtr& key_plan,
        const HandlePtr& key_masks,
        const std::vector<int64_t>& cu_q,
        const std::vector<int64_t>& cu_k,
        bool is_causal,
        double scale,
        const HandlePtr& key_mask = nullptr
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
    // `probe` tells whether the calling thread runs an autograd backward pass; such a
    // thread without a scope of its own reads the most recently opened scope of any
    // thread.
    void numerics_set_backward_probe(bool (*probe)());

    // ---- user programs (core/device/run_program.cpp, extension.cpp) ----------------

    // Runs the Metalium program described by `spec` (JSON from kohakutt.kernels or
    // kohakutt.dsl.metal) on the default stream. Tensor references in it index
    // `inputs`, then the outputs, which are allocated as TILE tensors of logical sizes
    // `out_shapes` and `out_dtypes`. Common argument {"value": i} takes `values[i]`.
    std::vector<HandlePtr> run_program(
        const std::string& spec,
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        const std::vector<int64_t>& values = {}
    );

    // A program description held for many launches.
    struct UserProgram;
    using UserProgramPtr = std::shared_ptr<const UserProgram>;

    // The UserProgram of `spec`; one per spec text.
    UserProgramPtr user_program(const std::string& spec);

    // run_program() of a held description: no work in the size of its text.
    std::vector<HandlePtr> run_program(
        const UserProgramPtr& program,
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        const std::vector<int64_t>& values = {}
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
