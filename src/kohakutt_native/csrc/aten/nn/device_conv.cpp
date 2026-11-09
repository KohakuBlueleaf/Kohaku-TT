// Bridge convolution: Kohaku-TT's implicit-GEMM convs (ktt_conv: dense stride 1;
// ktt_conv2: grouped / depthwise / strided, their input and weight gradients;
// ktt_conv_nhwc: "same" stride-1 convs on Nhwc storage) and TT-NN's sliding-window conv2d
// on channels-last activations.
#include <atomic>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>

#include "aten/nn/ktt_conv/program.hpp"
#include "aten/nn/ktt_conv2/program.hpp"
#include "aten/nn/ktt_conv_nhwc/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "core/device/program.hpp"
#include "core/device/trace.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ktt::bridge {
    namespace {

        // Prepared weights per (weight handle, conv geometry); the handle is held
        // weakly, so a freed or rebound weight never matches, and `epoch` is the
        // write_epoch() it was prepared at, so contents written in place never match.
        // `remake` re-prepares the weight from its current contents (TT-NN path,
        // entries read inside a capture: `traced`).
        struct Prepared {
            std::weak_ptr<Handle> weight;
            ttnn::Tensor tensor;
            int64_t epoch = -1;
            bool traced = false;
            std::function<ttnn::Tensor()> remake;
        };

        std::map<std::pair<const Handle*, std::string>, Prepared>& prepared() {
            // Never destroyed: device tensors must not be freed after the device
            // closes at exit.
            static auto& cache = *new std::map<std::pair<const Handle*, std::string>, Prepared>;
            return cache;
        }

        uint32_t u32(int64_t v) {
            return static_cast<uint32_t>(v);
        }

        // Stride-1 single-group convs with C a multiple of 32 run Kohaku-TT's
        // implicit-GEMM conv (aten/nn/ktt_conv); KTT_OWN_CONV=0: TT-NN's conv2d.
        bool own_conv_enabled() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_OWN_CONV");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // True unless KTT_CONV_LAYOUT=0 (TT-NN permute / pad / tilize around ktt_conv
        // instead of its layout kernels).
        bool own_layout_enabled() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_CONV_LAYOUT");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // ktt_conv: channels-last zero-padded input rows [N Hp Wp, C] (row-major), the
        // tap-major weight [KH KW C, O] (cached per weight handle: no size in it),
        // output [N OH Wp, O] cropped to OW columns and back to [N, O, OH, OW].
        ttnn::Tensor own_conv(
            const HandlePtr& x,
            const HandlePtr& w,
            const ktt_conv::Geometry& g,
            int64_t ph,
            int64_t pw,
            int64_t ow,
            const ttnn::ComputeKernelConfig& config,
            int64_t bytes,
            int64_t epoch,
            bool in_capture
        ) {
            using tt::tt_metal::Layout;
            const bool own_layout =
                (own_layout_enabled() && x->tensor.layout() == Layout::TILE &&
                 x->tensor.dtype() == tt::tt_metal::DataType::BFLOAT16);
            ttnn::Tensor rows;
            if (own_layout) {
                rows = ktt_conv::to_rows(x->tensor, g, u32(ph), u32(pw));
            } else {
                auto nhwc = ttnn::to_layout(
                    ttnn::permute(
                        x->tensor,
                        ttsl::SmallVector<int64_t>{0, 2, 3, 1},
                        tt::tt_metal::MemoryConfig{}
                    ),
                    Layout::ROW_MAJOR
                );
                if (ph > 0 || pw > 0) {
                    ttsl::SmallVector<std::array<uint32_t, 2>> widths{
                        {0, 0}, {u32(ph), u32(ph)}, {u32(pw), u32(pw)}, {0, 0}
                    };
                    nhwc = ttnn::pad(nhwc, widths, 0.0f);
                }
                rows = ttnn::reshape(nhwc, ttnn::Shape({g.n * g.hp * g.wp, g.c}));
            }

            auto& cache = prepared();
            const auto key = std::make_pair(w.get(), std::string("own"));
            // In a capture the weight is prepared inside the trace, so a replay reads
            // the weight's current contents.
            auto it = cache.find(key);
            const bool hit =
                (!in_capture && it != cache.end() && !it->second.weight.expired() &&
                 it->second.epoch == epoch);
            ttnn::Tensor matrix;
            if (hit) {
                matrix = it->second.tensor;
            } else {
                auto taps = ttnn::permute(
                    w->tensor, ttsl::SmallVector<int64_t>{2, 3, 1, 0}
                );
                matrix = ttnn::reshape(taps, ttnn::Shape({g.kh * g.kw * g.c, g.o}));
                if (!in_capture) {
                    cache.insert_or_assign(key, Prepared{w, matrix, epoch});
                }
            }

            auto out = ktt_conv::run(rows, matrix, g, config);
            if (own_layout) {
                return (
                    ktt_conv::to_nchw(out, g, u32(ow), detail::activation_config(bytes))
                );
            }
            auto grid = ttnn::reshape(
                ttnn::to_layout(out, Layout::ROW_MAJOR),
                ttnn::Shape({g.n, g.oh, g.wp, g.o})
            );
            if (u32(ow) != g.wp) {
                grid = ttnn::slice(
                    grid,
                    ttsl::SmallVector<int32_t>{0, 0, 0, 0},
                    ttsl::SmallVector<int32_t>{
                        static_cast<int32_t>(g.n),
                        static_cast<int32_t>(g.oh),
                        static_cast<int32_t>(ow),
                        static_cast<int32_t>(g.o)
                    },
                    ttsl::SmallVector<int32_t>{1, 1, 1, 1}
                );
            }
            auto nchw = ttnn::permute(grid, ttsl::SmallVector<int64_t>{0, 3, 1, 2});
            return (
                ttnn::to_layout(
                    nchw, Layout::TILE, std::nullopt, detail::activation_config(bytes)
                )
            );
        }

        // The sparse weight tiles' format for weights stored as `stored`: bfp8 stays
        // bfp8; bfp4 is held as bf16, exactly the stored values (re-quantizing it into
        // the tiles' blocks would add a second quantization).
        DType sparse_format(DType stored) {
            return stored == DType::BFloat8B ? DType::BFloat8B : DType::BFloat16;
        }

        // Block-float sparse tiles for groups of 32+ input channels; groups of fewer keep
        // the stored values as bf16 (their tiles would be requantized in other blocks).
        bool block_float(DType format, const ktt_conv2::Geometry& g) {
            return sparse_format(format) != DType::BFloat16 && g.c / g.groups >= 32;
        }

        uint32_t weight_tile_bytes(DType format, const ktt_conv2::Geometry& g) {
            return block_float(format, g) && sparse_format(format) == DType::BFloat8B ? 1088
                                                                                      : 2048;
        }

        // HiFi4 at least: bf16 products exact (the FPU does not bind these kernels).
        ttnn::ComputeKernelConfig exact_config() {
            return numerics::compute_config(
                numerics::effective({.floor = tt::tt_metal::MathFidelity::HiFi4}), true
            );
        }

        ttnn::Tensor tiled(const ttnn::Tensor& t) {
            using tt::tt_metal::Layout;
            return t.layout() == Layout::TILE ? t : ttnn::to_layout(t, Layout::TILE);
        }

        // Sizes of a 1-D conv operand [N, C, L] as 2-D [N, C, 1, L]; 2-D sizes as given.
        std::vector<int64_t> four(const std::vector<int64_t>& s) {
            return s.size() == 3 ? std::vector<int64_t>{s[0], s[1], 1, s[2]} : s;
        }

        // The zero-padded channels-last rows of x ([N, C, H, W], or [N, C, L] when
        // one_d) over an input extent hp x wp.
        ttnn::Tensor gather_rows(
            const ttnn::Tensor& x,
            bool one_d,
            uint32_t n,
            uint32_t c,
            uint32_t hp,
            uint32_t wp,
            std::array<uint32_t, 2> pad
        ) {
            if (one_d) {
                return ktt_conv::to_rows_1d(tiled(x), n, c, wp, pad[1]);
            }
            const ktt_conv::Geometry rows_g{n, c, hp, wp, c, 1, 1, 1, 1, hp};
            return ktt_conv::to_rows(tiled(x), rows_g, pad[0], pad[1]);
        }

        // A gather's output rows as NCHW, or NCL when one_d.
        ttnn::Tensor gather_output(
            const ttnn::Tensor& out,
            const ktt_conv::OutRows& layout,
            bool one_d,
            int64_t bytes
        ) {
            const auto memory = detail::activation_config(bytes);
            return one_d ? ktt_conv::to_ncl(out, layout, memory)
                         : ktt_conv::to_nchw(out, layout, memory);
        }

        // ktt_conv2's sparse weight tiles of w for g (cached per weight handle, format,
        // direction and kernel geometry; prepared inside a capture so a replay reads the
        // weight's current contents).
        ttnn::Tensor sparse_weights(
            const HandlePtr& w,
            const ktt_conv2::Geometry& g,
            bool transposed,
            DType format,
            int64_t epoch,
            bool in_capture
        ) {
            const bool one_d = w->shape.size() == 3;
            std::string kernel = (transposed ? "gather-t:" : "gather:") +
                std::to_string(static_cast<int>(sparse_format(format))) +
                (one_d ? ":1d" : ":2d");
            for (const int64_t v : {g.c, g.o, g.kh, g.kw, g.groups}) {
                kernel += "," + std::to_string(v);
            }
            auto& cache = prepared();
            const auto key = std::make_pair(w.get(), kernel);
            auto it = cache.find(key);
            if (!in_capture && it != cache.end() && !it->second.weight.expired() &&
                it->second.epoch == epoch) {
                return it->second.tensor;
            }
            auto weight = tiled(w->tensor);
            if (weight.dtype() != tt::tt_metal::DataType::BFLOAT16) {
                weight = ttnn::typecast(weight, tt::tt_metal::DataType::BFLOAT16);
            }
            auto sparse = ktt_conv2::prepare(weight, g, transposed, one_d);
            if (block_float(format, g)) {
                sparse = ttnn::typecast(sparse, detail::to_ttnn(sparse_format(format)));
            }
            if (!in_capture) {
                cache.insert_or_assign(key, Prepared{w, sparse, epoch});
            }
            return sparse;
        }

        // KTT_GROUPED_CONV (read per call): "own" (default) runs grouped, depthwise and
        // strided 1-D convs and their gradients on ktt_conv2, "ttnn" the forward on
        // TT-NN's conv2d, "composite" the per-tap composite.
        std::string grouped_route() {
            const char* v = std::getenv("KTT_GROUPED_CONV");
            return v ? std::string(v) : std::string("own");
        }

        bool own_grouped_enabled() {
            return program::own_kernels_enabled() && grouped_route() == "own";
        }

        // The convs ktt_conv2 claims: grouped and depthwise; one-group convs over one
        // input row with a stride.
        bool grouped_kind(
            const std::vector<int64_t>& x,
            const std::vector<int64_t>& w,
            std::array<int64_t, 2> stride,
            int64_t groups
        ) {
            if (x.size() != 4 || w.size() != 4 || groups < 1 || x[0] == 0 ||
                x[1] % groups != 0 || w[0] % groups != 0 || x[1] / groups != w[1]) {
                return false;
            }
            const bool strided_1d =
                (groups == 1 && x[2] == 1 && w[2] == 1 && (stride[0] > 1 || stride[1] > 1));
            return groups > 1 || strided_1d;
        }

        // A conv on TT-NN's conv2d: input sizes [N, C, H, W] and dtype, conv arguments,
        // and the compute config read where it was queued.
        struct TtnnConv {
            std::vector<int64_t> xs;
            DType x_dtype;
            std::array<int64_t, 2> stride, padding, dilation;
            int64_t groups;
            ttnn::ComputeKernelConfig config;
        };

        // TT-NN's conv2d of x (Nchw [N, C, H, W], or the Nhwc rows when nhwc_in) with w,
        // on the stream worker: TILE [1, 1, N OH OW, O]. The prepared weight is cached in
        // prepared() per weight handle and geometry (epoch / in_capture: write_epoch()
        // and capturing() where the conv was queued).
        ttnn::Tensor ttnn_conv_rows(
            const ttnn::Tensor& x,
            bool nhwc_in,
            const HandlePtr& w,
            const TtnnConv& tc,
            int64_t epoch,
            bool in_capture
        ) {
            const int64_t n = tc.xs[0], c = tc.xs[1], h = tc.xs[2], wd = tc.xs[3];
            const int64_t o = w->shape[0], kh = w->shape[2], kw = w->shape[3];
            const auto& stride = tc.stride;
            const auto& padding = tc.padding;
            const auto& dilation = tc.dilation;
            const int64_t groups = tc.groups;
            const auto config = tc.config;
            // Kernel geometry (prepared weights of the height-sharded config depend only
            // on it) and the full geometry (TT-NN's own config: its weights depend on H,
            // W).
            std::string kernel_key;
            for (
                int64_t v :
                {
                    c,
                    o,
                    kh,
                    kw,
                    stride[0],
                    stride[1],
                    padding[0],
                    padding[1],
                    dilation[0],
                    dilation[1],
                    groups
                }
            ) {
                kernel_key += std::to_string(v) + ',';
            }
            const std::string full_key =
                (kernel_key + std::to_string(n) + ',' + std::to_string(h) + ',' +
                 std::to_string(wd));
            const auto dtype = detail::to_ttnn(tc.x_dtype);
            // Channels last in DRAM: TT-NN's DRAM path slices large activations.
            auto nhwc = (
                !nhwc_in ? ttnn::permute(
                               x,
                               ttsl::SmallVector<int64_t>{0, 2, 3, 1},
                               tt::tt_metal::MemoryConfig{}
                           )
                : x.memory_config().is_l1()
                    ? ttnn::to_memory_config(x, tt::tt_metal::MemoryConfig{})
                    : x
            );
            // Shapes without a valid height-sharded config (never retried).
            static auto& unpinned = *new std::set<std::string>;
            // conv2d of `input` with `weight` (prepared, or host ROW_MAJOR): the
            // output and the prepared weight.
            auto conv = [=](const ttnn::Tensor& input,
                            const ttnn::Tensor& weight,
                            bool pinned) -> std::pair<ttnn::Tensor, ttnn::Tensor> {
                ttnn::Conv2dConfig conv_config;
                conv_config.weights_dtype = tt::tt_metal::DataType::BFLOAT16;
                conv_config.act_block_h_override = 32;
                if (pinned) {
                    conv_config.shard_layout = (
                        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED
                    );
                }
                auto result = ttnn::conv2d(
                    input,
                    weight,
                    detail::device(),
                    u32(c),
                    u32(o),
                    u32(n),
                    u32(h),
                    u32(wd),
                    {u32(kh), u32(kw)},
                    {u32(stride[0]), u32(stride[1])},
                    std::array<uint32_t, 2>{u32(padding[0]), u32(padding[1])},
                    {u32(dilation[0]), u32(dilation[1])},
                    u32(groups),
                    dtype,
                    std::nullopt,
                    conv_config,
                    config,
                    tt::tt_metal::MemoryConfig{},
                    std::nullopt,
                    false,  // return_output_dim
                    true    // return_weights_and_bias
                );
                auto& [out, weights] = std::get<std::tuple<
                    ttnn::Tensor,
                    std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>>>>(result);
                return {out, std::get<0>(weights)};
            };
            auto host_weight = [](const HandlePtr& weight) {
                return (
                    ttnn::to_layout(
                        ttnn::operations::core::from_device(weight->tensor),
                        tt::tt_metal::Layout::ROW_MAJOR
                    )
                );
            };
            auto run = [&](bool pinned) {
                auto& cache = prepared();
                const auto key = (
                    std::make_pair(
                        w.get(), (pinned ? "h:" + kernel_key : "a:" + full_key)
                    )
                );
                // Prepared through the host, which cannot run in a capture: a
                // capture reads the weights cached before it, refreshed before
                // every replay.
                auto it = cache.find(key);
                const bool hit =
                    (it != cache.end() && !it->second.weight.expired() &&
                     (it->second.epoch == epoch || in_capture));
                if (hit) {
                    if (in_capture && !it->second.traced) {
                        it->second.traced = true;
                        const auto spec = nhwc.tensor_spec();
                        std::weak_ptr<Handle> weak = w;
                        it->second.remake = [=]() -> ttnn::Tensor {
                            auto weight = weak.lock();
                            if (!weight) {
                                return ttnn::Tensor();
                            }
                            auto input = (
                                ttnn::empty(
                                    spec.logical_shape(),
                                    spec.data_type(),
                                    spec.layout(),
                                    detail::device(),
                                    spec.memory_config()
                                )
                            );
                            return conv(input, host_weight(weight), pinned).second;
                        };
                    }
                    return conv(nhwc, it->second.tensor, pinned).first;
                }
                auto [out, weights] = conv(nhwc, host_weight(w), pinned);
                cache[key] = Prepared{w, weights, epoch};
                return out;
            };
            ttnn::Tensor out;
            bool done = false;
            if (unpinned.count(full_key) == 0) {
                try {
                    out = run(true);
                    done = true;
                } catch (const std::exception& e) {
                    // No height-sharded config or slicing for this shape: TT-NN's
                    // own config from now on (L1_SMALL exhaustion is retried by
                    // timed_op instead).
                    if (detail::is_l1_small_full(e)) {
                        throw;
                    }
                    unpinned.insert(full_key);
                }
            }
            if (!done) {
                out = run(false);
            }
            return ttnn::to_layout(out, tt::tt_metal::Layout::TILE);
        }

        // Diagnostic, KTT_NHWC_CONV_L1=1 (read per call): the channels-last conv's own
        // program reads X and writes its output where the activation policy puts them
        // (L1 or DRAM) instead of DRAM.
        bool nhwc_l1_placement() {
            const char* v = std::getenv("KTT_NHWC_CONV_L1");
            return v && std::string(v) == "1";
        }

        // Channels-last convs run (nhwc_conv_runs()): own-program launches and 1x1
        // matmuls; those sent to conv2d()'s route (nhwc_conv_fallbacks()); inputs moved
        // from L1 to DRAM for the own program (nhwc_conv_input_moves()).
        std::atomic<int64_t> g_nhwc_runs{0};
        std::atomic<int64_t> g_nhwc_fallbacks{0};
        std::atomic<int64_t> g_nhwc_input_moves{0};

        // A prepared weight of the channels-last conv (W', or the 1x1 matmul weight) of
        // `weight` (held weakly) at its Handle::writes `writes`; `traced` once a capture
        // read it (then refreshed in place, never replaced); `make` builds a new one from
        // the weight's current values.
        struct NhwcWeight {
            std::weak_ptr<Handle> weight;
            HandlePtr prepared;
            int64_t writes = -1;
            bool traced = false;
            std::function<HandlePtr(const HandlePtr&)> make;
        };

        std::mutex& nhwc_weights_mutex() {
            static std::mutex m;
            return m;
        }

        // Per (weight handle, kind); traced entries whose weight was freed keep their
        // buffer in nhwc_retired() until close.
        std::map<std::pair<const Handle*, std::string>, NhwcWeight>& nhwc_weights() {
            static auto& cache = (
                *new std::map<std::pair<const Handle*, std::string>, NhwcWeight>
            );
            return cache;
        }

        std::vector<HandlePtr>& nhwc_retired() {
            static auto& retired = *new std::vector<HandlePtr>;
            return retired;
        }

        // A prepared weight of w's current values built on the host: w downloaded (a
        // buffer read), reordered by `reorder` (bf16 values), uploaded as device shape
        // `shape`. Not inside a capture.
        HandlePtr host_prepared(
            const HandlePtr& w,
            const std::vector<int64_t>& shape,
            const std::function<std::vector<uint16_t>(const std::vector<uint16_t>&)>& reorder
        ) {
            int64_t count = 1;
            for (const int64_t d : w->shape) {
                count *= d;
            }
            std::vector<uint16_t> values(static_cast<size_t>(count));
            download(w, values.data());
            const auto out = reorder(values);
            return upload(out.data(), shape, DType::BFloat16);
        }

        // Queues a copy of a new prepared weight of w's current values into `prepared`'s
        // buffer.
        void refresh_in_place(
            const HandlePtr& w, const HandlePtr& prepared, const NhwcWeight& entry
        ) {
            const auto fresh = entry.make(w);
            detail::counters().ops++;
            detail::submit([fresh, prepared] {
                ttnn::copy(fresh->tensor, prepared->tensor);
            });
        }

        // The prepared weight of w for `kind` (built by `make`), cached per weight handle
        // and its Handle::writes; null when the conv takes conv2d()'s route.
        // - A prepare pass or capture that wrote w earlier in the pass: null (both passes
        //   decide alike, so the prepare pass compiles what the capture runs).
        // - A capture reads the entry, stale or not, and marks it traced: a replay is
        //   preceded by an in-place refresh when w was written; without an entry, null.
        // - Otherwise (prepare pass included) a stale traced entry is refreshed in place,
        //   and other misses are built and cached.
        HandlePtr nhwc_weight(
            const HandlePtr& w,
            const std::string& kind,
            std::function<HandlePtr(const HandlePtr&)> make
        ) {
            if ((tracing() || capturing()) && written_in_pass(w)) {
                return nullptr;
            }
            const std::lock_guard lock(nhwc_weights_mutex());
            auto& cache = nhwc_weights();
            const auto key = std::make_pair(w.get(), kind);
            const int64_t writes = w->writes.load();
            const bool capture = capturing();
            auto it = cache.find(key);
            if (it != cache.end() && it->second.weight.lock() != w) {
                // Made for a freed handle at the same address.
                if (it->second.traced) {
                    nhwc_retired().push_back(it->second.prepared);
                }
                cache.erase(it);
                it = cache.end();
            }
            if (it != cache.end()) {
                auto& entry = it->second;
                if (entry.writes == writes || capture) {
                    entry.traced = entry.traced || capture;
                    return entry.prepared;
                }
                if (entry.traced) {
                    refresh_in_place(w, entry.prepared, entry);
                    entry.writes = writes;
                    return entry.prepared;
                }
            }
            if (capture) {
                return nullptr;
            }
            auto prepared = make(w);
            cache.insert_or_assign(
                key, NhwcWeight{w, prepared, writes, false, std::move(make)}
            );
            return prepared;
        }

        // y with bias [1, O] added as a [1, 1, 1, O] row, its tile padding back to zero
        // (y: Nhwc storage [1, 1, npix, O]).
        ttnn::Tensor add_bias(
            const ttnn::Tensor& y,
            const ttnn::Tensor& bias,
            int64_t npix,
            int64_t o,
            const tt::tt_metal::MemoryConfig& memory
        ) {
            const auto row = ttnn::reshape(
                bias, ttnn::Shape({1, 1, 1, static_cast<uint32_t>(o)})
            );
            auto out = ttnn::add(y, row, std::nullopt, memory);
            return detail::has_padding({1, 1, npix, o}) ? detail::zero_padding(out) : out;
        }

        // x as Nhwc: x itself when stored so, else a new handle converted on the stream into
        // DRAM (not cached, not counted as a layout conversion), as TT-NN's route does.
        HandlePtr nhwc_input(const HandlePtr& x) {
            if (x->format == Format::Nhwc) {
                return x;
            }
            const auto shape = x->shape;
            return detail::timed_op(
                x->dtype,
                shape,
                [x, shape] {
                    auto t = ttnn::permute(
                        x->tensor,
                        ttsl::SmallVector<int64_t>{0, 2, 3, 1},
                        tt::tt_metal::MemoryConfig{},
                        0.0f
                    );
                    return ttnn::reshape(
                        t,
                        detail::device_shape(detail::stored_dims(shape, Format::Nhwc)),
                        std::nullopt,
                        ttnn::PadValue(0.0f)
                    );
                },
                Format::Nhwc
            );
        }

        // `a`'s tensor as a handle of device shape `shape` stored as `format` (no copy).
        HandlePtr relabel(
            const HandlePtr& a, const std::vector<int64_t>& shape, Format format
        ) {
            return detail::timed_op(
                a->dtype,
                shape,
                [a] {
                    return a->tensor;
                },
                format
            );
        }

    }  // namespace

    bool own_grouped_conv(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        DType weight_format
    ) {
        const auto xs = four(x), ws = four(w);
        if (!own_grouped_enabled() || x.size() != w.size() ||
            !grouped_kind(xs, ws, stride, groups)) {
            return false;
        }
        // The forward, its input gradient and its weight gradient all fit: a conv is
        // claimed for training as a whole.
        std::array<uint32_t, 2> pad;
        const auto g = ktt_conv2::forward(xs, ws, stride, padding, dilation, groups, pad);
        if (g.h.out == 0 || g.w.out == 0) {
            return false;
        }
        const auto gi = ktt_conv2::input_grad(xs, ws, stride, padding, dilation, groups, pad);
        return ktt_conv2::plan(g, weight_tile_bytes(weight_format, g)).has_value() &&
            ktt_conv2::plan(gi, weight_tile_bytes(weight_format, gi)).has_value() &&
            ktt_conv2::weight_grad_plan(g).has_value();
    }

    bool ttnn_grouped_conv(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        int64_t groups
    ) {
        return grouped_route() == "ttnn" &&
            grouped_kind(four(x), four(w), stride, groups);
    }

    bool own_grouped_input_grad(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        DType weight_format
    ) {
        if (!own_grouped_conv(x, w, stride, padding, dilation, groups, weight_format)) {
            return false;
        }
        std::array<uint32_t, 2> pad;
        const auto g = ktt_conv2::input_grad(
            four(x), four(w), stride, padding, dilation, groups, pad
        );
        return ktt_conv2::plan(g, weight_tile_bytes(weight_format, g)).has_value();
    }

    HandlePtr conv2d_input_grad(
        const HandlePtr& go,
        const HandlePtr& w,
        const std::vector<int64_t>& x,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        DType weight_format
    ) {
        const bool one_d = x.size() == 3;
        const auto xs = four(x);
        std::array<uint32_t, 2> pad;
        const auto g = ktt_conv2::input_grad(
            xs, four(w->shape), stride, padding, dilation, groups, pad
        );
        const int64_t bytes = detail::padded_bytes(x, go->dtype);
        const int64_t epoch = write_epoch();
        const bool in_capture = capturing();
        return detail::timed_op(go->dtype, x, [=] {
            auto rows = gather_rows(go->tensor, one_d, g.n, g.c, g.h.in, g.w.in, pad);
            auto sparse = sparse_weights(w, g, true, weight_format, epoch, in_capture);
            auto out = ktt_conv2::run(rows, sparse, g, exact_config());
            const ktt_conv::OutRows layout{
                g.n, g.o, u32(xs[2]), u32(xs[3]), g.pitch, g.h.out,
                g.h.phases(), g.w.phases(), g.phase_tiles() * 32
            };
            return gather_output(out, layout, one_d, bytes);
        });
    }

    bool own_grouped_weight_grad(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups
    ) {
        if (!own_grouped_conv(x, w, stride, padding, dilation, groups, DType::BFloat16)) {
            return false;
        }
        std::array<uint32_t, 2> pad;
        const auto g = ktt_conv2::forward(
            four(x), four(w), stride, padding, dilation, groups, pad
        );
        return ktt_conv2::weight_grad_plan(g).has_value();
    }

    HandlePtr conv2d_weight_grad(
        const HandlePtr& go,
        const HandlePtr& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups
    ) {
        const bool one_d = x->shape.size() == 3;
        std::array<uint32_t, 2> pad;
        const auto g = ktt_conv2::forward(
            four(x->shape), four(w), stride, padding, dilation, groups, pad
        );
        return detail::timed_op(DType::BFloat16, w, [=] {
            auto x_rows = gather_rows(x->tensor, one_d, g.n, g.c, g.h.in, g.w.in, pad);
            // grad_out rows over the output pitch (zero past OW).
            auto g_rows = gather_rows(
                go->tensor, one_d, g.n, g.o, g.h.out, g.pitch, {0, 0}
            );
            return ktt_conv2::weight_grad(x_rows, g_rows, g, exact_config(), one_d);
        });
    }

    namespace detail {

        void refresh_traced_conv_weights(int64_t epoch) {
            for (auto& [key, entry] : prepared()) {
                if (!entry.traced || !entry.remake || entry.weight.expired() ||
                    entry.epoch == epoch) {
                    continue;
                }
                auto fresh = entry.remake();
                if (fresh.is_allocated()) {
                    ttnn::copy(fresh, entry.tensor);
                    entry.epoch = epoch;
                }
            }
        }

    }  // namespace detail

    HandlePtr conv2d(
        const HandlePtr& x,
        const HandlePtr& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        DType weight_format
    ) {
        // Kohaku-TT's convs read Nchw; TT-NN's conv2d reads an Nhwc input as stored and,
        // under nhwc_propagation(), returns its Nhwc output.
        const bool nhwc_out = nhwc_propagation();
        if (own_grouped_conv(
                x->shape, w->shape, stride, padding, dilation, groups, weight_format
            )) {
            const auto xn = to_nchw(x);
            const bool one_d = x->shape.size() == 3;
            std::array<uint32_t, 2> pad;
            const auto gg = ktt_conv2::forward(
                four(x->shape), four(w->shape), stride, padding, dilation, groups, pad
            );
            const std::vector<int64_t> out_shape = one_d
                ? std::vector<int64_t>{gg.n, gg.o, gg.w.out}
                : std::vector<int64_t>{gg.n, gg.o, gg.h.out, gg.w.out};
            const int64_t out_bytes = detail::padded_bytes(out_shape, x->dtype);
            const int64_t epoch = write_epoch();
            const bool in_capture = capturing();
            return detail::timed_op(x->dtype, out_shape, [=] {
                auto rows = gather_rows(
                    xn->tensor, one_d, gg.n, gg.c, gg.h.in, gg.w.in, pad
                );
                auto sparse = sparse_weights(
                    w, gg, false, weight_format, epoch, in_capture
                );
                auto out = ktt_conv2::run(rows, sparse, gg, exact_config());
                const ktt_conv::OutRows layout{
                    gg.n, gg.o, gg.h.out, gg.w.out, gg.pitch, gg.h.out,
                    1, 1, gg.phase_tiles() * 32
                };
                return gather_output(out, layout, one_d, out_bytes);
            });
        }
        const int64_t n = x->shape[0], c = x->shape[1], h = x->shape[2],
                      wd = x->shape[3];
        const int64_t o = w->shape[0], kh = w->shape[2], kw = w->shape[3];
        const int64_t oh = (
            (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) / stride[0] + 1
        );
        const int64_t ow = (
            (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) / stride[1] + 1
        );
        const auto config = numerics::compute_config(numerics::effective(), true);
        const std::vector<int64_t> shape{n, o, oh, ow};
        const int64_t bytes = detail::padded_bytes(shape, x->dtype);
        const ktt_conv::Geometry g{
            u32(n),
            u32(c),
            u32(h + 2 * padding[0]),
            u32(wd + 2 * padding[1]),
            u32(o),
            u32(kh),
            u32(kw),
            u32(dilation[0]),
            u32(dilation[1]),
            u32(oh)
        };
        if (!nhwc_out && own_conv_enabled() && groups == 1 && stride[0] == 1 &&
            stride[1] == 1 && c % 32 == 0 && ktt_conv::columns_per_core(g)) {
            const int64_t ph = padding[0], pw = padding[1];
            const int64_t epoch = write_epoch();
            const bool in_capture = capturing();
            const auto xn = to_nchw(x);
            return detail::timed_op(x->dtype, shape, [=] {
                return own_conv(xn, w, g, ph, pw, ow, config, bytes, epoch, in_capture);
            });
        }
        const TtnnConv tc{x->shape, x->dtype, stride, padding, dilation, groups, config};
        const int64_t epoch = write_epoch();
        const bool in_capture = capturing();
        const bool nhwc_in = x->format == Format::Nhwc;
        const auto run_conv = [=] {
            return ttnn_conv_rows(x->tensor, nhwc_in, w, tc, epoch, in_capture);
        };
        if (nhwc_out) {
            // TT-NN's [1, 1, N OH OW, O] as stored, tile padding canonical and zero.
            const std::vector<int64_t> rows{1, 1, n * oh * ow, o};
            return detail::timed_op(
                x->dtype,
                shape,
                [=] {
                    auto out = ttnn::reshape(run_conv(), detail::device_shape(rows));
                    return detail::has_padding(rows) ? detail::zero_padding(out) : out;
                },
                Format::Nhwc
            );
        }
        return detail::timed_op(x->dtype, shape, [=] {
            // [1, 1, N OH OW, O] -> [N, OH, OW, O] -> [N, O, OH, OW], tiled (a
            // row-major detour measured slower cold).
            auto rows = ttnn::reshape(
                run_conv(), ttnn::Shape({u32(n), u32(oh), u32(ow), u32(o)})
            );
            return ttnn::permute(
                rows,
                ttsl::SmallVector<int64_t>{0, 3, 1, 2},
                detail::activation_config(bytes)
            );
        });
    }

    namespace {

        // w's values [O, C, K, K] as [C, O, K, K] with each kernel's taps reversed: the
        // weight of the conv that gives the input gradient (stride 1, "same" padding).
        std::vector<uint16_t> flip_swap(
            const std::vector<uint16_t>& v, int64_t o, int64_t c, int64_t k
        ) {
            std::vector<uint16_t> out(v.size());
            const int64_t taps = k * k;
            for (int64_t oo = 0; oo < o; ++oo) {
                for (int64_t cc = 0; cc < c; ++cc) {
                    for (int64_t t = 0; t < taps; ++t) {
                        out[(cc * o + oo) * taps + taps - 1 - t] = v[(oo * c + cc) * taps + t];
                    }
                }
            }
            return out;
        }

    }  // namespace

    // The channels-last conv of geometry g: x (Nhwc, or Nchw converted in the op) with the
    // prepared weight `weight` (W', or the 1x1 matmul weight) plus bias (or null):
    // [N, O, H, W] stored as Nhwc. `conv_w` [O, C, K, K] is TT-NN conv2d's weight when no
    // plan fits.
    static HandlePtr nhwc_apply(
        const HandlePtr& x,
        const HandlePtr& weight,
        const HandlePtr& conv_w,
        const HandlePtr& bias,
        const ktt_conv_nhwc::Geometry& g
    );

    bool nhwc_conv(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups
    ) {
        if (!program::own_kernels_enabled() || x.size() != 4 || w.size() != 4 ||
            groups != 1) {
            return false;
        }
        const int64_t k = w[2];
        return x[0] > 0 && x[1] > 0 && x[2] > 0 && x[3] > 0 && w[0] > 0 && w[1] == x[1] &&
            w[3] == k && k % 2 == 1 && stride[0] == 1 && stride[1] == 1 &&
            dilation[0] == 1 && dilation[1] == 1 && padding[0] == (k - 1) / 2 &&
            padding[1] == (k - 1) / 2;
    }

    HandlePtr conv2d_nhwc(
        const HandlePtr& x,
        const HandlePtr& w,
        const HandlePtr& bias,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups
    ) {
        const auto bf16 = DType::BFloat16;
        if (!nhwc_conv(x->shape, w->shape, stride, padding, dilation, groups) ||
            (x->format != Format::Nhwc && !nhwc_propagation()) || x->dtype != bf16 ||
            w->dtype != bf16 || !x->placement.is_replicated() ||
            !w->placement.is_replicated() ||
            (bias && (bias->dtype != bf16 || !bias->placement.is_replicated()))) {
            return nullptr;
        }
        const int64_t n = x->shape[0], c = x->shape[1], h = x->shape[2],
                      wd = x->shape[3], o = w->shape[0], k = w->shape[2];
        const ktt_conv_nhwc::Geometry g{u32(n), u32(c), u32(h), u32(wd), u32(o), u32(k)};
        // The prepared weight, built on the host from the weight's values; a capture
        // without a current one takes conv2d()'s route.
        const auto weight = k == 1
            ? nhwc_weight(
                  w,
                  "1x1:" + std::to_string(c) + "," + std::to_string(o),
                  [g, c, o](const HandlePtr& wt) {
                      return host_prepared(wt, {c, o}, [g](const auto& v) {
                          return ktt_conv_nhwc::matmul_weight(v, g);
                      });
                  }
              )
            : nhwc_weight(
                  w,
                  "w':" + std::to_string(c) + "," + std::to_string(o) + "," +
                      std::to_string(k),
                  [g, c, o, k](const HandlePtr& wt) {
                      const std::vector<int64_t> shape{
                          (c + 31) / 32 * k * k * 32, (o + 31) / 32 * 32
                      };
                      return host_prepared(wt, shape, [g](const auto& v) {
                          return ktt_conv_nhwc::prepare(v, g);
                      });
                  }
              );
        if (!weight) {
            g_nhwc_fallbacks++;
            return nullptr;
        }
        return nhwc_apply(x, weight, w, bias, g);
    }

    HandlePtr conv2d_nhwc_input_grad(const HandlePtr& go, const HandlePtr& w) {
        const auto bf16 = DType::BFloat16;
        if (!program::own_kernels_enabled() || go->shape.size() != 4 || w->shape.size() != 4 ||
            go->dtype != bf16 || w->dtype != bf16 ||
            (go->format != Format::Nhwc && !nhwc_propagation()) ||
            !go->placement.is_replicated() || !w->placement.is_replicated() ||
            go->shape[1] != w->shape[0] || w->shape[2] % 2 == 0 || w->shape[3] != w->shape[2]) {
            return nullptr;
        }
        const int64_t n = go->shape[0], o = go->shape[1], h = go->shape[2],
                      wd = go->shape[3], c = w->shape[1], k = w->shape[2];
        // A conv from O to C channels with the flipped, swapped weight.
        const ktt_conv_nhwc::Geometry g{u32(n), u32(o), u32(h), u32(wd), u32(c), u32(k)};
        const std::string key = std::to_string(c) + "," + std::to_string(o) + "," + std::to_string(k);
        const auto weight = k == 1
            ? nhwc_weight(
                  w,
                  "input grad 1x1:" + key,
                  [g, o, c](const HandlePtr& wt) {
                      return host_prepared(wt, {o, c}, [g, o, c](const auto& v) {
                          return ktt_conv_nhwc::matmul_weight(flip_swap(v, o, c, 1), g);
                      });
                  }
              )
            : nhwc_weight(
                  w,
                  "input grad w':" + key,
                  [g, o, c, k](const HandlePtr& wt) {
                      const std::vector<int64_t> shape{
                          (o + 31) / 32 * k * k * 32, (c + 31) / 32 * 32
                      };
                      return host_prepared(wt, shape, [g, o, c, k](const auto& v) {
                          return ktt_conv_nhwc::prepare(flip_swap(v, o, c, k), g);
                      });
                  }
              );
        // TT-NN conv2d's weight [C, O, K, K] when no plan fits.
        const auto conv_w = !weight || k == 1
            ? weight
            : nhwc_weight(w, "input grad w:" + key, [o, c, k](const HandlePtr& wt) {
                  return host_prepared(wt, {c, o, k, k}, [o, c, k](const auto& v) {
                      return flip_swap(v, o, c, k);
                  });
              });
        if (!weight || !conv_w) {
            g_nhwc_fallbacks++;
            return nullptr;
        }
        return nhwc_apply(go, weight, conv_w, nullptr, g);
    }

    static HandlePtr nhwc_apply(
        const HandlePtr& x,
        const HandlePtr& weight,
        const HandlePtr& conv_w,
        const HandlePtr& bias,
        const ktt_conv_nhwc::Geometry& g
    ) {
        const auto bf16 = DType::BFloat16;
        const int64_t n = g.n, c = g.c, h = g.h, wd = g.w, o = g.o, k = g.k;
        const std::vector<int64_t> out_shape{n, o, h, wd};
        const int64_t out_bytes = detail::padded_bytes(
            detail::stored_dims(out_shape, Format::Nhwc), bf16
        );
        const int64_t npix = n * h * wd;
        const auto xn = nhwc_input(x);
        if (k == 1) {
            // The Nhwc rows [N H W, C] times [C, O]: the Nhwc output's rows, on the
            // bridge's matmul.
            const auto rows = relabel(xn, {1, 1, npix, c}, Format::Nchw);
            const auto y = linear(rows, weight, nullptr, true);
            g_nhwc_runs++;
            if (!bias) {
                return relabel(y, out_shape, Format::Nhwc);
            }
            return detail::timed_op(
                bf16,
                out_shape,
                [y, bias, npix, o, out_bytes] {
                    const auto memory = detail::activation_config(out_bytes);
                    return add_bias(y->tensor, bias->tensor, npix, o, memory);
                },
                Format::Nhwc
            );
        }
        const auto config = numerics::compute_config(numerics::effective(), true);
        const int64_t p = (k - 1) / 2;
        const TtnnConv tc{x->shape, bf16, {1, 1}, {p, p}, {1, 1}, 1, config};
        const int64_t epoch = write_epoch();
        const bool in_capture = capturing();
        const bool l1_placement = nhwc_l1_placement();
        // Copy tables: inside a capture, cached ones only (a capture cannot upload);
        // otherwise built and cached, or built for the call only while another trace is
        // alive.
        const auto tables = in_capture             ? ktt_conv_nhwc::Tables::CachedOnly
            : traces_alive() && !tracing() ? ktt_conv_nhwc::Tables::Once
                                           : ktt_conv_nhwc::Tables::Cache;
        return detail::timed_op(
            bf16,
            out_shape,
            [=] {
                // The program reads X and writes the output in DRAM (activation
                // placement under the diagnostic switch).
                const auto memory = l1_placement ? detail::activation_config(out_bytes)
                                                 : tt::tt_metal::MemoryConfig{};
                auto xt = xn->tensor;
                if (!l1_placement &&
                    xt.memory_config().buffer_type() != tt::tt_metal::BufferType::DRAM) {
                    xt = ttnn::to_memory_config(xt, memory);
                    g_nhwc_input_moves++;
                }
                auto out = ktt_conv_nhwc::run(
                    xt, weight->tensor, g, config, memory, tables
                );
                ttnn::Tensor y;
                if (out) {
                    y = *out;
                    g_nhwc_runs++;
                } else {
                    // No plan fits now: TT-NN's conv2d of the Nhwc rows (conv2d()'s
                    // route for an Nhwc input).
                    g_nhwc_fallbacks++;
                    const std::vector<int64_t> rows{1, 1, npix, o};
                    y = ttnn::reshape(
                        ttnn_conv_rows(xt, true, conv_w, tc, epoch, in_capture),
                        detail::device_shape(rows)
                    );
                    if (detail::has_padding(rows)) {
                        y = detail::zero_padding(y);
                    }
                }
                return bias ? add_bias(y, bias->tensor, npix, o, memory) : y;
            },
            Format::Nhwc
        );
    }

    int64_t nhwc_conv_runs() {
        return g_nhwc_runs.load();
    }

    int64_t nhwc_conv_fallbacks() {
        return g_nhwc_fallbacks.load();
    }

    int64_t nhwc_conv_input_moves() {
        return g_nhwc_input_moves.load();
    }

    namespace detail {

        void refresh_nhwc_conv_weights() {
            const std::lock_guard lock(nhwc_weights_mutex());
            for (auto& [key, entry] : nhwc_weights()) {
                if (!entry.traced) {
                    continue;
                }
                const auto w = entry.weight.lock();
                if (w && w->writes.load() != entry.writes) {
                    refresh_in_place(w, entry.prepared, entry);
                    entry.writes = w->writes.load();
                }
            }
        }

        void clear_conv_nhwc() {
            {
                const std::lock_guard lock(nhwc_weights_mutex());
                nhwc_weights().clear();
                nhwc_retired().clear();
            }
            post([] {
                ktt_conv_nhwc::clear();
            });
        }

    }  // namespace detail

}  // namespace ktt::bridge
