// Bridge-internal: the Handle definition and helpers shared by bridge translation
// units. TT-NN headers only; never included from torch-side files.
#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include <tracy/Tracy.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "core/device/stream.hpp"
#include "core/device/tt_bridge.hpp"

namespace ktt::bridge {

    namespace detail {

        // Live and peak device bytes held by handles (core/device/device.cpp).
        void track_bytes(int64_t delta);

        // Number of device opens so far (core/device/device.cpp).
        int64_t generation();

        // Padded device bytes of a TILE tensor of device shape `shape`.
        inline int64_t padded_bytes(const std::vector<int64_t>& shape, DType dtype) {
            auto up = [](int64_t v) {
                return (v + 31) / 32 * 32;
            };
            int64_t n = 1;
            for (size_t k = 0; k < shape.size(); ++k) {
                const bool tiled = k + 2 >= shape.size();
                n *= tiled ? up(shape[k]) : shape[k];
            }
            // Block-float tiles: 1024 values plus 64 exponent bytes.
            switch (dtype) {
                case DType::BFloat16:
                    return n * 2;
                case DType::BFloat8B:
                    return n / 1024 * 1088;
                case DType::BFloat4B:
                    return n / 1024 * 576;
                default:
                    return n * 4;
            }
        }

        // True for a device shape [1, n] with n a multiple of 32 (a 1-D tensor): its
        // Flat form is [n / w, w], flat_width(n) = w.
        inline bool flat_vector(const std::vector<int64_t>& shape) {
            return (
                shape.size() == 2 && shape[0] == 1 && shape[1] > 0 && shape[1] % 32 == 0
            );
        }

        // The widest w <= 1024 whose [n / w, w] has whole tiles; 32 when none has.
        inline int64_t flat_width(int64_t n) {
            int64_t w = 1024;
            while (w > 32 && n % (32 * w) != 0) {
                w /= 2;
            }
            return w;
        }

        // The TT-NN shape storing device shape `shape` as `format`: Nhwc [1, 1, N H W,
        // C], Flat [d0, d1 ... dk] ([n / w, w] for flat_vector()), HeadMajor `shape`
        // with dims -3 and -2 swapped, Fold `shape` with its last two dims merged.
        inline std::vector<int64_t> stored_dims(
            const std::vector<int64_t>& shape, Format format
        ) {
            if (format == Format::Nhwc) {
                return {1, 1, shape[0] * shape[2] * shape[3], shape[1]};
            }
            if (format == Format::Flat && flat_vector(shape)) {
                const int64_t w = flat_width(shape[1]);
                return {shape[1] / w, w};
            }
            if (format == Format::Flat) {
                int64_t rest = 1;
                for (size_t k = 1; k < shape.size(); ++k) {
                    rest *= shape[k];
                }
                return {shape[0], rest};
            }
            if (format == Format::Fold) {
                auto merged = shape;
                merged.pop_back();
                merged.back() *= shape.back();
                return merged;
            }
            if (format == Format::HeadMajor) {
                auto swapped = shape;
                std::swap(swapped[shape.size() - 3], swapped[shape.size() - 2]);
                return swapped;
            }
            return shape;
        }

    }  // namespace detail

    // A device tensor: device shape and dtype are known when it is created; `tensor` is
    // assigned on the stream by the op that produces it, before any later op reads it.
    // A non-replicated handle leaves `tensor` empty and holds its shards in `placed`
    // (local shape `local`); `shape` stays the global device shape. An Nhwc handle's
    // `tensor` has the logical shape detail::stored_dims(shape, Format::Nhwc).
    struct Handle {
        ttnn::Tensor tensor;
        std::vector<int64_t> shape;
        DType dtype;
        int64_t bytes = 0;
        Placement placement;
        ttnn::Tensor placed;
        std::vector<int64_t> local;

        // This tensor converted to other placements in write epoch `kept_epoch`
        // (converted(), core/mesh/device_mesh.cpp); a null copy was not kept.
        struct Kept {
            Placement to;
            HandlePtr copy;
        };

        std::vector<Kept> kept;
        int64_t kept_epoch = -1;
        HandlePtr view;  // local_view() cache
        Format format = Format::Nchw;
        HandlePtr twin;           // to_nchw() / to_nhwc() cache (layout.cpp)
        int64_t twin_epoch = -1;  // write epoch `twin` was built in
        // In-place writes of this handle's buffer: copy_into(), and each replay of a
        // trace whose capture copied into it (trace.cpp).
        std::atomic<int64_t> writes{0};
        HandlePtr owner;   // the handle whose buffer as_stored() views
        HandlePtr scaled;  // scaled_copy() cache: this * scale at `scaled_writes`
        double scale = 0;
        int64_t scaled_writes = -1;
        int64_t opened = detail::generation();  // the device open it belongs to
        int64_t made_by = detail::op_seq();     // the stream op that computes it

        Handle(std::vector<int64_t> s, DType d, Format f = Format::Nchw)
            : shape(std::move(s)),
              dtype(d),
              bytes(detail::padded_bytes(detail::stored_dims(shape, f), d)),
              format(f) {
            detail::track_bytes(bytes);
        }

        // A placed handle; `bytes` counts one chip's shard.
        Handle(std::vector<int64_t> s, DType d, Placement p, std::vector<int64_t> l)
            : shape(std::move(s)),
              dtype(d),
              bytes(detail::padded_bytes(l, d)),
              placement(std::move(p)),
              local(std::move(l)) {
            detail::track_bytes(bytes);
        }

        // The mesh tensor as stored: shards when placed, else `tensor`.
        const ttnn::Tensor& stored() const {
            return placement.is_replicated() ? tensor : placed;
        }

        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
    };

    namespace detail {

        // The open device; throws when no device is open.
        ttnn::MeshDevice* device();

        // KTT_L1_BUDGET: the fraction of each L1 bank activations may fill.
        double l1_budget();

        // L1 per bank for a program's circular buffers (stream worker): the bank
        // outside KTT_L1_BUDGET, less what live L1 buffers plus `reserve` bytes take
        // from it.
        int64_t l1_free_per_bank(int64_t reserve = 0);

        // The highest L1 address the circular buffers of the alive traces' programs
        // reach (0: none, or KTT_TRACE_L1_GUARD=0) (core/device/trace.cpp).
        uint64_t traced_cb_end();

        // Starts (device open) / stops (close) recording where each program's circular
        // buffers end; on the stream worker (core/device/trace.cpp).
        void track_program_regions(bool on);

        // The memory config of a new activation of `bytes` (on the stream worker): L1
        // interleaved while live L1 buffers stay within KTT_L1_BUDGET of each bank,
        // else DRAM interleaved.
        tt::tt_metal::MemoryConfig activation_config(int64_t bytes);

        // Queues activation_config's DRAM-only mode on (`dram`) or off.
        void trace_placement(bool dram);

        // Inner dim from which matmuls accumulate K blocks in L1 (packer_l1_acc).
        constexpr int64_t kPackerL1AccMinK = 1536;
        // Matmul-class compute config of the calling thread's numerics (call where the
        // op is queued, not on the worker); `fast`: LoFi.
        ttnn::ComputeKernelConfig matmul_config(int64_t k = 0, bool fast = false);

        // The first ceil(tiles / tiles_per_core) cores of the compute grid for an op
        // over a tensor of device `shape`; nullopt = the full grid
        // (core/device/device.cpp).
        std::optional<tt::tt_metal::CoreRangeSet> worker_cores(
            const std::vector<int64_t>& shape, int64_t tiles_per_core
        );

        // Tiles per core for elementwise families, balancing device time against host
        // cost.
        constexpr int64_t kUnaryTilesPerCore = 32;
        constexpr int64_t kBinaryTilesPerCore = 64;

        // Process-wide counters (core/device/device.cpp); ops counted at submission,
        // op_ns on the stream; own programs' launches, cache hits and phase times.
        struct AtomicCounters {
            std::atomic<int64_t> uploads{0}, downloads{0}, ops{0}, op_ns{0};
            std::atomic<int64_t> programs{0}, desc_hits{0}, program_entries{0};
            std::atomic<int64_t> alloc_ns{0}, args_ns{0}, desc_ns{0}, enqueue_ns{0};
            std::atomic<int64_t> attention_flash{0}, attention_own{0};
            std::atomic<int64_t> attention_ttnn{0}, attention_math{0};
            std::atomic<int64_t> attention_backward_flash{0},
                attention_backward_ttnn{0};
            std::atomic<int64_t> attention_backward_math{0};
        };

        AtomicCounters& counters();

        // The stream worker's end of the last timed phase (program::mark).
        inline std::chrono::steady_clock::time_point& phase_clock() {
            thread_local std::chrono::steady_clock::time_point t;
            return t;
        }

        // The device shape of logical torch sizes: rank 0 -> (1, 1), rank 1 (n) -> (1,
        // n).
        inline ttnn::Shape device_shape(const std::vector<int64_t>& sizes) {
            ttsl::SmallVector<uint32_t> dims;
            if (sizes.size() == 0) {
                dims.push_back(1);
            }
            if (sizes.size() <= 1) {
                dims.push_back(1);
            }
            for (int64_t d : sizes) {
                dims.push_back(static_cast<uint32_t>(d));
            }
            return ttnn::Shape(std::move(dims));
        }

        // device_shape() as plain sizes.
        inline std::vector<int64_t> device_dims(const std::vector<int64_t>& sizes) {
            auto shape = device_shape(sizes);
            return std::vector<int64_t>(shape.cbegin(), shape.cend());
        }

        inline tt::tt_metal::DataType to_ttnn(DType dtype) {
            switch (dtype) {
                case DType::BFloat16:
                    return tt::tt_metal::DataType::BFLOAT16;
                case DType::Float32:
                    return tt::tt_metal::DataType::FLOAT32;
                case DType::Int32:
                    return tt::tt_metal::DataType::INT32;
                case DType::BFloat8B:
                    return tt::tt_metal::DataType::BFLOAT8_B;
                case DType::BFloat4B:
                    return tt::tt_metal::DataType::BFLOAT4_B;
            }
            return tt::tt_metal::DataType::BFLOAT16;
        }

        // Holds `handle` for the active trace capture, if any (core/device/trace.cpp).
        void on_wrap(const HandlePtr& handle);

        // Device-lifetime state dropped before the device closes: traces with their
        // held handles (trace.cpp) and cached constants (ops/memory.cpp).
        void release_traces();
        void clear_constants();
        void clear_ranges();
        void clear_row_major();
        // The channels-last conv's prepared weights and copy tables
        // (aten/nn/device_conv.cpp).
        void clear_conv_nhwc();

        // Advances write_epoch() (core/device/trace.cpp).
        void bump_write_epoch();

        // Where a trace replay is queued: (weight handle, its Handle::writes) for the
        // TT-NN-path conv weights captures read (aten/nn/device_conv.cpp).
        std::vector<std::pair<const void*, int64_t>> traced_conv_weight_writes();
        // On the stream, before a trace replay: those weights' prepared buffers,
        // re-prepared for every weight whose count in `writes` is not the one its
        // buffer was prepared at.
        void refresh_traced_conv_weights(
            const std::vector<std::pair<const void*, int64_t>>& writes
        );

        // Before a trace replay, on the calling thread: queues the channels-last conv's
        // prepared weights read by captures, re-prepared in place, where their weight's
        // Handle::writes changed (aten/nn/device_conv.cpp).
        void refresh_nhwc_conv_weights();

        // True once mark_exiting() ran: TT-Metal may be torn down, so nothing is freed.
        bool exiting();

        // Frees a handle on the stream (TT-NN is only called there), or leaks it when
        // exiting. Stream-ordered free: bytes count as released now, the buffer is
        // freed on the worker.
        inline void release_handle(Handle* handle) {
            track_bytes(-handle->bytes);
            if (exiting()) {
                return;
            }
            post([handle] {
                delete handle;
            });
        }

        // A new handle of device shape `shape` stored as `format` (its tensor is
        // assigned later on the stream).
        inline HandlePtr make_handle(
            std::vector<int64_t> shape, DType dtype, Format format = Format::Nchw
        ) {
            auto handle = (HandlePtr(
                new Handle(std::move(shape), dtype, format), release_handle
            ));
            on_wrap(handle);
            return handle;
        }

        // On the stream worker: x (device shape `shape` [N, C, H, W]) as the Nhwc
        // tensor, and the Nhwc tensor y back as [N, C, H, W]; `bytes` sizes the output.
        ttnn::Tensor nhwc_of(
            const ttnn::Tensor& x, const std::vector<int64_t>& shape, int64_t bytes
        );
        ttnn::Tensor nchw_of(
            const ttnn::Tensor& y, const std::vector<int64_t>& shape, int64_t bytes
        );
        // On the stream worker: x of device shape `shape` [.., T, H, D] with dims -3
        // and -2 swapped, into (`to_major`) or out of the head-major form.
        ttnn::Tensor swap_heads(
            const ttnn::Tensor& x,
            const std::vector<int64_t>& shape,
            int64_t bytes,
            bool to_major
        );
        // True when a TILE tensor of logical shape `dims` has tile padding.
        bool has_padding(const std::vector<int64_t>& dims);
        // On the stream worker: `t` with zeros in its tile padding.
        ttnn::Tensor zero_padding(const ttnn::Tensor& t);

        // The format upload_auto() / empty_auto() store a replicated tensor of device
        // shape `dims` in: Flat for weight-like shapes (d0 >= 32, d1 > 1, last two dims
        // below 32) whose tile form is over 4x Flat's (KTT_FLAT=0: never), and for
        // flat_vector() shapes of kFlatVectorMin+ values (KTT_FLAT_1D=0: never), else
        // Nchw.
        constexpr int64_t kFlatVectorMin = 65536;
        Format auto_format(const std::vector<int64_t>& dims, DType dtype);

        // On the stream worker, from a conv weight `w` [O, C, KH, KW] ([O, C, KW]: KH =
        // 1), Flat or Nchw, as TILE [rows, O] (column o):
        // tap_major: row (kh KW + kw) C + c (ktt_conv's K order);
        // block_tap_major: Ct = ceil(C / 32) blocks, row ((c / 32) KH KW + kh KW + kw)
        // 32 + c % 32, rows of c >= C zero (the channels-last conv's K order).
        ttnn::Tensor tap_major(const Handle& w);
        ttnn::Tensor block_tap_major(const Handle& w);
        // On the stream worker: m TILE [O, taps C], column t C + c holding w[o, c, t],
        // as the Flat rows [O, C taps] (column c taps + t) of a weight of device
        // `shape`.
        ttnn::Tensor flat_from_taps(
            const ttnn::Tensor& m, const std::vector<int64_t>& shape
        );

        // Reads a non-replicated handle's global values into `dst`
        // (core/mesh/device_mesh.cpp).
        void download_placed(const HandlePtr& handle, void* dst);

        // Throws when TT-NN's result shape differs from the shape predicted for `h`.
        void check_shape(const Handle& h);

        // True for TT-Metal's error of an L1_SMALL buffer that does not fit (held by
        // cached programs, e.g. conv metadata per input shape).
        bool is_l1_small_full(const std::exception& e);

        // One TT-NN op on the stream: the handle (device shape `shape`, stored as
        // `format`) is returned at once; `call` runs on the worker, once, and its
        // result becomes the handle's tensor.
        template <typename F>
        HandlePtr timed_op(
            DType dtype,
            std::vector<int64_t> shape,
            F call,
            Format format = Format::Nchw,
            const char* label = __builtin_FUNCTION()
        ) {
            auto out = make_handle(std::move(shape), dtype, format);
            counters().ops++;
            submit(
                [out, call = std::move(call)]() mutable {
                    const auto start = std::chrono::steady_clock::now();
                    phase_clock() = start;
                    {
                        ZoneScopedN("ktt op call");
                        out->tensor = call();
                    }
                    const auto elapsed = std::chrono::steady_clock::now() - start;
                    using std::chrono::duration_cast, std::chrono::nanoseconds;
                    counters().op_ns += duration_cast<nanoseconds>(elapsed).count();
                    check_shape(*out);
                },
                label
            );
            return out;
        }

        // `dims` as "[a, b, c]" for messages.
        inline std::string dims_str(const std::vector<int64_t>& dims) {
            std::string s = "[";
            for (size_t k = 0; k < dims.size(); ++k) {
                s += (k ? ", " : "") + std::to_string(dims[k]);
            }
            return s + "]";
        }

        // The broadcast of two device shapes (right-aligned; size-1 dims stretch).
        inline std::vector<int64_t> broadcast_dims(
            const std::vector<int64_t>& a, const std::vector<int64_t>& b
        ) {
            std::vector<int64_t> out(std::max(a.size(), b.size()), 1);
            for (size_t k = 0; k < out.size(); ++k) {
                const int64_t x = k < a.size() ? a[a.size() - 1 - k] : 1;
                const int64_t y = k < b.size() ? b[b.size() - 1 - k] : 1;
                out[out.size() - 1 - k] = x == 1 ? y : x;
            }
            return out;
        }

        // Zero-size path: ops on zero-volume handles return an empty handle of the same
        // device shape without a TT-NN call (its programs divide by tile counts).
        inline bool is_empty(const HandlePtr& a) {
            for (auto d : a->shape) {
                if (d == 0) {
                    return true;
                }
            }
            return false;
        }

        inline HandlePtr empty_of(const HandlePtr& like, DType dtype) {
            return ktt::bridge::empty(like->shape, dtype);
        }

    }  // namespace detail
}  // namespace ktt::bridge
