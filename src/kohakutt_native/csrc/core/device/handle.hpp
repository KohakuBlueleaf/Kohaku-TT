// Bridge-internal: the Handle definition and helpers shared by bridge translation
// units. TT-NN headers only; never included from torch-side files.
#pragma once

#include <atomic>
#include <chrono>
#include <utility>

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

        // The logical shape of the TT-NN tensor holding device shape `shape` stored as
        // `format`: [1, 1, N H W, C] for Nhwc, else `shape`.
        inline std::vector<int64_t> stored_dims(
            const std::vector<int64_t>& shape, Format format
        ) {
            if (format != Format::Nhwc) {
                return shape;
            }
            return {1, 1, shape[0] * shape[2] * shape[3], shape[1]};
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
        HandlePtr full;           // replicated() cache (core/mesh/device_mesh.cpp)
        int64_t full_epoch = -1;  // write epoch `full` was built in
        HandlePtr view;           // local_view() cache
        Format format = Format::Nchw;
        HandlePtr twin;           // to_nchw() / to_nhwc() cache (layout.cpp)
        int64_t twin_epoch = -1;  // write epoch `twin` was built in
        // In-place writes of this handle's buffer: copy_into(), and each replay of a trace
        // whose capture copied into it (trace.cpp).
        std::atomic<int64_t> writes{0};

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

        // Free L1 per bank (bank size less live L1 buffers), on the stream worker.
        int64_t l1_free_per_bank();

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
            std::atomic<int64_t> programs{0}, desc_hits{0};
            std::atomic<int64_t> alloc_ns{0}, args_ns{0}, desc_ns{0}, enqueue_ns{0};
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

        // On the stream, before a trace replay: TT-NN-path conv weights prepared before
        // a capture and read by it, re-prepared into their buffers when the weights
        // were written after `epoch` (aten/nn/device_conv.cpp).
        void refresh_traced_conv_weights(int64_t epoch);

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
            auto handle = (
                HandlePtr(new Handle(std::move(shape), dtype, format), release_handle)
            );
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
        // True when a TILE tensor of logical shape `dims` has tile padding.
        bool has_padding(const std::vector<int64_t>& dims);
        // On the stream worker: `t` with zeros in its tile padding.
        ttnn::Tensor zero_padding(const ttnn::Tensor& t);

        // Reads a non-replicated handle's global values into `dst`
        // (core/mesh/device_mesh.cpp).
        void download_placed(const HandlePtr& handle, void* dst);

        // Throws when TT-NN's result shape differs from the shape predicted for `h`.
        void check_shape(const Handle& h);

        // True for TT-Metal's error of an L1_SMALL buffer that does not fit (held by
        // cached programs, e.g. conv metadata per input shape).
        bool is_l1_small_full(const std::exception& e);

        // One TT-NN op on the stream: the handle (device shape `shape`, stored as
        // `format`) is returned at once; `call` runs on the worker, once, and its result
        // becomes the handle's tensor.
        template <typename F>
        HandlePtr timed_op(
            DType dtype,
            std::vector<int64_t> shape,
            F call,
            Format format = Format::Nchw
        ) {
            auto out = make_handle(std::move(shape), dtype, format);
            counters().ops++;
            submit([out, call = std::move(call)]() mutable {
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
            });
            return out;
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
