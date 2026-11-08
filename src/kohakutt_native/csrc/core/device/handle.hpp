// Bridge-internal: the Handle definition and helpers shared by bridge translation
// units. TT-NN headers only; never included from torch-side files.
#pragma once

#include <atomic>
#include <chrono>
#include <utility>

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

    }  // namespace detail

    // A device tensor: device shape and dtype are known when it is created; `tensor` is
    // assigned on the stream by the op that produces it, before any later op reads it.
    struct Handle {
        ttnn::Tensor tensor;
        std::vector<int64_t> shape;
        DType dtype;
        int64_t bytes = 0;

        Handle(std::vector<int64_t> s, DType d)
            : shape(std::move(s)), dtype(d), bytes(detail::padded_bytes(shape, d)) {
            detail::track_bytes(bytes);
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

        // Process-wide counters (core/device/device.cpp); ops are counted at
        // submission, op_ns on the stream.
        struct AtomicCounters {
            std::atomic<int64_t> uploads{0}, downloads{0}, ops{0}, op_ns{0};
        };

        AtomicCounters& counters();

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

        // Records a new handle for spill_l1 (any thread).
        void register_handle(const HandlePtr& handle);

        // Device-lifetime state dropped before the device closes: traces with their
        // held handles (trace.cpp) and cached constants (ops/memory.cpp).
        void release_traces();
        void clear_constants();
        void clear_ranges();
        void clear_row_major();

        // Advances write_epoch() (core/device/trace.cpp).
        void bump_write_epoch();

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

        // A new handle of device shape `shape` (its tensor is assigned later on the
        // stream).
        inline HandlePtr make_handle(std::vector<int64_t> shape, DType dtype) {
            auto handle = (
                HandlePtr(new Handle(std::move(shape), dtype), release_handle)
            );
            on_wrap(handle);
            register_handle(handle);
            return handle;
        }

        // Throws when TT-NN's result shape differs from the shape predicted for `h`.
        void check_shape(const Handle& h);

        // Moves the L1 tensor of every live handle to DRAM (stream worker; not while a
        // trace captures); true when one moved. Handles sharing an L1 buffer each get
        // their own DRAM copy.
        bool spill_l1();

        // True for TT-Metal's error of a program whose circular buffers clash with L1
        // buffers, or of an L1 buffer that does not fit.
        bool is_l1_clash(const std::exception& e);

        // True for TT-Metal's error of an L1_SMALL buffer that does not fit (held by
        // cached programs, e.g. conv metadata per input shape).
        bool is_l1_small_full(const std::exception& e);

        // For its lifetime, activation_config returns DRAM on this thread.
        class DramOnly {
        public:
            DramOnly();
            ~DramOnly();
            DramOnly(const DramOnly&) = delete;
            DramOnly& operator=(const DramOnly&) = delete;
        };

        // One TT-NN op on the stream: the handle (device shape `shape`) is returned at
        // once; `call` runs on the worker and its result becomes the handle's tensor.
        // A program refused for L1 clashes runs again after L1 tensors spill to DRAM.
        template <typename F>
        HandlePtr timed_op(DType dtype, std::vector<int64_t> shape, F call) {
            auto out = make_handle(std::move(shape), dtype);
            counters().ops++;
            submit([out, call = std::move(call)]() mutable {
                const auto start = std::chrono::steady_clock::now();
                try {
                    out->tensor = call();
                } catch (const std::exception& e) {
                    if (is_l1_small_full(e)) {
                        // Cached programs' L1_SMALL buffers dropped, then again.
                        device()->clear_program_cache();
                        out->tensor = call();
                    } else if (is_l1_clash(e)) {
                        // Again with L1 activations spilled and this op's outputs in
                        // DRAM.
                        spill_l1();
                        const DramOnly dram;
                        out->tensor = call();
                    } else {
                        throw;
                    }
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
