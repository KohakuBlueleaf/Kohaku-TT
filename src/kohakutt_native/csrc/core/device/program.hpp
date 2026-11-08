// Device-side helpers for Kohaku-TT's own Metalium programs (aten/<family>/ktt_<name>/:
// program.cpp builds the program, reader / compute / writer.cpp run on the device).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "core/device/handle.hpp"

namespace ktt::bridge::program {

    // False when KTT_CUSTOM_KERNELS=0: every op runs TT-NN's implementation.
    bool own_kernels_enabled();

    // The path of device source `relative` (e.g. "eltwise/ktt_unary/reader.cpp") under
    // csrc/aten/ beside the loaded native library.
    std::string source(const std::string& relative);

    // The smallest tensor (tiles) an own program runs on: `measured`, or
    // KTT_CUSTOM_MIN_TILES when set (kernel tests at small sizes).
    int64_t min_tiles(int64_t measured);

    // Tiles of a TILE tensor of device shape `shape`.
    int64_t tile_count(const std::vector<int64_t>& shape);

    // `tiles` split evenly over the first cores of the grid (row-major), at least
    // `min_per_core` tiles per core; the first `extra` cores take one more.
    struct Split {
        std::vector<tt::tt_metal::CoreCoord> cores;
        tt::tt_metal::CoreRangeSet range;
        uint32_t base = 0, extra = 0;

        uint32_t count(size_t i) const {
            return base + (i < extra ? 1 : 0);
        }
    };

    Split split_tiles(uint32_t tiles, uint32_t min_per_core);

    // Per-core runtime args {c} of the first `cores` cores of the grid, core c at
    // (c % grid columns, c / grid columns).
    tt::tt_metal::KernelDescriptor::RuntimeArgs core_index_args(uint32_t cores);

    // Host-time phases of an own program before its enqueue.
    enum class Phase { Alloc, Args };

    // Charges the stream worker's time since the previous mark to `phase`.
    void mark(Phase phase);

    // A runtime argument holding the buffer address of io tensor `tensor`: argument
    // `position` of core entry `entry` of kernel `kernel` (kEveryCore: of every
    // entry; kCommon: of the kernel's common runtime arguments).
    struct Slot {
        static constexpr int32_t kEveryCore = -1, kCommon = -2;
        uint32_t kernel = 0;
        int32_t entry = kEveryCore;
        uint32_t position = 0;
        uint32_t tensor = 0;
    };

    // A program's descriptor and the slots of its buffer addresses.
    struct Built {
        tt::tt_metal::ProgramDescriptor desc;
        std::vector<Slot> slots;
    };

    // The cache key of a descriptor: the program's name, then every value the
    // descriptor depends on except buffer addresses.
    class Key {
    public:
        explicit Key(std::string_view program);
        Key& add(int64_t value);
        Key& add(std::string_view value);
        // dtype, layout, logical and padded shape, and memory config.
        Key& add(const ttnn::Tensor& tensor);

        const std::string& str() const {
            return key_;
        }

    private:
        std::string key_;
    };

    // False when KTT_DESC_CACHE=0: every own program builds its descriptor per call.
    bool desc_cache_enabled();

    // Runs the program `build` describes over io (inputs, then the output last) with
    // ttnn::generic_op and returns the output. With the descriptor cache on (or
    // `always_cached`: user programs), the descriptor is built once per key and reused
    // with io's addresses written to its slots (KTT_DESC_CACHE=verify: also rebuilt
    // and compared on every reuse).
    ttnn::Tensor launch(
        const Key& key,
        const std::vector<ttnn::Tensor>& io,
        const std::function<Built()>& build,
        bool always_cached = false
    );

    // launch() of a descriptor built by the caller (not cached).
    ttnn::Tensor launch(
        const std::vector<ttnn::Tensor>& io, const tt::tt_metal::ProgramDescriptor& desc
    );

    // Drops every cached descriptor (device close).
    void clear_desc_cache();

    // Queues `run` on the default stream over the tensors of `inputs`, then new TILE
    // outputs of logical sizes `out_shapes` and `out_dtypes`; returns the outputs.
    std::vector<HandlePtr> run_on_stream(
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        std::function<void(const std::vector<ttnn::Tensor>& tensors)> run
    );

}  // namespace ktt::bridge::program
