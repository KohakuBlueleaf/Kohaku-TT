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

    // Circular-buffer bytes per core an own program built now may use
    // (detail::l1_free_per_bank).
    uint32_t cb_budget();

    // Ranges [first, last] of logical grid columns whose Tensix columns are physically
    // adjacent in NoC0 on every chip of the device (multicast rectangles stay inside
    // one).
    const std::vector<std::pair<uint32_t, uint32_t>>& panels();

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

    // The runtime arguments kernel `kernel` of the descriptor reads: `common`, and per
    // core `per_core` plus any number of `per_item`-argument records (0: none).
    struct ArgCount {
        uint32_t kernel = 0, per_core = 0, common = 0, per_item = 0;
    };

    // How the kernels use circular buffer `cb`: the page counts of the pushes that
    // recur within a launch (not one made once per launch, nor one the kernel clamps at
    // the ring end), and the most bytes any kernel writes into it from its base (0: not
    // stated).
    // With `at` >= 0 the entry holds for desc.cbs[at] only (one buffer index sized per
    // core range).
    struct Ring {
        uint32_t cb = 0;
        std::vector<uint32_t> pushes;
        uint64_t written = 0;
        int32_t at = -1;
    };

    // A program's descriptor and the slots of its buffer addresses. `args` and `rings`
    // state what the kernel sources read and write; launch() checks the descriptor
    // against them.
    struct Built {
        tt::tt_metal::ProgramDescriptor desc;
        std::vector<Slot> slots;
        std::vector<ArgCount> args;
        std::vector<Ring> rings;
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

    // Common runtime argument `position` of kernel `kernel` holding `value`: a value
    // outside the key that changes from call to call.
    struct Value {
        uint32_t kernel = 0, position = 0, value = 0;
    };

    // Runs the program `build` describes over io (inputs, then the output last) with
    // ttnn::generic_op and returns the output. With the descriptor cache on (or
    // `always_cached`: user programs), the descriptor is built once per key and reused
    // with io's addresses written to its slots and `values` written to their arguments
    // (KTT_DESC_CACHE=verify: also rebuilt and compared on every reuse).
    ttnn::Tensor launch(
        const Key& key,
        const std::vector<ttnn::Tensor>& io,
        const std::function<Built()>& build,
        bool always_cached = false,
        const std::vector<Value>& values = {}
    );

    // launch() of a program whose runtime arguments change from call to call under one
    // key (same kernels, buffers and argument counts): built and checked on every call,
    // never cached.
    ttnn::Tensor launch_fresh(
        const Key& key,
        const std::vector<ttnn::Tensor>& io,
        const std::function<Built()>& build
    );

    // launch() of a descriptor built by the caller (not cached).
    ttnn::Tensor launch(
        const std::vector<ttnn::Tensor>& io, const tt::tt_metal::ProgramDescriptor& desc
    );

    // Drops every cached descriptor and the panel spans (device close).
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
