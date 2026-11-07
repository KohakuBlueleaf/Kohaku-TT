// Device-side helpers for Kohaku-TT's own Metalium programs (aten/<family>/ktt_<name>/:
// program.cpp builds the program, reader / compute / writer.cpp run on the device).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>

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

    // Queues `run` on the default stream over the tensors of `inputs`, then new TILE
    // outputs of logical sizes `out_shapes` and `out_dtypes`; returns the outputs.
    std::vector<HandlePtr> run_on_stream(
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        std::function<void(const std::vector<ttnn::Tensor>& tensors)> run
    );

}  // namespace ktt::bridge::program
