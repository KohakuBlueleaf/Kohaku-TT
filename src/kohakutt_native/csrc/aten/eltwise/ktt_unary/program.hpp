// Kohaku-TT's own unary program: any SFPU op chain on a tiled tensor, tiles split
// evenly over the grid, blocked reads / writes, one DST acquire per block.
#pragma once

#include <vector>

#include "core/device/handle.hpp"

namespace ktt::bridge::ktt_unary {

    // True when run(op, ...) handles `op` on a tiled tensor of `dtype` and device
    // `shape` (supported op, large enough to beat TT-NN's).
    bool supports(UnaryOp op, DType dtype, const std::vector<int64_t>& shape);

    // `op` applied elementwise to tiled interleaved `in` (TT-NN's SFPU op chain), with
    // SFPU approximations when `approx`, into a tensor of `out_config`.
    ttnn::Tensor run(
        UnaryOp op,
        const ttnn::Tensor& in,
        bool approx,
        const tt::tt_metal::MemoryConfig& out_config
    );

}  // namespace ktt::bridge::ktt_unary
