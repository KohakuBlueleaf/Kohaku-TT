// Kohaku-TT's softmax over the last dim: fp32 statistics on the SFPU for bf16 and fp32,
// any width. Design: docs/reference/kernels/softmax.md.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer_types.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ktt::bridge::ktt_softmax {

    // Groups of `group` cores share a tile row, each holding `seg` column tiles (the
    // last core fewer); `groups` groups take tile rows group-strided.
    struct Plan {
        uint32_t group = 1, seg = 1, groups = 1;
        uint32_t row_depth = 2;  // tile rows of x resident per core
        uint32_t l1 = 0;
        std::string str() const;
    };

    // The plan for `rows` tile rows of `wt` tiles of `x_tile` bytes; nullopt when no
    // split fits L1.
    std::optional<Plan> plan(uint32_t rows, uint32_t wt, uint32_t x_tile);

    // True when the kernels take x: TILE, bf16 / fp32, rank >= 2, interleaved.
    bool supports(const ttnn::Tensor& x);

    // softmax(x) over the last dim, in x's dtype; padding of y written as zeros.
    ttnn::Tensor forward(
        const ttnn::Tensor& x, const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_softmax
