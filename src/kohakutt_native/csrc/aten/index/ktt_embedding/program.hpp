// Kohaku-TT's fp32 row gather (aten/index/ktt_embedding/reader.cpp): every size a
// runtime argument.
#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_embedding {

    // table: TILE fp32 [..., V, D]; index: TILE int32 [1, n]. Returns TILE fp32 [...,
    // n, D] (rows of each [V, D] slice at index) in `memory`.
    ttnn::Tensor run(
        const ttnn::Tensor& table,
        const ttnn::Tensor& index,
        uint32_t n,
        const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_embedding
