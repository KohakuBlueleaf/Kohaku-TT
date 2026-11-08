// Kohaku-TT's cumulative sum along the last dim
// (aten/reduce/ktt_cumsum/{reader,compute, writer}.cpp): fp32 on the SFPU, every size a
// runtime argument.
#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_cumsum {

    // x: TILE bf16 / fp32 [..., R, W]. Returns the cumulative sum along W, same shape
    // and dtype, in `memory`.
    ttnn::Tensor run(const ttnn::Tensor& x, const tt::tt_metal::MemoryConfig& memory);

}  // namespace ktt::bridge::ktt_cumsum
