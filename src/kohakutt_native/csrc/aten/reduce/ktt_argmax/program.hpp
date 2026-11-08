// Kohaku-TT's argmax over the last dim (aten/reduce/ktt_argmax/{reader,compute,writer}
// .cpp): every size a runtime argument.
#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_argmax {

    // x: TILE bf16 / fp32 [P, R, W]. Returns TILE int32 [P, R] in `memory`: the first
    // index of the maximum along W.
    ttnn::Tensor run(const ttnn::Tensor& x, const tt::tt_metal::MemoryConfig& memory);

}  // namespace ktt::bridge::ktt_argmax
