// Kohaku-TT's gather along the last dim (aten/index/ktt_gather/reader.cpp): every size
// a runtime argument, work linear in the number of output elements.
#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_gather {

    // x: TILE bf16 / fp32 / int32 [..., H, W]; index: TILE int32 [..., Hi, Wi] with the
    // same leading dims and Hi <= H. Returns x's dtype [..., Hi, Wi] in `memory`.
    ttnn::Tensor run(
        const ttnn::Tensor& x,
        const ttnn::Tensor& index,
        const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_gather
