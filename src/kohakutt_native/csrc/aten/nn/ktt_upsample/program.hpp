// Kohaku-TT's nearest upsampling and pixel shuffle by factors dividing 32: one
// data-movement program each, every size a runtime argument
// (aten/nn/ktt_upsample/{reader,shuffle}.cpp).
#pragma once

#include <cstdint>

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_upsample {

    // x: TILE bf16 / fp32 [..., H, W]. Returns TILE [..., H kh, W kw] in `memory`.
    ttnn::Tensor run(
        const ttnn::Tensor& x,
        uint32_t kh,
        uint32_t kw,
        const tt::tt_metal::MemoryConfig& memory
    );

    // Pixel shuffle of x: TILE bf16 / fp32 [..., C r^2, H, W] (r dividing 32). Returns
    // TILE [..., C, r H, r W] in `memory`.
    ttnn::Tensor shuffle(
        const ttnn::Tensor& x, uint32_t r, const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_upsample
