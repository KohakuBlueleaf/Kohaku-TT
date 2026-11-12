// Kohaku-TT's nearest upsampling (any integer factors), pixel shuffle (factors dividing
// 32), strided column reads, members of a small last dim and column interleaving: one
// data-movement program each, every size a runtime argument (ktt_upsample/*.cpp).
#pragma once

#include <cstdint>

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_upsample {

    // Nearest upsampling by integer kh, kw as row copies: x TILE bf16 / fp32 [..., H,
    // W] -> [..., H kh, W kw] in `memory`.
    ttnn::Tensor rows(
        const ttnn::Tensor& x,
        uint32_t kh,
        uint32_t kw,
        const tt::tt_metal::MemoryConfig& memory
    );

    // The same on channels-last rows: x TILE [1, 1, images h w, C] -> [1, 1, images h
    // kh w kw, C].
    ttnn::Tensor rows_nhwc(
        const ttnn::Tensor& x,
        uint32_t images,
        uint32_t h,
        uint32_t w,
        uint32_t kh,
        uint32_t kw,
        const tt::tt_metal::MemoryConfig& memory
    );

    // Pixel shuffle of x: TILE bf16 / fp32 [..., C r^2, H, W] (r dividing 32). Returns
    // TILE [..., C, r H, r W] in `memory`.
    ttnn::Tensor shuffle(
        const ttnn::Tensor& x, uint32_t r, const tt::tt_metal::MemoryConfig& memory
    );

    // Every `step`-th column of x: TILE bf16 / fp32 / int32 [..., H, W] from column
    // `first`, `count` columns (step <= 16, first + step (count - 1) < W). Returns
    // TILE [..., H, count] in `memory`.
    ttnn::Tensor columns(
        const ttnn::Tensor& x,
        uint32_t first,
        uint32_t step,
        uint32_t count,
        const tt::tt_metal::MemoryConfig& memory
    );

    // Member j of x: TILE bf16 / fp32 / int32 [..., R, N, d] (d < 32, rank >= 2;
    // rank 2 reads as R = 1). Returns TILE [..., R, N] ([1, N] for rank 2) in `memory`.
    ttnn::Tensor select_last(
        const ttnn::Tensor& x, uint32_t j, const tt::tt_metal::MemoryConfig& memory
    );

    // xs: 2 .. 8 TILE bf16 / fp32 [..., H, N] of one shape, dtype and memory config,
    // their columns interleaved. Returns TILE [..., H, N k] in `memory`.
    ttnn::Tensor interleave(
        const std::vector<ttnn::Tensor>& xs, const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_upsample
