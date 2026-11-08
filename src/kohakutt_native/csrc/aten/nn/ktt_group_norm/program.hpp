// Kohaku-TT's group norm kernels: per-plane moments (sum x, sum x^2) and the per-plane
// affine y = x a + b, every size a runtime argument
// (aten/nn/ktt_group_norm/{moments,affine}_{reader,compute,writer}.cpp).
#pragma once

#include <cstdint>
#include <utility>

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_group_norm {

    // x: TILE bf16 / fp32 [N, C, H, W]. Returns (sum x, sum x^2) per (n, c) as TILE
    // fp32 [N, C, 1, 1].
    std::pair<ttnn::Tensor, ttnn::Tensor> moments(const ttnn::Tensor& x);

    struct Stats {
        ttnn::Tensor a, b, mean, rstd;
    };

    // s1, s2: moments() of [N, C, ...]; weight, bias: TILE bf16 / fp32 [1, C] or null.
    // Returns per plane a = rstd w, b = bias - mean a (TILE fp32 [N, C, 1, 1]) and
    // mean, rstd per (n, group) (TILE [N, groups] in `out`), count elements per group.
    Stats stats(
        const ttnn::Tensor& s1,
        const ttnn::Tensor& s2,
        const ttnn::Tensor* weight,
        const ttnn::Tensor* bias,
        uint32_t groups,
        float count,
        float eps,
        tt::tt_metal::DataType out
    );

    // x: TILE bf16 / fp32 [N, C, H, W]; a, b: TILE fp32 [N, C, 1, 1]. Returns y = x a +
    // b (per (n, c)) in x's dtype and `memory`.
    ttnn::Tensor affine(
        const ttnn::Tensor& x,
        const ttnn::Tensor& a,
        const ttnn::Tensor& b,
        const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_group_norm
