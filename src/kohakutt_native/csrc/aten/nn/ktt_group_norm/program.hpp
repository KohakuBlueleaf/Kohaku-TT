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

    // Group norm of x stored with its channels last: TILE bf16 / fp32 whose last dim is
    // C and whose rows are `images` blocks of `rows` (pixels of one image).
    struct Rows {
        uint32_t images = 1, rows = 1, channels = 1, groups = 1;
        float eps = 0;
    };

    // True when rows_norm() takes r: groups of whole channels reaching at most 8
    // channel tiles each, and no tile shared by three images (one image, `rows` a
    // multiple of 32 or >= 32).
    bool rows_supported(const Rows& r);

    // y = (x - mean) rstd weight + bias per (image, group) in x's dtype, shape and
    // `memory`, tile padding zero; x's tile padding must be zero. weight, bias: TILE
    // bf16 / fp32 [1, C] or null. Two programs: per-channel sums (rows_moments_*.cpp),
    // then statistics and the affine (rows_affine_*.cpp). `sums`, when given, receives
    // (mean, rstd) of each channel's group per (image, channel) as TILE fp32 [images,
    // C].
    ttnn::Tensor rows_norm(
        const ttnn::Tensor& x,
        const ttnn::Tensor* weight,
        const ttnn::Tensor* bias,
        const Rows& r,
        const tt::tt_metal::MemoryConfig& memory,
        std::pair<ttnn::Tensor, ttnn::Tensor>* sums = nullptr
    );

    // The input gradient of rows_norm() for x, dy (same shape and dtype, tile padding
    // zero) with the forward's mean and rstd expanded per channel (TILE fp32 [images,
    // C]) and weight (TILE bf16 / fp32 [1, C] or null), in `memory`. `terms` receives
    // per (image, channel) the weight and bias gradient terms as TILE fp32 [images, C]
    // (their sums over the images are the gradients). Two programs (rows_bwd_*.cpp).
    ttnn::Tensor rows_backward(
        const ttnn::Tensor& x,
        const ttnn::Tensor& dy,
        const ttnn::Tensor& mean,
        const ttnn::Tensor& rstd,
        const ttnn::Tensor* weight,
        const Rows& r,
        const tt::tt_metal::MemoryConfig& memory,
        std::pair<ttnn::Tensor, ttnn::Tensor>& terms
    );

}  // namespace ktt::bridge::ktt_group_norm
