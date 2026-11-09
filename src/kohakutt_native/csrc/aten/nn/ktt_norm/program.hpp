// Kohaku-TT's own layer norm and RMS norm over the last dim (aten/nn/ktt_norm/): fp32
// statistics and arithmetic on the SFPU for bf16 and fp32 inputs, any width, sizes as runtime
// arguments. Design: docs/reference/kernels/layernorm.md.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer_types.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ktt::bridge::ktt_norm {

    // How tile rows are split over the grid: groups of `group` cores share a tile row, each
    // core holding `seg` column tiles of it (the last core of a group fewer); `groups` groups
    // take tile rows group-strided.
    struct Plan {
        uint32_t group = 1, seg = 1, groups = 1;
        uint32_t row_depth = 2;  // tile rows of x (and dy) resident per core
        uint32_t l1 = 0;
        std::string str() const;
    };

    // The plan for `rows` tile rows of `wt` tiles; nullopt when no split fits L1. `x_tile`,
    // `g_tile`: tile bytes of x and of gamma / beta (0: absent), `params` of them present.
    std::optional<Plan> plan(
        uint32_t rows, uint32_t wt, uint32_t x_tile, uint32_t g_tile, uint32_t params,
        bool backward
    );

    // True when the kernels take x (TILE, bf16 / fp32, rank >= 2, interleaved) with gamma /
    // beta (null, or [1, W] TILE bf16 / fp32).
    bool supports(
        const ttnn::Tensor& x, const ttnn::Tensor* gamma, const ttnn::Tensor* beta
    );

    struct Forward {
        ttnn::Tensor y;
        std::optional<ttnn::Tensor> mean, rstd;  // fp32 [..., M, 1]
    };

    // y = (x - mean) * rstd * gamma + beta over the last dim (rms: y = x * rstd * gamma (+
    // beta), rstd = 1 / sqrt(mean(x^2) + eps)); `stats`: also return mean (layer norm) and
    // rstd. `fidelity`, `approx`: the compute config's, from numerics (no FPU multiply runs;
    // approx selects the SFPU rsqrt).
    Forward forward(
        const ttnn::Tensor& x,
        const ttnn::Tensor* gamma,
        const ttnn::Tensor* beta,
        float eps,
        bool rms,
        bool stats,
        const tt::tt_metal::MemoryConfig& memory,
        tt::tt_metal::MathFidelity fidelity,
        bool approx
    );

    struct Backward {
        std::optional<ttnn::Tensor> dx, dgamma, dbeta;
    };

    // Gradients of forward() from its saved mean (layer norm) / rstd (fp32 [..., M, 1]): dx
    // in x's dtype, dgamma / dbeta [1, W] in gamma's dtype (`param_dtype` when gamma is
    // absent).
    Backward backward(
        const ttnn::Tensor& dy,
        const ttnn::Tensor& x,
        const ttnn::Tensor* mean,
        const ttnn::Tensor& rstd,
        const ttnn::Tensor* gamma,
        bool rms,
        bool want_dx,
        bool want_dgamma,
        bool want_dbeta,
        tt::tt_metal::DataType param_dtype,
        const tt::tt_metal::MemoryConfig& memory,
        tt::tt_metal::MathFidelity fidelity,
        bool approx
    );

}  // namespace ktt::bridge::ktt_norm
