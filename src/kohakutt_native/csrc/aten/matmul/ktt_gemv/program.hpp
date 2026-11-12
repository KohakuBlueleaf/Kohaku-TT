// Weight-bound matmul: out [..., M, N] = a [..., M, K] @ w for few rows of a, the
// output columns split over the grid (device kernels in aten/matmul/ktt_gemv/).
#pragma once

#include <cstdint>
#include <optional>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_gemv {

    // Row tiles of a (leading dims folded) up to which this kernel takes a matmul.
    constexpr uint32_t kMaxRowTiles = 4;

    // True for bf16 TILE a with up to kMaxRowTiles row tiles @ w (2-D TILE bf16 / bfp8
    // / bfp4, [N, K] with w_nk) of 48+ output column tiles, or block-float with 2 rows.
    bool supports(const ttnn::Tensor& a, const ttnn::Tensor& w, bool w_nk);

    // a @ w (w [K, N] or, w_nk, [N, K]) + bias (row 0 of a row of bf16 tiles), K pads
    // zero: TILE bf16 [..., M, N] in `memory`, or nullopt when L1 is short.
    std::optional<ttnn::Tensor> run(
        const ttnn::Tensor& a,
        const ttnn::Tensor& w,
        bool w_nk,
        const tt::tt_metal::MemoryConfig& memory,
        const ttnn::ComputeKernelConfig& config,
        const ttnn::Tensor* bias = nullptr
    );

}  // namespace ktt::bridge::ktt_gemv
