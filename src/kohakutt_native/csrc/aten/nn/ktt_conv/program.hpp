// Kohaku-TT's 2-D convolution (stride 1, one group) as an implicit GEMM: output rows of
// the padded-width image [N OH Wp, O] = sum over taps (i, j) of input rows shifted by
// i dh Wp + j dw [.., C] @ weight rows of the tap [C, O]. One program per (C, O, taps)
// class: every size is a runtime argument
// (aten/nn/ktt_conv/{reader,compute,writer}.cpp).
#pragma once

#include <cstdint>
#include <optional>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_conv {

    struct Geometry {
        uint32_t n, c, hp, wp, o, kh, kw, dh, dw, oh;
    };

    // Output column tiles per core (weight block resident in L1) for `g`, or nullopt
    // when the weight rows of one column tile do not fit.
    std::optional<uint32_t> columns_per_core(const Geometry& g);

    // x_rows: row-major [N Hp Wp, C] bf16 (zero-padded input, channels last, one page
    // per pixel); w: TILE [KH KW C, O] bf16 (rows ordered tap-major, then channel).
    // Returns TILE [N OH Wp, O] bf16: every padded-width column (the last Wp - OW are
    // not output pixels).
    ttnn::Tensor run(
        const ttnn::Tensor& x_rows,
        const ttnn::Tensor& w,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config
    );

}  // namespace ktt::bridge::ktt_conv
