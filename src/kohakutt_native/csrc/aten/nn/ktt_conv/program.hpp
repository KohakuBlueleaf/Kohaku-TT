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

    // x_rows: row-major zero-padded channels-last input rows (N Hp Wp pixel rows of C
    // bf16, P rows per page: to_rows() or [N Hp Wp, C]); w: TILE [KH KW C, O] bf16
    // (rows ordered tap-major, then channel).
    // Returns TILE [N OH Wp, O] bf16: every padded-width column (the last Wp - OW are
    // not output pixels).
    ttnn::Tensor run(
        const ttnn::Tensor& x_rows,
        const ttnn::Tensor& w,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config
    );

    // x: NCHW TILE bf16 [N, C, H, W] (C a multiple of 32). Returns the zero-padded
    // channels-last rows that run() reads (row-major [ceil(N Hp Wp / 32), 32 C]: 32
    // pixel rows per page); padding ph rows on top, pw columns on the left.
    ttnn::Tensor to_rows(
        const ttnn::Tensor& x, const Geometry& g, uint32_t ph, uint32_t pw
    );

    // out: run()'s TILE [N OH Wp, O]. Returns NCHW TILE bf16 [N, O, OH, ow] in
    // `memory`.
    ttnn::Tensor to_nchw(
        const ttnn::Tensor& out,
        const Geometry& g,
        uint32_t ow,
        const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_conv
