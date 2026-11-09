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

    // x: NCHW TILE bf16 [N, C, H, W]. Returns the zero-padded channels-last rows that
    // run() reads (row-major [ceil(N Hp Wp / 32), 32 Cp]: 32 pixel rows per page, Cp = C
    // rounded up to 32, channels from C on zero); padding ph rows on top, pw columns on
    // the left, the rest of g's Hp x Wp at the bottom / right.
    ttnn::Tensor to_rows(
        const ttnn::Tensor& x, const Geometry& g, uint32_t ph, uint32_t pw
    );

    // The pixel rows of a conv output TILE [rows, O]: Sh x Sw phases of phase_rows rows
    // each; output pixel (image, y, x) is row ((y mod Sh) Sw + x mod Sw) phase_rows +
    // (image OHq + y / Sh) pitch + x / Sw (one phase: (image OH + y) pitch + x).
    struct OutRows {
        uint32_t n, o, oh, ow, pitch, ohq, sh = 1, sw = 1, phase_rows = 0;
    };

    // out: run()'s TILE [N OH Wp, O]. Returns NCHW TILE bf16 [N, O, OH, ow] in
    // `memory`.
    ttnn::Tensor to_nchw(
        const ttnn::Tensor& out,
        const Geometry& g,
        uint32_t ow,
        const tt::tt_metal::MemoryConfig& memory
    );

    // out: TILE [rows, O] bf16 laid out as `r` says (Sw <= 8). Returns NCHW TILE bf16
    // [N, O, OH, OW] in `memory`.
    ttnn::Tensor to_nchw(
        const ttnn::Tensor& out,
        const OutRows& r,
        const tt::tt_metal::MemoryConfig& memory
    );

    // x: NCL TILE bf16 [N, C, L]. Returns the zero-padded channels-last rows [N Wp, Cp]
    // as to_rows() lays them out (one image row of Wp), padding pw positions on the
    // left, the rest of Wp on the right.
    ttnn::Tensor to_rows_1d(
        const ttnn::Tensor& x, uint32_t n, uint32_t c, uint32_t wp, uint32_t pw
    );

    // out: TILE [rows, O] bf16 laid out as `r` says (OH = 1, Sw <= 8). Returns NCL
    // TILE bf16 [N, O, OW] in `memory`.
    ttnn::Tensor to_ncl(
        const ttnn::Tensor& out,
        const OutRows& r,
        const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_conv
