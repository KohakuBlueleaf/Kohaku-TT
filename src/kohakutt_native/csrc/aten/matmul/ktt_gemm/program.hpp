// Block matmul: out [..., M, N] = a [..., M, K] @ w (device kernels in aten/matmul/ktt_gemm/;
// design in docs/reference/kernels/gemm.md).
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_gemm {

    // A launch plan. 2D layout: grid rows x cols; A multicast along each panel row from the
    // panel's left column, W multicast down each column from the top row. 1D layout: every
    // core owns pcm full output rows and reads its own A; each column's top core reads W and
    // multicasts it down the column.
    struct Plan {
        bool one_d = false;
        uint32_t rows = 0, cols = 0;  // cores used (y, x)
        uint32_t pcm = 0, pcn = 0;    // output tiles per core (rows, columns)
        uint32_t bm = 0, bn = 0;      // block tiles (rows, columns)
        uint32_t nbh = 0, nbw = 0;    // blocks per core
        uint32_t kb = 0, nk = 0;      // K tiles per block, K blocks
        uint32_t sh = 0, sw = 0;      // output subblock
        uint32_t l1 = 0;              // circular-buffer bytes per core

        std::string str() const;
    };

    // The plan for [mt, kt] @ [kt, nt] tiles with w tiles of `w_tile` bytes on a grid of
    // gx x gy cores, its circular buffers within `budget` bytes per core; nullopt when none
    // fits.
    std::optional<Plan> plan(
        uint32_t mt, uint32_t kt, uint32_t nt, uint32_t w_tile, uint32_t gx, uint32_t gy, uint32_t budget
    );

    // True when this kernel takes a (bf16 TILE [..., M, K]) @ w (2-D TILE, bf16 / bfp8 /
    // bfp4).
    bool supports(const ttnn::Tensor& a, const ttnn::Tensor& w, bool w_nk);

    // a: TILE bf16 [..., M, K]; w: TILE [K, N] (w_nk false) or [N, K] (w_nk true), bf16,
    // bfp8 or bfp4; K's tile padding zero in both. Returns TILE bf16 [..., M, N] in
    // `memory`, or nullopt when no plan fits the L1 left free by live L1 buffers.
    std::optional<ttnn::Tensor> run(
        const ttnn::Tensor& a,
        const ttnn::Tensor& w,
        bool w_nk,
        const tt::tt_metal::MemoryConfig& memory,
        const ttnn::ComputeKernelConfig& config
    );

}  // namespace ktt::bridge::ktt_gemm
