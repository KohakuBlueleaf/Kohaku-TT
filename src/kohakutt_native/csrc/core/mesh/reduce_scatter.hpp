// Kohaku-TT's own collectives on a 1x2 mesh (TT-NN headers; device side only).
#pragma once

#include <optional>
#include <utility>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::mesh_rs {

    // The fp32 sum over the two chips of `t` (each chip's partial value, rank >= 2,
    // leading dims 1, TILE, DRAM interleaved), scattered on the rows: chip k gets rows
    // [k H / 2, (k + 1) H / 2). fp32 adds on the SFPU (round to nearest even); tiles
    // cross every fabric link between the chips. nullopt when the mesh is not 1x2 over
    // `axis` 1, `t` is not fp32 TILE in DRAM, or H is not a multiple of 64.
    std::optional<ttnn::Tensor> reduce_scatter_rows(
        const ttnn::Tensor& t, uint32_t axis
    );

    // reduce_scatter_rows on the columns: chip k gets columns [k W / 2, (k + 1) W / 2);
    // nullopt when W is not a multiple of 64 (other conditions as for the rows).
    std::optional<ttnn::Tensor> reduce_scatter_columns(
        const ttnn::Tensor& t, uint32_t axis
    );

    // The sum over the two chips of `t` (each chip's partial value; bf16 or fp32, TILE,
    // DRAM interleaved), plus `addend` (each chip's copy; same dtype and padded shape)
    // when given, on both chips, in one program: every tile crosses the fabric once
    // each way and is added in fp32 on the SFPU. nullopt when the mesh is not 1x2 over
    // `axis` 1 or an operand does not qualify.
    std::optional<ttnn::Tensor> all_reduce(
        const ttnn::Tensor& t, const ttnn::Tensor* addend, uint32_t axis
    );

    // all_reduce(t, addend, 1) of a bf16 [M, K] operand and, in the same program, its
    // sum x times w (each chip's 2-D TILE weight, [K, N] or with `w_nk` [N, K]): {x, x
    // @ w}. The tiles are exchanged column by column and ktt_gemm takes each K block
    // once the drains have written it. nullopt when all_reduce or ktt_gemm does not
    // take the operands or K does not divide into whole K blocks per link.
    std::optional<std::pair<ttnn::Tensor, ttnn::Tensor>> all_reduce_matmul(
        const ttnn::Tensor& t,
        const ttnn::Tensor* addend,
        const ttnn::Tensor& w,
        bool w_nk,
        const ttnn::ComputeKernelConfig& config
    );

    // Frees the global semaphores (before the device closes).
    void clear();

}  // namespace ktt::bridge::mesh_rs
