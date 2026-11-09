// Kohaku-TT's matmul with the weight block resident per core: out [..., M, N] = a [...,
// M, K] @ w, every size a runtime argument (aten/matmul/ktt_matmul/{reader,compute,
// writer}.cpp).
#pragma once

#include <cstdint>
#include <optional>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_matmul {

    // Output column tiles per core for K (tiles) and w's tile bytes, or nullopt when
    // one column's weight tiles do not fit in L1.
    std::optional<uint32_t> columns_per_core(uint32_t kt, uint32_t nt, uint32_t w_tile);

    // Circular-buffer bytes per core run() allocates for K (tiles), N (tiles) and w's
    // tile bytes.
    uint32_t l1_bytes(uint32_t kt, uint32_t nt, uint32_t w_tile);

    // a: TILE bf16 [..., M, K]; w: TILE bf16 / bfp8 [K, N] (w_nk false) or [N, K] (w_nk
    // true). Returns TILE bf16 [..., M, N] in `memory`.
    ttnn::Tensor run(
        const ttnn::Tensor& a,
        const ttnn::Tensor& w,
        bool w_nk,
        const tt::tt_metal::MemoryConfig& memory,
        const ttnn::ComputeKernelConfig& config
    );

}  // namespace ktt::bridge::ktt_matmul
