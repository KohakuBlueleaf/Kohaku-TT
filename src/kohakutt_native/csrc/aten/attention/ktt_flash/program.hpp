// Flash attention (aten/attention/ktt_flash/{reader,compute,writer}.cpp; design sheet
// docs/reference/kernels/flash.md).
#pragma once

#include <cstdint>
#include <optional>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_flash {

    // True when run() takes q [B, H, Sq, D], k [B, Hk, Sk, D], v [B, Hk, Sk, Dv] (TILE bf16, H a
    // multiple of Hk, D and Dv multiples of 32 up to 256, causal only with Sq == Sk and no
    // mask) and a mask [Bm, Hm, Sqm, Sk] (Bm 1 or B, Hm 1 or H, Sqm 1 or Sq).
    bool supports(
        const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask, bool causal
    );

    // The mask operand of run() from an additive mask in logits: bf16, clamped at -1e4 and
    // divided by `scale`.
    ttnn::Tensor mask_operand(const ttnn::Tensor& mask, float scale);

    // softmax(q k^T scale [+ mask] [+ causal mask]) v: TILE bf16 [B, H, Sq, Dv]. `mask` is
    // mask_operand()'s output; `approx` selects the approximate exponential.
    ttnn::Tensor run(
        const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask, float scale, bool causal,
        const ttnn::ComputeKernelConfig& config, bool approx
    );

}  // namespace ktt::bridge::ktt_flash
