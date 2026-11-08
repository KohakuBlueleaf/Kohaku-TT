// Kohaku-TT's scaled dot-product attention (flash attention): per (batch, head, q
// chunk) unit, softmax(Q K^T scale) V with the online softmax over K chunks, every size
// a runtime argument (aten/attention/ktt_attention/{reader,compute,writer}.cpp).
#pragma once

#include <cstdint>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_attention {

    // q [B, H, Sq, D], k [B, Hk, Sk, D], v [B, Hk, Sk, Dv]: TILE bf16, H a multiple of
    // Hk. Returns TILE [B, H, Sq, Dv] bf16. causal: key j attends query i iff j <= i.
    // fp32_dest: fp32 DST and fp32 intermediates.
    ttnn::Tensor run(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        float scale,
        bool causal,
        const ttnn::ComputeKernelConfig& config,
        bool fp32_dest
    );

}  // namespace ktt::bridge::ktt_attention
