// ktt_adam: one Adam / AdamW step of one parameter as one program (TT-NN headers;
// device side only).
#pragma once

#include <array>

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_adam {

    // The step's coefficients:
    //   g' = (negate ? -g : g) + l2 p;  m' = b1 m + (1 - b1) g';  v' = b2 v + (1 - b2)
    //   g'^2 p' = decay p - step m' / (sqrt(v') scale + eps)
    struct Coefficients {
        float b1 = 0, b2 = 0, scale = 1, eps = 0, step = 0, decay = 1, l2 = 0;
        bool negate = false;
    };

    // True for bf16 / fp32 TILE tensors of one dtype and padded shape, interleaved.
    bool supports(const std::array<const ttnn::Tensor*, 4>& pgmv);

    // Runs the step over p, g, m, v into p', m', v' (allocated by the caller like p);
    // fp32 arithmetic on the SFPU. The coefficients are runtime arguments: one program
    // per tensor geometry.
    void run(
        const std::array<const ttnn::Tensor*, 4>& pgmv,
        const std::array<const ttnn::Tensor*, 3>& out,
        const Coefficients& c
    );

}  // namespace ktt::bridge::ktt_adam
