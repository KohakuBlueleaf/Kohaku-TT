// Kohaku-TT's own broadcast binary program: add / sub / mul / div of a and a b
// broadcast over a's leading dims or rows, b's tiles reused across their repeats.
#pragma once

#include <optional>
#include <vector>

#include "core/device/handle.hpp"

namespace ktt::bridge::ktt_binary {

    // How the program walks its tiles: (j, r) pairs j-major, output tile
    // r * sr + j * sj, b's tile j (filled from row / column 0 when `fill` is 1 / 2).
    struct Plan {
        uint32_t pairs = 0, nr = 1, sr = 0, sj = 1, fill = 0;
    };

    // The plan for `op` on device shapes `a` and `b` of `dtype`; nullopt where
    // TT-NN's op runs (cases not measured faster).
    std::optional<Plan> plan(
        BinaryOp op,
        DType dtype,
        const std::vector<int64_t>& a,
        const std::vector<int64_t>& b
    );

    // a `op` b by `plan`, both tiled interleaved of one dtype, into a tensor of
    // `out_config`.
    ttnn::Tensor run(
        BinaryOp op,
        const ttnn::Tensor& a,
        const ttnn::Tensor& b,
        const Plan& plan,
        const tt::tt_metal::MemoryConfig& out_config
    );

    // The plan for the complex multiply of interleaved fp32 pairs a [..., 2 N] and b:
    // b of a's shape, or broadcast over a's leading dims; nullopt for other shapes.
    std::optional<Plan> complex_plan(
        const std::vector<int64_t>& a, const std::vector<int64_t>& b
    );

    // a * b as complex numbers by `plan` (complex_plan()): fp32 tiles with (re, im) at
    // columns (2 i, 2 i + 1); conj_a / conj_b conjugate that operand.
    ttnn::Tensor run_complex_mul(
        const ttnn::Tensor& a,
        const ttnn::Tensor& b,
        const Plan& plan,
        bool conj_a,
        bool conj_b,
        const tt::tt_metal::MemoryConfig& out_config
    );

}  // namespace ktt::bridge::ktt_binary
