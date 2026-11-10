// Numerics: the math settings ops are queued with (fidelity of FPU multiplies, SFPU
// approximations), as a per-thread scope read when an op is queued.
#pragma once

#include <optional>

#include <tt-metalium/base_types.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::numerics {

    using tt::tt_metal::MathFidelity;

    // Attention inputs: as given; K centered over the sequence (softmax-invariant);
    // centered K and Q / K in bfp8.
    enum class Attention { Exact = 0, Centered = 1, CenteredBfp8 = 2 };

    // fidelity: FPU multiply passes (LoFi < HiFi2 < HiFi3 < HiFi4); approx: SFPU
    // approximations (exp, reciprocal, ...); strict: kernel requirements ignored;
    // attention: how attention prepares Q / K.
    struct Numerics {
        MathFidelity fidelity = MathFidelity::HiFi2;
        bool approx = false;
        bool strict = false;
        Attention attention = Attention::Exact;
    };

    // What a kernel needs or can use: the effective fidelity is the scope's, raised to
    // `floor` and lowered to `ceiling`; `exact` turns approximations off.
    struct Requirements {
        std::optional<MathFidelity> floor;
        std::optional<MathFidelity> ceiling;
        bool exact = false;
    };

    // The calling thread's numerics: the innermost scope, else (in a backward pass, see
    // set_backward_probe) the most recently opened scope of any thread, else the
    // process default (KTT_FIDELITY / KTT_MATMUL_FIDELITY, KTT_APPROX; HiFi2, exact).
    Numerics current();
    void push(const Numerics& n);
    void pop();

    // `probe` tells whether the calling thread runs an autograd backward pass.
    void set_backward_probe(bool (*probe)());

    // current() with `r` applied (unless strict).
    Numerics effective(const Requirements& r = {});

    // A TT-NN compute config of `n`: fp32 destination accumulation always on.
    ttnn::ComputeKernelConfig compute_config(
        const Numerics& n, bool packer_l1_acc = false
    );

    // Fidelity rank 0..3 (LoFi, HiFi2, HiFi3, HiFi4) and back.
    int rank(MathFidelity f);
    MathFidelity from_rank(int r);

}  // namespace ktt::bridge::numerics
