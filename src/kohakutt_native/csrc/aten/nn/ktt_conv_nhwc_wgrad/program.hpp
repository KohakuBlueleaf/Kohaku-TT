// Weight gradient of Kohaku-TT's channels-last convolution (aten/nn/ktt_conv_nhwc): per
// tap, X^T (channels x pixels) times dY shifted by the tap (pixels x output channels),
// on Nhwc tiles. The shifted dY tiles are built from channel planes by local copies, as
// the conv's A tiles (aten/nn/ktt_conv_nhwc_wgrad/{reader,writer,feeder,compute}.cpp;
// X^T by rotate_{compute,writer}.cpp).
#pragma once

#include <cstdint>
#include <string>

#include <ttnn/tensor/tensor.hpp>

#include "aten/nn/ktt_conv_nhwc/program.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_conv_nhwc_wgrad {

    using ktt_conv_nhwc::Geometry;
    using ktt_conv_nhwc::Tables;

    // X^T of x (Nhwc TILE [1, 1, N H W, C] bf16, zero tile padding): TILE [Ct 32, Mt
    // 32] bf16 in DRAM (written into `into` when given), tile (cb, m) the transpose of
    // x's tile (m, cb). On the stream worker.
    ttnn::Tensor rotate(
        const ttnn::Tensor& x, const Geometry& g, const ttnn::Tensor* into = nullptr
    );

    // The weight gradient of the conv g from rotate()'s `xt` and dy (Nhwc TILE [1, 1, N
    // H W, O] bf16, zero tile padding): TILE [Ct K K 32, Nt 32] bf16 in DRAM, row (cb K
    // K + t) 32 + c and column o holding dW[o, 32 cb + c, t]
    // (ktt_conv_nhwc::prepare()'s layout). Throws when no block size fits the
    // circular-buffer budget or `tables` has no copy table. On the stream worker.
    ttnn::Tensor run(
        const ttnn::Tensor& xt,
        const ttnn::Tensor& dy,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config,
        Tables tables
    );

    // Test hook. With go > 0, run() takes only the plan (go Cin-group columns per slot,
    // bc Cin tiles per block, kb pixel tiles per block, bo Cout tiles per group), still
    // subject to the budget; go 0: the planner's choice. `iters`: passes of the
    // kernels' loops per launch.
    void debug_plan(uint32_t go, uint32_t bc, uint32_t kb, uint32_t bo, uint32_t iters);

    // The plan of the last run(): "go G bc B x NQ kb K x NPB bo O slots S x ROWS".
    std::string last_plan();

    // Drops the cached plans and copy tables (device close, on the stream worker).
    void clear();

}  // namespace ktt::bridge::ktt_conv_nhwc_wgrad
