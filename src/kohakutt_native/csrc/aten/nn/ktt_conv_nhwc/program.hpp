// Kohaku-TT's channels-last convolution: one group, stride 1, dilation 1, an odd K x K
// kernel with padding (K - 1) / 2, on Nhwc tiles [1, 1, N H W, C] (32 pixels x 32
// channels each). Each core owns bm output pixel tiles x bn Cout tiles and walks them
// in blocks of bm_b x bn_b tiles; its A tiles (pixels shifted per tap) are built from
// channel planes by local copies
// (aten/nn/ktt_conv_nhwc/{reader,writer,feeder,compute}.cpp).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_conv_nhwc {

    // A conv of x [N, C, H, W] with w [O, C, K, K] (K odd).
    struct Geometry {
        uint32_t n = 1, c = 1, h = 1, w = 1, o = 1, k = 1;
    };

    // W' of the weight's bf16 values `w` (row-major [O, C, K, K]), on the host:
    // row-major [Ct K K 32, Nt 32] bf16, row (cb K K + t) 32 + c holding tap t's
    // weights (taps row-major) from input channel 32 cb + c; zero past C and O.
    std::vector<uint16_t> prepare(const std::vector<uint16_t>& w, const Geometry& g);

    // The 1x1 conv's matmul weight of the weight's bf16 values `w` (row-major [O, C, 1,
    // 1]), on the host: row-major [C, O].
    std::vector<uint16_t> matmul_weight(
        const std::vector<uint16_t>& w, const Geometry& g
    );

    // Appends the copy-table words of shifted tile (m, t) of g's pixels to `words`: 32
    // pixels from 32 m on, shifted by tap t's offset (taps row-major over K x K). Run
    // count, then per run (dst byte offset in the tile, src byte offset in the planes
    // slot or 0xFFFFFFFF for zeros, bytes); a face's 16 rows split only where validity
    // changes (rows outside the image, or every row when !live, are zeros). The slot
    // holds input tiles [tile_lo, tile_lo + t_in) as two planes of slot_tiles / 2
    // tiles.
    void copy_runs(
        const Geometry& g,
        uint32_t m,
        uint32_t t,
        bool live,
        uint32_t tile_lo,
        uint32_t t_in,
        uint32_t slot_tiles,
        std::vector<uint32_t>& words
    );

    // Where run() takes its copy table from: built and cached, built for this call
    // only, or the cache only.
    enum class Tables { Cache, Once, CachedOnly };

    // The conv of x (Nhwc TILE [1, 1, N H W, C] bf16, zero tile padding) with
    // prepare()'s W' `wp`: Nhwc TILE [1, 1, N H W, O] bf16 with zero tile padding, in
    // `memory`. nullopt when no plan's circular buffers fit the budget or no fitting
    // plan has a copy table under `tables`. With `xt` (TILE [Ct 32, Mt 32] bf16), the
    // program also writes X^T into it: tile (cb, m) the transpose of x's tile (m, cb);
    // nothing is written on nullopt. On the stream worker.
    std::optional<ttnn::Tensor> run(
        const ttnn::Tensor& x,
        const ttnn::Tensor& wp,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config,
        const tt::tt_metal::MemoryConfig& memory,
        Tables tables,
        const ttnn::Tensor* xt = nullptr
    );

    // Test hook. With go > 0, run() takes only the plan (go columns per pixel block,
    // bm_b output tiles x npb pixel blocks per core, bn_b Cout tiles per W' block),
    // still subject to the budget; go 0: the planner's choice. `iters`: passes of the
    // kernels' loops per launch (timing; 1 otherwise).
    void debug_plan(
        uint32_t go, uint32_t bm_b, uint32_t npb, uint32_t bn_b, uint32_t iters
    );

    // The plan of the last run() that launched: "go G bn_b B x NNB bm_b M x NPB sub S
    // blocks X x ROWS".
    std::string last_plan();

    // Drops the cached plans and copy tables (device close, on the stream worker).
    void clear();

}  // namespace ktt::bridge::ktt_conv_nhwc
