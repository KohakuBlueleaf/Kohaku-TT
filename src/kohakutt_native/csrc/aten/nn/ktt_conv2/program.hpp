// Kohaku-TT's grouped, depthwise and strided convolution as a sparse-K implicit GEMM
// over channels-last rows: output column tile j (32 output channels) sums, over the
// taps, only the input channel tiles of its groups (a group's own tiles, a
// block-diagonal tile when a group has under 32 channels, a diagonal tile for
// depthwise). The forward conv and the input gradient (as phases, any stride) are one
// gather program; the weight gradient is a second program. Every size is a runtime
// argument (aten/nn/ktt_conv2/*.cpp).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <ttnn/tensor/tensor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_conv2 {

    // The most phases per axis and taps per axis a gather takes.
    constexpr uint32_t kMaxPhases = 8;
    constexpr uint32_t kMaxTaps = 16;

    // One spatial axis of a gather over a zero-padded input of extent `in`: output
    // position y of phase f reads input position y stride + offset for each of f's taps
    // (offset, kernel index).
    struct Axis {
        uint32_t in = 1, out = 1, stride = 1;
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> taps{{{0, 0}}};

        uint32_t phases() const {
            return static_cast<uint32_t>(taps.size());
        }
        uint32_t most_taps() const;
    };

    // A gather conv: N images of C input channels (Cp: C rounded up to 32) to O output
    // channels in `groups` groups, weight kernel KH x KW, axes h / w, `pitch` output
    // rows per output image row (>= w.out). Output rows: phases (fh, fw) in order, each
    // phase_tiles() row tiles of [N, h.out, pitch].
    struct Geometry {
        uint32_t n = 1, c = 32, cp = 32, o = 32, kh = 1, kw = 1, groups = 1;
        Axis h, w;
        uint32_t pitch = 1;

        uint32_t taps() const {
            return kh * kw;
        }
        uint32_t phases() const {
            return h.phases() * w.phases();
        }
        uint32_t phase_tiles() const {
            return (n * h.out * pitch + 31) / 32;
        }
        uint32_t out_rows() const {
            return phases() * phase_tiles() * 32;
        }
        uint32_t most_taps() const {
            return h.most_taps() * w.most_taps();
        }
    };

    // The forward conv of x [N, C, H, W] with w [O, C / groups, KH, KW]; `pad` gets the
    // input padding (top, left) of its to_rows().
    Geometry forward(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        std::array<uint32_t, 2>& pad
    );

    // The input gradient [N, C, H, W] (sizes x) of that forward conv as a gather over
    // grad_out [N, O, OH, OW]: stride x stride phases of stride-1 taps, weight
    // transposed per group (prepare(..., transposed = true)); `pad` as in forward().
    Geometry input_grad(
        const std::vector<int64_t>& x,
        const std::vector<int64_t>& w,
        std::array<int64_t, 2> stride,
        std::array<int64_t, 2> padding,
        std::array<int64_t, 2> dilation,
        int64_t groups,
        std::array<uint32_t, 2>& pad
    );

    // True when g's groups have one input channel (depthwise, channel multipliers): the
    // weights travel packed (prepare() returns ROW_MAJOR [Ot, taps 32 Cg] bf16, expanded
    // into tiles in L1) and the weight gradient's partials are packed alike.
    bool packed(const Geometry& g);

    // Input channel tiles [k0, k1) that output column tile j reads.
    void channel_tiles(const Geometry& g, uint32_t j, uint32_t& k0, uint32_t& k1);

    // The most channel tiles one output column tile reads.
    uint32_t channel_tiles_max(const Geometry& g);

    // The gather program's choice for `g` with `weight_tile` bytes per weight tile:
    // output column tiles per core, taps per K block, the tap window, taps per read
    // barrier, and the tap-row split (split > 1: each work item takes a range of tap rows
    // and writes fp32 partial sums, summed by a second program); nullopt when one column
    // tile's weights do not fit the circular-buffer budget.
    struct Plan {
        uint32_t nt, block;
        bool window;
        uint32_t group = 1, split = 1;
    };
    std::optional<Plan> plan(const Geometry& g, uint32_t weight_tile);

    // Static circular-buffer bytes per core of `p` for `g`.
    uint32_t cb_bytes(const Geometry& g, const Plan& p, uint32_t weight_tile);

    // weight: the torch weight TILE [O, Cg, KH, KW] bf16 (KH, KW <= 32). Returns the
    // sparse weight TILE [Ot KH KW Kmax 32, 32] bf16: tile (j, t, k) holds tap t's
    // weights from channel tile k0(j) + k to output tile j (zero outside the group
    // blocks). `transposed`: weight is the forward weight [C, Og, KH, KW] of the conv
    // whose input gradient `g` describes (g's C, O are the forward's O, C): its
    // per-group transpose. `one_d`: weight is TILE [O, Cg, KW] (KH = 1). packed(g): the
    // packed rows instead, row j holding per tap t and column cc the Cg weights of
    // output channel 32 j + cc.
    ttnn::Tensor prepare(
        const ttnn::Tensor& weight, const Geometry& g, bool transposed, bool one_d
    );

    // x_rows: ktt_conv::to_rows() of the input ([N h.in w.in, Cp] rows); w: prepare()'s
    // tiles (bf16 or bfp8). Returns TILE [out_rows(), O] bf16 (rows past each phase's
    // N h.out pitch and columns past each image row's output width are not pixels).
    ttnn::Tensor run(
        const ttnn::Tensor& x_rows,
        const ttnn::Tensor& w,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config
    );

    // The weight gradient's split: output column tiles per core, output row tiles per
    // chunk (one core's K) and chunks; nullopt when it does not fit the budget.
    struct WeightGradPlan {
        uint32_t nt, per_chunk, chunks;
        bool window;
    };
    std::optional<WeightGradPlan> weight_grad_plan(const Geometry& g);

    // Weight gradient of the one-phase conv `g` describes: x_rows its to_rows() input
    // rows, g_rows the to_rows() rows of grad_out [N, O, OH, OW] over g.pitch columns
    // (zero from OW on). Per weight tile and pixel chunk, G^T @ A (fp32 partials); then
    // their sum per weight in the torch layout TILE [O, Cg, KH, KW] bf16 (`one_d`:
    // [O, Cg, KW]; round to nearest even).
    ttnn::Tensor weight_grad(
        const ttnn::Tensor& x_rows,
        const ttnn::Tensor& g_rows,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config,
        bool one_d
    );

}  // namespace ktt::bridge::ktt_conv2
