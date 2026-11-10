// Flash attention (aten/attention/ktt_flash/{reader,compute,writer}.cpp; design sheet
// docs/reference/kernels/flash.md).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <ttnn/tensor/tensor.hpp>

#include "aten/attention/ktt_flash/varlen_plan.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::ktt_flash {

    // True when run() takes q [B, H, Sq, D], k [B, Hk, Sk, D], v [B, Hk, Sk, Dv] (TILE
    // bf16, H a multiple of Hk, D and Dv multiples of 32 up to 256, causal only with Sq
    // == Sk and no mask) and a mask [Bm, Hm, Sqm, Sk] (Bm 1 or B, Hm 1 or H, Sqm 1 or
    // Sq).
    bool supports(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask,
        bool causal
    );

    // The mask operand of run() from an additive mask in logits: bf16, clamped at -1e4
    // and divided by `scale`; the key columns past the last key (tile padding) masked.
    ttnn::Tensor mask_operand(const ttnn::Tensor& mask, float scale);

    // Debug dump of one query tile row (global index over B x H x Sq tiles): per step
    // (key chunk of up to 3 tiles) 8 + W fp32 tiles, W = Dv / 32 + 1 (+ 1 with stats):
    // S tiles 0-2 and the state tile before the vector step, P tiles 0-2 and the state
    // tile after it, then the row's O tiles, l and (stats) lane sums after the step's
    // accumulation. `tiles` [pages x 32, 32].
    struct Dump {
        uint32_t row = 0;
        uint32_t pages = 0;
        ttnn::Tensor tiles;
    };

    // softmax(q k^T scale [+ mask] [+ causal mask]) v: TILE bf16 [B, H, Sq, Dv]. `mask`
    // is mask_operand()'s output; `approx` selects the approximate exponential. With
    // `stats`, also writes there each query row's lse = ln sum_j exp(scale s_j) as TILE
    // fp32 [B, H, Sq, 32] (row r's lse in every column of row r).
    ttnn::Tensor run(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask,
        float scale,
        bool causal,
        const ttnn::ComputeKernelConfig& config,
        bool approx,
        ttnn::Tensor* stats = nullptr,
        Dump* dump = nullptr
    );

    // True when run_varlen() takes the packed q [Tq, H D], k [Tk, Hk D], v [Tk, Hk Dv]
    // (TILE bf16, a token's `h` / `hk` heads side by side, H a multiple of Hk, D and Dv
    // multiples of 32 up to 256) and its circular buffers fit.
    bool supports_varlen(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        uint32_t h,
        uint32_t hk
    );

    // Packed variable-length attention: every query token attends to the keys of its
    // row record in `rows` (varlen::plan of the sequence offsets, one record per query
    // tile row). `plan`: the records as TILE int32 [pages x 32, 32] (flash.hpp:
    // kRecordWords words per row); `masks`: the plan's mask tiles as TILE bf16 [tiles x
    // 32, 32] (at least one tile). Returns TILE bf16 [Tq, H Dv]; with `stats` also each
    // query row's log-sum-exp as TILE fp32 [Tq, H 32] (head h's value in columns 32 h
    // .. 32 h + 31). The kernels read and write the heads in place (no per-head copy),
    // zero the rows past the last token of the tiles they read, and one program serves
    // every set of lengths (they reach the kernels as runtime arguments and through
    // `plan`). `key_mask`: TILE bf16 [1, Tk], added to the scores of each key
    // (additive, already divided by `scale`; finite).
    ttnn::Tensor run_varlen(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        uint32_t h,
        uint32_t hk,
        const ttnn::Tensor& plan,
        const ttnn::Tensor& masks,
        const std::vector<varlen::Row>& rows,
        float scale,
        const ttnn::ComputeKernelConfig& config,
        bool approx,
        ttnn::Tensor* stats = nullptr,
        const ttnn::Tensor* key_mask = nullptr
    );

    // True when backward_varlen() takes the q, k, v of a run_varlen(): the backward
    // programs' circular buffers fit.
    bool supports_varlen_backward(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        uint32_t h,
        uint32_t hk
    );

    // Gradients {dq [Tq, H D], dk [Tk, Hk D], dv [Tk, Hk Dv]} (TILE bf16) of run_varlen
    // from its output `out`, the output gradient `grad` [Tq, H Dv] and its `stats`.
    // `plan`, `masks`, `rows`: run_varlen's; `key_plan`, `key_masks`, `key_rows`:
    // varlen::key_plan of the same offsets in the same tensor forms (one record per key
    // tile row). Two programs (dQ with the row statistics, then dK / dV), each serving
    // every set of lengths. `key_mask`: run_varlen's.
    std::array<ttnn::Tensor, 3> backward_varlen(
        const ttnn::Tensor& grad,
        const ttnn::Tensor& out,
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const ttnn::Tensor& stats,
        uint32_t h,
        uint32_t hk,
        const ttnn::Tensor& plan,
        const ttnn::Tensor& masks,
        const std::vector<varlen::Row>& rows,
        const ttnn::Tensor& key_plan,
        const ttnn::Tensor& key_masks,
        const std::vector<varlen::Row>& key_rows,
        float scale,
        const ttnn::ComputeKernelConfig& config,
        const ttnn::Tensor* key_mask = nullptr
    );

    // True when backward() takes q, k, v of a run() without mask: the backward
    // programs' circular buffers and runtime args fit (either orientation; a sequence
    // length that is not a multiple of 32 runs the default one).
    bool supports_backward(
        const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v, bool causal
    );

    // Gradients {dq, dk, dv} (TILE bf16) of run(q, k, v, no mask, scale, causal) from
    // its output `out`, the output gradient `grad` and run()'s `stats`; dk and dv sum
    // over each K / V head's query heads. `key_rows`: dK / dV from S^T = K Q^T instead
    // of the forward's S = Q K^T (for ordinary logits only: its scores differ from the
    // forward's by the matrix unit's operand-role rounding, see
    // docs/reference/kernels/flash.md, Backward orientation).
    std::array<ttnn::Tensor, 3> backward(
        const ttnn::Tensor& grad,
        const ttnn::Tensor& out,
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const ttnn::Tensor& stats,
        float scale,
        bool causal,
        const ttnn::ComputeKernelConfig& config,
        bool key_rows = false
    );

}  // namespace ktt::bridge::ktt_flash
