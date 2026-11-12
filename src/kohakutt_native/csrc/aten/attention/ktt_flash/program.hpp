// Flash attention (aten/attention/ktt_flash/{reader,compute,writer}.cpp; design sheet
// docs/reference/kernels/flash.md).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <ttnn/tensor/tensor.hpp>

#include "aten/attention/ktt_flash/varlen_plan.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ktt::bridge::program {
    class Key;
}  // namespace ktt::bridge::program

namespace ktt::bridge::ktt_flash {

    // True when run() takes q [B, H, Sq, D], k [B, Hk, Sk, D], v [B, Hk, Sk, Dv] (TILE
    // bf16, H a multiple of Hk, head dims whose buffers fit, causal (top-left, any Sq,
    // Sk) only without a mask) and a mask [Bm, Hm, Sqm, Sk] (Bm 1 or B, Hm 1 or H, Sqm
    // 1 or Sq).
    bool supports(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask,
        bool causal,
        bool lengths = false
    );

    // Empty when supports(); else the first condition the call fails. `lengths`: for
    // the key-lengths variant (its extra buffers).
    std::string refusal(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const std::optional<ttnn::Tensor>& mask,
        bool causal,
        bool lengths = false
    );

    // Empty when supports_backward(); else the first condition the call fails.
    std::string backward_refusal(
        const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v, bool causal
    );

    // The mask operand of run() from an additive mask in logits: bf16, clamped at -5e5
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

    // K or V in run()'s descriptor key; with key lengths without the key count (dim 2),
    // so one cached program serves every Sk (sharded tensors keep it).
    void add_kv(program::Key& key, const tt::tt_metal::TensorSpec& t, bool lengths);

    // Checks run()'s key-lengths plan for `shape` {B, H, Hk, Sq, Sk} on a gx x gy grid
    // with R = r: every lane core's job takes the lane sender's batch and key tiles
    // (`old_batch`: the batch from the job's first row, as before). "ok ..." or why.
    std::string debug_lengths_plan(
        const std::array<uint32_t, 5>& shape,
        bool causal,
        uint32_t r,
        uint32_t gx,
        uint32_t gy,
        bool old_batch
    );

    // add_kv()'s key of a bf16 TILE DRAM K / V of shape `kv`, in hex (tests).
    std::string debug_kv_key(const std::vector<int64_t>& kv, bool lengths);

    // Q, K and V read in place from TILE bf16 [B, S, C] bases, heads side by side: each
    // operand's view [B, H, S, D] and the column of its head 0 (D, columns: k x 32).
    // `out`: O written as TILE bf16 [B, Sq, H Dv] (heads side by side) instead of
    // [B, H, Sq, Dv].
    struct Pitch {
        std::array<uint32_t, 4> q, k, v;
        std::array<uint32_t, 3> cols;
        bool out = false;
    };

    // Why run() with `p` does not take the bases q, k, v ("" when it does).
    std::string pitched_refusal(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        const Pitch& p,
        bool causal
    );

    // softmax(q k^T scale [+ mask] [+ causal mask]) v: TILE bf16 [B, H, Sq, Dv]. `mask`
    // is mask_operand()'s output, or a raw additive bf16 mask of one query row;
    // `approx` selects the approximate exponential. With `stats`, also writes there
    // each query row's lse = ln sum_j exp(scale s_j) as TILE fp32 [B, H, Sq, 32] (row
    // r's lse in every column of row r). `lengths` (no mask, no dump): TILE int32 [B],
    // batch b's queries see its keys below lengths[b] only. `pitch` (no mask, stats,
    // dump or lengths): q, k, v are the bases it describes.
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
        Dump* dump = nullptr,
        const ttnn::Tensor* lengths = nullptr,
        const Pitch* pitch = nullptr
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
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        bool causal,
        bool lengths = false
    );

    // Gradients {dq, dk, dv} (TILE bf16) of run(q, k, v, no mask, scale, causal) from
    // its output `out`, the output gradient `grad` and run()'s `stats`; dk and dv sum
    // over each K / V head's query heads. `key_rows`: dK / dV from S^T = K Q^T instead
    // of the forward's S = Q K^T (for ordinary logits only: its scores differ from the
    // forward's by the matrix unit's operand-role rounding, see
    // docs/reference/kernels/flash.md, Backward orientation). `lengths`: run()'s key
    // lengths (default orientation); keys at or past them get zero gradients.
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
        bool key_rows = false,
        const ttnn::Tensor* lengths = nullptr
    );

}  // namespace ktt::bridge::ktt_flash
