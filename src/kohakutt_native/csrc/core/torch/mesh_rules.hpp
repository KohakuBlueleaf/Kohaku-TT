// Mesh placement rules (mesh plan R-1): an op family's factor mapping, the engine that
// picks placements per mesh axis, and running an op on its inputs' local shards.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "core/device/tt_bridge.hpp"

namespace ktt::mesh {

    // How a factor maps through the op: Pass appears in the output, Reduce is summed
    // (or maxed) away, Replicate must be whole on every chip (never sharded).
    enum class FactorKind { Pass, Reduce, Replicate };

    // An op's factor mapping over device dims. inputs[i][d] / output[d]: the factor of
    // that dim, or -1 for a dim of size 1 that broadcasts.
    struct Factors {
        std::vector<std::vector<int>> inputs;
        std::vector<int> output;
        std::vector<FactorKind> kinds;
        bridge::ReduceKind reduce = bridge::ReduceKind::Sum;
        // P(sum) inputs passing through to a P(sum) output: All (add, sub, neg,
        // scaling: every input P), One (mul: one input P, the others R), None (P
        // converts).
        enum class Partial { None, All, One } partial = Partial::None;
    };

    // Elementwise over `n` inputs of device rank `rank` (same shape): factor d per dim.
    Factors elementwise(int n, size_t rank);

    // Elementwise with broadcasting: input device shapes `in` (each of the output's
    // rank) to output device shape `out`; an input dim of size 1 under a larger output
    // dim has no factor.
    Factors broadcast(
        const std::vector<std::vector<int64_t>>& in, const std::vector<int64_t>& out
    );

    // Matmul-class: a [..., M, K] (or [..., K, M] with `ta`) @ b [K, N] (or [N, K] with
    // `tb`) of device ranks `ra` / 2 -> [..., M, N]; batch dims and M, N pass through,
    // K reduces.
    Factors matmul(size_t ra, bool ta, bool tb);

    // Batched matmul a [B, M, K] (or [B, K, M] with `ta`) @ b [B, K, N] (or [B, N, K]
    // with `tb`) -> [B, M, N].
    Factors bmm(bool ta, bool tb);

    // The placements an op runs at: per input, and the output's.
    struct Choice {
        std::vector<bridge::Placement> inputs;
        bridge::Placement output;
    };

    // The least-cost choice for `f` from the inputs' current placements (device
    // shapes `shapes`, element bytes `elem`), per mesh axis of size > 1, input i's
    // conversion cost times (*scale)[i] when given; nullopt when every input is
    // replicated (the plain path applies).
    std::optional<Choice> choose(
        const Factors& f,
        const std::vector<bridge::Placement>& current,
        const std::vector<std::vector<int64_t>>& shapes,
        int64_t elem,
        const std::vector<double>* scale = nullptr
    );

    // Persistent tensors re-placed for their consumers since load (R-9).
    int64_t migrations();

    // The storage base of tt tensor `t` as placed when `t` is a view the placement
    // survives and the base is mesh-placed; null otherwise.
    bridge::HandlePtr placed_base(const at::Tensor& t);

    // Placed `h` (logical sizes `from`) reshaped contiguously to `to`, placed; its
    // replicated copy reshaped (a counted fallback of `op`) when the shard cannot
    // follow.
    bridge::HandlePtr reshape_placed(
        const std::string& op,
        const bridge::HandlePtr& h,
        const std::vector<int64_t>& from,
        const std::vector<int64_t>& to
    );

    // Counts a gather fallback of op `op` (a placed input read as replicated).
    void count_fallback(const std::string& op);
    // Fallbacks per op since load or the last reset, as "op=count".
    std::vector<std::string> fallback_ops();
    void reset_fallback_ops();

    // Rows `index` (replicated int32, device [1, n]) of mesh-placed 2-D `table`
    // (mesh plan R-2): an S(1) table looks up locally (S(1) result); an S(0) table
    // whose rows divide the mesh looks up its own row range, rows outside it zeroed
    // (P(sum) result). nullopt for other placements.
    std::optional<bridge::HandlePtr> placed_rows(
        const bridge::HandlePtr& table, const bridge::HandlePtr& index, int64_t n
    );

    // Rows of replicated 2-D `table` (version `version`) at mesh-placed `index`
    // (device [1, n], sharded evenly along n on one axis): rows looked up per chip,
    // the result [n, width] sharded on dim 0. nullopt for other placements.
    std::optional<bridge::HandlePtr> placed_index_rows(
        const bridge::HandlePtr& table,
        int64_t version,
        const bridge::HandlePtr& index,
        int64_t n
    );

    // Runs `run` (local input handles -> local output handle) on `inputs` at the
    // placements chosen for `f`, returning the output placed at device shape
    // `out_shape`; nullopt when no input is placed (callers take their plain path).
    std::optional<bridge::HandlePtr> run_placed(
        const std::string& op,
        const Factors& f,
        const std::vector<bridge::HandlePtr>& inputs,
        const std::vector<int64_t>& out_shape,
        const std::function<bridge::HandlePtr(const std::vector<bridge::HandlePtr>&)>&
            run
    );

}  // namespace ktt::mesh
