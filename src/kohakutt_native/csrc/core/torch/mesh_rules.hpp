// Mesh placement rules: an op family's factor mapping, the engine that
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
        // The dtype the op computes in, when the caller casts its inputs to it.
        std::optional<bridge::DType> dtype;
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

    // The placements an op runs at: per input, and the output's; `reason` names the
    // rule picked per mesh axis of size > 1 (replicated, shard, reduce, partial).
    struct Choice {
        std::vector<bridge::Placement> inputs;
        bridge::Placement output;
        std::string reason;
    };

    // One input of an op: its placement, device shape and dtype, the placements a
    // copy of it exists at (`ready`), and a multiplier on its conversion cost.
    struct Operand {
        bridge::Placement at;
        std::vector<int64_t> shape;
        bridge::DType dtype = bridge::DType::BFloat16;
        std::vector<bridge::Placement> ready;
        double weight = 1.0;
    };

    // What a choice depends on beyond the op and its inputs: chips per mesh axis,
    // the bytes below which a partial sum is never completed to a shard (0: no
    // floor), and whether a conversion to a `ready` placement costs nothing.
    struct Hints {
        std::vector<int> mesh;
        int64_t floor = 0;
        bool free_ready = true;
    };

    // The hints of an eager call: the open mesh, bridge::shard_min_bytes() as the
    // floor (none in a prepare pass or capture), free_ready unless
    // KTT_MESH_CONV_FREE=0.
    Hints eager_hints();

    // The least-cost choice for `f` on `inputs` under `hints`, per mesh axis of size
    // > 1, from these arguments alone; nullopt when every input is replicated (the
    // plain path applies) or no candidate converts. Equal costs go to the candidate
    // that is cheaper without `ready`, then to a placed output.
    std::optional<Choice> decide(
        const Factors& f, const std::vector<Operand>& inputs, const Hints& hints
    );

    // Persistent tensors re-placed for their consumers since load.
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

    // Drops the placed constants of the rules (before the device closes).
    void clear_constants();

    // Counts a gather fallback of op `op` (a placed input read as replicated).
    void count_fallback(const std::string& op);
    // Fallbacks per op since load or the last reset, as "op=count".
    std::vector<std::string> fallback_ops();
    void reset_fallback_ops();

    // Rows `index` (replicated int32, device [1, n]) of mesh-placed 2-D `table`:
    // an S(1) table looks up locally (S(1) result); an S(0) table
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
    // `chosen`, when given, holds those placements before `run` is called.
    std::optional<bridge::HandlePtr> run_placed(
        const std::string& op,
        const Factors& f,
        const std::vector<bridge::HandlePtr>& inputs,
        const std::vector<int64_t>& out_shape,
        const std::function<bridge::HandlePtr(const std::vector<bridge::HandlePtr>&)>&
            run,
        Choice* chosen = nullptr
    );

}  // namespace ktt::mesh
