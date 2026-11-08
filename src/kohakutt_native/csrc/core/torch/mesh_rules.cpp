// Mesh placement rules: the factor engine and placed execution (mesh plan R-1, R-3,
// R-7, R-8).
#include "core/torch/mesh_rules.hpp"

#include <algorithm>
#include <limits>

#include "core/torch/tt_tensor.hpp"

namespace ktt::mesh {
    namespace {

        using bridge::AxisKind;
        using bridge::AxisPlacement;
        using bridge::Placement;

        constexpr double kInf = std::numeric_limits<double>::infinity();

        AxisPlacement axis_of(const Placement& p, size_t a) {
            return a < p.axes.size() ? p.axes[a] : AxisPlacement{};
        }

        // Bytes moved converting `from` to `to` on one mesh axis of `n` chips, for a
        // tensor of `bytes` global bytes (Alpa-style per-axis formulas).
        double cost(const AxisPlacement& from, const AxisPlacement& to, double bytes, int n) {
            if (from == to) {
                return 0.0;
            }
            const double spread = bytes * (n - 1) / n;
            switch (to.kind) {
                case AxisKind::R:
                    return from.kind == AxisKind::P ? 2 * spread : spread;
                case AxisKind::S:
                    if (from.kind == AxisKind::R) {
                        return bytes / n * 0.1;  // local partition
                    }
                    return from.kind == AxisKind::P ? spread : 2 * spread;
                case AxisKind::P:
                    return kInf;  // a value never becomes partial by conversion
            }
            return kInf;
        }

        // One candidate on one axis: per input and the output.
        struct Candidate {
            std::vector<AxisPlacement> inputs;
            AxisPlacement output;
        };

        std::vector<Candidate> candidates(
            const Factors& f,
            const std::vector<Placement>& current,
            const std::vector<std::vector<int64_t>>& shapes,
            size_t axis,
            int n
        ) {
            const size_t k = f.inputs.size();
            std::vector<Candidate> out;
            out.push_back({std::vector<AxisPlacement>(k), AxisPlacement{}});
            for (size_t fac = 0; fac < f.kinds.size(); ++fac) {
                Candidate c{std::vector<AxisPlacement>(k), AxisPlacement{}};
                bool padded = false;
                for (size_t i = 0; i < k; ++i) {
                    const auto& dims = f.inputs[i];
                    const auto it = std::find(dims.begin(), dims.end(), static_cast<int>(fac));
                    if (it != dims.end()) {
                        const int d = static_cast<int>(it - dims.begin());
                        c.inputs[i] = AxisPlacement::shard(d);
                        padded = padded || shapes[i][d] % n != 0;
                    }
                }
                if (f.kinds[fac] == FactorKind::Pass) {
                    const auto it = std::find(f.output.begin(), f.output.end(), static_cast<int>(fac));
                    if (it == f.output.end()) {
                        continue;
                    }
                    c.output = AxisPlacement::shard(static_cast<int>(it - f.output.begin()));
                } else {
                    // Padding is undefined (R-3): no reduction over a padded shard.
                    if (padded) {
                        continue;
                    }
                    c.output = AxisPlacement::partial(f.reduce);
                }
                out.push_back(c);
            }
            // P(sum) passing through ops linear in it (R-7).
            if (f.partial != Factors::Partial::None) {
                const auto sum = AxisPlacement::partial(bridge::ReduceKind::Sum);
                std::vector<size_t> partial_inputs;
                for (size_t i = 0; i < k; ++i) {
                    if (axis_of(current[i], axis) == sum) {
                        partial_inputs.push_back(i);
                    }
                }
                if (f.partial == Factors::Partial::All && !partial_inputs.empty()) {
                    out.push_back({std::vector<AxisPlacement>(k, sum), sum});
                }
                if (f.partial == Factors::Partial::One && partial_inputs.size() == 1) {
                    Candidate c{std::vector<AxisPlacement>(k), sum};
                    c.inputs[partial_inputs[0]] = sum;
                    out.push_back(c);
                }
            }
            return out;
        }

        int64_t volume(const std::vector<int64_t>& s) {
            int64_t v = 1;
            for (int64_t d : s) {
                v *= d;
            }
            return v;
        }

    }  // namespace

    Factors elementwise(int n, size_t rank) {
        Factors f;
        std::vector<int> dims(rank);
        for (size_t d = 0; d < rank; ++d) {
            dims[d] = static_cast<int>(d);
        }
        f.inputs.assign(n, dims);
        f.output = dims;
        f.kinds.assign(rank, FactorKind::Pass);
        return f;
    }

    std::optional<Choice> choose(
        const Factors& f,
        const std::vector<Placement>& current,
        const std::vector<std::vector<int64_t>>& shapes,
        int64_t elem
    ) {
        if (std::all_of(current.begin(), current.end(), [](const Placement& p) {
                return p.is_replicated();
            })) {
            return std::nullopt;
        }
        const auto mesh = bridge::mesh_shape();
        const size_t k = f.inputs.size();
        Choice choice{std::vector<Placement>(k), Placement{}};
        for (auto& p : choice.inputs) {
            p.axes.resize(mesh.size());
        }
        choice.output.axes.resize(mesh.size());
        for (size_t a = 0; a < mesh.size(); ++a) {
            if (mesh[a] == 1) {
                continue;
            }
            double best = kInf;
            const Candidate* pick = nullptr;
            const auto cands = candidates(f, current, shapes, a, mesh[a]);
            for (const auto& c : cands) {
                double total = 0.0;
                for (size_t i = 0; i < k; ++i) {
                    const double bytes = static_cast<double>(volume(shapes[i]) * elem);
                    total += cost(axis_of(current[i], a), c.inputs[i], bytes, mesh[a]);
                }
                // Ties keep the output placed, so memory stays spread over the mesh.
                const bool better = total < best ||
                                    (total == best && pick && pick->output.kind == AxisKind::R &&
                                     c.output.kind != AxisKind::R);
                if (better) {
                    best = total;
                    pick = &c;
                }
            }
            if (!pick || best == kInf) {
                return std::nullopt;
            }
            for (size_t i = 0; i < k; ++i) {
                choice.inputs[i].axes[a] = pick->inputs[i];
            }
            choice.output.axes[a] = pick->output;
        }
        return choice;
    }

    bridge::HandlePtr placed_base(const at::Tensor& t) {
        if (!t.is_privateuseone() || t.numel() == 0 || !t.is_contiguous() ||
            t.storage_offset() != 0) {
            return nullptr;
        }
        auto& st = storage_of(t);
        if (!st.base || st.dtype != t.scalar_type() ||
            st.sizes != to_vector(t.sizes()) || bridge::is_block(bridge::dtype(st.base)) ||
            bridge::placement(st.base).is_replicated()) {
            return nullptr;
        }
        return st.base;
    }

    std::optional<bridge::HandlePtr> run_placed(
        const std::string& op,
        const Factors& f,
        const std::vector<bridge::HandlePtr>& inputs,
        const std::vector<int64_t>& out_shape,
        const std::function<bridge::HandlePtr(const std::vector<bridge::HandlePtr>&)>& run
    ) {
        std::vector<Placement> current;
        std::vector<std::vector<int64_t>> shapes;
        int64_t elem = 4;
        for (const auto& h : inputs) {
            current.push_back(bridge::placement(h));
            shapes.push_back(bridge::device_sizes(h));
            elem = bridge::dtype(h) == bridge::DType::BFloat16 ? 2 : 4;
        }
        auto choice = choose(f, current, shapes, elem);
        if (!choice || choice->output.is_replicated()) {
            return std::nullopt;
        }
        std::vector<bridge::HandlePtr> locals;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto h = (
                current[i] == choice->inputs[i]
                    ? inputs[i]
                    : bridge::distribute(inputs[i], choice->inputs[i])
            );
            locals.push_back(bridge::local_view(h));
        }
        (void)op;
        return bridge::adopt_placed(run(locals), out_shape, choice->output);
    }

}  // namespace ktt::mesh
