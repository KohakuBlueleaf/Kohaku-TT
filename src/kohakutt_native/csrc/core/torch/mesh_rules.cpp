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

        // µs to convert a tensor of `bytes` global bytes from `from` to `to` on one
        // mesh axis: latency + bytes / bandwidth per conversion kind, measured on a 1x2
        // P150A mesh (TT-NN collectives, fabric 1-D; global bytes per µs).
        double cost(const AxisPlacement& from, const AxisPlacement& to, double bytes, int n) {
            (void)n;
            if (from == to) {
                return 0.0;
            }
            const auto gather = [&] {
                return 60.0 + bytes / 170e3;
            };
            const auto partition = [&] {
                return 35.0 + bytes / 367e3;
            };
            switch (to.kind) {
                case AxisKind::R:
                    return from.kind == AxisKind::P ? 120.0 + bytes / 77e3 : gather();
                case AxisKind::S:
                    if (from.kind == AxisKind::R) {
                        return partition();
                    }
                    if (from.kind == AxisKind::P) {
                        return 100.0 + bytes / 136e3;  // reduce-scatter
                    }
                    return gather() + partition();
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
                if (f.kinds[fac] == FactorKind::Replicate) {
                    continue;
                }
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

    Factors broadcast(
        const std::vector<std::vector<int64_t>>& in, const std::vector<int64_t>& out
    ) {
        Factors f = elementwise(static_cast<int>(in.size()), out.size());
        for (size_t i = 0; i < in.size(); ++i) {
            for (size_t d = 0; d < out.size(); ++d) {
                if (in[i][d] == 1 && out[d] != 1) {
                    f.inputs[i][d] = -1;
                }
            }
        }
        return f;
    }

    Factors matmul(size_t ra, bool ta, bool tb) {
        // Factors: batch dims 0 .. ra-3, then M, K, N.
        const int m = static_cast<int>(ra) - 2, k = m + 1, n = m + 2;
        Factors f;
        std::vector<int> a;
        for (int d = 0; d < m; ++d) {
            a.push_back(d);
        }
        std::vector<int> out = a;
        a.push_back(ta ? k : m);
        a.push_back(ta ? m : k);
        f.inputs.push_back(a);
        f.inputs.push_back(tb ? std::vector<int>{n, k} : std::vector<int>{k, n});
        out.push_back(m);
        out.push_back(n);
        f.output = out;
        f.kinds.assign(n + 1, FactorKind::Pass);
        f.kinds[k] = FactorKind::Reduce;
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

    namespace {

        int64_t product(const std::vector<int64_t>& v, size_t from, size_t to) {
            int64_t p = 1;
            for (size_t i = from; i < to; ++i) {
                p *= v[i];
            }
            return p;
        }

        // The S axes of `p` with their logical base dims (device dim - lift); false
        // when a sharded dim is a lifted unit dim.
        bool sharded_dims(
            const Placement& p, int64_t lift, std::vector<std::pair<size_t, int64_t>>& out
        ) {
            for (size_t a = 0; a < p.axes.size(); ++a) {
                if (p.axes[a].kind == AxisKind::S) {
                    const int64_t d = p.axes[a].dim - lift;
                    if (d < 0) {
                        return false;
                    }
                    out.push_back({a, d});
                }
            }
            return true;
        }

        // Placed `base` (logical sizes `b`) viewed contiguously as sizes `v` (R-4): the
        // shard survives a 1:1 dim, an even split (kept on the outermost new dim) or a
        // merge it leads, unpadded; null otherwise (the gather fallback).
        bridge::HandlePtr reshape_view(
            const bridge::HandlePtr& base,
            const std::vector<int64_t>& b,
            const std::vector<int64_t>& v
        ) {
            const auto p = bridge::placement(base);
            const auto mesh = bridge::mesh_shape();
            const int64_t lift_b = std::max<int64_t>(0, 2 - static_cast<int64_t>(b.size()));
            const int64_t lift_v = std::max<int64_t>(0, 2 - static_cast<int64_t>(v.size()));
            std::vector<std::pair<size_t, int64_t>> shards;
            if (!sharded_dims(p, lift_b, shards) || shards.size() > 1) {
                return nullptr;
            }
            // Partial axes only: the view of a partial sum is the partial sum of the view.
            if (shards.empty()) {
                Placement q = p;
                q.valid.clear();
                auto h = bridge::reshape(bridge::local_view(base), v);
                return bridge::adopt_placed(h, bridge::device_sizes_of(v), q);
            }
            const auto [axis, d] = shards[0];
            const int64_t m = mesh[axis];
            // The segment of base dims [i0, i1) and view dims [j0, j1) holding d.
            size_t i = 0, j = 0;
            while (i < b.size() || j < v.size()) {
                const size_t i0 = i, j0 = j;
                int64_t pb = i < b.size() ? b[i++] : 1, pv = j < v.size() ? v[j++] : 1;
                while (pb != pv) {
                    if (pb < pv && i < b.size()) {
                        pb *= b[i++];
                    } else if (pv < pb && j < v.size()) {
                        pv *= v[j++];
                    } else {
                        return nullptr;
                    }
                }
                if (static_cast<size_t>(d) < i0 || static_cast<size_t>(d) >= i) {
                    continue;
                }
                const size_t i1 = i, j1 = j;
                const bool padded = b[d] % m != 0;
                int64_t new_dim = -1;
                if (i1 - i0 == 1 && j1 - j0 == 1) {
                    new_dim = static_cast<int64_t>(j0);
                } else if (!padded && i1 - i0 == 1 && v[j0] % m == 0) {
                    new_dim = static_cast<int64_t>(j0);  // split: outermost new dim
                } else if (!padded && j1 - j0 == 1 && static_cast<size_t>(d) == i0) {
                    new_dim = static_cast<int64_t>(j0);  // merge led by the sharded dim
                }
                if (new_dim < 0) {
                    return nullptr;
                }
                Placement q = p;
                q.valid.clear();
                q.axes[axis] = AxisPlacement::shard(static_cast<int>(new_dim + lift_v));
                auto local = v;
                local[new_dim] = (v[new_dim] + m - 1) / m;
                auto h = bridge::reshape(bridge::local_view(base), local);
                return bridge::adopt_placed(h, bridge::device_sizes_of(v), q);
            }
            return nullptr;
        }

        // Placed `base` (logical sizes `b`) viewed as a permutation (sizes `v`, strides
        // `s`); null when the view is not one.
        bridge::HandlePtr permute_view(
            const bridge::HandlePtr& base,
            const std::vector<int64_t>& b,
            const std::vector<int64_t>& v,
            const std::vector<int64_t>& s
        ) {
            const size_t r = b.size();
            if (v.size() != r || r < 2) {
                return nullptr;
            }
            // order[i]: the view dim that is base dim i (strides descending).
            std::vector<int64_t> order(r);
            for (size_t k = 0; k < r; ++k) {
                order[k] = static_cast<int64_t>(k);
            }
            std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
                return s[x] > s[y];
            });
            int64_t stride = 1;
            for (size_t i = r; i-- > 0;) {
                if (v[order[i]] != b[i] || (v[order[i]] > 1 && s[order[i]] != stride)) {
                    return nullptr;
                }
                stride *= b[i];
            }
            std::vector<int64_t> inv(r);
            for (size_t i = 0; i < r; ++i) {
                inv[order[i]] = static_cast<int64_t>(i);
            }
            Placement q = bridge::placement(base);
            q.valid.clear();
            for (auto& ax : q.axes) {
                if (ax.kind == AxisKind::S) {
                    ax.dim = static_cast<int>(order[ax.dim]);
                }
            }
            auto h = bridge::permute(bridge::local_view(base), inv);
            return bridge::adopt_placed(h, bridge::device_sizes_of(v), q);
        }

    }  // namespace

    bridge::HandlePtr placed_base(const at::Tensor& t) {
        if (!t.is_privateuseone() || t.numel() == 0 || t.storage_offset() != 0) {
            return nullptr;
        }
        auto& st = storage_of(t);
        if (!st.base || st.dtype != t.scalar_type() || bridge::is_block(bridge::dtype(st.base)) ||
            bridge::placement(st.base).is_replicated()) {
            return nullptr;
        }
        const auto b = st.sizes;
        const auto v = to_vector(t.sizes());
        if (product(b, 0, b.size()) != t.numel()) {
            return nullptr;
        }
        if (t.is_contiguous()) {
            return v == b ? st.base : reshape_view(st.base, b, v);
        }
        return permute_view(st.base, b, v, to_vector(t.strides()));
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
