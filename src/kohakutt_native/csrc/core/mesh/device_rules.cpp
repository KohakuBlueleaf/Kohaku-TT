// Mesh placement rules: the factor engine and placed execution.
#include "core/mesh/rules.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

#include "core/device/trace.hpp"

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
        double cost(
            const AxisPlacement& from, const AxisPlacement& to, double bytes, int n
        ) {
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
                    return 100.0 + bytes / 130e3;  // all-to-all (S(i) -> S(j))
                case AxisKind::P:
                    return kInf;  // a value never becomes partial by conversion
            }
            return kInf;
        }

        // True when `to` splits one of the two tiled (last) dims of `shape` into shards
        // off 32-element boundaries (a conversion through the host).
        bool off_tile(
            const AxisPlacement& to, const std::vector<int64_t>& shape, int n
        ) {
            if (to.kind != AxisKind::S || n <= 1) {
                return false;
            }
            const auto size = static_cast<int64_t>(shape.size());
            const int64_t rank = std::max<int64_t>(2, size);
            const int64_t d = to.dim - (rank - size);
            if (to.dim + 2 < rank || d < 0) {
                return false;
            }
            return (shape[d] + n - 1) / n % 32 != 0;
        }

        // µs of a conversion through the host (every shard read, new shards written):
        // about 2 ms per 0.3 MB activation on the 1x2 P150A mesh.
        double host_cost(double bytes) {
            return 1500.0 + bytes / 2e3;
        }

        // One candidate on one axis: per input and the output, and its rule's name.
        struct Candidate {
            std::vector<AxisPlacement> inputs;
            AxisPlacement output;
            const char* rule = "replicated";
        };

        std::vector<Candidate> candidates(
            const Factors& f, const std::vector<Operand>& ops, size_t axis, int n
        ) {
            const size_t k = f.inputs.size();
            std::vector<Candidate> out;
            out.push_back({std::vector<AxisPlacement>(k), AxisPlacement{}});
            for (size_t fac = 0; fac < f.kinds.size(); ++fac) {
                if (f.kinds[fac] == FactorKind::Replicate) {
                    continue;
                }
                Candidate c{std::vector<AxisPlacement>(k), AxisPlacement{}, "shard"};
                bool empty_chip = false;
                for (size_t i = 0; i < k; ++i) {
                    const auto& dims = f.inputs[i];
                    const auto it = (std::find(
                        dims.begin(), dims.end(), static_cast<int>(fac)
                    ));
                    if (it != dims.end()) {
                        const int d = static_cast<int>(it - dims.begin());
                        const int64_t size = ops[i].shape[d];
                        c.inputs[i] = AxisPlacement::shard(d);
                        empty_chip = empty_chip || (size + n - 1) / n * (n - 1) >= size;
                    }
                }
                // A shard leaving a chip only padding (a size-1 batch dim) is never
                // chosen.
                if (empty_chip) {
                    continue;
                }
                if (f.kinds[fac] == FactorKind::Pass) {
                    const auto it = (std::find(
                        f.output.begin(), f.output.end(), static_cast<int>(fac)
                    ));
                    if (it == f.output.end()) {
                        continue;
                    }
                    c.output = (AxisPlacement::shard(
                        static_cast<int>(it - f.output.begin())
                    ));
                } else {
                    // A padded shard is reduced with its padding masked (run_placed).
                    c.output = AxisPlacement::partial(f.reduce);
                    c.rule = "reduce";
                }
                out.push_back(c);
            }
            // P(sum) passing through ops linear in it.
            if (f.partial != Factors::Partial::None) {
                const auto sum = AxisPlacement::partial(bridge::ReduceKind::Sum);
                std::vector<size_t> partial_inputs;
                for (size_t i = 0; i < k; ++i) {
                    if (axis_of(ops[i].at, axis) == sum) {
                        partial_inputs.push_back(i);
                    }
                }
                if (f.partial == Factors::Partial::All && !partial_inputs.empty()) {
                    out.push_back({std::vector<AxisPlacement>(k, sum), sum, "partial"});
                }
                if (f.partial == Factors::Partial::One && partial_inputs.size() == 1) {
                    Candidate c{std::vector<AxisPlacement>(k), sum, "partial"};
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

    Factors bmm(bool ta, bool tb) {
        // Factors: batch 0, M 1, K 2, N 3.
        Factors f;
        f.inputs = {
            ta ? std::vector<int>{0, 2, 1} : std::vector<int>{0, 1, 2},
            tb ? std::vector<int>{0, 3, 2} : std::vector<int>{0, 2, 3}
        };
        f.output = {0, 1, 3};
        f.kinds = {
            FactorKind::Pass, FactorKind::Pass, FactorKind::Reduce, FactorKind::Pass
        };
        return f;
    }

    namespace {

        // True when a copy of `op` exists that is `to` on mesh axis `a` and where
        // `op` is on every other axis.
        bool ready_at(
            const Operand& op, const AxisPlacement& to, size_t a, size_t axes
        ) {
            for (const auto& r : op.ready) {
                bool match = axis_of(r, a) == to;
                for (size_t b = 0; b < axes && match; ++b) {
                    match = b == a || axis_of(r, b) == axis_of(op.at, b);
                }
                if (match) {
                    return true;
                }
            }
            return false;
        }

    }  // namespace

    Hints eager_hints() {
        Hints h;
        h.mesh = bridge::mesh_shape();
        const bool eager = !(bridge::tracing() || bridge::capturing());
        h.floor = eager ? bridge::shard_min_bytes() : 0;
        const char* v = std::getenv("KTT_MESH_CONV_FREE");
        h.free_ready = !(v && std::string(v) == "0");
        return h;
    }

    std::optional<Choice> decide(
        const Factors& f, const std::vector<Operand>& ops, const Hints& hints
    ) {
        if (std::all_of(ops.begin(), ops.end(), [](const Operand& o) {
                return o.at.is_replicated();
            })) {
            return std::nullopt;
        }
        const auto& mesh = hints.mesh;
        const size_t k = f.inputs.size();
        // Each input is priced by its own element size.
        std::vector<int64_t> nbytes(k);
        std::vector<bool> small(k);
        for (size_t i = 0; i < k; ++i) {
            const int64_t elem = ops[i].dtype == bridge::DType::BFloat16 ? 2 : 4;
            nbytes[i] = volume(ops[i].shape) * elem;
            small[i] = nbytes[i] < hints.floor;
        }
        Choice choice{std::vector<Placement>(k), Placement{}, ""};
        for (auto& p : choice.inputs) {
            p.axes.resize(mesh.size());
        }
        choice.output.axes.resize(mesh.size());
        for (size_t a = 0; a < mesh.size(); ++a) {
            if (mesh[a] == 1) {
                continue;
            }
            // `plain`: the cost with no copy ready; `floored`: the least cost the
            // floor ruled out.
            double best = kInf, best_plain = kInf, plain_best = kInf, floored = kInf;
            const Candidate* pick = nullptr;
            const Candidate* plain_pick = nullptr;
            const auto cands = candidates(f, ops, a, mesh[a]);
            for (const auto& c : cands) {
                double total = 0.0, plain = 0.0;
                bool below = false;
                for (size_t i = 0; i < k; ++i) {
                    const auto from = axis_of(ops[i].at, a);
                    const auto& to = c.inputs[i];
                    if (from == to) {
                        continue;
                    }
                    // A partial below the floor completes to replicated only.
                    below = below ||
                        (small[i] && from.kind == AxisKind::P &&
                         to.kind == AxisKind::S);
                    const auto bytes = static_cast<double>(nbytes[i]);
                    const double price =
                        (ops[i].weight *
                         (off_tile(to, ops[i].shape, mesh[a])
                              ? host_cost(bytes)
                              : cost(from, to, bytes, mesh[a])));
                    plain += price;
                    // A partial sum's copy is not free where the op can pass the
                    // partial through in another dtype.
                    const bool recast =
                        (from.kind == AxisKind::P && f.dtype &&
                         *f.dtype != ops[i].dtype &&
                         f.partial != Factors::Partial::None);
                    const bool ready =
                        (hints.free_ready && to.kind != AxisKind::P && !recast &&
                         ready_at(ops[i], to, a, mesh.size()));
                    total += ready ? 0.0 : price;
                }
                if (below) {
                    floored = std::min(floored, total);
                    continue;
                }
                // Ties keep the output placed, so memory stays spread over the mesh.
                const auto placed_tie = [&](const Candidate* held) {
                    return held && held->output.kind == AxisKind::R &&
                        c.output.kind != AxisKind::R;
                };
                if (plain < plain_best ||
                    (plain == plain_best && placed_tie(plain_pick))) {
                    plain_best = plain;
                    plain_pick = &c;
                }
                const bool better =
                    (total < best ||
                     (total == best &&
                      (plain < best_plain ||
                       (plain == best_plain && placed_tie(pick)))));
                if (better) {
                    best = total;
                    best_plain = plain;
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
            choice.reason += (choice.reason.empty() ? "" : ", ");
            choice.reason += pick->rule;
            if (floored < best) {
                choice.reason += " (size floor)";
            }
            if (pick != plain_pick) {
                choice.reason += " (memory)";
            }
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
            const Placement& p,
            int64_t lift,
            std::vector<std::pair<size_t, int64_t>>& out
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

        // Placed `base` (logical sizes `b`) viewed contiguously as sizes `v`: the
        // shard survives a 1:1 dim, an even split (kept on the outermost new dim) or a
        // merge it leads, unpadded; null otherwise (the gather fallback).
        bridge::HandlePtr reshape_view(
            const bridge::HandlePtr& base,
            const std::vector<int64_t>& b,
            const std::vector<int64_t>& v
        ) {
            const auto p = bridge::placement(base);
            const auto mesh = bridge::mesh_shape();
            const int64_t lift_b = (std::max<int64_t>(
                0, 2 - static_cast<int64_t>(b.size())
            ));
            const int64_t lift_v = (std::max<int64_t>(
                0, 2 - static_cast<int64_t>(v.size())
            ));
            std::vector<std::pair<size_t, int64_t>> shards;
            if (!sharded_dims(p, lift_b, shards) || shards.size() > 1) {
                return nullptr;
            }
            // Partial axes only: the view of a partial sum is the partial sum of the
            // view.
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
                // A split's first new dim of size > 1 carries the shard.
                size_t lead = j0;
                while (lead + 1 < j1 && v[lead] == 1) {
                    ++lead;
                }
                if (i1 - i0 == 1 && j1 - j0 == 1) {
                    new_dim = static_cast<int64_t>(j0);
                } else if (!padded && i1 - i0 == 1 && v[lead] % m == 0) {
                    new_dim = static_cast<int64_t>(
                        lead
                    );  // split: outermost real new dim
                } else if (!padded && j1 - j0 == 1 && product(b, i0, d) == 1) {
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

        // Placed `base` (logical sizes `b`) viewed as a box (sizes `v`, the base's
        // contiguous strides `s`, storage offset `off`): a local slice when no sharded
        // dim is cut; null otherwise.
        bridge::HandlePtr box_view(
            const bridge::HandlePtr& base,
            const std::vector<int64_t>& b,
            const std::vector<int64_t>& view_sizes,
            const std::vector<int64_t>& view_strides,
            int64_t off
        ) {
            const size_t r = b.size();
            if (view_sizes.size() != r) {
                return nullptr;
            }
            // order[i]: the view dim that is base dim i (strides descending); the box
            // is checked in base order, then permuted back.
            std::vector<int64_t> order(r);
            for (size_t k = 0; k < r; ++k) {
                order[k] = static_cast<int64_t>(k);
            }
            std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
                return view_strides[x] > view_strides[y];
            });
            std::vector<int64_t> v(r), s(r);
            for (size_t i = 0; i < r; ++i) {
                v[i] = view_sizes[order[i]];
                s[i] = view_strides[order[i]];
            }
            std::vector<int64_t> stride(r, 1);
            for (size_t i = r; i-- > 1;) {
                stride[i - 1] = stride[i] * b[i];
            }
            std::vector<int64_t> begin(r, 0);
            for (size_t d = 0; d < r; ++d) {
                if (v[d] > 1 && s[d] != stride[d]) {
                    return nullptr;
                }
                begin[d] = (off / stride[d]) % b[d];
                if (begin[d] + v[d] > b[d]) {
                    return nullptr;
                }
            }
            // Offsets beyond what the begins account for: not a box.
            int64_t rebuilt = 0;
            for (size_t d = 0; d < r; ++d) {
                rebuilt += begin[d] * stride[d];
            }
            if (rebuilt != off) {
                return nullptr;
            }
            const auto p = bridge::placement(base);
            const int64_t lift = std::max<int64_t>(0, 2 - static_cast<int64_t>(r));
            const auto local = bridge::device_sizes(bridge::local_view(base));
            std::vector<int64_t> lo(lift, 0), hi(local.begin(), local.begin() + lift);
            for (size_t d = 0; d < r; ++d) {
                const auto dev = static_cast<int>(d + lift);
                for (const auto& ax : p.axes) {
                    if (ax.kind == AxisKind::S && ax.dim == dev && v[d] != b[d]) {
                        return nullptr;  // the slice cuts a sharded dim
                    }
                }
                const bool sharded = std::any_of(
                    p.axes.begin(), p.axes.end(), [&](const AxisPlacement& ax) {
                        return ax.kind == AxisKind::S && ax.dim == dev;
                    }
                );
                lo.push_back(sharded ? 0 : begin[d]);
                hi.push_back(sharded ? local[dev] : begin[d] + v[d]);
            }
            std::vector<int64_t> steps(lo.size(), 1);
            Placement q = p;
            q.valid.clear();
            auto h = (bridge::adopt_placed(
                bridge::slice(bridge::local_view(base), lo, hi, steps),
                bridge::device_sizes_of(v),
                q
            ));
            bool identity = true;
            for (size_t i = 0; i < r; ++i) {
                identity = identity && order[i] == static_cast<int64_t>(i);
            }
            if (identity) {
                return h;
            }
            if (r < 2) {
                return nullptr;
            }
            // Back to view order: view dim k is base dim inv[k].
            std::vector<int64_t> inv(r);
            for (size_t i = 0; i < r; ++i) {
                inv[order[i]] = static_cast<int64_t>(i);
            }
            for (auto& ax : q.axes) {
                if (ax.kind == AxisKind::S) {
                    ax.dim = static_cast<int>(order[ax.dim]);
                }
            }
            auto permuted = bridge::permute(bridge::local_view(h), inv);
            return (
                bridge::adopt_placed(permuted, bridge::device_sizes_of(view_sizes), q)
            );
        }

        // Base sizes `b` split into the view's dims in stride order (stride ratios, 1
        // for size-1 dims); empty when the view's strides (`s`) do not tile the base.
        std::vector<int64_t> refined_base(
            const std::vector<int64_t>& b,
            const std::vector<int64_t>& v,
            const std::vector<int64_t>& s
        ) {
            const size_t r = v.size();
            std::vector<int64_t> order(r);
            for (size_t k = 0; k < r; ++k) {
                order[k] = static_cast<int64_t>(k);
            }
            std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
                return s[x] > s[y];
            });
            std::vector<int64_t> out(r, 1);
            int64_t outer = product(b, 0, b.size()), last = -1;
            for (size_t i = 0; i < r; ++i) {
                const int64_t d = order[i];
                if (v[d] == 1) {
                    continue;
                }
                if (s[d] <= 0 || outer % s[d] != 0) {
                    return {};
                }
                out[i] = outer / s[d];
                outer = s[d];
                last = s[d];
            }
            if (last > 1) {
                return {};
            }
            return out;
        }

    }  // namespace

    bridge::HandlePtr view_of(
        const bridge::HandlePtr& base,
        const std::vector<int64_t>& b,
        const std::vector<int64_t>& v,
        const std::vector<int64_t>& s,
        int64_t offset,
        bool whole
    ) {
        if (whole) {
            return v == b ? base : reshape_view(base, b, v);
        }
        // A strided view: the base reshaped (contiguously) to the view's own dims in
        // stride order, of which the view is a permuted box.
        const auto b2 = refined_base(b, v, s);
        if (b2.empty()) {
            return nullptr;
        }
        auto refined = b2 == b ? base : reshape_view(base, b, b2);
        if (!refined) {
            return nullptr;
        }
        if (product(b2, 0, b2.size()) == product(v, 0, v.size()) && offset == 0) {
            return permute_view(refined, b2, v, s);
        }
        return box_view(refined, b2, v, s, offset);
    }

    bridge::HandlePtr reshape_placed(
        const std::string& op,
        const bridge::HandlePtr& h,
        const std::vector<int64_t>& from,
        const std::vector<int64_t>& to
    ) {
        if (bridge::placement(h).is_replicated()) {
            return from == to ? h : bridge::reshape(h, to);
        }
        if (from == to) {
            return h;
        }
        if (auto placed = reshape_view(h, from, to)) {
            return placed;
        }
        fallback(
            op,
            h,
            "a reshape keeps the shards only along a dim it leaves as it is, splits "
            "evenly or merges with the sharded dim first"
        );
        return bridge::reshape(bridge::replicated(h), to);
    }

    namespace {

        std::atomic<WarningSink> g_warning_sink{nullptr};
        std::atomic<MigrationHook> g_migration_hook{nullptr};

        std::mutex g_fallback_mutex;
        std::map<std::string, int64_t> g_fallbacks;
        // "op|placement" of the fallbacks warned about since the last reset.
        std::set<std::string> g_fallbacks_warned;

        bool env_is(const char* name, const char* value) {
            const char* v = std::getenv(name);
            return v && std::string(v) == value;
        }

        // Placed constants of the rules by key, uploaded on first use (also usable
        // inside trace captures then). Never destroyed at exit; clear_constants()
        // empties it.
        std::mutex g_constants_mutex;

        std::map<std::string, bridge::HandlePtr>& placed_constants() {
            static auto* map = new std::map<std::string, bridge::HandlePtr>();
            return *map;
        }

        // The placed constant `key`: host `values` of device `shape` and `dtype`
        // uploaded at placement `p` on first use.
        template <typename T>
        bridge::HandlePtr placed_constant(
            const std::string& key,
            const std::vector<T>& values,
            const std::vector<int64_t>& shape,
            bridge::DType dtype,
            const Placement& p
        ) {
            std::lock_guard lock(g_constants_mutex);
            auto& slot = placed_constants()[key + "|" + p.str()];
            if (!slot) {
                slot = bridge::upload_placed(values.data(), shape, dtype, p);
            }
            return slot;
        }

    }  // namespace

    void clear_constants() {
        std::lock_guard lock(g_constants_mutex);
        placed_constants().clear();
    }

    namespace {

        constexpr size_t kLogRecords = 65536;
        std::atomic<bool> g_recording{env_is("KTT_MESH_EXPLAIN", "1")};
        std::mutex g_log_mutex;
        std::deque<std::string> g_log;

        // `p` over every mesh axis, e.g. "[R, S(0)]".
        std::string shown(Placement p, size_t axes) {
            p.axes.resize(std::max(axes, p.axes.size()));
            return p.str();
        }

        // The conversion kinds from `from` to `to`, one per mesh axis that changes.
        std::string kind_of(const Placement& from, const Placement& to, size_t axes) {
            std::string out;
            for (size_t a = 0; a < axes; ++a) {
                const auto f = axis_of(from, a), t = axis_of(to, a);
                if (f == t) {
                    continue;
                }
                const bool gather = t.kind == AxisKind::R;
                const char* kind =
                    (f.kind == AxisKind::P ? (gather ? "all-reduce" : "reduce-scatter")
                         : f.kind == AxisKind::S
                         ? (gather ? "all-gather" : "all-to-all")
                         : "partition");
                out += (out.empty() ? "" : "+") + std::string(kind);
            }
            return out;
        }

    }  // namespace

    bool recording() {
        return g_recording.load(std::memory_order_relaxed);
    }

    bool set_recording(bool on) {
        return g_recording.exchange(on);
    }

    void record(
        const std::string& op,
        const std::vector<Placement>& inputs,
        const Placement& output,
        const std::vector<Conversion>& conversions,
        const std::string& reason
    ) {
        if (!recording()) {
            return;
        }
        const size_t axes = bridge::mesh_shape().size();
        std::string line = op + "\t";
        for (size_t i = 0; i < inputs.size(); ++i) {
            line += (i ? ";" : "") + shown(inputs[i], axes);
        }
        line += "\t" + shown(output, axes) + "\t";
        for (size_t k = 0; k < conversions.size(); ++k) {
            const auto& c = conversions[k];
            line += (k ? "|" : "") + std::to_string(c.input) + ">" +
                shown(c.from, axes) + ">" + shown(c.to, axes) + ">" +
                kind_of(c.from, c.to, axes) + ">" + (c.reused ? "1" : "0");
        }
        line += "\t" + reason;
        std::lock_guard lock(g_log_mutex);
        if (g_log.size() == kLogRecords) {
            g_log.pop_front();
        }
        g_log.push_back(std::move(line));
    }

    std::vector<std::string> placement_log() {
        std::lock_guard lock(g_log_mutex);
        return {g_log.begin(), g_log.end()};
    }

    void reset_placement_log() {
        std::lock_guard lock(g_log_mutex);
        g_log.clear();
    }

    void fallback(
        const std::string& op, const bridge::HandlePtr& input, const char* why
    ) {
        if (recording() && !env_is("KTT_MESH_STRICT", "1")) {
            const auto at = bridge::placement(input);
            const auto made = bridge::conversions_of(input);
            const bool reused = std::find(made.begin(), made.end(), Placement{}) !=
                made.end();
            record(
                op,
                {at},
                Placement{},
                {{0, at, Placement{}, reused}},
                std::string("fallback: ") + why
            );
        }
        const std::string placed = bridge::placement(input).str();
        if (env_is("KTT_MESH_STRICT", "1")) {
            throw std::runtime_error(
                "kohakutt: " + op + " has no placed form for an input placed " +
                placed + " (KTT_MESH_STRICT=1): " + why +
                ". Replicate the input before the op: kohakutt.replicate(x), or "
                "kohakutt.replicate(module) for a parameter."
            );
        }
        bool first = false;
        {
            std::lock_guard lock(g_fallback_mutex);
            ++g_fallbacks[op];
            first = g_fallbacks_warned.insert(op + "|" + placed).second;
        }
        if (first && !env_is("KTT_FALLBACK_WARN", "0")) {
            const std::string message = "kohakutt: " + op +
                " gathers an input placed " + placed + " and runs replicated: " + why +
                ". kohakutt.replicate(x) before the op makes the conversion explicit; "
                "KTT_MESH_STRICT=1 raises here; counts in "
                "torch.tt.mesh_fallback_ops().";
            if (const auto sink = g_warning_sink.load()) {
                sink(message);
            } else {
                std::fprintf(stderr, "%s\n", message.c_str());
            }
        }
    }

    void set_warning_sink(WarningSink sink) {
        g_warning_sink.store(sink);
    }

    void set_migration_hook(MigrationHook hook) {
        g_migration_hook.store(hook);
    }

    std::vector<std::string> fallback_ops() {
        std::lock_guard lock(g_fallback_mutex);
        std::vector<std::string> out;
        for (const auto& [op, n] : g_fallbacks) {
            out.push_back(op + "=" + std::to_string(n));
        }
        return out;
    }

    void reset_fallback_ops() {
        std::lock_guard lock(g_fallback_mutex);
        g_fallbacks.clear();
        g_fallbacks_warned.clear();
    }

    std::optional<bridge::HandlePtr> placed_index_rows(
        const bridge::HandlePtr& table,
        int64_t version,
        const bridge::HandlePtr& index,
        int64_t n
    ) {
        const auto p = bridge::placement(index);
        const auto mesh = bridge::mesh_shape();
        const auto dims = bridge::device_sizes(table);
        if (p.has_partial() || dims.size() != 2 ||
            bridge::device_sizes(index).size() != 2) {
            return std::nullopt;
        }
        std::optional<size_t> axis;
        for (size_t a = 0; a < p.axes.size(); ++a) {
            if (p.axes[a].kind != AxisKind::S || mesh[a] == 1) {
                continue;
            }
            if (axis || p.axes[a].dim != 1 || n % mesh[a] != 0) {
                return std::nullopt;
            }
            axis = a;
        }
        if (!axis) {
            return std::nullopt;
        }
        auto local = bridge::local_view(index);
        if (bridge::dtype(local) != bridge::DType::Int32) {
            local = bridge::typecast(local, bridge::DType::Int32);
        }
        const int64_t local_n = n / mesh[*axis];
        auto rows = bridge::embedding_rows(table, version, local, local_n);
        Placement q = p;
        q.valid.clear();
        q.axes[*axis] = AxisPlacement::shard(0);
        return bridge::adopt_placed(rows, {n, dims[1]}, q);
    }

    std::optional<bridge::HandlePtr> placed_rows(
        const bridge::HandlePtr& table, const bridge::HandlePtr& index, int64_t n
    ) {
        using B = bridge::BinaryOp;
        const auto p = bridge::placement(table);
        const auto mesh = bridge::mesh_shape();
        const auto dims = bridge::device_sizes(table);
        if (p.has_partial() || dims.size() != 2) {
            return std::nullopt;
        }
        std::vector<size_t> axes;
        for (size_t a = 0; a < p.axes.size(); ++a) {
            if (p.axes[a].kind == AxisKind::S && mesh[a] > 1) {
                axes.push_back(a);
            }
        }
        if (axes.size() != 1) {
            return std::nullopt;
        }
        const size_t a = axes[0];
        const int m = mesh[a];
        auto local = bridge::local_view(table);
        auto idx = bridge::replicated(index);
        if (p.axes[a].dim == 1) {
            auto rows = bridge::embedding_rows(local, 0, idx, n);
            return bridge::adopt_placed(rows, {n, dims[1]}, p);
        }
        if (p.axes[a].dim != 0 || dims[0] % m != 0) {
            return std::nullopt;
        }
        // Each chip's first row, uploaded at S(0) so chip c holds c * shard.
        const int64_t shard = dims[0] / m;
        std::vector<int32_t> starts(m);
        for (int c = 0; c < m; ++c) {
            starts[c] = static_cast<int32_t>(c * shard);
        }
        Placement q;
        q.axes.resize(mesh.size());
        q.axes[a] = AxisPlacement::shard(1);
        auto offsets = bridge::local_view(placed_constant(
            "row_starts|" + std::to_string(shard), starts, {m}, bridge::DType::Int32, q
        ));
        auto shifted = bridge::binary(B::Sub, idx, offsets);
        auto inside = (bridge::binary(
            B::Mul,
            bridge::binary_scalar(B::Ge, shifted, 0.0),
            bridge::binary_scalar(B::Lt, shifted, static_cast<double>(shard))
        ));
        auto clamped = bridge::clamp(shifted, 0.0, static_cast<double>(shard - 1));
        auto rows = bridge::embedding_rows(local, 0, clamped, n);
        auto mask = bridge::reshape(inside, {n, 1});
        if (bridge::dtype(mask) != bridge::dtype(rows)) {
            mask = bridge::typecast(mask, bridge::dtype(rows));
        }
        Placement out;
        out.axes.resize(mesh.size());
        out.axes[a] = AxisPlacement::partial(bridge::ReduceKind::Sum);
        return bridge::adopt_placed(
            bridge::binary(B::Mul, rows, mask), {n, dims[1]}, out
        );
    }

    namespace {

        // Placed `h` (input `i` of `f`, at placement `p`) with the shard padding of
        // every sharded reduce factor set to the reduction's identity; `h` itself
        // when no reduced shard is padded.
        bridge::HandlePtr mask_reduced_padding(
            bridge::HandlePtr h, const Factors& f, size_t i, const Placement& p
        ) {
            const auto mesh = bridge::mesh_shape();
            const auto dims = bridge::device_sizes(h);
            for (size_t a = 0; a < p.axes.size(); ++a) {
                if (p.axes[a].kind != AxisKind::S || mesh[a] == 1) {
                    continue;
                }
                const int d = p.axes[a].dim;
                const int fac =
                    (d < static_cast<int>(f.inputs[i].size()) ? f.inputs[i][d] : -1);
                if (fac < 0 || f.kinds[fac] != FactorKind::Reduce ||
                    dims[d] % mesh[a] == 0) {
                    continue;
                }
                const int64_t total = (dims[d] + mesh[a] - 1) / mesh[a] * mesh[a];
                std::vector<float> valid(total);
                for (int64_t k = 0; k < total; ++k) {
                    valid[k] = k < dims[d] ? 1.0f : 0.0f;
                }
                std::vector<int64_t> shape(dims.size(), 1);
                shape[d] = total;
                Placement q;
                q.axes.resize(mesh.size());
                q.axes[a] = AxisPlacement::shard(d);
                std::string key = "valid|" + std::to_string(dims[d]);
                for (const auto s : shape) {
                    key += "," + std::to_string(s);
                }
                auto cond = (bridge::local_view(
                    placed_constant(key, valid, shape, bridge::DType::Float32, q)
                ));
                const auto dtype = bridge::dtype(h);
                if (dtype != bridge::DType::Float32) {
                    cond = bridge::typecast(cond, dtype);
                }
                const double identity =
                    (f.reduce == bridge::ReduceKind::Sum       ? 0.0
                         : f.reduce == bridge::ReduceKind::Max ? -kInf
                                                               : kInf);
                auto fill = bridge::full({1}, identity, dtype);
                auto masked = bridge::where(cond, bridge::local_view(h), fill);
                Placement out = p;
                out.valid.clear();
                h = bridge::adopt_placed(masked, dims, out);
            }
            return h;
        }

    }  // namespace

    std::optional<bridge::HandlePtr> run_placed(
        const std::string& op,
        const Factors& f,
        const std::vector<bridge::HandlePtr>& inputs,
        const std::vector<int64_t>& out_shape,
        const std::function<bridge::HandlePtr(const std::vector<bridge::HandlePtr>&)>&
            run,
        Choice* chosen
    ) {
        std::vector<Operand> ops;
        bool placed = false;
        for (const auto& h : inputs) {
            ops.push_back(
                {bridge::placement(h), bridge::device_sizes(h), bridge::dtype(h), {}}
            );
            placed = placed || !ops.back().at.is_replicated();
        }
        if (!placed) {
            return std::nullopt;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            ops[i].ready = bridge::conversions_of(inputs[i]);
        }
        const auto hints = eager_hints();
        auto choice = decide(f, ops, hints);
        if (!choice) {
            return std::nullopt;
        }
        // Migration: a persistent input's conversion priced as amortized over
        // the window; the placement its consumers keep wanting becomes permanent.
        auto in = inputs;
        if (const auto migrate = g_migration_hook.load()) {
            if (migrate(f, ops, in, hints)) {
                choice = decide(f, ops, hints);
            }
        }
        std::vector<bridge::HandlePtr> locals;
        std::vector<Conversion> conversions;
        for (size_t i = 0; i < in.size(); ++i) {
            // A chosen conversion: made once per write epoch, then reused.
            auto h = in[i];
            if (!(ops[i].at == choice->inputs[i])) {
                bool reused = false;
                h = bridge::converted(in[i], choice->inputs[i], &reused);
                conversions.push_back({i, ops[i].at, choice->inputs[i], reused});
            }
            h = mask_reduced_padding(h, f, i, choice->inputs[i]);
            locals.push_back(bridge::local_view(h));
        }
        if (recording()) {
            std::vector<Placement> from;
            for (const auto& o : ops) {
                from.push_back(o.at);
            }
            record(op, from, choice->output, conversions, choice->reason);
        }
        if (chosen) {
            *chosen = *choice;
        }
        return bridge::adopt_placed(run(locals), out_shape, choice->output);
    }

}  // namespace ktt::mesh
