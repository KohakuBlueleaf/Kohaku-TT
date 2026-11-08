// Mesh placements of device tensors: per mesh axis R, S(dim) or P(op) (no TT-NN or
// torch headers).
#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ktt::bridge {

    enum class AxisKind : uint8_t { R, S, P };
    enum class ReduceKind : uint8_t { Sum, Max, Min };

    // One mesh axis: R replicated, S(dim) equal contiguous shards of device dim `dim`,
    // P(op) pending reduction `op` over the axis.
    struct AxisPlacement {
        AxisKind kind = AxisKind::R;
        int dim = -1;
        ReduceKind op = ReduceKind::Sum;

        static AxisPlacement replicate() {
            return {};
        }
        static AxisPlacement shard(int d) {
            return {AxisKind::S, d, ReduceKind::Sum};
        }
        static AxisPlacement partial(ReduceKind r = ReduceKind::Sum) {
            return {AxisKind::P, -1, r};
        }
        bool operator==(const AxisPlacement&) const = default;
    };

    // One entry per mesh axis (rows, cols); missing entries are R.
    struct Placement {
        std::vector<AxisPlacement> axes;

        static Placement replicated() {
            return {};
        }

        bool is_replicated() const {
            return std::all_of(axes.begin(), axes.end(), [](const AxisPlacement& a) {
                return a.kind == AxisKind::R;
            });
        }

        bool has_partial() const {
            return std::any_of(axes.begin(), axes.end(), [](const AxisPlacement& a) {
                return a.kind == AxisKind::P;
            });
        }

        bool operator==(const Placement& o) const {
            const size_t n = std::max(axes.size(), o.axes.size());
            for (size_t k = 0; k < n; ++k) {
                const auto a = k < axes.size() ? axes[k] : AxisPlacement{};
                const auto b = k < o.axes.size() ? o.axes[k] : AxisPlacement{};
                if (!(a == b)) {
                    return false;
                }
            }
            return true;
        }

        // e.g. "[R, S(1)]", "[P(sum), R]".
        std::string str() const {
            std::ostringstream s;
            s << "[";
            for (size_t k = 0; k < axes.size(); ++k) {
                const auto& a = axes[k];
                s << (k ? ", " : "");
                if (a.kind == AxisKind::R) {
                    s << "R";
                } else if (a.kind == AxisKind::S) {
                    s << "S(" << a.dim << ")";
                } else {
                    s << "P("
                      << (a.op == ReduceKind::Sum   ? "sum"
                          : a.op == ReduceKind::Max ? "max"
                                                    : "min")
                      << ")";
                }
            }
            s << "]";
            return s.str();
        }
    };

    // Shard geometry of device shape `shape` placed by `p` on mesh sizes `m`: equal
    // ceil(n / m) shards (tail is padding); chips are row-major over mesh coords.
    struct ShardLayout {
        std::vector<int64_t> global;
        std::vector<int64_t> local;
        std::vector<int> mesh;
        std::vector<AxisPlacement> axes;

        ShardLayout(
            const std::vector<int64_t>& shape, const Placement& p, std::vector<int> m
        )
            : global(shape), local(shape), mesh(std::move(m)), axes(p.axes) {
            if (axes.size() > mesh.size()) {
                throw std::invalid_argument(
                    "kohakutt_native: placement " + p.str() +
                    " has more axes than the mesh"
                );
            }
            axes.resize(mesh.size());
            for (size_t a = 0; a < axes.size(); ++a) {
                if (axes[a].kind != AxisKind::S) {
                    continue;
                }
                const int d = axes[a].dim;
                if (d < 0 || d >= static_cast<int>(shape.size())) {
                    throw std::invalid_argument(
                        "kohakutt_native: placement " + p.str() + " shards dim " +
                        std::to_string(d) + " of a rank-" +
                        std::to_string(shape.size()) + " device tensor"
                    );
                }
                local[d] = (local[d] + mesh[a] - 1) / mesh[a];
            }
        }

        int chips() const {
            int n = 1;
            for (int m : mesh) {
                n *= m;
            }
            return n;
        }

        // Mesh coordinate of chip `c`.
        std::vector<int> coord(int c) const {
            std::vector<int> out(mesh.size());
            for (size_t a = mesh.size(); a-- > 0;) {
                out[a] = c % mesh[a];
                c /= mesh[a];
            }
            return out;
        }

        // Global start of chip `c`'s shard per device dim.
        std::vector<int64_t> offset(int c) const {
            const auto xy = coord(c);
            // Shard index per dim: mixed radix over its splitting axes, axis 0 outermost.
            std::vector<int64_t> index(global.size(), 0);
            for (size_t a = 0; a < axes.size(); ++a) {
                if (axes[a].kind == AxisKind::S) {
                    index[axes[a].dim] = index[axes[a].dim] * mesh[a] + xy[a];
                }
            }
            std::vector<int64_t> off(global.size(), 0);
            for (size_t d = 0; d < global.size(); ++d) {
                off[d] = index[d] * local[d];
            }
            return off;
        }

        // True when chip `c` has coordinate 0 on every R axis.
        bool replica_root(int c) const {
            const auto xy = coord(c);
            for (size_t a = 0; a < axes.size(); ++a) {
                if (axes[a].kind == AxisKind::R && xy[a] != 0) {
                    return false;
                }
            }
            return true;
        }
    };

}  // namespace ktt::bridge
