// Bridge mesh placements: placed upload / download, distribute and the gather fallback
// (host transfers; on the stream).
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>

#include "core/device/handle.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"

namespace ktt::bridge {
    namespace {

        std::mutex g_full_mutex;
        std::atomic<int64_t> g_gathers{0};
        std::atomic<int64_t> g_gather_bytes{0};
        std::atomic<int64_t> g_host_conversions{0};
        std::atomic<int64_t> g_ccl_failures{0};

        // Device-side conversions, per kind.
        enum Conversion { kAllGather, kAllReduce, kReduceScatter, kPartition, kKinds };

        std::atomic<int64_t> g_conversions[kKinds];

        // Counts a collective that raised (the host path runs instead); the first
        // message goes to stderr.
        void note_ccl_failure(const std::exception& e) {
            if (g_ccl_failures++ == 0) {
                std::fprintf(
                    stderr,
                    "kohakutt_native: mesh collective failed, host path: %s\n",
                    e.what()
                );
            }
        }

        bool strict() {
            const char* v = std::getenv("KTT_MESH_STRICT");
            return v && std::string(v) == "1";
        }

        int64_t volume(const std::vector<int64_t>& shape) {
            int64_t n = 1;
            for (int64_t d : shape) {
                n *= d;
            }
            return n;
        }

        size_t element_size(DType dtype) {
            return dtype == DType::BFloat16 ? 2 : 4;
        }

        // TILE layout in interleaved DRAM at device shape `dims`.
        tt::tt_metal::TensorSpec spec_of(
            const std::vector<int64_t>& dims, DType dtype
        ) {
            ttsl::SmallVector<uint32_t> s;
            for (int64_t d : dims) {
                s.push_back(static_cast<uint32_t>(d));
            }
            return tt::tt_metal::TensorSpec(
                ttnn::Shape(std::move(s)),
                tt::tt_metal::TensorLayout(
                    detail::to_ttnn(dtype),
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    tt::tt_metal::MemoryConfig{}
                )
            );
        }

        ShardLayout layout_of(const std::vector<int64_t>& shape, const Placement& p) {
            return ShardLayout(shape, p, mesh_shape());
        }

        // Calls f(global_index, local_index, run) for each contiguous run of the last
        // dim of chip `c`'s shard that lies inside the global tensor.
        template <typename F>
        void for_runs(const ShardLayout& l, int c, F f) {
            const auto off = l.offset(c);
            const size_t rank = l.global.size();
            if (rank == 0) {
                f(0, 0, 1);
                return;
            }
            const int64_t last = rank - 1;
            const int64_t run = (
                std::max<int64_t>(
                    0, std::min(l.local[last], l.global[last] - off[last])
                )
            );
            if (run == 0) {
                return;
            }
            std::vector<int64_t> idx(rank, 0);
            const int64_t outer = volume(l.local) / std::max<int64_t>(1, l.local[last]);
            for (int64_t o = 0; o < outer; ++o) {
                // idx: local multi-index of the row's start (last dim 0).
                bool inside = true;
                int64_t g = 0, li = 0;
                for (size_t d = 0; d < rank; ++d) {
                    const int64_t gd = off[d] + idx[d];
                    inside = inside && gd < l.global[d];
                    g = g * l.global[d] + gd;
                    li = li * l.local[d] + idx[d];
                }
                if (inside) {
                    f(g, li, run);
                }
                for (int64_t d = last - 1; d >= 0; --d) {
                    if (++idx[d] < l.local[d]) {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        }

        // True when chip `c` has coordinate 0 on every P axis of `l`.
        bool partial_root(const ShardLayout& l, int c) {
            const auto xy = l.coord(c);
            for (size_t a = 0; a < l.axes.size(); ++a) {
                if (l.axes[a].kind == AxisKind::P && xy[a] != 0) {
                    return false;
                }
            }
            return true;
        }

        template <typename T>
        T combine(ReduceKind op, T a, T b) {
            if constexpr (std::is_same_v<T, bfloat16>) {
                return bfloat16(combine<float>(op, float(a), float(b)));
            } else {
                switch (op) {
                    case ReduceKind::Sum:
                        return a + b;
                    case ReduceKind::Max:
                        return std::max(a, b);
                    case ReduceKind::Min:
                        return std::min(a, b);
                }
                return a;
            }
        }

        ReduceKind partial_op(const ShardLayout& l) {
            for (const auto& a : l.axes) {
                if (a.kind == AxisKind::P) {
                    return a.op;
                }
            }
            return ReduceKind::Sum;
        }

        // Writes the global row-major values of `t` (device tensor at placement `p`,
        // device shape `shape`) into `out`: shards assembled, partial axes reduced.
        template <typename T>
        void read_global_into(
            T* out,
            const ttnn::Tensor& t,
            const std::vector<int64_t>& shape,
            const Placement& p
        ) {
            auto chips = ttnn::distributed::get_device_tensors(t);
            if (p.is_replicated()) {
                const auto values =
                    (chips.empty() ? t.to_vector<T>() : chips[0].to_vector<T>());
                std::copy(values.begin(), values.end(), out);
                return;
            }
            const auto l = layout_of(shape, p);
            const ReduceKind op = partial_op(l);
            // Partial roots assign first, other partial chips combine after.
            for (int pass = 0; pass < 2; ++pass) {
                for (int c = 0; c < l.chips(); ++c) {
                    if (!l.replica_root(c) || partial_root(l, c) != (pass == 0)) {
                        continue;
                    }
                    const auto local = chips.at(c).to_vector<T>();
                    for_runs(l, c, [&](int64_t g, int64_t li, int64_t run) {
                        if (pass == 0) {
                            std::copy_n(local.data() + li, run, out + g);
                        } else {
                            for (int64_t k = 0; k < run; ++k) {
                                out[g + k] = combine(op, out[g + k], local[li + k]);
                            }
                        }
                    });
                }
            }
        }

        // read_global_into() as a new vector.
        template <typename T>
        std::vector<T> read_global(
            const ttnn::Tensor& t, const std::vector<int64_t>& shape, const Placement& p
        ) {
            std::vector<T> out(volume(shape));
            read_global_into(out.data(), t, shape, p);
            return out;
        }

        // A device tensor holding global row-major `values` (device shape `shape`) at
        // placement `p`: shard padding and non-root partial chips hold zeros.
        template <typename T>
        ttnn::Tensor write_global(
            const T* values,
            const std::vector<int64_t>& shape,
            DType dtype,
            const Placement& p
        ) {
            const int64_t n = volume(shape);
            if (p.is_replicated()) {
                return (
                    ttnn::Tensor::from_span(
                        ttsl::Span<const T>(values, n),
                        spec_of(shape, dtype),
                        detail::device()
                    )
                );
            }
            const auto l = layout_of(shape, p);
            const auto spec = spec_of(l.local, dtype);
            std::vector<ttnn::Tensor> shards;
            for (int c = 0; c < l.chips(); ++c) {
                std::vector<T> local(volume(l.local), T(0));
                if (partial_root(l, c)) {
                    for_runs(l, c, [&](int64_t g, int64_t li, int64_t run) {
                        std::copy_n(values + g, run, local.data() + li);
                    });
                }
                shards.push_back(ttnn::Tensor::from_vector(std::move(local), spec));
            }
            int shard_dim = 0;
            for (const auto& a : l.axes) {
                if (a.kind == AxisKind::S) {
                    shard_dim = a.dim;
                    break;
                }
            }
            auto host = (
                ttnn::distributed::from_host_shards(
                    shards, detail::device()->shape(), shard_dim
                )
            );
            return host.to_device(detail::device(), tt::tt_metal::MemoryConfig{});
        }

        // The replicated tensor of placed `h` through TT-NN collectives (inner axes
        // first, then the padding tail sliced off); nullopt with KTT_MESH_CCL=0.
        std::optional<ttnn::Tensor> gather_on_device(const Handle& h) {
            const char* ccl = std::getenv("KTT_MESH_CCL");
            if (ccl && std::string(ccl) == "0") {
                return std::nullopt;
            }
            const auto l = layout_of(h.shape, h.placement);
            ttnn::Tensor t = h.placed;
            for (size_t a = l.axes.size(); a-- > 0;) {
                const auto& ax = l.axes[a];
                if (l.mesh[a] == 1 || ax.kind == AxisKind::R) {
                    continue;
                }
                const auto axis = static_cast<uint32_t>(a);
                if (ax.kind == AxisKind::S) {
                    t = ttnn::all_gather(t, ax.dim, axis);
                    g_conversions[kAllGather]++;
                } else if (h.dtype == DType::Float32 || ax.op != ReduceKind::Sum) {
                    // fp32 sums, max, min: the chips' values gathered on dim 0, then
                    // combined on each chip.
                    const auto& s = t.logical_shape();
                    const uint32_t rows = s[0];
                    auto all = ttnn::all_gather(t, 0, axis);
                    g_conversions[kAllGather]++;
                    ttsl::SmallVector<uint32_t> begin(s.rank(), 0), end,
                        step(s.rank(), 1);
                    for (size_t d = 0; d < s.rank(); ++d) {
                        end.push_back(s[d]);
                    }
                    t = ttnn::slice(all, begin, end, step);
                    for (int k = 1; k < l.mesh[a]; ++k) {
                        begin[0] = k * rows;
                        end[0] = (k + 1) * rows;
                        const auto part = ttnn::slice(all, begin, end, step);
                        t =
                            (ax.op == ReduceKind::Max       ? ttnn::maximum(t, part)
                                 : ax.op == ReduceKind::Min ? ttnn::minimum(t, part)
                                                            : ttnn::add(t, part));
                    }
                } else {
                    t = ttnn::all_reduce(t, axis);
                    g_conversions[kAllReduce]++;
                }
            }
            const auto& got = t.logical_shape();
            if (std::vector<int64_t>(got.cbegin(), got.cend()) != l.global) {
                ttsl::SmallVector<uint32_t> begin(l.global.size(), 0), end, step;
                for (int64_t d : l.global) {
                    end.push_back(static_cast<uint32_t>(d));
                    step.push_back(1);
                }
                t = ttnn::slice(t, begin, end, step);
            }
            return t;
        }

        // `t` (replicated, device shape `global`) zero-padded to the equal shards of
        // layout `l`.
        ttnn::Tensor pad_to_shards(const ttnn::Tensor& t, const ShardLayout& l) {
            ttsl::SmallVector<std::array<uint32_t, 2>> pads;
            bool any = false;
            std::vector<int64_t> ways(l.global.size(), 1);
            for (size_t a = 0; a < l.axes.size(); ++a) {
                if (l.axes[a].kind == AxisKind::S) {
                    ways[l.axes[a].dim] *= l.mesh[a];
                }
            }
            for (size_t d = 0; d < l.global.size(); ++d) {
                const int64_t extra = l.local[d] * ways[d] - l.global[d];
                pads.push_back({0, static_cast<uint32_t>(extra)});
                any = any || extra > 0;
            }
            return any ? ttnn::pad(t, pads, 0.0f) : t;
        }

        // The stored tensor of `a` converted to placement `p` with TT-NN collectives:
        // P(sum) -> S(d) on one axis by reduce-scatter; otherwise gather to R
        // (all-gather / all-reduce), then partition per S axis of `p`. nullopt for
        // targets with P axes, P(max / min) sources, or KTT_MESH_CCL=0.
        std::optional<ttnn::Tensor> convert_on_device(
            const Handle& a, const Placement& p
        ) {
            const char* ccl = std::getenv("KTT_MESH_CCL");
            if ((ccl && std::string(ccl) == "0") || p.has_partial()) {
                return std::nullopt;
            }
            const auto from = layout_of(a.shape, a.placement);
            const auto to = layout_of(a.shape, p);
            // TT-NN partitions and scatters TILE tensors correctly only at tile
            // boundaries: every target shard on the two tiled dims is a multiple of 32.
            const size_t rank = to.global.size();
            for (size_t ax = 0; ax < to.axes.size(); ++ax) {
                const auto& t = to.axes[ax];
                if (t.kind == AxisKind::S && to.mesh[ax] > 1 &&
                    static_cast<size_t>(t.dim) + 2 >= rank &&
                    to.local[t.dim] % 32 != 0) {
                    return std::nullopt;
                }
            }
            // Reduce-scatter: the source is P(sum) on exactly the axes where the target
            // is S, and both are R elsewhere.
            bool scatter = a.placement.has_partial() && a.dtype != DType::Float32;
            for (size_t ax = 0; ax < from.axes.size() && scatter; ++ax) {
                const auto& f = from.axes[ax];
                const auto& t = to.axes[ax];
                const bool here =
                    (f.kind == AxisKind::P && f.op == ReduceKind::Sum &&
                     t.kind == AxisKind::S && from.mesh[ax] > 1);
                const bool idle = f.kind == AxisKind::R && t.kind == AxisKind::R;
                scatter = here || idle;
            }
            if (scatter) {
                ttnn::Tensor t = pad_to_shards(a.placed, to);
                for (size_t ax = 0; ax < to.axes.size(); ++ax) {
                    if (to.axes[ax].kind == AxisKind::S && to.mesh[ax] > 1) {
                        t = (
                            ttnn::reduce_scatter(
                                t, to.axes[ax].dim, static_cast<uint32_t>(ax)
                            )
                        );
                        g_conversions[kReduceScatter]++;
                    }
                }
                return t;
            }
            ttnn::Tensor t;
            if (a.placement.is_replicated()) {
                t = a.tensor;
            } else if (auto r = gather_on_device(a)) {
                t = std::move(*r);
            } else {
                return std::nullopt;
            }
            if (p.is_replicated()) {
                return t;
            }
            t = pad_to_shards(t, to);
            // Axis 0 outermost: partition it first, then inner axes within its shard.
            for (size_t ax = 0; ax < to.axes.size(); ++ax) {
                if (to.axes[ax].kind == AxisKind::S && to.mesh[ax] > 1) {
                    t = (
                        ttnn::mesh_partition(
                            t, to.axes[ax].dim, static_cast<uint32_t>(ax)
                        )
                    );
                    g_conversions[kPartition]++;
                }
            }
            return t;
        }

        // Calls f(T*) with the element type of `dtype`.
        template <typename F>
        void with_type(DType dtype, F f) {
            switch (dtype) {
                case DType::BFloat16:
                    return f(static_cast<bfloat16*>(nullptr));
                case DType::Float32:
                    return f(static_cast<float*>(nullptr));
                case DType::Int32:
                    return f(static_cast<int32_t*>(nullptr));
                default:
                    throw std::runtime_error(
                        "kohakutt_native: block floats are not placed on the mesh"
                    );
            }
        }

        // A new handle of `shape` at placement `p` (its tensor assigned on the stream).
        HandlePtr make_placed(
            const std::vector<int64_t>& shape, DType dtype, const Placement& p
        ) {
            if (p.is_replicated()) {
                return detail::make_handle(shape, dtype);
            }
            const auto layout = layout_of(shape, p);
            Placement placed = p;
            std::vector<int64_t> ways(shape.size(), 1);
            for (size_t a = 0; a < layout.axes.size(); ++a) {
                if (layout.axes[a].kind == AxisKind::S) {
                    ways[layout.axes[a].dim] *= layout.mesh[a];
                }
            }
            placed.valid.assign(shape.size(), 0);
            for (size_t d = 0; d < shape.size(); ++d) {
                if (layout.local[d] * ways[d] != shape[d]) {
                    placed.valid[d] = shape[d];
                }
            }
            auto handle = (
                HandlePtr(
                    new Handle(shape, dtype, std::move(placed), layout.local),
                    detail::release_handle
                )
            );
            detail::on_wrap(handle);
            detail::register_handle(handle);
            return handle;
        }

        void assign(Handle& h, ttnn::Tensor t) {
            if (h.placement.is_replicated()) {
                h.tensor = std::move(t);
            } else {
                h.placed = std::move(t);
            }
        }

    }  // namespace

    std::vector<int> mesh_shape() {
        const auto& s = detail::device()->shape();
        std::vector<int> out;
        for (size_t a = 0; a < s.dims(); ++a) {
            out.push_back(static_cast<int>(s[a]));
        }
        return out;
    }

    Placement placement(const HandlePtr& handle) {
        return handle->placement;
    }

    HandlePtr upload_placed(
        const void* data,
        const std::vector<int64_t>& shape,
        DType dtype,
        const Placement& p
    ) {
        if (p.is_replicated()) {
            return upload(data, shape, dtype);
        }
        if (is_block(dtype)) {
            throw std::runtime_error("kohakutt_native: block floats are device-only");
        }
        if (capturing()) {
            throw std::runtime_error(
                "kohakutt_native: host upload inside a trace capture"
            );
        }
        detail::counters().uploads++;
        const auto dims = detail::device_dims(shape);
        auto bytes = (
            std::make_shared<std::vector<uint8_t>>(volume(dims) * element_size(dtype))
        );
        std::memcpy(bytes->data(), data, bytes->size());
        auto out = make_placed(dims, dtype, p);
        detail::submit([out, bytes, dims, dtype, p] {
            with_type(dtype, [&]<typename T>(T*) {
                out->placed = (
                    write_global(
                        reinterpret_cast<const T*>(bytes->data()), dims, dtype, p
                    )
                );
            });
        });
        return out;
    }

    HandlePtr distribute(const HandlePtr& a, const Placement& p) {
        if (a->placement == p) {
            return a;
        }
        if (capturing()) {
            throw std::runtime_error(
                "kohakutt_native: mesh redistribution inside a trace capture"
            );
        }
        auto out = make_placed(a->shape, a->dtype, p);
        detail::submit([out, a, p] {
            try {
                if (auto t = convert_on_device(*a, p)) {
                    assign(*out, std::move(*t));
                    return;
                }
            } catch (const std::exception& e) {
                note_ccl_failure(e);
            }
            g_host_conversions++;
            with_type(a->dtype, [&]<typename T>(T*) {
                auto values = read_global<T>(a->stored(), a->shape, a->placement);
                assign(*out, write_global(values.data(), a->shape, a->dtype, p));
            });
        });
        return out;
    }

    HandlePtr replicated(const HandlePtr& a) {
        if (a->placement.is_replicated()) {
            return a;
        }
        std::lock_guard lock(g_full_mutex);
        if (a->full) {
            return a->full;
        }
        if (strict()) {
            throw std::runtime_error(
                "kohakutt_native: gather fallback of a tensor placed " +
                a->placement.str() + " (KTT_MESH_STRICT=1)"
            );
        }
        g_gathers++;
        g_gather_bytes += volume(a->shape) * element_size(a->dtype);
        a->full = distribute(a, Placement::replicated());
        return a->full;
    }

    MeshCounters mesh_counters() {
        // Queued conversions count on the stream: drained so earlier work is included.
        detail::drain();
        return MeshCounters{
            g_gathers.load(),
            g_gather_bytes.load(),
            g_conversions[kAllGather].load(),
            g_conversions[kAllReduce].load(),
            g_conversions[kReduceScatter].load(),
            g_conversions[kPartition].load(),
            g_host_conversions.load(),
            g_ccl_failures.load()
        };
    }

    void reset_mesh_counters() {
        detail::drain();
        g_gathers = 0;
        g_gather_bytes = 0;
        for (auto& c : g_conversions) {
            c = 0;
        }
        g_host_conversions = 0;
        g_ccl_failures = 0;
    }

    Placement auto_placement(const std::vector<int64_t>& dims, DType dtype) {
        const char* on = std::getenv("KTT_MESH_AUTO");
        if ((on && std::string(on) == "0") || is_block(dtype) || mesh_size() < 2) {
            return Placement::replicated();
        }
        const char* min_env = std::getenv("KTT_MESH_SHARD_MIN");
        const int64_t min_bytes = (
            min_env ? std::strtoll(min_env, nullptr, 10) : (1 << 20)
        );
        if (volume(dims) * static_cast<int64_t>(element_size(dtype)) < min_bytes) {
            return Placement::replicated();
        }
        // The largest mesh axis carries the shard; other axes replicate.
        const auto mesh = mesh_shape();
        const size_t axis = std::max_element(mesh.begin(), mesh.end()) - mesh.begin();
        const int m = mesh[axis];
        // Leading dim when it divides the axis, else the largest dim that divides.
        int dim = -1;
        if (dims[0] % m == 0) {
            dim = 0;
        } else {
            for (size_t d = 0; d < dims.size(); ++d) {
                if (dims[d] % m == 0 && (dim < 0 || dims[d] > dims[dim])) {
                    dim = static_cast<int>(d);
                }
            }
        }
        if (dim < 0) {
            return Placement::replicated();
        }
        Placement p;
        p.axes.resize(mesh.size());
        p.axes[axis] = AxisPlacement::shard(dim);
        return p;
    }

    HandlePtr empty_placed(
        const std::vector<int64_t>& dims, DType dtype, const Placement& p
    ) {
        auto out = make_placed(dims, dtype, p);
        detail::submit([out, dtype] {
            const auto& local = local_sizes(out);
            assign(
                *out,
                ttnn::empty(
                    spec_of(local, dtype).logical_shape(),
                    detail::to_ttnn(dtype),
                    tt::tt_metal::Layout::TILE,
                    detail::device(),
                    tt::tt_metal::MemoryConfig{}
                )
            );
        });
        return out;
    }

    HandlePtr upload_auto(
        const void* data, const std::vector<int64_t>& shape, DType dtype
    ) {
        const auto p =
            (capturing() ? Placement::replicated()
                         : auto_placement(detail::device_dims(shape), dtype));
        return upload_placed(data, shape, dtype, p);
    }

    HandlePtr empty_auto(const std::vector<int64_t>& shape, DType dtype) {
        const auto dims = detail::device_dims(shape);
        const auto p = (
            capturing() ? Placement::replicated() : auto_placement(dims, dtype)
        );
        return p.is_replicated() ? empty(shape, dtype) : empty_placed(dims, dtype, p);
    }

    HandlePtr full_placed(
        const std::vector<int64_t>& dims, double value, DType dtype, const Placement& p
    ) {
        auto out = make_placed(dims, dtype, p);
        detail::submit([out, value, dtype] {
            const auto spec = spec_of(local_sizes(out), dtype);
            auto fill = [&](auto typed_value) {
                return (
                    ttnn::full(
                        spec.logical_shape(),
                        typed_value,
                        detail::to_ttnn(dtype),
                        tt::tt_metal::Layout::TILE,
                        std::ref(*detail::device())
                    )
                );
            };
            assign(
                *out,
                dtype == DType::Int32 ? fill(static_cast<int>(value))
                                      : fill(static_cast<float>(value))
            );
        });
        return out;
    }

    MemoryInfo mem_info() {
        MemoryInfo info;
        detail::run_sync([&] {
            const auto& alloc = detail::device()->allocator();
            const auto dram = tt::tt_metal::BufferType::DRAM;
            const auto stats = alloc->get_statistics(dram);
            const int64_t banks = alloc->get_num_banks(dram);
            const int64_t chips = mesh_size();
            info.free = static_cast<int64_t>(stats.total_free_bytes) * banks * chips;
            info.total =
                (static_cast<int64_t>(stats.total_allocatable_size_bytes) * banks *
                 chips);
        });
        return info;
    }

    HandlePtr local_view(const HandlePtr& a) {
        if (a->placement.is_replicated()) {
            return a;
        }
        std::lock_guard lock(g_full_mutex);
        if (a->view) {
            return a->view;
        }
        auto view = detail::make_handle(a->local, a->dtype);
        a->view = view;
        // The view shares a's buffers: its bytes are a's.
        detail::track_bytes(-view->bytes);
        view->bytes = 0;
        detail::submit([view, a] {
            view->tensor = a->placed;
        });
        return view;
    }

    HandlePtr adopt_placed(
        const HandlePtr& local, const std::vector<int64_t>& shape, const Placement& p
    ) {
        if (p.is_replicated()) {
            return local;
        }
        auto out = make_placed(shape, local->dtype, p);
        if (out->local != local->shape) {
            throw std::logic_error(
                "kohakutt_native: a rule's local result has the wrong shard shape "
                "for " +
                p.str()
            );
        }
        detail::submit([out, local] {
            out->placed = local->tensor;
        });
        return out;
    }

    std::vector<int64_t> local_sizes(const HandlePtr& handle) {
        return handle->placement.is_replicated() ? handle->shape : handle->local;
    }

    void download_local(const HandlePtr& handle, void* dst, int chip) {
        if (chip < 0 || chip >= mesh_size()) {
            throw std::out_of_range(
                "kohakutt_native: no mesh chip " + std::to_string(chip)
            );
        }
        detail::run_sync([&] {
            auto chips = ttnn::distributed::get_device_tensors(handle->stored());
            with_type(handle->dtype, [&]<typename T>(T*) {
                auto values = chips.at(chip).to_vector<T>();
                std::copy(values.begin(), values.end(), static_cast<T*>(dst));
            });
        });
    }

    namespace detail {

        void download_placed(const HandlePtr& handle, void* dst) {
            run_sync([&] {
                with_type(handle->dtype, [&]<typename T>(T*) {
                    read_global_into(
                        static_cast<T*>(dst),
                        handle->stored(),
                        handle->shape,
                        handle->placement
                    );
                });
            });
        }

    }  // namespace detail
}  // namespace ktt::bridge
