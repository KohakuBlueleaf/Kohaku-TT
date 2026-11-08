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
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

namespace ktt::bridge {
    namespace {

        std::mutex g_full_mutex;
        std::atomic<int64_t> g_gathers{0};
        std::atomic<int64_t> g_gather_bytes{0};
        std::atomic<int64_t> g_device_gathers{0};
        std::atomic<int64_t> g_ccl_failures{0};

        // Counts a collective that raised (the host path runs instead); the first
        // message goes to stderr.
        void note_ccl_failure(const std::exception& e) {
            if (g_ccl_failures++ == 0) {
                std::fprintf(
                    stderr, "kohakutt_native: mesh collective failed, host path: %s\n", e.what()
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
        tt::tt_metal::TensorSpec spec_of(const std::vector<int64_t>& dims, DType dtype) {
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
            const int64_t run =
                std::max<int64_t>(0, std::min(l.local[last], l.global[last] - off[last]));
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

        // The global row-major values of `t` (device tensor at placement `p`, device
        // shape `shape`): shards assembled, partial axes reduced.
        template <typename T>
        std::vector<T> read_global(
            const ttnn::Tensor& t, const std::vector<int64_t>& shape, const Placement& p
        ) {
            if (p.is_replicated()) {
                auto chips = ttnn::distributed::get_device_tensors(t);
                return chips.empty() ? t.to_vector<T>() : chips[0].to_vector<T>();
            }
            const auto l = layout_of(shape, p);
            auto chips = ttnn::distributed::get_device_tensors(t);
            std::vector<T> out(volume(shape));
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
                            std::copy_n(local.data() + li, run, out.data() + g);
                        } else {
                            for (int64_t k = 0; k < run; ++k) {
                                out[g + k] = combine(op, out[g + k], local[li + k]);
                            }
                        }
                    });
                }
            }
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
                return ttnn::Tensor::from_span(
                    ttsl::Span<const T>(values, n), spec_of(shape, dtype), detail::device()
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
            auto host = ttnn::distributed::from_host_shards(
                shards, detail::device()->shape(), shard_dim
            );
            return host.to_device(detail::device(), tt::tt_metal::MemoryConfig{});
        }

        // The replicated tensor of placed `h` through TT-NN collectives (inner axes
        // first, then the padding tail sliced off); nullopt for placements they do not
        // cover (P(max / min)) or with KTT_MESH_CCL=0.
        std::optional<ttnn::Tensor> gather_on_device(const Handle& h) {
            const char* ccl = std::getenv("KTT_MESH_CCL");
            if (ccl && std::string(ccl) == "0") {
                return std::nullopt;
            }
            const auto l = layout_of(h.shape, h.placement);
            for (const auto& ax : l.axes) {
                if (ax.kind == AxisKind::P && ax.op != ReduceKind::Sum) {
                    return std::nullopt;
                }
            }
            ttnn::Tensor t = h.placed;
            for (size_t a = l.axes.size(); a-- > 0;) {
                const auto& ax = l.axes[a];
                if (l.mesh[a] == 1 || ax.kind == AxisKind::R) {
                    continue;
                }
                const auto axis = static_cast<uint32_t>(a);
                t = ax.kind == AxisKind::S ? ttnn::all_gather(t, ax.dim, axis)
                                           : ttnn::all_reduce(t, axis);
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
            auto local = layout_of(shape, p).local;
            auto handle = HandlePtr(
                new Handle(shape, dtype, p, std::move(local)), detail::release_handle
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
        auto bytes = std::make_shared<std::vector<uint8_t>>(
            volume(dims) * element_size(dtype)
        );
        std::memcpy(bytes->data(), data, bytes->size());
        auto out = make_placed(dims, dtype, p);
        detail::submit([out, bytes, dims, dtype, p] {
            with_type(dtype, [&]<typename T>(T*) {
                out->placed = write_global(
                    reinterpret_cast<const T*>(bytes->data()), dims, dtype, p
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
            if (p.is_replicated()) {
                try {
                    if (auto t = gather_on_device(*a)) {
                        out->tensor = std::move(*t);
                        g_device_gathers++;
                        return;
                    }
                } catch (const std::exception& e) {
                    note_ccl_failure(e);
                }
            }
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
        return MeshCounters{
            g_gathers.load(),
            g_gather_bytes.load(),
            g_device_gathers.load(),
            g_ccl_failures.load()
        };
    }

    void reset_mesh_counters() {
        g_gathers = 0;
        g_gather_bytes = 0;
        g_device_gathers = 0;
        g_ccl_failures = 0;
    }

    Placement auto_placement(const std::vector<int64_t>& dims, DType dtype) {
        const char* on = std::getenv("KTT_MESH_AUTO");
        if (!on || std::string(on) != "1" || is_block(dtype) || mesh_size() < 2) {
            return Placement::replicated();
        }
        const char* min_env = std::getenv("KTT_MESH_SHARD_MIN");
        const int64_t min_bytes = min_env ? std::strtoll(min_env, nullptr, 10) : (1 << 20);
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
                static_cast<int64_t>(stats.total_allocatable_size_bytes) * banks * chips;
        });
        return info;
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
                    auto values = read_global<T>(
                        handle->stored(), handle->shape, handle->placement
                    );
                    std::copy(values.begin(), values.end(), static_cast<T*>(dst));
                });
            });
        }

    }  // namespace detail
}  // namespace ktt::bridge
