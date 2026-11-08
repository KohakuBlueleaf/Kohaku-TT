// Helpers shared by Kohaku-TT's own Metalium program builders.
#include "core/device/program.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/device/generic_op_device_operation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::program {

    bool own_kernels_enabled() {
        static const bool on = [] {
            const char* v = std::getenv("KTT_CUSTOM_KERNELS");
            return !(v && std::string(v) == "0");
        }();
        return on;
    }

    std::string source(const std::string& relative) {
        static const std::filesystem::path dir = [] {
            Dl_info info{};
            dladdr(reinterpret_cast<void*>(&own_kernels_enabled), &info);
            return (
                std::filesystem::path(info.dli_fname).parent_path() / "csrc" / "aten"
            );
        }();
        return (dir / relative).string();
    }

    int64_t min_tiles(int64_t measured) {
        static const char* v = std::getenv("KTT_CUSTOM_MIN_TILES");
        return v ? std::atoll(v) : measured;
    }

    int64_t tile_count(const std::vector<int64_t>& shape) {
        int64_t tiles = 1;
        const size_t r = shape.size();
        for (size_t i = 0; i + 2 < r; ++i) {
            tiles *= shape[i];
        }
        return tiles * ((shape[r - 2] + 31) / 32) * ((shape[r - 1] + 31) / 32);
    }

    Split split_tiles(uint32_t tiles, uint32_t min_per_core) {
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t total = grid.x * grid.y;
        const uint32_t wanted = (tiles + min_per_core - 1) / min_per_core;
        const uint32_t used = std::clamp<uint32_t>(wanted, 1u, total);
        Split s;
        s.range = tt::tt_metal::num_cores_to_corerangeset(used, grid, true);
        s.cores = corerange_to_cores(s.range, std::nullopt, true);
        s.base = tiles / used;
        s.extra = tiles % used;
        return s;
    }

    tt::tt_metal::KernelDescriptor::RuntimeArgs core_index_args(uint32_t cores) {
        const auto grid = detail::device()->compute_with_storage_grid_size();
        tt::tt_metal::KernelDescriptor::RuntimeArgs args;
        args.reserve(cores);
        for (uint32_t c = 0; c < cores; ++c) {
            args.push_back({tt::tt_metal::CoreCoord(c % grid.x, c / grid.x), {c}});
        }
        return args;
    }

    namespace {

        // ns since the phase clock, which moves to now.
        int64_t lap() {
            const auto now = std::chrono::steady_clock::now();
            auto& clock = detail::phase_clock();
            const auto ns = (
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - clock)
                    .count()
            );
            clock = now;
            return ns;
        }

    }  // namespace

    void mark(Phase phase) {
        auto& c = detail::counters();
        (phase == Phase::Alloc ? c.alloc_ns : c.args_ns) += lap();
    }

    Key::Key(std::string_view program) : key_(program) {
        key_.push_back('\0');
    }

    Key& Key::add(int64_t value) {
        key_.append(reinterpret_cast<const char*>(&value), sizeof(value));
        return *this;
    }

    Key& Key::add(std::string_view value) {
        add(static_cast<int64_t>(value.size()));
        key_.append(value);
        return *this;
    }

    Key& Key::add(const ttnn::Tensor& tensor) {
        add(static_cast<int64_t>(tensor.dtype()));
        add(static_cast<int64_t>(tensor.layout()));
        for (const auto* shape : {&tensor.logical_shape(), &tensor.padded_shape()}) {
            add(static_cast<int64_t>(shape->rank()));
            for (const auto d : *shape) {
                add(static_cast<int64_t>(d));
            }
        }
        const auto& config = tensor.memory_config();
        add(static_cast<int64_t>(config.buffer_type()));
        add(static_cast<int64_t>(config.memory_layout()));
        if (config.is_sharded()) {
            add(
                static_cast<int64_t>(ttsl::hash::hash_objects_with_default_seed(config))
            );
        }
        return *this;
    }

    bool desc_cache_enabled() {
        static const bool on = [] {
            const char* v = std::getenv("KTT_DESC_CACHE");
            return !(v && std::string(v) == "0");
        }();
        return on;
    }

    namespace {

        using tt::tt_metal::ProgramDescriptor;
        using tt::tt_metal::experimental::MeshProgramDescriptor;

        bool verify_desc_cache() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_DESC_CACHE");
                return v && std::string(v) == "verify";
            }();
            return on;
        }

        // Descriptors kept at most; the cache is emptied when it would grow beyond.
        constexpr size_t kMaxCached = 4096;

        struct Cached {
            MeshProgramDescriptor mesh;
            std::vector<Slot> slots;
        };

        std::unordered_map<std::string, Cached>& desc_cache() {
            static std::unordered_map<std::string, Cached> cache;
            return cache;
        }

        // `desc` as generic_op's one-program mesh descriptor over io's mesh.
        MeshProgramDescriptor over_mesh(
            const std::vector<ttnn::Tensor>& io, ProgramDescriptor desc
        ) {
            MeshProgramDescriptor mesh;
            mesh.mesh_programs.emplace_back(
                ttnn::MeshCoordinateRange(io.front().device()->shape()), std::move(desc)
            );
            return mesh;
        }

        // Writes io's buffer addresses into `desc` at `slots`.
        void patch(
            ProgramDescriptor& desc,
            const std::vector<Slot>& slots,
            const std::vector<ttnn::Tensor>& io
        ) {
            for (const auto& s : slots) {
                const uint32_t addr = io.at(s.tensor).buffer()->address();
                auto& kernel = desc.kernels[s.kernel];
                if (s.entry == Slot::kCommon) {
                    kernel.common_runtime_args.at(s.position) = addr;
                } else if (s.entry == Slot::kEveryCore) {
                    for (auto& [core, args] : kernel.runtime_args) {
                        args.at(s.position) = addr;
                    }
                } else {
                    kernel.runtime_args.at(s.entry).second.at(s.position) = addr;
                }
            }
        }

        // Throws when the descriptors differ in anything generic_op reads.
        void check_same(
            const std::string& name,
            const ProgramDescriptor& a,
            const ProgramDescriptor& b
        ) {
            bool same =
                (a.kernels.size() == b.kernels.size() && a.cbs.size() == b.cbs.size());
            for (size_t k = 0; same && k < a.kernels.size(); ++k) {
                const auto &x = a.kernels[k], &y = b.kernels[k];
                same =
                    (x.kernel_source == y.kernel_source &&
                     x.core_ranges == y.core_ranges &&
                     x.compile_time_args == y.compile_time_args &&
                     x.defines == y.defines && x.runtime_args == y.runtime_args &&
                     x.common_runtime_args == y.common_runtime_args &&
                     x.config.index() == y.config.index());
            }
            for (size_t i = 0; same && i < a.cbs.size(); ++i) {
                const auto &x = a.cbs[i], &y = b.cbs[i];
                same =
                    (x.total_size == y.total_size && x.core_ranges == y.core_ranges &&
                     x.format_descriptors.size() == y.format_descriptors.size());
                for (size_t f = 0; same && f < x.format_descriptors.size(); ++f) {
                    const auto &p = x.format_descriptors[f],
                               &q = y.format_descriptors[f];
                    same =
                        (p.buffer_index == q.buffer_index &&
                         p.data_format == q.data_format && p.page_size == q.page_size);
                }
            }
            if (!same) {
                throw std::logic_error(
                    "KTT_DESC_CACHE=verify: the cached descriptor of " + name +
                    " differs from a fresh build"
                );
            }
        }

        // ttnn::generic_op over io, its time counted.
        ttnn::Tensor enqueue(
            const std::vector<ttnn::Tensor>& io, const MeshProgramDescriptor& mesh
        ) {
            auto& c = detail::counters();
            c.desc_ns += lap();
            auto out = ttnn::generic_op(io, mesh);
            c.enqueue_ns += lap();
            c.programs++;
            return out;
        }

    }  // namespace

    ttnn::Tensor launch(
        const Key& key,
        const std::vector<ttnn::Tensor>& io,
        const std::function<Built()>& build,
        bool always_cached
    ) {
        if (!always_cached && !desc_cache_enabled()) {
            return enqueue(io, over_mesh(io, build().desc));
        }
        auto& cache = desc_cache();
        auto it = cache.find(key.str());
        if (it == cache.end()) {
            if (cache.size() >= kMaxCached) {
                cache.clear();
            }
            auto built = build();
            Cached entry{over_mesh(io, std::move(built.desc)), std::move(built.slots)};
            // The structural hash generic_op computes per call, computed once.
            const ttnn::operations::generic::tensor_args_t args{io, io.back()};
            entry.mesh.mesh_programs.front()
                .second.custom_program_hash = (
                    ttnn::operations::generic::GenericOpDeviceOperation::
                        compute_program_hash(entry.mesh, args)
                );
            it = cache.emplace(key.str(), std::move(entry)).first;
        } else {
            auto& desc = it->second.mesh.mesh_programs.front().second;
            patch(desc, it->second.slots, io);
            detail::counters().desc_hits++;
            if (verify_desc_cache()) {
                const auto phase = detail::phase_clock();
                check_same(key.str().c_str(), desc, build().desc);
                detail::phase_clock() = phase;
            }
        }
        return enqueue(io, it->second.mesh);
    }

    ttnn::Tensor launch(
        const std::vector<ttnn::Tensor>& io, const tt::tt_metal::ProgramDescriptor& desc
    ) {
        return enqueue(io, over_mesh(io, desc));
    }

    void clear_desc_cache() {
        desc_cache().clear();
    }

    std::vector<HandlePtr> run_on_stream(
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        std::function<void(const std::vector<ttnn::Tensor>& tensors)> run
    ) {
        std::vector<HandlePtr> outputs;
        for (size_t i = 0; i < out_shapes.size(); ++i) {
            outputs.push_back(
                detail::make_handle(detail::device_dims(out_shapes[i]), out_dtypes[i])
            );
        }
        detail::counters().ops++;
        detail::submit([inputs, outputs, run = std::move(run)] {
            const auto start = std::chrono::steady_clock::now();
            detail::phase_clock() = start;
            // Inputs, then outputs allocated per the activation policy.
            auto operands = [&] {
                std::vector<ttnn::Tensor> tensors;
                for (const auto& h : inputs) {
                    tensors.push_back(h->tensor);
                }
                for (const auto& h : outputs) {
                    h->tensor = (
                        ttnn::empty(
                            detail::device_shape(h->shape),
                            detail::to_ttnn(h->dtype),
                            tt::tt_metal::Layout::TILE,
                            detail::device(),
                            detail::activation_config(h->bytes)
                        )
                    );
                    tensors.push_back(h->tensor);
                }
                return tensors;
            };
            try {
                run(operands());
            } catch (const std::exception& e) {
                if (!detail::is_l1_clash(e)) {
                    throw;
                }
                // Again with L1 activations spilled and the outputs in DRAM.
                for (const auto& h : outputs) {
                    h->tensor = ttnn::Tensor();
                }
                detail::spill_l1();
                const detail::DramOnly dram;
                run(operands());
            }
            const auto elapsed = std::chrono::steady_clock::now() - start;
            using std::chrono::duration_cast, std::chrono::nanoseconds;
            detail::counters().op_ns += duration_cast<nanoseconds>(elapsed).count();
        });
        return outputs;
    }

}  // namespace ktt::bridge::program
