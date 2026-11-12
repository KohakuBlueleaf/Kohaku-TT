// Helpers shared by Kohaku-TT's own Metalium program builders.
#include "core/device/program.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <variant>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
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
            auto root = std::filesystem::path(info.dli_fname).parent_path();
            // A wheel holds one library per torch series in torch<x.y>/, the
            // kernel sources once beside those directories.
            if (!std::filesystem::exists(root / "csrc")) {
                root = root.parent_path();
            }
            return root / "csrc" / "aten";
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

    uint32_t cb_budget() {
        return static_cast<uint32_t>(std::max<int64_t>(0, detail::l1_free_per_bank()));
    }

    namespace {

        // Panel spans of the open device: computed on first use, dropped at close.
        std::optional<std::vector<std::pair<uint32_t, uint32_t>>>& panels_cache() {
            static auto*
                cache = new std::optional<std::vector<std::pair<uint32_t, uint32_t>>>();
            return *cache;
        }

    }  // namespace

    const std::vector<std::pair<uint32_t, uint32_t>>& panels() {
        auto& cache = panels_cache();
        if (cache) {
            return *cache;
        }
        cache = [] {
            const auto& mesh = detail::device();
            const auto grid = mesh->compute_with_storage_grid_size();
            std::vector<bool> breaks(grid.x, false);
            for (auto* chip : mesh->get_devices()) {
                const auto& soc = (tt::tt_metal::MetalContext::instance()
                                       .get_cluster()
                                       .get_soc_desc(chip->id()));
                auto phys_x = [&](uint32_t x) {
                    const auto t = mesh->worker_core_from_logical_core({x, 0});
                    return soc
                        .translate_coord_to(
                            tt::umd::CoreCoord(
                                t.x,
                                t.y,
                                tt::CoreType::TENSIX,
                                tt::CoordSystem::TRANSLATED
                            ),
                            tt::CoordSystem::NOC0
                        )
                        .x;
                };
                for (uint32_t x = 1; x < grid.x; ++x) {
                    if (phys_x(x) != phys_x(x - 1) + 1) {
                        breaks[x] = true;
                    }
                }
            }
            std::vector<std::pair<uint32_t, uint32_t>> spans;
            for (uint32_t x = 0; x < grid.x; ++x) {
                if (x == 0 || breaks[x]) {
                    spans.emplace_back(x, x);
                } else {
                    spans.back().second = x;
                }
            }
            return spans;
        }();
        return *cache;
    }

    namespace {

        // ns since the phase clock, which moves to now.
        int64_t lap() {
            const auto now = std::chrono::steady_clock::now();
            auto& clock = detail::phase_clock();
            const auto ns = (std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 now - clock
            )
                                 .count());
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

        using Workload = tt::tt_metal::distributed::MeshWorkload;

        // A cached descriptor and, once launched, its program on `device`: a repeat
        // launch writes only the slots and values into the program and enqueues it.
        struct Cached {
            MeshProgramDescriptor mesh;
            std::vector<Slot> slots;
            std::shared_ptr<Workload> workload;
            ttnn::MeshDevice* device = nullptr;
        };

        // Never destroyed at exit (its programs outlive TT-Metal's teardown); close()
        // empties it.
        std::unordered_map<std::string, Cached>& desc_cache() {
            static auto* cache = new std::unordered_map<std::string, Cached>();
            return *cache;
        }

        // Programs of entries dropped from a full cache: a trace may still replay
        // them, so they live until clear_desc_cache().
        std::vector<std::shared_ptr<Workload>>& retired_workloads() {
            static auto* retired = new std::vector<std::shared_ptr<Workload>>();
            return *retired;
        }

        // Programs built by launch() since the cache was last cleared.
        int64_t& own_program_count() {
            static int64_t count = 0;
            return count;
        }

        // Held by launch() from lookup through enqueue, and by clear_desc_cache().
        std::mutex& desc_cache_mutex() {
            static std::mutex m;
            return m;
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

        // True when kernel configs `a` and `b` are the same variant with equal fields.
        bool same_config(
            const tt::tt_metal::KernelDescriptor::ConfigDescriptor& a,
            const tt::tt_metal::KernelDescriptor::ConfigDescriptor& b
        ) {
            using namespace tt::tt_metal;
            if (a.index() != b.index()) {
                return false;
            }
            if (const auto* x = std::get_if<ComputeConfigDescriptor>(&a)) {
                const auto& y = std::get<ComputeConfigDescriptor>(b);
                return (
                    x->math_fidelity == y.math_fidelity &&
                    x->fp32_dest_acc_en == y.fp32_dest_acc_en &&
                    x->dst_full_sync_en == y.dst_full_sync_en &&
                    x->unpack_to_dest_mode == y.unpack_to_dest_mode &&
                    x->bfp8_pack_precise == y.bfp8_pack_precise &&
                    x->math_approx_mode == y.math_approx_mode &&
                    x->enable_trisc2_rvv == y.enable_trisc2_rvv
                );
            }
            if (const auto* x = std::get_if<DataMovementConfigDescriptor>(&a)) {
                const auto& y = std::get<DataMovementConfigDescriptor>(b);
                return (
                    x->processor == y.processor && x->noc == y.noc &&
                    x->noc_mode == y.noc_mode
                );
            }
            return true;  // Reader / Writer configs carry no fields.
        }

        // True when CB format lists `a` and `b` are equal.
        bool same_formats(
            const tt::tt_metal::CBDescriptor::FormatDescriptors& a,
            const tt::tt_metal::CBDescriptor::FormatDescriptors& b
        ) {
            if (a.size() != b.size()) {
                return false;
            }
            for (size_t f = 0; f < a.size(); ++f) {
                const auto &p = a[f], &q = b[f];
                if (!(p.buffer_index == q.buffer_index &&
                      p.data_format == q.data_format && p.page_size == q.page_size &&
                      p.tile == q.tile &&
                      p.face_geometry.has_value() == q.face_geometry.has_value())) {
                    return false;
                }
            }
            return true;
        }

        // Throws when the descriptors differ in anything generic_op reads.
        void check_same(
            const std::string& name,
            const ProgramDescriptor& a,
            const ProgramDescriptor& b
        ) {
            bool same =
                (a.kernels.size() == b.kernels.size() && a.cbs.size() == b.cbs.size() &&
                 a.semaphores.size() == b.semaphores.size());
            for (size_t k = 0; same && k < a.kernels.size(); ++k) {
                const auto &x = a.kernels[k], &y = b.kernels[k];
                same =
                    (x.kernel_source == y.kernel_source &&
                     x.source_type == y.source_type && x.core_ranges == y.core_ranges &&
                     x.compile_time_args == y.compile_time_args &&
                     x.named_compile_time_args == y.named_compile_time_args &&
                     x.defines == y.defines && x.runtime_args == y.runtime_args &&
                     x.common_runtime_args == y.common_runtime_args &&
                     x.opt_level == y.opt_level &&
                     x.compiler_include_paths == y.compiler_include_paths &&
                     same_config(x.config, y.config));
            }
            for (size_t i = 0; same && i < a.cbs.size(); ++i) {
                const auto &x = a.cbs[i], &y = b.cbs[i];
                same =
                    (x.total_size == y.total_size && x.core_ranges == y.core_ranges &&
                     same_formats(x.format_descriptors, y.format_descriptors) &&
                     same_formats(
                         x.remote_format_descriptors, y.remote_format_descriptors
                     ) &&
                     (x.buffer == nullptr) == (y.buffer == nullptr) &&
                     (x.tensor == nullptr) == (y.tensor == nullptr) &&
                     x.address_offset == y.address_offset);
            }
            for (size_t i = 0; same && i < a.semaphores.size(); ++i) {
                const auto &x = a.semaphores[i], &y = b.semaphores[i];
                same =
                    (x.id == y.id && x.core_type == y.core_type &&
                     x.core_ranges == y.core_ranges &&
                     x.initial_value == y.initial_value);
            }
            if (!same) {
                throw std::logic_error(
                    "KTT_DESC_CACHE=verify: the cached descriptor of " + name +
                    " differs from a fresh build"
                );
            }
        }

        // TT-Metal's program-cache entries plus the programs launch() holds.
        int64_t program_entries() {
            return static_cast<int64_t>(detail::device()->num_program_cache_entries()) +
                own_program_count();
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
            c.program_entries = program_entries();
            return out;
        }

        // True when generic_op's cache-hit path does nothing for `mesh` beyond copying
        // its runtime arguments: one program, no circular buffer on a tensor's memory.
        bool direct_ok(const MeshProgramDescriptor& mesh) {
            if (mesh.mesh_programs.size() != 1) {
                return false;
            }
            for (const auto& cb : mesh.mesh_programs.front().second.cbs) {
                if (cb.buffer || cb.tensor || cb.global_circular_buffer) {
                    return false;
                }
            }
            return true;
        }

        // Writes io's buffer addresses at `slots` and `values` at their arguments into
        // the program of `entry` (the live argument storage, as TT-Metal's own
        // patching).
        void write_args(
            Cached& entry,
            const std::vector<ttnn::Tensor>& io,
            const std::vector<Value>& values
        ) {
            using tt::tt_metal::GetCommonRuntimeArgs;
            using tt::tt_metal::GetRuntimeArgs;
            auto& program = entry.workload->get_programs().begin()->second;
            const auto& desc = entry.mesh.mesh_programs.front().second;
            for (const auto& s : entry.slots) {
                const uint32_t addr = io.at(s.tensor).buffer()->address();
                if (s.entry == Slot::kCommon) {
                    GetCommonRuntimeArgs(program, s.kernel).at(s.position) = addr;
                    continue;
                }
                auto& table = GetRuntimeArgs(program, s.kernel);
                const auto& cores = desc.kernels[s.kernel].runtime_args;
                if (s.entry == Slot::kEveryCore) {
                    for (const auto& [core, args] : cores) {
                        table[core.x][core.y].at(s.position) = addr;
                    }
                } else {
                    const auto& core = cores.at(s.entry).first;
                    table[core.x][core.y].at(s.position) = addr;
                }
            }
            for (const auto& v : values) {
                if (v.entry == Value::kCommon) {
                    GetCommonRuntimeArgs(program, v.kernel).at(v.position) = v.value;
                } else {
                    const auto&
                        core = desc.kernels[v.kernel].runtime_args.at(v.entry).first;
                    GetRuntimeArgs(program, v.kernel)[core.x][core.y].at(
                        v.position
                    ) = v.value;
                }
            }
        }

        // The program of `entry` enqueued over io, as generic_op launches it on a
        // program-cache hit without re-applying the whole descriptor; time counted.
        ttnn::Tensor enqueue_direct(
            const std::vector<ttnn::Tensor>& io, Cached& entry
        ) {
            namespace op = ttnn::device_operation::detail;
            using ttnn::operations::generic::GenericOpDeviceOperation;
            auto& c = detail::counters();
            c.desc_ns += lap();
            auto* mesh = io.front().device();
            for (const auto& t : io) {
                if (!t.is_allocated()) {
                    throw std::runtime_error("own program: a tensor is not allocated");
                }
            }
            const ttnn::operations::generic::tensor_args_t args{io, io.back()};
            const std::vector<std::reference_wrapper<const ttnn::Tensor>> inputs(
                io.begin(), io.end()
            );
            ttnn::Tensor out = io.back();
            op::place_output_tensors<GenericOpDeviceOperation>(
                entry.mesh, args, inputs, mesh, out
            );
            auto& workload = *entry.workload;
            const auto id = ttnn::CoreIDs::instance()
                                .fetch_and_increment_device_operation_id();
            for (auto& [range, program] : workload.get_programs()) {
                program.set_runtime_id(id);
            }
            namespace inspector = tt::tt_metal::experimental::inspector;
            if (inspector::IsEnabled()) {
                inspector::EmitMeshWorkloadDebugEntry(
                    workload,
                    id,
                    "ktt::program",
                    {},
                    inspector::GetCurrentMeshTraceId(mesh)
                );
            }
            if (!ttnn::device_operation::mesh_device_operation_utils::track_workload(
                    workload, mesh
                )) {
                tt::tt_metal::distributed::EnqueueMeshWorkload(
                    mesh->mesh_command_queue(), workload, false
                );
                ttnn::graph::track_mesh_workload_execution(workload, mesh, id);
            }
            c.enqueue_ns += lap();
            c.programs++;
            c.program_entries = program_entries();
            return out;
        }

        // `entry`'s program built from its descriptor for io's device.
        void build_workload(Cached& entry, const std::vector<ttnn::Tensor>& io) {
            auto& [range, desc] = entry.mesh.mesh_programs.front();
            auto workload = std::make_shared<Workload>();
            workload->add_program(range, tt::tt_metal::Program(desc));
            entry.workload = std::move(workload);
            entry.device = io.front().device();
        }

    }  // namespace

    namespace {

        // generic_op's program hash with the program key folded in. generic_op's own
        // hash counts the cores with runtime arguments but not each core's argument
        // count, so two programs differing only in per-core argument length would share
        // one TT-Metal program-cache entry and run with shifted arguments
        // (tt-metal#60938).
        void seal_hash(
            const Key& key,
            const std::vector<ttnn::Tensor>& io,
            MeshProgramDescriptor& mesh
        ) {
            const ttnn::operations::generic::tensor_args_t args{io, io.back()};
            auto& program = mesh.mesh_programs.front().second;
            const size_t
                structural = (ttnn::operations::generic::GenericOpDeviceOperation::
                                  compute_program_hash(mesh, args));
            program.custom_program_hash =
                (structural ^
                 (std::hash<std::string>{}(key.str()) + 0x9e3779b97f4a7c15ull +
                  (structural << 6) + (structural >> 2)));
        }

    }  // namespace

    namespace {

        // Checks every own program's descriptor against the host conventions; throws
        // std::logic_error naming the program and the rule.
        // - Circular buffers: a positive size that is a whole number of pages.
        // - Runtime arguments: at most max_runtime_args per core and in common.
        // - Semaphore ids are distinct.
        // - Two data-movement kernels on the same core (RISCV_0 and RISCV_1) use
        //   different NoCs.
        // - Each kernel of `built.args` gets the stated runtime argument counts.
        // - For each of `built.rings`: the circular buffer's page count is a multiple
        // of
        //   every push, and no kernel writes past its end.
        const ProgramDescriptor& validated(const Key& key, const Built& built) {
            using namespace tt::tt_metal;
            const ProgramDescriptor& desc = built.desc;
            const auto fail = [&](const std::string& what) {
                // The key starts with the program's name, then a separator byte.
                std::string name;
                for (const char c : key.str()) {
                    if (static_cast<unsigned char>(c) < 0x20 || name.size() >= 48) {
                        break;
                    }
                    name += c;
                }
                throw std::logic_error("own program " + name + ": " + what);
            };
            for (const auto& count : built.args) {
                if (count.kernel >= desc.kernels.size()) {
                    fail(
                        "argument counts given for kernel " +
                        std::to_string(count.kernel)
                    );
                }
                const auto& kernel = desc.kernels[count.kernel];
                bool ok = kernel.common_runtime_args.size() == count.common;
                size_t seen = count.per_core;
                for (const auto& [core, args] : kernel.runtime_args) {
                    const bool fits =
                        (count.per_item == 0 ? args.size() == count.per_core
                                             : args.size() >= count.per_core &&
                                 (args.size() - count.per_core) % count.per_item == 0);
                    if (!fits) {
                        ok = false;
                        seen = args.size();
                    }
                }
                if (!ok) {
                    fail(
                        kernel.kernel_source + " reads " +
                        std::to_string(count.per_core) +
                        (count.per_item ? " + n x " + std::to_string(count.per_item)
                                        : "") +
                        " per core and " + std::to_string(count.common) +
                        " common runtime arguments, the host passes " +
                        std::to_string(seen) + " and " +
                        std::to_string(kernel.common_runtime_args.size())
                    );
                }
            }
            for (const auto& ring : built.rings) {
                bool found = false;
                for (size_t i = 0; i < desc.cbs.size(); ++i) {
                    if (ring.at >= 0 && static_cast<size_t>(ring.at) != i) {
                        continue;
                    }
                    const auto& cb = desc.cbs[i];
                    for (const auto& f : cb.format_descriptors) {
                        if (f.buffer_index != ring.cb || f.page_size == 0) {
                            continue;
                        }
                        found = true;
                        const uint64_t pages = cb.total_size / f.page_size;
                        for (const uint32_t push : ring.pushes) {
                            if (push == 0 || pages % push != 0) {
                                fail(
                                    "circular buffer " + std::to_string(ring.cb) +
                                    " of " + std::to_string(pages) +
                                    " pages takes pushes of " + std::to_string(push)
                                );
                            }
                        }
                        if (ring.written > cb.total_size) {
                            fail(
                                "circular buffer " + std::to_string(ring.cb) +
                                " holds " + std::to_string(cb.total_size) +
                                " B, a kernel writes " + std::to_string(ring.written)
                            );
                        }
                    }
                }
                if (!found) {
                    fail("no circular buffer " + std::to_string(ring.cb));
                }
            }
            for (const auto& cb : desc.cbs) {
                for (const auto& f : cb.format_descriptors) {
                    if (cb.total_size == 0 || f.page_size == 0 ||
                        cb.total_size % f.page_size != 0) {
                        fail(
                            "circular buffer " + std::to_string(f.buffer_index) +
                            " of " + std::to_string(cb.total_size) +
                            " B is not a whole number of " +
                            std::to_string(f.page_size) + " B pages"
                        );
                    }
                }
            }
            for (const auto& kernel : desc.kernels) {
                bool over = kernel.common_runtime_args.size() > max_runtime_args;
                for (const auto& [core, args] : kernel.runtime_args) {
                    over = over || args.size() > max_runtime_args;
                }
                if (over) {
                    fail(
                        kernel.kernel_source +
                        " takes more than max_runtime_args arguments"
                    );
                }
            }
            for (size_t i = 0; i < desc.semaphores.size(); ++i) {
                for (size_t j = i + 1; j < desc.semaphores.size(); ++j) {
                    if (desc.semaphores[i].id == desc.semaphores[j].id &&
                        desc.semaphores[i].core_ranges.intersects(
                            desc.semaphores[j].core_ranges
                        )) {
                        fail(
                            "semaphore id " + std::to_string(desc.semaphores[i].id) +
                            " is used twice"
                        );
                    }
                }
            }
            for (size_t i = 0; i < desc.kernels.size(); ++i) {
                const auto* a = std::get_if<DataMovementConfigDescriptor>(
                    &desc.kernels[i].config
                );
                for (size_t j = i + 1; a && j < desc.kernels.size(); ++j) {
                    const auto* b = std::get_if<DataMovementConfigDescriptor>(
                        &desc.kernels[j].config
                    );
                    if (b && a->processor != b->processor && a->noc == b->noc &&
                        desc.kernels[i].core_ranges.intersects(
                            desc.kernels[j].core_ranges
                        )) {
                        fail(
                            desc.kernels[i].kernel_source + " and " +
                            desc.kernels[j].kernel_source + " share a NoC on one core"
                        );
                    }
                }
            }
            return desc;
        }

    }  // namespace

    ttnn::Tensor launch(
        const Key& key,
        const std::vector<ttnn::Tensor>& io,
        const std::function<Built()>& build,
        bool always_cached,
        const std::vector<Value>& values
    ) {
        if (!always_cached && !desc_cache_enabled()) {
            const auto built = build();
            auto mesh = over_mesh(io, validated(key, built));
            seal_hash(key, io, mesh);
            return enqueue(io, mesh);
        }
        const std::lock_guard lock(desc_cache_mutex());
        auto& cache = desc_cache();
        auto it = cache.find(key.str());
        if (it == cache.end()) {
            if (cache.size() >= kMaxCached) {
                for (auto& [name, e] : cache) {
                    if (e.workload) {
                        retired_workloads().push_back(std::move(e.workload));
                    }
                }
                cache.clear();
            }
            auto built = build();
            validated(key, built);
            Cached entry{over_mesh(io, std::move(built.desc)), std::move(built.slots)};
            seal_hash(key, io, entry.mesh);
            if (!direct_ok(entry.mesh)) {
                auto out = enqueue(io, entry.mesh);
                cache.emplace(key.str(), std::move(entry));
                return out;
            }
            // Cached once its first launch (the compile) succeeded.
            build_workload(entry, io);
            auto out = enqueue_direct(io, entry);
            // The program counts from its first launch on, as in TT-Metal's program cache.
            own_program_count()++;
            detail::counters().program_entries = program_entries();
            cache.emplace(key.str(), std::move(entry));
            return out;
        }
        auto& entry = it->second;
        detail::counters().desc_hits++;
        const bool direct = static_cast<bool>(entry.workload);
        if (!direct || verify_desc_cache()) {
            auto& desc = entry.mesh.mesh_programs.front().second;
            patch(desc, entry.slots, io);
            for (const auto& v : values) {
                auto& kernel = desc.kernels[v.kernel];
                if (v.entry == Value::kCommon) {
                    kernel.common_runtime_args.at(v.position) = v.value;
                } else {
                    kernel.runtime_args.at(v.entry).second.at(v.position) = v.value;
                }
            }
            if (verify_desc_cache()) {
                const auto phase = detail::phase_clock();
                check_same(key.str().c_str(), desc, build().desc);
                detail::phase_clock() = phase;
            }
        }
        if (!direct) {
            return enqueue(io, entry.mesh);
        }
        if (entry.device != io.front().device()) {
            retired_workloads().push_back(std::move(entry.workload));
            build_workload(entry, io);
            own_program_count()++;
        }
        write_args(entry, io, values);
        return enqueue_direct(io, entry);
    }

    ttnn::Tensor launch_fresh(
        const Key& key,
        const std::vector<ttnn::Tensor>& io,
        const std::function<Built()>& build
    ) {
        const auto built = build();
        auto mesh = over_mesh(io, validated(key, built));
        seal_hash(key, io, mesh);
        return enqueue(io, mesh);
    }

    ttnn::Tensor launch(
        const std::vector<ttnn::Tensor>& io, const tt::tt_metal::ProgramDescriptor& desc
    ) {
        return enqueue(io, over_mesh(io, desc));
    }

    void clear_desc_cache() {
        const std::lock_guard lock(desc_cache_mutex());
        desc_cache().clear();
        retired_workloads().clear();
        own_program_count() = 0;
        panels_cache().reset();
    }

    std::vector<HandlePtr> run_on_stream(
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        std::function<void(const std::vector<ttnn::Tensor>& tensors)> run,
        const std::vector<int64_t>& rows
    ) {
        std::vector<HandlePtr> outputs;
        for (size_t i = 0; i < out_shapes.size(); ++i) {
            outputs.push_back(
                detail::make_handle(detail::device_dims(out_shapes[i]), out_dtypes[i])
            );
        }
        detail::counters().ops++;
        detail::submit([inputs, outputs, rows, run = std::move(run)] {
            using tt::tt_metal::Layout;
            const auto start = std::chrono::steady_clock::now();
            detail::phase_clock() = start;
            auto is_rows = [&](size_t position) {
                return std::find(rows.begin(), rows.end(), position) != rows.end();
            };
            // Inputs, then outputs allocated per the activation policy.
            auto operands = [&] {
                std::vector<ttnn::Tensor> tensors;
                for (const auto& h : inputs) {
                    const bool row = is_rows(tensors.size());
                    tensors.push_back(
                        row ? ttnn::to_layout(h->tensor, Layout::ROW_MAJOR) : h->tensor
                    );
                }
                for (const auto& h : outputs) {
                    const bool row = is_rows(tensors.size());
                    h->tensor = (ttnn::empty(
                        detail::device_shape(h->shape),
                        detail::to_ttnn(h->dtype),
                        row ? Layout::ROW_MAJOR : Layout::TILE,
                        detail::device(),
                        detail::activation_config(h->bytes)
                    ));
                    tensors.push_back(h->tensor);
                }
                return tensors;
            };
            const auto tensors = operands();
            mark(Phase::Alloc);
            run(tensors);
            // Row-major outputs back to the TILE layout every other op reads.
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (is_rows(inputs.size() + i)) {
                    outputs[i]->tensor = ttnn::to_layout(
                        tensors[inputs.size() + i], Layout::TILE
                    );
                }
            }
            const auto elapsed = std::chrono::steady_clock::now() - start;
            using std::chrono::duration_cast, std::chrono::nanoseconds;
            detail::counters().op_ns += duration_cast<nanoseconds>(elapsed).count();
        });
        return outputs;
    }

}  // namespace ktt::bridge::program
