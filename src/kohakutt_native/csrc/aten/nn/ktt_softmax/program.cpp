// Host side of ktt_softmax: the split of tile rows and columns over the grid (plan),
// the circular buffers, semaphores and runtime args of the program.
#include "aten/nn/ktt_softmax/program.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ktt::bridge::ktt_softmax {
    namespace {

        using namespace tt::tt_metal;

        constexpr uint32_t kF32 = 4096;
        constexpr uint32_t kMaxGroup = 16;
        constexpr uint32_t kBlock = 4;  // output tiles per writer flush
        constexpr uint32_t kOut = 2 * kBlock;
        constexpr uint32_t kSentinel = 0x4b534d58u;

        // Runtime arguments the kernel sources read: common (sentinel included), per
        // core; the writer then reads kPeerArgs per core of the group.
        constexpr uint32_t kReaderCommon = 10, kComputeCommon = 6, kWriterCommon = 8;
        constexpr uint32_t kReaderArgs = 3, kComputeArgs = 4, kWriterArgs = 4;
        constexpr uint32_t kPeerArgs = 2;

        // Circular buffers.
        constexpr uint8_t kX = 0, kGather = 4, kM = 5, kR = 6, kE = 7, kY = 16,
                          kPart = 17;
        constexpr uint8_t kRowCompute = 24, kRowWriter = 25;
        constexpr uint32_t kRowPage = 16, kRowPages = 4;

        // Semaphore ids: the exchange's data and ready counts, the shared row counter.
        constexpr uint32_t kDataSem = 0, kReadySem = 1, kCounterSem = 2;

        // Reader NoC; the writer takes the other one (dedicated NoC mode).
        constexpr NOC kReaderNoc = NOC::NOC_0;

        uint32_t ceil_div(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        uint32_t tile_bytes(DataType t) {
            return t == DataType::FLOAT32 ? 4096 : 2048;
        }

        tt::DataFormat format_of(DataType t) {
            return (
                t == DataType::FLOAT32 ? tt::DataFormat::Float32
                                       : tt::DataFormat::Float16_b
            );
        }

        std::pair<DataMovementConfigDescriptor, DataMovementConfigDescriptor> movers() {
            const NOC writer = kReaderNoc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            return {
                DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = kReaderNoc,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                },
                DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = writer,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                }
            };
        }

        // Circular-buffer bytes per core: x rows, e, y ring, m and r tiles, row
        // indices; gather and partials when g > 1.
        uint32_t l1_bytes(
            uint32_t seg, uint32_t depth, uint32_t group, uint32_t x_tile
        ) {
            uint32_t bytes = depth * seg * x_tile + seg * kF32 + kOut * x_tile;
            bytes += 2 * kF32 + 2 * kRowPages * kRowPage;
            if (group > 1) {
                bytes += 6 * kF32;
            }
            return bytes;
        }

        struct Shape {
            uint32_t rows = 0;  // tile rows (leading dims flattened)
            uint32_t wt = 0, wlast = 0, mt = 0, hlast = 0;
        };

        Shape shape_of(const ttnn::Tensor& x) {
            const auto& s = x.logical_shape();
            Shape r;
            uint32_t slices = 1;
            for (uint32_t i = 0; i + 2 < s.rank(); ++i) {
                slices *= s[i];
            }
            r.wt = ceil_div(s[-1], 32);
            r.wlast = s[-1] - (r.wt - 1) * 32;
            r.mt = ceil_div(s[-2], 32);
            r.hlast = s[-2] - (r.mt - 1) * 32;
            r.rows = slices * r.mt;
            return r;
        }

        CoreCoord logical(uint32_t c) {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            return CoreCoord(c % grid.x, c / grid.x);
        }

        void add_cb(
            ProgramDescriptor& desc,
            const CoreRangeSet& range,
            uint8_t index,
            uint32_t pages,
            tt::DataFormat format,
            uint32_t page
        ) {
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = pages * page,
                    .core_ranges = range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index, .data_format = format, .page_size = page
                    }}
                }
            );
        }

        void add_tiles(
            ProgramDescriptor& desc,
            const CoreRangeSet& range,
            uint8_t index,
            uint32_t tiles,
            tt::DataFormat format
        ) {
            const uint32_t page = format == tt::DataFormat::Float32 ? 4096 : 2048;
            add_cb(desc, range, index, tiles, format, page);
        }

        // Compute config: fp32 DST, half-sync; fp32 circular buffers unpack to DST
        // unrounded.
        ComputeConfigDescriptor compute_config(
            const ProgramDescriptor& desc, MathFidelity fidelity
        ) {
            ComputeConfigDescriptor c{
                .math_fidelity = fidelity,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = ComputeConfigDescriptor::UnpackToDestModes(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                ),
                .math_approx_mode = false
            };
            for (const auto& cb : desc.cbs) {
                const auto& f = cb.format_descriptors.front();
                if (f.data_format == tt::DataFormat::Float32) {
                    c.unpack_to_dest_mode
                        [f.buffer_index] = (UnpackToDestMode::UnpackToDestFp32);
                }
            }
            return c;
        }

        // The sentinel appended to every kernel's common args, their count and the per
        // core arg count to its compile-time args.
        void seal(ProgramDescriptor& desc) {
            for (auto& k : desc.kernels) {
                k.common_runtime_args.push_back(kSentinel);
                k.compile_time_args.push_back(
                    static_cast<uint32_t>(k.common_runtime_args.size())
                );
                const size_t per_core =
                    (k.runtime_args.empty() ? 0 : k.runtime_args.front().second.size());
                for (const auto& [core, args] : k.runtime_args) {
                    if (args.size() != per_core) {
                        throw std::runtime_error(
                            "ktt_softmax: per-core argument counts differ"
                        );
                    }
                }
                k.compile_time_args.push_back(static_cast<uint32_t>(per_core));
            }
        }

        // Tiles on the busiest core with groups of g cores (an exchange counts two); 0
        // when g cores do not split wt evenly into segments or exceed the grid.
        uint64_t group_cost(uint32_t rows, uint32_t wt, uint32_t g, uint32_t cores) {
            const uint32_t seg = ceil_div(wt, g);
            if (ceil_div(wt, seg) != g || g > cores) {
                return 0;
            }
            const uint32_t groups = std::min(rows, cores / g);
            return uint64_t{ceil_div(rows, groups)} * (seg + (g > 1 ? 2 : 0));
        }

        // Dynamic rows (each core takes its next tile row from the shared counter) for
        // single-core groups with more tile rows than groups.
        bool dynamic_rows(const Plan& p, uint32_t rows) {
            return p.group == 1 && rows > p.groups;
        }

    }  // namespace

    std::string Plan::str() const {
        char buf[120];
        std::snprintf(
            buf,
            sizeof(buf),
            "group %u seg %u groups %u depth %u l1 %u",
            group,
            seg,
            groups,
            row_depth,
            l1
        );
        return buf;
    }

    std::optional<Plan> plan(uint32_t rows, uint32_t wt, uint32_t x_tile) {
        static std::unordered_map<std::string, std::optional<Plan>> cache;
        const std::string id =
            (std::to_string(rows) + "," + std::to_string(wt) + "," +
             std::to_string(x_tile));
        const uint32_t budget = program::cb_budget();
        // A cached plan is reused while its circular buffers fit the current L1 gap.
        if (const auto hit = cache.find(id);
            hit != cache.end() && hit->second && hit->second->l1 <= budget) {
            return hit->second;
        }
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t cores = grid.x * grid.y;
        // Among the group sizes whose segment fits (two resident rows, else one), the
        // fewest tiles on the busiest core.
        std::optional<Plan> best;
        for (uint32_t depth : {2u, 1u}) {
            uint64_t best_cost = ~uint64_t{0};
            for (uint32_t g = 1; g <= std::min(kMaxGroup, wt); ++g) {
                const uint64_t cost = group_cost(rows, wt, g, cores);
                const uint32_t seg = ceil_div(wt, g);
                const uint32_t l1 = l1_bytes(seg, depth, g, x_tile);
                if (cost == 0 || l1 > budget) {
                    continue;
                }
                const uint32_t groups = std::min(rows, cores / g);
                if (cost < best_cost) {
                    best_cost = cost;
                    best = Plan{g, seg, groups, depth, l1};
                }
            }
            if (best) {
                break;
            }
        }
        cache.insert_or_assign(id, best);
        return best;
    }

    bool supports(const ttnn::Tensor& x) {
        return (
            (x.dtype() == DataType::BFLOAT16 || x.dtype() == DataType::FLOAT32) &&
            x.layout() == Layout::TILE && !x.memory_config().is_sharded() &&
            x.logical_shape().rank() >= 2 && x.logical_shape().volume() > 0
        );
    }

    ttnn::Tensor forward(const ttnn::Tensor& x, const MemoryConfig& memory) {
        const auto sh = shape_of(x);
        const uint32_t x_tile = tile_bytes(x.dtype());
        // The output first: an L1-resident y lowers the gap the circular buffers fit.
        auto y = (
            ttnn::empty(
                x.logical_shape(), x.dtype(), Layout::TILE, detail::device(), memory
            )
        );
        const auto p = plan(sh.rows, sh.wt, x_tile);
        if (!p) {
            throw std::runtime_error(
                "ktt_softmax: no row split fits the program L1 budget"
            );
        }
        static const bool trace = std::getenv("KTT_SOFTMAX_TRACE") != nullptr;
        if (trace) {
            std::fprintf(
                stderr,
                "ktt_softmax rows %u wt %u: %s\n",
                sh.rows,
                sh.wt,
                p->str().c_str()
            );
        }
        program::mark(program::Phase::Alloc);
        const bool dynamic = dynamic_rows(*p, sh.rows);
        program::Key key("ktt_softmax");
        key.add(x).add(y);
        key.add(p->group).add(p->seg).add(p->groups).add(p->row_depth).add(dynamic);
        return program::launch(key, {x, y}, [&] {
            program::Built built;
            auto& desc = built.desc;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t count = p->groups * p->group;
            const CoreRangeSet range = num_cores_to_corerangeset(count, grid, true);
            const auto f32 = tt::DataFormat::Float32;
            const auto xf = format_of(x.dtype());
            add_tiles(desc, range, kX, p->row_depth * p->seg, xf);
            add_tiles(desc, range, kM, 1, f32);
            add_tiles(desc, range, kR, 1, f32);
            add_tiles(desc, range, kE, p->seg, f32);
            add_tiles(desc, range, kY, kOut, xf);
            for (const uint8_t index : {kRowCompute, kRowWriter}) {
                add_cb(desc, range, index, kRowPages, tt::DataFormat::UInt32, kRowPage);
            }
            if (p->group > 1) {
                add_tiles(desc, range, kGather, 4, f32);
                add_tiles(desc, range, kPart, 2, f32);
            }
            for (uint32_t id = 0; id <= kCounterSem; ++id) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = range,
                        .initial_value = id == kCounterSem ? p->groups : 0
                    }
                );
            }
            const auto counter = (
                detail::device()->worker_core_from_logical_core(CoreCoord(0, 0))
            );

            KernelDescriptor::RuntimeArgs reader_args, compute_args, writer_args;
            for (uint32_t c = 0; c < count; ++c) {
                const uint32_t gid = c / p->group, index = c % p->group;
                const uint32_t first = index * p->seg;
                const uint32_t seg = std::min(p->seg, sh.wt - first);
                const uint32_t wlast = first + seg == sh.wt ? sh.wlast : 32;
                const CoreCoord core = logical(c);
                reader_args.push_back({core, {gid, first, seg}});
                compute_args.push_back({core, {gid, index, seg, wlast}});
                std::vector<uint32_t> w{gid, index, first, seg};
                for (uint32_t q = 0; q < p->group; ++q) {
                    const auto noc = (
                        detail::device()->worker_core_from_logical_core(
                            logical(gid * p->group + q)
                        )
                    );
                    w.push_back(static_cast<uint32_t>(noc.x));
                    w.push_back(static_cast<uint32_t>(noc.y));
                }
                writer_args.push_back({core, w});
            }
            program::mark(program::Phase::Args);
            const auto [reader_cfg, writer_cfg] = movers();

            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_softmax/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*x.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = reader_args,
                    .common_runtime_args =
                        {
                            x.buffer()->address(),
                            sh.wt,
                            sh.rows,
                            p->groups,
                            p->seg,
                            dynamic ? 1u : 0u,
                            static_cast<uint32_t>(counter.x),
                            static_cast<uint32_t>(counter.y),
                            kCounterSem
                        },
                    .config = reader_cfg
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_softmax/compute.cpp"),
                    .core_ranges = range,
                    .runtime_args = compute_args,
                    .common_runtime_args = {sh.rows, p->group, p->seg, sh.mt, sh.hlast},
                    .config = compute_config(desc, MathFidelity::HiFi4)
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_softmax/writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*y.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = writer_args,
                    .common_runtime_args =
                        {
                            y.buffer()->address(),
                            sh.wt,
                            sh.rows,
                            p->group,
                            kDataSem,
                            kReadySem,
                            kBlock
                        },
                    .config = writer_cfg
                }
            );
            built.slots = {
                {
                    .kernel = 0,
                    .entry = program::Slot::kCommon,
                    .position = 0,
                    .tensor = 0
                },
                {
                    .kernel = 2,
                    .entry = program::Slot::kCommon,
                    .position = 0,
                    .tensor = 1
                },
            };
            seal(desc);
            built.args = {
                {.kernel = 0, .per_core = kReaderArgs, .common = kReaderCommon},
                {.kernel = 1, .per_core = kComputeArgs, .common = kComputeCommon},
                {
                    .kernel = 2,
                    .per_core = kWriterArgs + kPeerArgs * p->group,
                    .common = kWriterCommon
                },
            };
            built.rings = {
                {kX, {p->seg}, 0},
                {kE, {p->seg}, 0},
                {kM, {1}, 0},
                {kR, {1}, 0},
                {kY, {1}, 0},
                {kRowCompute, {1}, 0},
                {kRowWriter, {1}, 0},
            };
            if (p->group > 1) {
                // Gather: four tiles zeroed and filled by the peers from its base,
                // pushed two at a time.
                built.rings.push_back({kGather, {2}, 4 * kF32});
                built.rings.push_back({kPart, {2}, 0});
            }
            return built;
        });
    }

}  // namespace ktt::bridge::ktt_softmax
