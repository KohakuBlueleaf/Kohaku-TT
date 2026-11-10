// Host side of ktt_norm: the split of tile rows and columns over the grid (plan), the
// circular buffers, semaphores and runtime args of the forward, backward and column-sum
// programs.
#include "aten/nn/ktt_norm/program.hpp"

#include <algorithm>
#include <bit>
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

namespace ktt::bridge::ktt_norm {
    namespace {

        using namespace tt::tt_metal;

        constexpr uint32_t kF32 = 4096;
        constexpr uint32_t kMaxGroup = 16;
        constexpr uint32_t kBlock = 4;  // output tiles per writer flush

        // Output ring tiles: a whole segment (at least two flushes).
        uint32_t out_tiles(uint32_t seg) {
            return std::max(2 * kBlock, seg + (seg & 1));
        }

        constexpr uint32_t kSentinel = 0x4b4e4f52u;

        // Runtime arguments the kernel sources read. Common: the count each kernel
        // checks (args_ok, the sentinel included). Per core: the fixed ones; the
        // forward's and backward's writers then read kPeerArgs per core of the group.
        constexpr uint32_t kFwdReaderCommon = 14, kFwdComputeCommon = 11,
                           kFwdWriterCommon = 12;
        constexpr uint32_t kFwdReaderArgs = 3, kFwdComputeArgs = 6, kFwdWriterArgs = 4;
        constexpr uint32_t kBwdReaderCommon = 12, kBwdComputeCommon = 13,
                           kBwdWriterCommon = 14;
        constexpr uint32_t kBwdReaderArgs = 4, kBwdComputeArgs = 5, kBwdWriterArgs = 5;
        constexpr uint32_t kSumReaderCommon = 5, kSumComputeCommon = 4,
                           kSumWriterCommon = 4;
        constexpr uint32_t kSumArgs = 1, kPeerArgs = 2;

        // Circular buffer `cb` taking pushes of `pages` that recur within a launch.
        program::Ring pushed(uint32_t cb, uint32_t pages) {
            return {cb, {pages}, 0};
        }

        // Circular buffer `cb` written from its base, at most `bytes`.
        program::Ring filled(uint32_t cb, uint64_t bytes) {
            return {cb, {}, bytes};
        }

        // Reader NoC; the writer of the same core takes the other one (dedicated NoC
        // mode).
        constexpr NOC kReaderNoc = NOC::NOC_0;

        uint32_t ceil_div(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        uint32_t tile_bytes(DataType t) {
            return t == DataType::FLOAT32 ? 4096 : 2048;
        }

        tt::DataFormat format_of(DataType t) {
            return t == DataType::FLOAT32 ? tt::DataFormat::Float32
                                          : tt::DataFormat::Float16_b;
        }

        uint32_t bits(float f) {
            return std::bit_cast<uint32_t>(f);
        }

        // Reader and writer configs of one core: different NoCs (two RISCs on one NoC
        // share its command buffers and barrier counters).
        std::pair<DataMovementConfigDescriptor, DataMovementConfigDescriptor> movers() {
            const NOC writer = kReaderNoc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
            if (writer == kReaderNoc) {
                throw std::logic_error("ktt_norm: reader and writer on one NoC");
            }
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

        // Circular-buffer bytes per core for `seg` column tiles and `depth` resident
        // rows.
        uint32_t l1_bytes(
            uint32_t seg,
            uint32_t depth,
            uint32_t group,
            uint32_t x_tile,
            uint32_t g_tile,
            uint32_t params,
            bool backward
        ) {
            uint32_t bytes = 0;
            if (backward) {
                // x, dy rows; gamma; mean / rstd; S, S2; g, xh rows; A / B; dx; dgamma,
                // dbeta sums; gather, partials.
                bytes += 2 * depth * seg * x_tile + seg * g_tile * params;
                bytes += kF32 * (2 * depth + 2 + 2 * seg + 2 + 2 * seg);
                bytes += out_tiles(seg) * x_tile;
                if (group > 1) {
                    bytes += kF32 * 6;
                }
            } else {
                // x rows; gamma / beta; P; S, statistics, statistics out; y; row
                // indices; gather, count, partials, n.
                bytes += depth * seg * x_tile + seg * g_tile * params;
                bytes += kF32 * (ceil_div(seg, 2) + 1 + 2 + 2);
                bytes += out_tiles(seg) * x_tile + 2 * 64;
                if (group > 1) {
                    bytes += kF32 * 8;
                }
            }
            return bytes;
        }

        struct Shape {
            uint32_t rows = 0;  // tile rows (leading dims flattened)
            uint32_t wt = 0, w = 0, wlast = 0;
            ttsl::SmallVector<uint32_t> stat_dims;
        };

        Shape shape_of(const ttnn::Tensor& x) {
            const auto& s = x.logical_shape();
            Shape r;
            const uint32_t rank = s.rank();
            uint32_t slices = 1;
            for (uint32_t i = 0; i + 2 < rank; ++i) {
                slices *= s[i];
                r.stat_dims.push_back(s[i]);
            }
            r.stat_dims.push_back(s[-2]);
            r.stat_dims.push_back(1);
            r.w = s[-1];
            r.wt = ceil_div(r.w, 32);
            r.wlast = r.w - (r.wt - 1) * 32;
            r.rows = slices * ceil_div(s[-2], 32);
            return r;
        }

        // Cores of plan `p` in row-major order, core c at (c % grid x, c / grid x).
        CoreRangeSet cores_of(const Plan& p, uint32_t& count) {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            count = p.groups * p.group;
            return num_cores_to_corerangeset(count, grid, true);
        }

        CoreCoord logical(uint32_t c) {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            return CoreCoord(c % grid.x, c / grid.x);
        }

        void add_cb(
            ProgramDescriptor& desc,
            const CoreRangeSet& range,
            uint8_t index,
            uint32_t tiles,
            tt::DataFormat format
        ) {
            const uint32_t page = format == tt::DataFormat::Float32 ? 4096 : 2048;
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = tiles * page,
                    .core_ranges = range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index, .data_format = format, .page_size = page
                    }}
                }
            );
        }

        void append_accessor(std::vector<uint32_t>& ct, const ttnn::Tensor& t) {
            const auto more = TensorAccessorArgs(*t.buffer()).get_compile_time_args();
            ct.insert(ct.end(), more.begin(), more.end());
        }

        // Compute config: fp32 DST, half-sync, fidelity and SFPU approximations from
        // numerics; fp32 circular buffers unpack to DST unrounded.
        ComputeConfigDescriptor compute_config(
            const ProgramDescriptor& desc, MathFidelity fidelity, bool approx
        ) {
            ComputeConfigDescriptor c{
                .math_fidelity = fidelity,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = ComputeConfigDescriptor::UnpackToDestModes(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                ),
                .math_approx_mode = approx
            };
            for (const auto& cb : desc.cbs) {
                const auto& f = cb.format_descriptors.front();
                if (f.data_format == tt::DataFormat::Float32) {
                    c.unpack_to_dest_mode[f.buffer_index] = UnpackToDestMode::
                        UnpackToDestFp32;
                }
            }
            return c;
        }

        // Appends the sentinel to every kernel's common runtime args and their count to
        // its compile-time args (checked by the kernels before reading any arg), then
        // the largest per-core runtime arg count (a program cache key; tt-metal#60938).
        void seal(ProgramDescriptor& desc) {
            for (auto& k : desc.kernels) {
                k.common_runtime_args.push_back(kSentinel);
                k.compile_time_args.push_back(
                    static_cast<uint32_t>(k.common_runtime_args.size())
                );
                size_t per_core = k.runtime_args.empty()
                    ? 0
                    : k.runtime_args.front().second.size();
                for (const auto& [core, args] : k.runtime_args) {
                    if (args.size() != per_core) {
                        throw std::runtime_error(
                            "ktt_norm: per-core argument counts differ"
                        );
                    }
                }
                k.compile_time_args.push_back(static_cast<uint32_t>(per_core));
            }
        }

        // One core's place in its group: the group, its index, its first column tile
        // and its column tiles.
        struct Seat {
            uint32_t gid, index, first, seg;
        };

        Seat seat(const Plan& p, const Shape& sh, uint32_t c) {
            const uint32_t gid = c / p.group, index = c % p.group;
            const uint32_t first = index * p.seg;
            return {gid, index, first, std::min(p.seg, sh.wt - first)};
        }

        // Semaphore ids: the exchange's data and ready counts, and the shared row
        // counter.
        constexpr uint32_t kCounterSem = 2;
        // Row-index CBs (reader to compute, reader to writer): 16-byte pages.
        constexpr uint8_t kRowCompute = 24, kRowWriter = 25;
        constexpr uint32_t kRowPage = 16, kRowPages = 4;

        // Dynamic rows (each core takes its next tile row from the shared counter) for
        // single-core groups with more tile rows than groups; KTT_NORM_DYNAMIC=0 turns
        // them off.
        bool dynamic_rows(const Plan& p, uint32_t rows) {
            static const bool on = [] {
                const char* v = std::getenv("KTT_NORM_DYNAMIC");
                return v == nullptr || std::atoi(v) != 0;
            }();
            return on && p.group == 1 && rows > p.groups;
        }

        void add_row_cbs(ProgramDescriptor& desc, const CoreRangeSet& range) {
            for (const uint8_t index : {kRowCompute, kRowWriter}) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = kRowPages * kRowPage,
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index,
                            .data_format = tt::DataFormat::UInt32,
                            .page_size = kRowPage
                        }}
                    }
                );
            }
        }

        // The exchange semaphores (0) and the row counter (starting after the groups'
        // first rows).
        void add_semaphores(
            ProgramDescriptor& desc, const CoreRangeSet& range, uint32_t groups
        ) {
            for (uint32_t id = 0; id <= kCounterSem; ++id) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = range,
                        .initial_value = id == kCounterSem ? groups : 0
                    }
                );
            }
        }

        // NoC coordinates of the core holding the row counter (logical core 0).
        std::pair<uint32_t, uint32_t> row_counter() {
            const auto noc = detail::device()->worker_core_from_logical_core(
                CoreCoord(0, 0)
            );
            return {static_cast<uint32_t>(noc.x), static_cast<uint32_t>(noc.y)};
        }

        // The NoC coordinates of the group's cores, as (x, y) pairs appended to `args`.
        void append_peers(std::vector<uint32_t>& args, const Plan& p, uint32_t gid) {
            const auto& mesh = detail::device();
            for (uint32_t q = 0; q < p.group; ++q) {
                const auto noc = mesh->worker_core_from_logical_core(
                    logical(gid * p.group + q)
                );
                args.push_back(static_cast<uint32_t>(noc.x));
                args.push_back(static_cast<uint32_t>(noc.y));
            }
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

    std::optional<Plan> plan(
        uint32_t rows,
        uint32_t wt,
        uint32_t x_tile,
        uint32_t g_tile,
        uint32_t params,
        bool backward
    ) {
        static std::unordered_map<std::string, std::optional<Plan>> cache;
        const std::string id =
            (std::to_string(rows) + "," + std::to_string(wt) + "," +
             std::to_string(x_tile) + "," + std::to_string(g_tile) + "," +
             std::to_string(params) + "," + std::to_string(backward));
        const uint32_t budget = program::cb_budget();
        // A cached plan is reused while its circular buffers fit the current L1 gap.
        if (const auto hit = cache.find(id);
            hit != cache.end() && hit->second && hit->second->l1 <= budget) {
            return hit->second;
        }
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t cores = grid.x * grid.y;
        std::optional<Plan> best;
        // Among the group sizes whose segment fits (two resident rows, else one), the
        // one with the fewest tiles on the busiest core: ceil(rows / groups) rows of
        // `seg` tiles, plus kExchange tiles' worth per row exchange when g > 1.
        constexpr uint32_t kExchange = 2;
        static const uint32_t force_depth = [] {
            const char* v = std::getenv("KTT_NORM_DEPTH");
            return v ? static_cast<uint32_t>(std::atoi(v)) : 0u;
        }();
        static const uint32_t force_group = [] {
            const char* v = std::getenv("KTT_NORM_GROUP");
            return v ? static_cast<uint32_t>(std::atoi(v)) : 0u;
        }();
        for (uint32_t depth : {2u, 1u}) {
            if (force_depth && depth != force_depth) {
                continue;
            }
            uint64_t best_cost = ~uint64_t{0};
            for (uint32_t g = 1; g <= std::min(kMaxGroup, wt); ++g) {
                if (force_group && g != force_group) {
                    continue;
                }
                const uint32_t seg = ceil_div(wt, g);
                const uint32_t group = ceil_div(wt, seg);
                if (group != g) {
                    continue;
                }
                const uint32_t l1 = l1_bytes(
                    seg, depth, group, x_tile, g_tile, params, backward
                );
                if (l1 > budget || group > cores) {
                    continue;
                }
                const uint32_t groups = std::min(rows, cores / group);
                const uint64_t cost = uint64_t{ceil_div(rows, groups)} *
                    (seg + (group > 1 ? kExchange : 0));
                if (cost < best_cost) {
                    best_cost = cost;
                    best = Plan{group, seg, groups, depth, l1};
                }
            }
            if (best) {
                break;
            }
        }
        cache.insert_or_assign(id, best);
        return best;
    }

    bool supports(
        const ttnn::Tensor& x, const ttnn::Tensor* gamma, const ttnn::Tensor* beta
    ) {
        const auto ok = [](const ttnn::Tensor& t) {
            return (
                (t.dtype() == DataType::BFLOAT16 || t.dtype() == DataType::FLOAT32) &&
                t.layout() == Layout::TILE && !t.memory_config().is_sharded()
            );
        };
        if (!ok(x) || x.logical_shape().rank() < 2) {
            return false;
        }
        const uint32_t w = x.logical_shape()[-1];
        for (const auto* p : {gamma, beta}) {
            if (p != nullptr &&
                (!ok(*p) || p->logical_shape().rank() != 2 ||
                 p->logical_shape()[0] != 1 || p->logical_shape()[1] != w)) {
                return false;
            }
        }
        return true;
    }

    Forward forward(
        const ttnn::Tensor& x,
        const ttnn::Tensor* gamma,
        const ttnn::Tensor* beta,
        float eps,
        bool rms,
        bool stats,
        const MemoryConfig& memory,
        MathFidelity fidelity,
        bool approx
    ) {
        const auto sh = shape_of(x);
        const uint32_t x_tile = tile_bytes(x.dtype());
        uint32_t g_tile = 0, params = 0;
        for (const auto* p : {gamma, beta}) {
            if (p != nullptr) {
                g_tile = std::max(g_tile, tile_bytes(p->dtype()));
                ++params;
            }
        }
        // Outputs first: an L1-resident y lowers the gap the circular buffers must fit.
        Forward out;
        out.y = ttnn::empty(
            x.logical_shape(), x.dtype(), Layout::TILE, detail::device(), memory
        );
        if (stats) {
            const ttnn::Shape ss(sh.stat_dims);
            const MemoryConfig dram{};
            if (!rms) {
                out.mean = ttnn::empty(
                    ss, DataType::FLOAT32, Layout::TILE, detail::device(), dram
                );
            }
            out.rstd = ttnn::empty(
                ss, DataType::FLOAT32, Layout::TILE, detail::device(), dram
            );
        }
        const auto p = plan(sh.rows, sh.wt, x_tile, g_tile, params, false);
        if (!p) {
            throw std::runtime_error(
                "ktt_norm: no row split fits the program L1 budget"
            );
        }
        static const bool trace = std::getenv("KTT_NORM_TRACE") != nullptr;
        if (trace) {
            std::fprintf(
                stderr,
                "ktt_norm fwd rows %u wt %u: %s\n",
                sh.rows,
                sh.wt,
                p->str().c_str()
            );
        }
        std::vector<ttnn::Tensor> io{x};
        if (gamma) {
            io.push_back(*gamma);
        }
        if (beta) {
            io.push_back(*beta);
        }
        const uint32_t gamma_io = gamma ? 1 : 0, beta_io = gamma_io + (beta ? 1 : 0);
        uint32_t mean_io = 0, rstd_io = 0;
        if (out.mean) {
            mean_io = static_cast<uint32_t>(io.size());
            io.push_back(*out.mean);
        }
        if (out.rstd) {
            rstd_io = static_cast<uint32_t>(io.size());
            io.push_back(*out.rstd);
        }
        const uint32_t y_io = static_cast<uint32_t>(io.size());
        io.push_back(out.y);

        program::mark(program::Phase::Alloc);
        program::Key key("ktt_norm_fwd");
        for (const auto& t : io) {
            key.add(t);
        }
        const bool dynamic = dynamic_rows(*p, sh.rows);
        key.add(gamma != nullptr).add(beta != nullptr).add(rms).add(stats);
        key.add(bits(eps)).add(p->group).add(p->seg).add(p->groups).add(p->row_depth);
        key.add(dynamic).add(static_cast<int64_t>(fidelity)).add(approx);
        program::launch(key, io, [&] {
            program::Built built;
            auto& desc = built.desc;
            uint32_t count = 0;
            const CoreRangeSet range = cores_of(*p, count);
            const auto f32 = tt::DataFormat::Float32;
            const auto xf = format_of(x.dtype());
            add_cb(desc, range, 0, p->row_depth * p->seg, xf);
            add_row_cbs(desc, range);
            if (gamma) {
                add_cb(desc, range, 1, p->seg, format_of(gamma->dtype()));
            }
            if (beta) {
                add_cb(desc, range, 2, p->seg, format_of(beta->dtype()));
            }
            add_cb(desc, range, 3, ceil_div(p->seg, 2), f32);
            add_cb(desc, range, 5, 1, f32);
            add_cb(desc, range, 6, 2, f32);
            add_cb(desc, range, 16, out_tiles(p->seg), xf);
            add_cb(desc, range, 19, 2, f32);
            if (p->group > 1) {
                add_cb(desc, range, 4, 4, f32);
                add_cb(desc, range, 7, 1, f32);
                add_cb(desc, range, 17, 2, f32);
                add_cb(desc, range, 18, 1, f32);
            }
            add_semaphores(desc, range, p->groups);
            const auto counter = row_counter();

            KernelDescriptor::RuntimeArgs reader_args, compute_args, writer_args;
            for (uint32_t c = 0; c < count; ++c) {
                const Seat s = seat(*p, sh, c);
                const uint32_t last_rows = s.first + s.seg == sh.wt ? sh.wlast : 32;
                const float n = static_cast<float>(32 * (s.seg - 1) + last_rows);
                const CoreCoord core = logical(c);
                reader_args.push_back({core, {s.gid, s.first, s.seg}});
                compute_args.push_back(
                    {core, {s.gid, s.index, s.seg, last_rows, bits(1.0f / n), bits(n)}}
                );
                std::vector<uint32_t> w{s.gid, s.index, s.first, s.seg};
                append_peers(w, *p, s.gid);
                writer_args.push_back({core, w});
            }
            program::mark(program::Phase::Args);
            const auto [reader_cfg, writer_cfg] = movers();

            std::vector<uint32_t> ct;
            append_accessor(ct, x);
            append_accessor(ct, gamma ? *gamma : x);
            append_accessor(ct, beta ? *beta : x);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_norm/fwd_reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .runtime_args = reader_args,
                    .common_runtime_args =
                        {x.buffer()->address(),
                         gamma ? gamma->buffer()->address() : 0u,
                         beta ? beta->buffer()->address() : 0u,
                         sh.wt,
                         sh.rows,
                         p->groups,
                         gamma ? 1u : 0u,
                         beta ? 1u : 0u,
                         p->seg,
                         dynamic ? 1u : 0u,
                         counter.first,
                         counter.second,
                         kCounterSem},
                    .config = reader_cfg
                }
            );
            built.slots.push_back(
                {.kernel = 0,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = 0}
            );
            if (gamma) {
                built.slots.push_back(
                    {.kernel = 0,
                     .entry = program::Slot::kCommon,
                     .position = 1,
                     .tensor = gamma_io}
                );
            }
            if (beta) {
                built.slots.push_back(
                    {.kernel = 0,
                     .entry = program::Slot::kCommon,
                     .position = 2,
                     .tensor = beta_io}
                );
            }

            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_norm/fwd_compute.cpp"),
                    .core_ranges = range,
                    .runtime_args = compute_args,
                    .common_runtime_args =
                        {sh.rows,
                         p->groups,
                         p->group,
                         rms ? 1u : 0u,
                         gamma ? 1u : 0u,
                         beta ? 1u : 0u,
                         bits(eps),
                         stats ? 1u : 0u,
                         p->seg,
                         bits(1.0f / static_cast<float>(sh.w))},
                    .config = compute_config(desc, fidelity, approx)
                }
            );

            std::vector<uint32_t> wct;
            append_accessor(wct, out.y);
            append_accessor(wct, out.mean ? *out.mean : out.y);
            append_accessor(wct, out.rstd ? *out.rstd : out.y);
            const uint32_t stat_bits = (out.mean ? 1u : 0u) | (out.rstd ? 2u : 0u);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_norm/fwd_writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = wct,
                    .runtime_args = writer_args,
                    .common_runtime_args =
                        {out.y.buffer()->address(),
                         out.mean ? out.mean->buffer()->address() : 0u,
                         out.rstd ? out.rstd->buffer()->address() : 0u,
                         sh.wt,
                         sh.rows,
                         p->groups,
                         p->group,
                         stat_bits,
                         0,
                         1,
                         kBlock},
                    .config = writer_cfg
                }
            );
            built.slots.push_back(
                {.kernel = 2,
                 .entry = program::Slot::kCommon,
                 .position = 0,
                 .tensor = y_io}
            );
            if (out.mean) {
                built.slots.push_back(
                    {.kernel = 2,
                     .entry = program::Slot::kCommon,
                     .position = 1,
                     .tensor = mean_io}
                );
            }
            if (out.rstd) {
                built.slots.push_back(
                    {.kernel = 2,
                     .entry = program::Slot::kCommon,
                     .position = 2,
                     .tensor = rstd_io}
                );
            }
            seal(built.desc);
            built.args = {
                {.kernel = 0, .per_core = kFwdReaderArgs, .common = kFwdReaderCommon},
                {.kernel = 1, .per_core = kFwdComputeArgs, .common = kFwdComputeCommon},
                {.kernel = 2,
                 .per_core = kFwdWriterArgs + kPeerArgs * p->group,
                 .common = kFwdWriterCommon},
            };
            built.rings = {
                pushed(0, p->seg),
                pushed(kRowCompute, 1),
                pushed(kRowWriter, 1),
                pushed(5, 1),
                pushed(6, 2),
                pushed(16, 1),
                pushed(19, 2),
                // P tiles: one push of the core's own pairs, once per launch.
                filled(3, uint64_t{ceil_div(p->seg, 2)} * kF32),
            };
            // gamma / beta: one push of the core's own column tiles, once per launch.
            if (gamma) {
                built.rings.push_back(
                    filled(1, uint64_t{p->seg} * tile_bytes(gamma->dtype()))
                );
            }
            if (beta) {
                built.rings.push_back(
                    filled(2, uint64_t{p->seg} * tile_bytes(beta->dtype()))
                );
            }
            if (p->group > 1) {
                // Gather: four tiles zeroed and filled by the peers from its base,
                // pushed two at a time. Count and n: pushed once per launch.
                built.rings.push_back({4, {2}, 4 * kF32});
                built.rings.push_back(filled(7, kF32));
                built.rings.push_back(pushed(17, 2));
                built.rings.push_back(filled(18, kF32));
            }
            return built;
        });
        return out;
    }

    namespace {

        // out [1, W] = sum over groups and rows of the [groups * 32, W] fp32 partials.
        void column_sum(
            const ttnn::Tensor& partials,
            const ttnn::Tensor& out,
            uint32_t groups,
            MathFidelity fidelity,
            bool approx
        ) {
            const uint32_t wt = ceil_div(out.logical_shape()[-1], 32);
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = std::min<uint32_t>(wt, grid.x * grid.y);
            program::Key key("ktt_norm_colsum");
            key.add(partials)
                .add(out)
                .add(groups)
                .add(static_cast<int64_t>(fidelity))
                .add(approx);
            program::launch(key, {partials, out}, [&] {
                program::Built built;
                auto& desc = built.desc;
                const CoreRangeSet range = num_cores_to_corerangeset(cores, grid, true);
                add_cb(desc, range, 0, 8, tt::DataFormat::Float32);
                add_cb(desc, range, 16, 2, format_of(out.dtype()));
                KernelDescriptor::RuntimeArgs args;
                for (uint32_t c = 0; c < cores; ++c) {
                    args.push_back({logical(c), {c}});
                }
                std::vector<uint32_t> rct, wct;
                append_accessor(rct, partials);
                append_accessor(wct, out);
                const auto [reader_cfg, writer_cfg] = movers();
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "nn/ktt_norm/colsum_reader.cpp"
                        ),
                        .core_ranges = range,
                        .compile_time_args = rct,
                        .runtime_args = args,
                        .common_runtime_args =
                            {partials.buffer()->address(), wt, groups, cores},
                        .config = reader_cfg
                    }
                );
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "nn/ktt_norm/colsum_compute.cpp"
                        ),
                        .core_ranges = range,
                        .runtime_args = args,
                        .common_runtime_args = {wt, groups, cores},
                        .config = compute_config(desc, fidelity, approx)
                    }
                );
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "nn/ktt_norm/colsum_writer.cpp"
                        ),
                        .core_ranges = range,
                        .compile_time_args = wct,
                        .runtime_args = args,
                        .common_runtime_args = {out.buffer()->address(), wt, cores},
                        .config = writer_cfg
                    }
                );
                built.slots.push_back(
                    {.kernel = 0, .entry = program::Slot::kCommon, .tensor = 0}
                );
                built.slots.push_back(
                    {.kernel = 2, .entry = program::Slot::kCommon, .tensor = 1}
                );
                seal(built.desc);
                built.args = {
                    {.kernel = 0, .per_core = kSumArgs, .common = kSumReaderCommon},
                    {.kernel = 1, .per_core = kSumArgs, .common = kSumComputeCommon},
                    {.kernel = 2, .per_core = kSumArgs, .common = kSumWriterCommon},
                };
                // Partials: 1 to 4 pages per push, clamped at the ring end by the
                // reader.
                built.rings = {filled(0, 8 * kF32), pushed(16, 1)};
                return built;
            });
        }

    }  // namespace

    Backward backward(
        const ttnn::Tensor& dy,
        const ttnn::Tensor& x,
        const ttnn::Tensor* mean,
        const ttnn::Tensor& rstd,
        const ttnn::Tensor* gamma,
        bool rms,
        bool want_dx,
        bool want_dgamma,
        bool want_dbeta,
        DataType param_dtype,
        const MemoryConfig& memory,
        MathFidelity fidelity,
        bool approx
    ) {
        const auto sh = shape_of(x);
        const uint32_t m = x.logical_shape()[-2];
        const uint32_t mt = ceil_div(m, 32), hlast = m - (mt - 1) * 32;
        const uint32_t x_tile = tile_bytes(x.dtype());
        const uint32_t g_tile = gamma ? tile_bytes(gamma->dtype()) : 0;
        const auto dev = detail::device();
        const DataType pdt = gamma ? gamma->dtype() : param_dtype;
        Backward out;
        if (!want_dx && !want_dgamma && !want_dbeta) {
            return out;
        }
        // dx first: an L1-resident dx lowers the gap the circular buffers must fit.
        if (want_dx) {
            out.dx = ttnn::empty(
                x.logical_shape(), x.dtype(), Layout::TILE, dev, memory
            );
        }
        const auto p = plan(sh.rows, sh.wt, x_tile, g_tile, gamma ? 1 : 0, true);
        if (!p) {
            throw std::runtime_error(
                "ktt_norm: no row split fits the program L1 budget"
            );
        }
        static const bool trace = std::getenv("KTT_NORM_TRACE") != nullptr;
        if (trace) {
            std::fprintf(
                stderr,
                "ktt_norm bwd rows %u wt %u: %s\n",
                sh.rows,
                sh.wt,
                p->str().c_str()
            );
        }
        const ttnn::Shape partial_shape({p->groups * 32, sh.w});
        std::optional<ttnn::Tensor> pg, pb;
        if (want_dgamma) {
            pg = ttnn::empty(
                partial_shape, DataType::FLOAT32, Layout::TILE, dev, MemoryConfig{}
            );
        }
        if (want_dbeta) {
            pb = ttnn::empty(
                partial_shape, DataType::FLOAT32, Layout::TILE, dev, MemoryConfig{}
            );
        }
        std::vector<ttnn::Tensor> io{dy, x};
        if (mean) {
            io.push_back(*mean);
        }
        const uint32_t rstd_io = static_cast<uint32_t>(io.size());
        io.push_back(rstd);
        uint32_t gamma_io = 0, pg_io = 0, pb_io = 0, dx_io = 0;
        if (gamma) {
            gamma_io = static_cast<uint32_t>(io.size());
            io.push_back(*gamma);
        }
        if (pg) {
            pg_io = static_cast<uint32_t>(io.size());
            io.push_back(*pg);
        }
        if (pb) {
            pb_io = static_cast<uint32_t>(io.size());
            io.push_back(*pb);
        }
        if (out.dx) {
            dx_io = static_cast<uint32_t>(io.size());
            io.push_back(*out.dx);
        }

        program::mark(program::Phase::Alloc);
        program::Key key("ktt_norm_bwd");
        for (const auto& t : io) {
            key.add(t);
        }
        key.add(mean != nullptr).add(gamma != nullptr).add(rms);
        key.add(want_dx).add(want_dgamma).add(want_dbeta);
        key.add(p->group).add(p->seg).add(p->groups).add(p->row_depth);
        key.add(static_cast<int64_t>(fidelity)).add(approx);
        program::launch(key, io, [&] {
            program::Built built;
            auto& desc = built.desc;
            uint32_t count = 0;
            const CoreRangeSet range = cores_of(*p, count);
            const auto f32 = tt::DataFormat::Float32;
            const auto xf = format_of(x.dtype());
            add_cb(desc, range, 0, p->row_depth * p->seg, xf);
            add_cb(desc, range, 9, p->row_depth * p->seg, xf);
            if (gamma) {
                add_cb(desc, range, 1, p->seg, format_of(gamma->dtype()));
            }
            add_cb(desc, range, 5, 1, f32);
            add_cb(desc, range, 8, 2 * p->row_depth, f32);
            add_cb(desc, range, 10, p->seg, f32);
            add_cb(desc, range, 11, p->seg, f32);
            add_cb(desc, range, 12, 2, f32);
            add_cb(desc, range, 13, 1, f32);
            add_cb(desc, range, 16, out_tiles(p->seg), xf);
            add_cb(desc, range, 21, p->seg, f32);
            add_cb(desc, range, 22, p->seg, f32);
            if (p->group > 1) {
                add_cb(desc, range, 4, 4, f32);
                add_cb(desc, range, 17, 2, f32);
            }
            for (uint32_t id = 0; id < 2; ++id) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = range,
                        .initial_value = 0
                    }
                );
            }

            KernelDescriptor::RuntimeArgs reader_args, compute_args, writer_args;
            for (uint32_t c = 0; c < count; ++c) {
                const Seat s = seat(*p, sh, c);
                const uint32_t wlast = s.first + s.seg == sh.wt ? sh.wlast : 32;
                const uint32_t rot = s.gid % s.seg;
                const CoreCoord core = logical(c);
                reader_args.push_back({core, {s.gid, s.first, s.seg, rot}});
                compute_args.push_back({core, {s.gid, s.index, s.seg, rot, wlast}});
                std::vector<uint32_t> w{s.gid, s.index, s.first, s.seg, rot};
                append_peers(w, *p, s.gid);
                writer_args.push_back({core, w});
            }
            program::mark(program::Phase::Args);
            const auto [reader_cfg, writer_cfg] = movers();

            std::vector<uint32_t> ct;
            append_accessor(ct, x);
            append_accessor(ct, dy);
            append_accessor(ct, mean ? *mean : rstd);
            append_accessor(ct, rstd);
            append_accessor(ct, gamma ? *gamma : x);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_norm/bwd_reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .runtime_args = reader_args,
                    .common_runtime_args =
                        {x.buffer()->address(),
                         dy.buffer()->address(),
                         mean ? mean->buffer()->address() : 0u,
                         rstd.buffer()->address(),
                         gamma ? gamma->buffer()->address() : 0u,
                         sh.wt,
                         sh.rows,
                         p->groups,
                         mean ? 1u : 0u,
                         gamma ? 1u : 0u,
                         p->seg},
                    .config = reader_cfg
                }
            );
            const auto slot = [&](uint32_t kernel, uint32_t position, uint32_t tensor) {
                built.slots.push_back(
                    {.kernel = kernel,
                     .entry = program::Slot::kCommon,
                     .position = position,
                     .tensor = tensor}
                );
            };
            slot(0, 0, 1);
            slot(0, 1, 0);
            if (mean) {
                slot(0, 2, 2);
            }
            slot(0, 3, rstd_io);
            if (gamma) {
                slot(0, 4, gamma_io);
            }

            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_norm/bwd_compute.cpp"),
                    .core_ranges = range,
                    .runtime_args = compute_args,
                    .common_runtime_args =
                        {sh.rows,
                         p->groups,
                         p->group,
                         rms ? 1u : 0u,
                         gamma ? 1u : 0u,
                         bits(1.0f / static_cast<float>(sh.w)),
                         mt,
                         hlast,
                         want_dx ? 1u : 0u,
                         want_dgamma ? 1u : 0u,
                         want_dbeta ? 1u : 0u,
                         p->seg},
                    .config = compute_config(desc, fidelity, approx)
                }
            );

            std::vector<uint32_t> wct;
            const ttnn::Tensor& any = out.dx ? *out.dx : (pg ? *pg : *pb);
            append_accessor(wct, out.dx ? *out.dx : any);
            append_accessor(wct, pg ? *pg : any);
            append_accessor(wct, pb ? *pb : any);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_norm/bwd_writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = wct,
                    .runtime_args = writer_args,
                    .common_runtime_args =
                        {out.dx ? out.dx->buffer()->address() : 0u,
                         pg ? pg->buffer()->address() : 0u,
                         pb ? pb->buffer()->address() : 0u,
                         sh.wt,
                         sh.rows,
                         p->groups,
                         p->group,
                         want_dx ? 1u : 0u,
                         want_dgamma ? 1u : 0u,
                         want_dbeta ? 1u : 0u,
                         0,
                         1,
                         kBlock},
                    .config = writer_cfg
                }
            );
            if (out.dx) {
                slot(2, 0, dx_io);
            }
            if (pg) {
                slot(2, 1, pg_io);
            }
            if (pb) {
                slot(2, 2, pb_io);
            }
            seal(built.desc);
            built.args = {
                {.kernel = 0, .per_core = kBwdReaderArgs, .common = kBwdReaderCommon},
                {.kernel = 1, .per_core = kBwdComputeArgs, .common = kBwdComputeCommon},
                {.kernel = 2,
                 .per_core = kBwdWriterArgs + kPeerArgs * p->group,
                 .common = kBwdWriterCommon},
            };
            built.rings = {
                pushed(0, p->seg),
                pushed(9, p->seg),
                pushed(5, 1),
                pushed(8, 2),
                pushed(10, p->seg),
                pushed(11, p->seg),
                pushed(12, 2),
                pushed(13, 1),
                pushed(16, 1),
                // dgamma / dbeta sums: the core's own column tiles accumulated in
                // place, pushed once at the end of the launch.
                filled(21, uint64_t{p->seg} * kF32),
                filled(22, uint64_t{p->seg} * kF32),
            };
            // gamma: one push of the core's own column tiles, once per launch.
            if (gamma) {
                built.rings.push_back(
                    filled(1, uint64_t{p->seg} * tile_bytes(gamma->dtype()))
                );
            }
            if (p->group > 1) {
                // Gather: four tiles zeroed and filled by the peers from its base,
                // pushed two at a time.
                built.rings.push_back({4, {2}, 4 * kF32});
                built.rings.push_back(pushed(17, 2));
            }
            return built;
        });
        if (pg) {
            out.dgamma = ttnn::empty(
                ttnn::Shape({1, sh.w}), pdt, Layout::TILE, dev, MemoryConfig{}
            );
            column_sum(*pg, *out.dgamma, p->groups, fidelity, approx);
        }
        if (pb) {
            out.dbeta = ttnn::empty(
                ttnn::Shape({1, sh.w}), pdt, Layout::TILE, dev, MemoryConfig{}
            );
            column_sum(*pb, *out.dbeta, p->groups, fidelity, approx);
        }
        return out;
    }

}  // namespace ktt::bridge::ktt_norm
