// Host side of ktt_gemv: output columns split over the grid by the weight bytes, x
// multicast once per panel, each core streaming its weight columns on both movers.
#include "aten/matmul/ktt_gemv/program.hpp"

#include "aten/matmul/ktt_gemv/args.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ktt::bridge::ktt_gemv {
    namespace {

        constexpr uint32_t kTile = 2048;          // bf16 x and output tiles
        constexpr uint32_t kMinColumnTiles = 48;  // see supports()
        constexpr uint32_t kBlockBytes = 65536;   // W bytes per read barrier, at most
        constexpr uint32_t kXChunk = 32;          // x tiles per multicast
        constexpr uint32_t kMinBlocks = 8;  // W blocks per core, at least, if K allows
        constexpr uint32_t kMinBlockTiles = 8;  // W tiles per read barrier, at least
        constexpr uint32_t kDepth = 4;          // W blocks in flight per mover, at most
        // x senders per panel: one (more senders measured slower: their multicasts
        // contend at the receivers).
        constexpr uint32_t kMaxSenders = 1;
        constexpr uint32_t kSelfRead = 8;  // up to this many cores read x themselves
        constexpr uint32_t kNone = 0xffff;
        // Runtime args: mover common, per core (plus 2 per x sender); compute.
        constexpr uint32_t kMoverCommon = 13, kMoverArgs = 10;
        constexpr uint32_t kComputeArgs = 3, kComputeCommon = 6;

        // KTT_GEMV_TUNE="senders,x chunk,kb,self read cores" (kernel tuning; 0 keeps
        // a field's default).
        std::array<uint32_t, 4> tune() {
            static const std::array<uint32_t, 4> t = [] {
                std::array<uint32_t, 4> v = {0, 0, 0, 0};
                if (const char* e = std::getenv("KTT_GEMV_TUNE")) {
                    std::sscanf(e, "%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3]);
                }
                return v;
            }();
            return t;
        }

        uint32_t ceil_div(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        tt::DataFormat format_of(tt::tt_metal::DataType d) {
            using tt::tt_metal::DataType;
            return d == DataType::BFLOAT8_B ? tt::DataFormat::Bfp8_b
                : d == DataType::BFLOAT4_B  ? tt::DataFormat::Bfp4_b
                                            : tt::DataFormat::Float16_b;
        }

        // A tensor's part of the key: dtype, layout, padded shape, memory (rows inside
        // one tile share a program).
        void add_padded(program::Key& key, const ttnn::Tensor& t) {
            key.add(static_cast<int64_t>(t.dtype()));
            key.add(static_cast<int64_t>(t.layout()));
            for (const auto d : t.padded_shape()) {
                key.add(static_cast<int64_t>(d));
            }
            key.add(static_cast<int64_t>(t.memory_config().buffer_type()));
        }

    }  // namespace

    bool supports(const ttnn::Tensor& a, const ttnn::Tensor& w, bool w_nk) {
        using tt::tt_metal::DataType;
        const auto wd = w.dtype();
        if (a.dtype() != DataType::BFLOAT16 || w.logical_shape().rank() != 2 ||
            (wd != DataType::BFLOAT16 && wd != DataType::BFLOAT8_B &&
             wd != DataType::BFLOAT4_B) ||
            a.layout() != tt::tt_metal::Layout::TILE ||
            w.layout() != tt::tt_metal::Layout::TILE ||
            a.memory_config().is_sharded() || w.memory_config().is_sharded()) {
            return false;
        }
        const auto& s = a.logical_shape();
        uint32_t mt = ceil_div(s[-2], 32);
        for (uint32_t i = 0; i + 2 < s.rank(); ++i) {
            mt *= s[i];
        }
        const uint32_t nt = ceil_div(
            w_nk ? w.logical_shape()[0] : w.logical_shape()[1], 32
        );
        return mt <= kMaxRowTiles &&
            (nt >= kMinColumnTiles || (wd != DataType::BFLOAT16 && mt <= 2));
    }

    std::optional<ttnn::Tensor> run(
        const ttnn::Tensor& a,
        const ttnn::Tensor& w,
        bool w_nk,
        const tt::tt_metal::MemoryConfig& memory,
        const ttnn::ComputeKernelConfig& config,
        const ttnn::Tensor* bias
    ) {
        using namespace tt::tt_metal;
        const auto& s = a.logical_shape();
        const uint32_t rank = s.rank(), k = s[-1], m = s[-2];
        const auto& ws = w.logical_shape();
        const uint32_t n = w_nk ? ws[-2] : ws[-1];
        const uint32_t kt = ceil_div(k, 32), nt = ceil_div(n, 32);
        uint32_t slices = 1;
        ttsl::SmallVector<uint32_t> dims;
        for (uint32_t i = 0; i + 2 < rank; ++i) {
            slices *= s[i];
            dims.push_back(s[i]);
        }
        dims.push_back(m);
        dims.push_back(n);
        const uint32_t mt = slices * ceil_div(m, 32);
        const auto w_format = format_of(w.dtype());
        const uint32_t w_tile = tt::tile_size(w_format);
        const auto& mesh = detail::device();
        const auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t gx = grid.x, gy = grid.y;

        auto out = ttnn::empty(
            ttnn::Shape(dims), DataType::BFLOAT16, Layout::TILE, mesh, memory
        );

        // Cores: one per output column up to the grid, as whole rows of the grid.
        const uint32_t want = std::min(nt, gx * gy);
        const uint32_t cols = std::min(gx, want), rows = ceil_div(want, cols);
        const uint32_t cores = rows * cols, pcn = ceil_div(nt, cores);
        const uint32_t sw = std::max<uint32_t>(1, std::min(4 / mt, pcn));
        // x: read by each core itself on few cores, else multicast per panel.
        const auto tuned = tune();
        const bool self_read = cores <= (tuned[3] ? tuned[3] : kSelfRead);
        const uint32_t senders = self_read
            ? 1
            : std::min(rows, tuned[0] ? tuned[0] : kMaxSenders);
        const uint32_t groups = ceil_div(pcn, sw);
        uint32_t kb = std::min(kt, std::max<uint32_t>(1, kBlockBytes / (sw * w_tile)));
        while (
            sw * ceil_div(kb, 2) >= kMinBlockTiles &&
            groups * ceil_div(kt, kb) < kMinBlocks
        ) {
            kb = ceil_div(kb, 2);
        }
        if (tuned[2]) {
            kb = std::min(kt, tuned[2]);
        }
        const uint32_t blocks = groups * ceil_div(kt, kb);
        const uint32_t x_bytes = mt * kt * kTile,
                       out_bytes = (mt * groups + (bias ? 2 : 0)) * sw * kTile;
        const uint32_t budget = program::cb_budget();
        const uint32_t w_block = sw * kb * w_tile;
        if (x_bytes + out_bytes + 2 * w_block > budget) {
            return std::nullopt;
        }
        const uint32_t room = (budget - x_bytes - out_bytes) / (2 * w_block);
        const uint32_t d0 = std::min({kDepth, room, blocks});
        const uint32_t d1 = d0;  // either mover may read every block (x senders)
        const uint32_t xs = 0;   // dm0 takes x (on dm1's NoC 1 measured 2-3x slower)
        const uint32_t xc = tuned[1] ? tuned[1]
                                     : std::max<uint32_t>(1, kXChunk / mt);  // x chunk

        program::mark(program::Phase::Alloc);
        program::Key key("ktt_gemv");
        add_padded(key, a);
        add_padded(key, w);
        add_padded(key, out);
        if (bias) {
            add_padded(key, *bias);
        }
        key.add(bias != nullptr)
            .add(w_nk)
            .add(static_cast<int64_t>(config.math_fidelity));
        for (const uint32_t v : {cols, rows, sw, kb, d0, d1, senders, xs, xc}) {
            key.add(static_cast<int64_t>(v));
        }
        const auto describe = [&] {
            auto tr = [&](uint32_t x, uint32_t y) {
                const auto c = mesh->worker_core_from_logical_core(CoreCoord(x, y));
                return std::pair<uint32_t, uint32_t>{
                    static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)
                };
            };
            const CoreRangeSet all_set(
                CoreRange(CoreCoord(0, 0), CoreCoord(cols - 1, rows - 1))
            );
            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            struct Cb {
                uint8_t index;
                uint32_t tiles;
                tt::DataFormat format;
                uint32_t page;
            };
            std::vector<Cb> cbs = {
                {0, mt * kt, bf16, kTile},
                {1, d0 * sw * kb, w_format, w_tile},
                {2, d1 * sw * kb, w_format, w_tile},
                {16, mt * sw * groups, bf16, kTile},
            };
            if (bias) {
                cbs.push_back({3, 2 * sw, bf16, kTile});  // bias tiles per group
            }
            for (const auto& cb : cbs) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = cb.tiles * cb.page,
                        .core_ranges = all_set,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = cb.index,
                            .data_format = cb.format,
                            .page_size = cb.page
                        }}
                    }
                );
            }
            for (uint32_t id = 0; id <= senders; ++id) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = all_set,
                        .initial_value = 0
                    }
                );
            }

            // Per core: its columns (the first nt % cores cores take one more), its
            // panel's rectangle and x senders (the panel's first column, from the top).
            using Args = std::vector<uint32_t>;
            KernelDescriptor::RuntimeArgs mover_args[2], compute_args;
            const uint32_t base = nt / cores, extra = nt % cores;
            std::vector<std::pair<uint32_t, uint32_t>> panels;
            for (const auto& [x0, x1] : program::panels()) {
                if (x0 < cols) {
                    panels.emplace_back(x0, std::min(x1, cols - 1));
                }
            }
            for (const auto& [x0, x1] : panels) {
                for (uint32_t x = x0; x <= x1; ++x) {
                    for (uint32_t y = 0; y < rows; ++y) {
                        const uint32_t i = y * cols + x;
                        const uint32_t nc = base + (i < extra ? 1 : 0);
                        const uint32_t c0 = i * base + std::min(i, extra);
                        const CoreCoord core(x, y);
                        std::pair<uint32_t, uint32_t> lo = tr(x, y), hi = lo;
                        uint32_t receivers = 0, me = 0;
                        Args sender_cores = {lo.first, lo.second};
                        if (!self_read) {
                            lo = tr(x0, 0), hi = tr(x1, rows - 1);
                            receivers = (x1 - x0 + 1) * rows - 1;
                            me = x == x0 && y < senders ? y : kNone;
                            sender_cores.clear();
                            for (uint32_t s = 0; s < senders; ++s) {
                                const auto c = tr(x0, s);
                                sender_cores.insert(
                                    sender_cores.end(), {c.first, c.second}
                                );
                            }
                        }
                        // A sender's x mover is busy with x: the other takes its W.
                        const bool busy = me != kNone && receivers > 0;
                        uint32_t from = 0;
                        if (busy) {
                            from = xs == 0 ? 2 : 1;
                        }
                        for (uint32_t side = 0; side < 2; ++side) {
                            // NoC 1 rectangles run from the high corner.
                            const auto start = side == 0 ? lo : hi,
                                       end = side == 0 ? hi : lo;
                            const uint32_t mine = busy ? (side == xs ? 2 : 1) : 0;
                            // Columns (first, count), own W blocks (0 shared, 1 all, 2
                            // none), x rectangle, receivers, sender index, senders.
                            Args args = {
                                c0,
                                nc,
                                mine,
                                start.first,
                                start.second,
                                end.first,
                                end.second,
                                receivers,
                                me
                            };
                            args.insert(
                                args.end(), sender_cores.begin(), sender_cores.end()
                            );
                            args.push_back(::ktt_gemv::kArgSentinel);
                            mover_args[side].push_back({core, args});
                        }
                        compute_args.push_back(
                            {core, Args{nc, from, ::ktt_gemv::kArgSentinel}}
                        );
                    }
                }
            }
            program::mark(program::Phase::Args);

            std::vector<program::Slot> slots;
            std::vector<program::ArgCount> counts;
            // x, w, output addresses; M, K, N tiles; sw; kb; w stored [N, K]; x chunk
            // (K tiles); x senders per panel; bias address (0: none); sentinel.
            const std::vector<uint32_t> mover_common = {
                a.buffer()->address(),
                w.buffer()->address(),
                out.buffer()->address(),
                mt,
                kt,
                nt,
                sw,
                kb,
                static_cast<uint32_t>(w_nk),
                xc,
                senders,
                bias ? bias->buffer()->address() : 0,
                ::ktt_gemv::kArgSentinel
            };
            const uint32_t out_index = bias ? 3 : 2;
            for (uint32_t side = 0; side < 2; ++side) {
                std::vector<uint32_t> ct = {side};
                for (
                    const ttnn::Tensor* t :
                    {&a, &w, static_cast<const ttnn::Tensor*>(&out)}
                ) {
                    const auto c = TensorAccessorArgs(*t->buffer())
                                       .get_compile_time_args();
                    ct.insert(ct.end(), c.begin(), c.end());
                }
                if (bias) {
                    const auto c = TensorAccessorArgs(*bias->buffer())
                                       .get_compile_time_args();
                    ct.insert(ct.end(), c.begin(), c.end());
                }
                KernelDescriptor::Defines defines;
                if (side == xs) {
                    defines.emplace_back("HAS_X", "1");
                }
                if (side == 1) {
                    defines.emplace_back("HAS_OUT", "1");
                    if (bias) {
                        defines.emplace_back("HAS_BIAS", "1");
                    }
                }
                const auto index = static_cast<uint32_t>(desc.kernels.size());
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source("matmul/ktt_gemv/mover.cpp"),
                        .core_ranges = all_set,
                        .compile_time_args = ct,
                        .defines = defines,
                        .runtime_args = mover_args[side],
                        .common_runtime_args = mover_common,
                        .config = DataMovementConfigDescriptor{
                            .processor = side == 0 ? DataMovementProcessor::RISCV_1
                                                   : DataMovementProcessor::RISCV_0,
                            .noc = side == 0 ? NOC::NOC_0 : NOC::NOC_1,
                            .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                        }
                    }
                );
                for (uint32_t t = 0; t < 3; ++t) {
                    slots.push_back(
                        {.kernel = index,
                         .entry = program::Slot::kCommon,
                         .position = t,
                         .tensor = t < 2 ? t : out_index}
                    );
                }
                if (bias) {
                    slots.push_back(
                        {.kernel = index,
                         .entry = program::Slot::kCommon,
                         .position = 11,
                         .tensor = 2}
                    );
                }
                counts.push_back(
                    {.kernel = index,
                     .per_core = kMoverArgs + 2 * senders,
                     .common = kMoverCommon}
                );
            }
            ComputeConfigDescriptor compute{
                .math_fidelity = config.math_fidelity,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = ComputeConfigDescriptor::UnpackToDestModes(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                ),
                .math_approx_mode = false
            };
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("matmul/ktt_gemv/compute.cpp"),
                    .core_ranges = all_set,
                    .defines = bias ? KernelDescriptor::Defines{{"HAS_BIAS", "1"}}
                                    : KernelDescriptor::Defines{},
                    .runtime_args = compute_args,
                    .common_runtime_args =
                        {mt,
                         kt,
                         sw,
                         kb,
                         static_cast<uint32_t>(w_nk),
                         ::ktt_gemv::kArgSentinel},
                    .config = compute
                }
            );
            counts.push_back({
                .kernel = static_cast<uint32_t>(desc.kernels.size() - 1),
                .per_core = kComputeArgs,
                .common = kComputeCommon,
            });
            std::vector<program::Ring> rings = {
                {.cb = 1, .pushes = {sw * kb}},
                {.cb = 2, .pushes = {sw * kb}},
                {.cb = 16, .pushes = {mt * sw}},
            };
            if (bias) {
                rings.push_back({.cb = 3, .pushes = {sw}});
            }
            return program::Built{
                std::move(desc), std::move(slots), std::move(counts), std::move(rings)
            };
        };
        if (bias) {
            return program::launch(key, {a, w, *bias, out}, describe);
        }
        return program::launch(key, {a, w, out}, describe);
    }

}  // namespace ktt::bridge::ktt_gemv
