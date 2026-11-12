// Host side of ktt_gemm: the plan (layout, blocks, subblock, K block in the free L1
// span), the multicast roles and NoCs per core, circular buffers and runtime args.
#include "aten/matmul/ktt_gemm/program.hpp"

#include "aten/matmul/ktt_gemm/args.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ktt::bridge::ktt_gemm {
    namespace {

        constexpr uint32_t kTile = 2048;      // bf16 A and output tiles
        constexpr uint32_t kPartTile = 4096;  // fp32 partials
        // Runtime args the kernel sources read (sentinels included), per core and
        // common: A stream with the writer, W stream, compute.
        constexpr uint32_t kAArgs = 12, kWArgs = 9, kComputeArgs = 0;
        constexpr uint32_t kACommon = 21, kWCommon = 14, kComputeCommon = 12;

        uint32_t ceil_div(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        // Circular-buffer bytes per core: double-buffered A and W blocks, two output
        // subblocks, the fp32 partials block when K takes more than one block.
        uint32_t l1_bytes(
            uint32_t bm,
            uint32_t bn,
            uint32_t kb,
            uint32_t nk,
            uint32_t sub,
            uint32_t w_tile
        ) {
            return (
                2 * bm * kb * kTile + 2 * kb * bn * w_tile + 2 * sub * kTile +
                (nk > 1 ? bm * bn * kPartTile : 0)
            );
        }

        // Subblocks with fp32 Dst in half-sync mode (at most 4 tiles), the ones with
        // the fewest unpacked tiles per product first.
        constexpr std::array<std::pair<uint32_t, uint32_t>, 8> kSubblocks = {
            {{1, 4}, {4, 1}, {2, 2}, {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 1}}
        };

        // The fastest config per call measured on the card for bf16 [K, N] weights
        // (per-call autotune, 2026-11-09), keyed by (Mt, Kt, Nt).
        struct Measured {
            bool one_d;
            uint32_t bm, bn, nbh, nbw, kb, sh, sw;
        };

        const std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Measured>& measured() {
            static const std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Measured>
                table = {
                    {{32, 32, 32}, {false, 4, 3, 1, 1, 16, 4, 1}},
                    {{64, 64, 64}, {false, 7, 6, 1, 1, 8, 1, 3}},
                    {{128, 128, 128}, {false, 13, 12, 1, 1, 2, 1, 4}},
                    {{80, 96, 128}, {false, 8, 12, 1, 1, 4, 1, 4}},
                    {{512, 32, 32}, {true, 5, 16, 1, 2, 8, 1, 4}},
                    {{512, 12, 12}, {true, 5, 12, 1, 1, 4, 1, 4}},
                    {{32, 256, 128}, {false, 4, 12, 1, 1, 8, 1, 4}},
                    {{128, 32, 256}, {false, 13, 8, 1, 3, 4, 1, 4}},
            };
            return table;
        }

        // Fills the grid fields of p for its layout.
        void place(Plan& p, uint32_t mt, uint32_t nt, uint32_t gx, uint32_t gy) {
            if (p.one_d) {
                p.pcm = ceil_div(mt, gx * gy);
                p.pcn = nt;
                p.rows = gy;
                p.cols = gx;
            } else {
                p.pcm = ceil_div(mt, gy);
                p.rows = ceil_div(mt, p.pcm);
                p.pcn = ceil_div(nt, gx);
                p.cols = ceil_div(nt, p.pcn);
            }
        }

    }  // namespace

    std::string Plan::str() const {
        char buf[200];
        std::snprintf(
            buf,
            sizeof(buf),
            "%s grid %ux%u pc %ux%u block %ux%u x %ux%u kb %u nk %u sub %ux%u l1 %u",
            one_d ? "1d" : "2d",
            rows,
            cols,
            pcm,
            pcn,
            bm,
            bn,
            nbh,
            nbw,
            kb,
            nk,
            sh,
            sw,
            l1
        );
        return buf;
    }

    namespace {

        // Config m as a plan, when it covers each core's range and fits `budget`.
        std::optional<Plan> fixed(
            const Measured& m,
            uint32_t mt,
            uint32_t kt,
            uint32_t nt,
            uint32_t w_tile,
            uint32_t gx,
            uint32_t gy,
            uint32_t budget
        ) {
            Plan p;
            p.one_d = m.one_d;
            place(p, mt, nt, gx, gy);
            p.bm = m.bm, p.bn = m.bn, p.nbh = m.nbh, p.nbw = m.nbw;
            p.kb = std::min(m.kb, kt), p.nk = ceil_div(kt, p.kb), p.sh = m.sh,
            p.sw = m.sw;
            p.l1 = l1_bytes(p.bm, p.bn, p.kb, p.nk, p.sh * p.sw, w_tile);
            const bool whole =
                (p.bm > 0 && p.bn > 0 && p.kb > 0 && p.sh > 0 && p.sw > 0 &&
                 p.sh * p.sw <= 4 && p.bm % p.sh == 0 && p.bn % p.sw == 0);
            if (whole && p.l1 <= budget && p.bm * p.nbh >= p.pcm &&
                p.bn * p.nbw >= p.pcn) {
                return p;
            }
            return std::nullopt;
        }

    }  // namespace

    std::optional<Plan> plan(
        uint32_t mt,
        uint32_t kt,
        uint32_t nt,
        uint32_t w_tile,
        uint32_t gx,
        uint32_t gy,
        uint32_t budget
    ) {
        // KTT_GEMM_PLAN="1d,bm,bn,nbh,nbw,kb,sh,sw" (kernel tuning): that config when
        // valid.
        if (const char* forced = std::getenv("KTT_GEMM_PLAN")) {
            unsigned v[8] = {};
            if (std::sscanf(
                    forced,
                    "%u,%u,%u,%u,%u,%u,%u,%u",
                    v,
                    v + 1,
                    v + 2,
                    v + 3,
                    v + 4,
                    v + 5,
                    v + 6,
                    v + 7
                ) == 8) {
                return fixed(
                    {v[0] != 0, v[1], v[2], v[3], v[4], v[5], v[6], v[7]},
                    mt,
                    kt,
                    nt,
                    w_tile,
                    gx,
                    gy,
                    budget
                );
            }
        }
        // The measured config when it fits.
        if (w_tile == kTile) {
            if (const auto it = measured().find({mt, kt, nt}); it != measured().end()) {
                if (auto p = fixed(it->second, mt, kt, nt, w_tile, gx, gy, budget)) {
                    return p;
                }
            }
        }
        // Layout by reuse: 2D reuses each received A tile bn times and each W tile bm
        // times per core; 1D reads A locally and reuses each W tile pcm times.
        Plan p;
        p.one_d = false;
        place(p, mt, nt, gx, gy);
        const uint32_t reuse_2d = std::min(p.pcm, p.pcn);
        const uint32_t pcm_1d = ceil_div(mt, gx * gy);
        if (pcm_1d >= 3 && pcm_1d > reuse_2d) {
            p.one_d = true;
            place(p, mt, nt, gx, gy);
        }
        // Output blocks: fewest first, then the least re-reading; 1D splits N into at
        // least two blocks when N allows, so earlier blocks' writes overlap later
        // blocks' compute.
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>
            splits;  // count, cost, nbh, nbw
        for (uint32_t nbh = 1; nbh <= p.pcm; ++nbh) {
            const uint32_t bm = ceil_div(p.pcm, nbh);
            if (nbh > 1 && (nbh - 1) * bm >= p.pcm) {
                continue;
            }
            for (uint32_t nbw = 1; nbw <= p.pcn; ++nbw) {
                const uint32_t bn = ceil_div(p.pcn, nbw);
                if (nbw > 1 && (nbw - 1) * bn >= p.pcn) {
                    continue;
                }
                if (p.one_d && nt >= 8 && nbw < 2) {
                    continue;
                }
                splits.emplace_back(nbh * nbw, nbw * mt + nbh * nt, nbh, nbw);
            }
        }
        std::sort(splits.begin(), splits.end());
        for (const auto& [count, cost, nbh, nbw] : splits) {
            const uint32_t bm = ceil_div(p.pcm, nbh), bn = ceil_div(p.pcn, nbw);
            auto sub = kSubblocks.back();
            for (const auto& s : kSubblocks) {
                if (bm % s.first == 0 && bn % s.second == 0) {
                    sub = s;
                    break;
                }
            }
            for (const uint32_t kb_max : {16u, 8u, 4u, 2u, 1u}) {
                const uint32_t kb = std::min(kb_max, kt), nk = ceil_div(kt, kb);
                const uint32_t l1 = l1_bytes(
                    bm, bn, kb, nk, sub.first * sub.second, w_tile
                );
                if (l1 <= budget) {
                    p.bm = bm, p.bn = bn, p.nbh = nbh, p.nbw = nbw, p.kb = kb,
                    p.nk = nk;
                    p.sh = sub.first, p.sw = sub.second, p.l1 = l1;
                    return p;
                }
            }
        }
        return std::nullopt;
    }

    bool supports(const ttnn::Tensor& a, const ttnn::Tensor& w, bool) {
        using tt::tt_metal::DataType;
        const auto wd = w.dtype();
        return (
            a.dtype() == DataType::BFLOAT16 && w.logical_shape().rank() == 2 &&
            (wd == DataType::BFLOAT16 || wd == DataType::BFLOAT8_B ||
             wd == DataType::BFLOAT4_B) &&
            a.layout() == tt::tt_metal::Layout::TILE &&
            w.layout() == tt::tt_metal::Layout::TILE
        );
    }

    std::optional<ttnn::Tensor> run(
        const ttnn::Tensor& a,
        const ttnn::Tensor& w,
        bool w_nk,
        const tt::tt_metal::MemoryConfig& memory,
        const ttnn::ComputeKernelConfig& config,
        Stream* stream,
        const ttnn::Tensor* bias
    ) {
        using namespace tt::tt_metal;
        const auto& s = a.logical_shape();
        const uint32_t rank = s.rank(), k = s[-1], m = s[-2];
        const auto& ws = w.logical_shape();
        const uint32_t n = w_nk ? ws[-2] : ws[-1];
        const uint32_t kt = (k + 31) / 32, nt = (n + 31) / 32;
        uint32_t slices = 1;
        ttsl::SmallVector<uint32_t> dims;
        for (uint32_t i = 0; i + 2 < rank; ++i) {
            slices *= s[i];
            dims.push_back(s[i]);
        }
        dims.push_back(m);
        dims.push_back(n);
        const uint32_t mt = slices * ((m + 31) / 32);

        const auto wd = w.dtype();
        const auto w_format =
            (wd == DataType::BFLOAT8_B       ? tt::DataFormat::Bfp8_b
                 : wd == DataType::BFLOAT4_B ? tt::DataFormat::Bfp4_b
                                             : tt::DataFormat::Float16_b);
        const uint32_t w_tile = tt::tile_size(w_format);
        const auto& mesh = detail::device();
        const auto grid = mesh->compute_with_storage_grid_size();
        const uint32_t y0 = stream ? stream->y0 : 0,
                       gy = static_cast<uint32_t>(grid.y) - y0;
        const uint32_t parts = stream ? stream->parts : 1;

        // The output first: in L1 it lowers the free span the plan must fit.
        auto out = ttnn::empty(
            ttnn::Shape(dims), DataType::BFLOAT16, Layout::TILE, mesh, memory
        );
        // Plans per shape and budget class (the free span rounded down to 32 KiB).
        constexpr uint32_t kBudgetClass = 32 * 1024;
        const uint32_t budget = program::cb_budget() / kBudgetClass * kBudgetClass;
        static std::mutex mutex;
        static std::map<std::array<uint32_t, 7>, std::optional<Plan>> plans;
        std::optional<Plan> p;
        {
            const std::array<uint32_t, 7> id = {
                mt, kt, nt, w_tile, static_cast<uint32_t>(grid.x), gy, budget
            };
            std::lock_guard lock(mutex);
            if (std::getenv("KTT_GEMM_PLAN")) {
                p = plan(mt, kt, nt, w_tile, grid.x, gy, budget);
                if (std::getenv("KTT_GEMM_TRACE")) {
                    std::fprintf(
                        stderr, "ktt_gemm forced: %s\n", p ? p->str().c_str() : "none"
                    );
                }
            } else if (const auto hit = plans.find(id); hit != plans.end()) {
                p = hit->second;
            } else {
                p = plans.emplace(id, plan(mt, kt, nt, w_tile, grid.x, gy, budget))
                        .first->second;
                static const bool trace = std::getenv("KTT_GEMM_TRACE") != nullptr;
                if (trace) {
                    std::fprintf(
                        stderr,
                        "ktt_gemm %ux%ux%u tiles w_tile %u budget %u: %s\n",
                        mt,
                        kt,
                        nt,
                        w_tile,
                        budget,
                        p ? p->str().c_str() : "none"
                    );
                }
            }
        }
        // A bias block (bn tiles, double-buffered) beside the plan's buffers.
        if (!p || (bias && (stream || p->l1 + 2 * p->bn * kTile > budget))) {
            return std::nullopt;
        }
        if (stream) {
            // Whole K blocks, the same number in each of the K ranges.
            while (p->kb > 1 && (kt % p->kb != 0 || (kt / p->kb) % parts != 0)) {
                p->kb /= 2;
            }
            p->nk = kt / p->kb;
            if (p->one_d || kt % p->kb != 0 || p->nk % parts != 0) {
                return std::nullopt;
            }
        }
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_gemm");
        key.add(a).add(w).add(out).add(bias != nullptr);
        if (bias) {
            key.add(*bias);
        }
        key.add(w_nk).add(static_cast<int64_t>(config.math_fidelity));
        for (
            const uint32_t v :
            {static_cast<uint32_t>(p->one_d),
             p->rows,
             p->cols,
             p->pcm,
             p->pcn,
             p->bm,
             p->bn,
             p->nbh,
             p->nbw,
             p->kb,
             p->sh,
             p->sw}
        ) {
            key.add(static_cast<int64_t>(v));
        }
        const auto describe = [&] {
            auto tr = [&](uint32_t x, uint32_t y) {
                const auto c = mesh->worker_core_from_logical_core(
                    CoreCoord(x, y + y0)
                );
                return std::pair<uint32_t, uint32_t>{
                    static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)
                };
            };
            const uint32_t rows = p->rows, cols = p->cols;
            const CoreRangeSet all_set(
                CoreRange(CoreCoord(0, y0), CoreCoord(cols - 1, y0 + rows - 1))
            );

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            const uint32_t sub = p->sh * p->sw;
            struct Cb {
                uint8_t index;
                uint32_t tiles;
                tt::DataFormat format;
                uint32_t page;
            };
            std::vector<Cb> cbs = {
                {0, 2 * p->bm * p->kb, bf16, kTile},
                {1, 2 * p->kb * p->bn, w_format, w_tile},
                {16, 2 * sub, bf16, kTile},
            };
            if (p->nk > 1) {
                cbs.push_back({24, p->bm * p->bn, tt::DataFormat::Float32, kPartTile});
            }
            if (bias) {
                cbs.push_back({3, 2 * p->bn, bf16, kTile});
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
            // Semaphores: A sender / receiver, W sender / receiver.
            std::vector<uint32_t> semaphores = {0, 1, 2, 3};
            if (stream && stream->per_block > 0) {
                for (uint32_t part = 0; part < parts; ++part) {
                    semaphores.push_back(stream->progress + part);
                }
            }
            for (const uint32_t id : semaphores) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = all_set,
                        .initial_value = 0
                    }
                );
            }

            // Panels: physically contiguous column spans (multicasts stay inside one).
            std::vector<std::pair<uint32_t, uint32_t>> panels;
            for (const auto& [x0, x1] : program::panels()) {
                if (x0 < cols) {
                    panels.emplace_back(x0, std::min(x1, cols - 1));
                }
            }
            // Per core and stream (A with the output on NoC 1, W on NoC 0): sender or
            // not, args.
            using Args = std::vector<uint32_t>;
            std::map<bool, KernelDescriptor::RuntimeArgs> a_args, w_args;
            std::map<bool, std::vector<CoreRange>> a_cores, w_cores;
            const Args none = {0, 0, 0, 0, 0};
            auto rect =
                [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t num) {
                    const auto a0 = tr(x0, y0), a1 = tr(x1, y1);
                    return Args{a0.first, a0.second, a1.first, a1.second, num};
                };
            auto add_core = [&](uint32_t x,
                                uint32_t y,
                                bool a_sender,
                                bool w_sender,
                                uint32_t m0,
                                uint32_t n0,
                                const Args& mc0,
                                std::pair<uint32_t, uint32_t> s0,
                                const Args& mc1,
                                std::pair<uint32_t, uint32_t> s1) {
                const CoreCoord core(x, y + y0);
                if (stream && a_sender) {
                    stream->a_senders.push_back(core);
                }
                Args aa = {
                    m0, n0, std::min(m0 + p->pcm, mt), std::min(n0 + p->pcn, nt)
                };
                aa.insert(aa.end(), mc0.begin(), mc0.end());
                aa.insert(aa.end(), {s0.first, s0.second, ::ktt_gemm::kArgSentinel});
                a_args[a_sender].push_back({core, aa});
                a_cores[a_sender].emplace_back(core, core);
                Args wa = {n0};
                wa.insert(wa.end(), mc1.begin(), mc1.end());
                wa.insert(wa.end(), {s1.first, s1.second, ::ktt_gemm::kArgSentinel});
                w_args[w_sender].push_back({core, wa});
                w_cores[w_sender].emplace_back(core, core);
            };
            if (p->one_d) {
                uint32_t i = 0;
                for (uint32_t x = 0; x < cols; ++x) {
                    for (uint32_t y = 0; y < rows; ++y, ++i) {
                        const bool top = y == 0;
                        const Args mc1 = top && rows > 1
                            ? rect(x, 1, x, rows - 1, rows - 1)
                            : none;
                        add_core(
                            x, y, true, top, i * p->pcm, 0, none, {0, 0}, mc1, tr(x, 0)
                        );
                    }
                }
            } else {
                // A from the panel's last column leftwards (a NoC 1 rectangle runs high
                // to low).
                for (const auto& [x0, x1] : panels) {
                    for (uint32_t x = x0; x <= x1; ++x) {
                        for (uint32_t y = 0; y < rows; ++y) {
                            const Args mc0 = x1 > x0 ? rect(x1 - 1, y, x0, y, x1 - x0)
                                                     : none;
                            const Args mc1 = rows > 1
                                ? rect(x, 1, x, rows - 1, rows - 1)
                                : none;
                            add_core(
                                x,
                                y,
                                x == x1,
                                y == 0,
                                y * p->pcm,
                                x * p->pcn,
                                mc0,
                                tr(x1, y),
                                mc1,
                                tr(x, 0)
                            );
                        }
                    }
                }
            }
            program::mark(program::Phase::Args);

            std::vector<program::Slot> slots;
            std::vector<program::ArgCount> counts;
            auto a_ct = TensorAccessorArgs(*a.buffer()).get_compile_time_args();
            const auto out_ct = TensorAccessorArgs(*out.buffer())
                                    .get_compile_time_args();
            a_ct.insert(a_ct.end(), out_ct.begin(), out_ct.end());
            if (bias) {
                const auto bias_ct = TensorAccessorArgs(*bias->buffer())
                                         .get_compile_time_args();
                a_ct.insert(a_ct.end(), bias_ct.begin(), bias_ct.end());
            }
            const uint32_t out_index = bias ? 3 : 2;
            const std::vector<uint32_t> a_common = {
                a.buffer()->address(),
                out.buffer()->address(),
                kt,
                nt,
                mt,
                p->kb,
                p->nk,
                p->bm,
                p->bn,
                p->nbh,
                p->nbw,
                1,
                0,
                1,
                p->sh,
                p->sw,
                parts,
                stream ? stream->progress : 0,
                stream ? stream->per_block : 0,
                bias ? bias->buffer()->address() : 0,
                ::ktt_gemm::kArgSentinel
            };
            for (const auto& [sender, args] : a_args) {
                const auto index = static_cast<uint32_t>(desc.kernels.size());
                KernelDescriptor::Defines defines;
                if (sender) {
                    defines.emplace_back("IN0_SENDER", "1");
                }
                if (bias) {
                    defines.emplace_back("HAS_BIAS", "1");
                }
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "matmul/ktt_gemm/a_writer.cpp"
                        ),
                        .core_ranges = CoreRangeSet(a_cores[sender]).merge_ranges(),
                        .compile_time_args = a_ct,
                        .defines = defines,
                        .runtime_args = args,
                        .common_runtime_args = a_common,
                        .config = DataMovementConfigDescriptor{
                            .processor = DataMovementProcessor::RISCV_0,
                            .noc = NOC::NOC_1,
                            .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                        }
                    }
                );
                slots.push_back(
                    {.kernel = index,
                     .entry = program::Slot::kCommon,
                     .position = 0,
                     .tensor = 0}
                );
                slots.push_back(
                    {.kernel = index,
                     .entry = program::Slot::kCommon,
                     .position = 1,
                     .tensor = out_index}
                );
                if (bias) {
                    slots.push_back(
                        {.kernel = index,
                         .entry = program::Slot::kCommon,
                         .position = 19,
                         .tensor = 2}
                    );
                }
                counts.push_back(
                    {.kernel = index, .per_core = kAArgs, .common = kACommon}
                );
            }
            const std::vector<uint32_t> w_common = {
                w.buffer()->address(),
                kt,
                nt,
                p->kb,
                p->nk,
                p->bn,
                p->nbh,
                p->nbw,
                1,
                2,
                3,
                static_cast<uint32_t>(w_nk),
                parts,
                ::ktt_gemm::kArgSentinel
            };
            for (const auto& [sender, args] : w_args) {
                const auto index = static_cast<uint32_t>(desc.kernels.size());
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "matmul/ktt_gemm/w_reader.cpp"
                        ),
                        .core_ranges = CoreRangeSet(w_cores[sender]).merge_ranges(),
                        .compile_time_args = TensorAccessorArgs(*w.buffer())
                                                 .get_compile_time_args(),
                        .defines = sender
                            ? KernelDescriptor::Defines{{"IN1_SENDER", "1"}}
                            : KernelDescriptor::Defines{},
                        .runtime_args = args,
                        .common_runtime_args = w_common,
                        .config = DataMovementConfigDescriptor{
                            .processor = DataMovementProcessor::RISCV_1,
                            .noc = NOC::NOC_0,
                            .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                        }
                    }
                );
                slots.push_back(
                    {.kernel = index,
                     .entry = program::Slot::kCommon,
                     .position = 0,
                     .tensor = 1}
                );
                counts.push_back(
                    {.kernel = index, .per_core = kWArgs, .common = kWCommon}
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
            compute.unpack_to_dest_mode[24] = UnpackToDestMode::UnpackToDestFp32;
            const uint32_t kb_last = kt - (p->nk - 1) * p->kb;
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("matmul/ktt_gemm/compute.cpp"),
                    .core_ranges = all_set,
                    .defines = bias ? KernelDescriptor::Defines{{"HAS_BIAS", "1"}}
                                    : KernelDescriptor::Defines{},
                    .common_runtime_args =
                        {p->bm,
                         p->bn,
                         p->kb,
                         p->nk,
                         kb_last,
                         p->nbh,
                         p->nbw,
                         p->sh,
                         p->sw,
                         1,
                         static_cast<uint32_t>(w_nk),
                         ::ktt_gemm::kArgSentinel},
                    .config = compute
                }
            );
            counts.push_back({
                .kernel = static_cast<uint32_t>(desc.kernels.size() - 1),
                .per_core = kComputeArgs,
                .common = kComputeCommon,
            });
            std::vector<program::Ring> rings = {
                {.cb = 0, .pushes = {p->bm * p->kb}},
                {.cb = 1, .pushes = {p->kb * p->bn}},
                {.cb = 16, .pushes = {sub}},
            };
            if (p->nk > 1) {
                rings.push_back({.cb = 24, .pushes = {sub}});
            }
            if (bias) {
                rings.push_back({.cb = 3, .pushes = {p->bn}});
            }
            return program::Built{
                std::move(desc), std::move(slots), std::move(counts), std::move(rings)
            };
        };
        if (stream) {
            stream->desc = describe().desc;
            stream->kb = p->kb;
            stream->plan = p->str();
            return out;
        }
        if (bias) {
            return program::launch(key, {a, w, *bias, out}, describe);
        }
        return program::launch(key, {a, w, out}, describe);
    }

}  // namespace ktt::bridge::ktt_gemm
