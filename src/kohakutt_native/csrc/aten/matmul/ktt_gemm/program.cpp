// Host side of ktt_gemm: the plan (layout, blocks, subblock, K block in the free L1 span), the
// multicast roles and NoCs per core, circular buffers and runtime args.
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

        uint32_t ceil_div(uint32_t a, uint32_t b) {
            return (a + b - 1) / b;
        }

        // Circular-buffer bytes per core: double-buffered A and W blocks, two output
        // subblocks, the fp32 partials block when K takes more than one block.
        uint32_t l1_bytes(uint32_t bm, uint32_t bn, uint32_t kb, uint32_t nk, uint32_t sub, uint32_t w_tile) {
            return (
                2 * bm * kb * kTile + 2 * kb * bn * w_tile + 2 * sub * kTile +
                (nk > 1 ? bm * bn * kPartTile : 0)
            );
        }

        // Subblocks with fp32 Dst in half-sync mode (at most 4 tiles), the ones with the
        // fewest unpacked tiles per product first.
        constexpr std::array<std::pair<uint32_t, uint32_t>, 8> kSubblocks = {{
            {1, 4}, {4, 1}, {2, 2}, {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 1}
        }};

        // The fastest config per call measured on the card for bf16 [K, N] weights
        // (per-call autotune, 2026-11-09), keyed by (Mt, Kt, Nt).
        struct Measured {
            bool one_d;
            uint32_t bm, bn, nbh, nbw, kb, sh, sw;
        };

        const std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Measured>& measured() {
            static const std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Measured> table = {
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
            buf, sizeof(buf), "%s grid %ux%u pc %ux%u block %ux%u x %ux%u kb %u nk %u sub %ux%u l1 %u",
            one_d ? "1d" : "2d", rows, cols, pcm, pcn, bm, bn, nbh, nbw, kb, nk, sh, sw, l1
        );
        return buf;
    }

    std::optional<Plan> plan(
        uint32_t mt, uint32_t kt, uint32_t nt, uint32_t w_tile, uint32_t gx, uint32_t gy, uint32_t budget
    ) {
        // The measured config when it fits.
        if (w_tile == kTile) {
            if (const auto it = measured().find({mt, kt, nt}); it != measured().end()) {
                const auto& m = it->second;
                Plan p;
                p.one_d = m.one_d;
                place(p, mt, nt, gx, gy);
                p.bm = m.bm, p.bn = m.bn, p.nbh = m.nbh, p.nbw = m.nbw;
                p.kb = std::min(m.kb, kt), p.nk = ceil_div(kt, p.kb), p.sh = m.sh, p.sw = m.sw;
                p.l1 = l1_bytes(p.bm, p.bn, p.kb, p.nk, p.sh * p.sw, w_tile);
                if (p.l1 <= budget && p.bm * p.nbh >= p.pcm && p.bn * p.nbw >= p.pcn) {
                    return p;
                }
            }
        }
        // Layout by reuse: 2D reuses each received A tile bn times and each W tile bm times
        // per core; 1D reads A locally and reuses each W tile pcm times.
        Plan p;
        p.one_d = false;
        place(p, mt, nt, gx, gy);
        const uint32_t reuse_2d = std::min(p.pcm, p.pcn);
        const uint32_t pcm_1d = ceil_div(mt, gx * gy);
        if (pcm_1d >= 3 && pcm_1d > reuse_2d) {
            p.one_d = true;
            place(p, mt, nt, gx, gy);
        }
        // Output blocks: fewest first, then the least re-reading; 1D splits N into at least
        // two blocks when N allows, so earlier blocks' writes overlap later blocks' compute.
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> splits;  // count, cost, nbh, nbw
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
                const uint32_t l1 = l1_bytes(bm, bn, kb, nk, sub.first * sub.second, w_tile);
                if (l1 <= budget) {
                    p.bm = bm, p.bn = bn, p.nbh = nbh, p.nbw = nbw, p.kb = kb, p.nk = nk;
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
            (wd == DataType::BFLOAT16 || wd == DataType::BFLOAT8_B || wd == DataType::BFLOAT4_B) &&
            a.layout() == tt::tt_metal::Layout::TILE && w.layout() == tt::tt_metal::Layout::TILE
        );
    }

    std::optional<ttnn::Tensor> run(
        const ttnn::Tensor& a,
        const ttnn::Tensor& w,
        bool w_nk,
        const tt::tt_metal::MemoryConfig& memory,
        const ttnn::ComputeKernelConfig& config
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
        const auto w_format = (
            wd == DataType::BFLOAT8_B   ? tt::DataFormat::Bfp8_b
            : wd == DataType::BFLOAT4_B ? tt::DataFormat::Bfp4_b
                                        : tt::DataFormat::Float16_b
        );
        const uint32_t w_tile = tt::tile_size(w_format);
        const auto& mesh = detail::device();
        const auto grid = mesh->compute_with_storage_grid_size();

        // The output first: in L1 it lowers the free span the plan must fit.
        auto out = ttnn::empty(ttnn::Shape(dims), DataType::BFLOAT16, Layout::TILE, mesh, memory);
        // Plans per shape and budget class (the free span rounded down to 32 KiB).
        constexpr uint32_t kBudgetClass = 32 * 1024;
        const uint32_t budget = program::cb_budget() / kBudgetClass * kBudgetClass;
        static std::mutex mutex;
        static std::map<std::array<uint32_t, 7>, std::optional<Plan>> plans;
        std::optional<Plan> p;
        {
            const std::array<uint32_t, 7> id = {
                mt, kt, nt, w_tile, static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y), budget
            };
            std::lock_guard lock(mutex);
            if (const auto hit = plans.find(id); hit != plans.end()) {
                p = hit->second;
            } else {
                p = plans.emplace(id, plan(mt, kt, nt, w_tile, grid.x, grid.y, budget)).first->second;
                static const bool trace = std::getenv("KTT_GEMM_TRACE") != nullptr;
                if (trace) {
                    std::fprintf(
                        stderr, "ktt_gemm %ux%ux%u tiles w_tile %u budget %u: %s\n", mt, kt, nt,
                        w_tile, budget, p ? p->str().c_str() : "none"
                    );
                }
            }
        }
        if (!p) {
            return std::nullopt;
        }
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_gemm");
        key.add(a).add(w).add(out).add(w_nk).add(static_cast<int64_t>(config.math_fidelity));
        for (const uint32_t v :
             {static_cast<uint32_t>(p->one_d), p->rows, p->cols, p->pcm, p->pcn, p->bm, p->bn, p->nbh,
              p->nbw, p->kb, p->sh, p->sw}) {
            key.add(static_cast<int64_t>(v));
        }
        return program::launch(key, {a, w, out}, [&] {
            auto tr = [&](uint32_t x, uint32_t y) {
                const auto c = mesh->worker_core_from_logical_core(CoreCoord(x, y));
                return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)};
            };
            const uint32_t rows = p->rows, cols = p->cols;
            const CoreRangeSet all_set(CoreRange(CoreCoord(0, 0), CoreCoord(cols - 1, rows - 1)));

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
            for (const auto& cb : cbs) {
                desc.cbs.push_back(CBDescriptor{
                    .total_size = cb.tiles * cb.page,
                    .core_ranges = all_set,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = cb.index, .data_format = cb.format, .page_size = cb.page
                    }}
                });
            }
            // Semaphores: A sender / receiver, W sender / receiver.
            for (uint32_t id = 0; id < 4; ++id) {
                desc.semaphores.push_back(SemaphoreDescriptor{
                    .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = all_set, .initial_value = 0
                });
            }

            // Panels: physically contiguous column spans (multicasts stay inside one).
            std::vector<std::pair<uint32_t, uint32_t>> panels;
            for (const auto& [x0, x1] : program::panels()) {
                if (x0 < cols) {
                    panels.emplace_back(x0, std::min(x1, cols - 1));
                }
            }
            // Per core: reader role (A sender, W sender), reader args, writer args.
            using Args = std::vector<uint32_t>;
            std::map<std::pair<bool, bool>, KernelDescriptor::RuntimeArgs> reader_args;
            std::map<std::pair<bool, bool>, std::vector<CoreRange>> reader_cores;
            std::map<uint32_t, KernelDescriptor::RuntimeArgs> writer_args;
            std::map<uint32_t, std::vector<CoreRange>> writer_cores;
            const Args none = {0, 0, 0, 0, 0};
            auto rect = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t num) {
                const auto a0 = tr(x0, y0), a1 = tr(x1, y1);
                return Args{a0.first, a0.second, a1.first, a1.second, num};
            };
            auto add_core = [&](uint32_t x, uint32_t y, bool a_sender, bool w_sender, uint32_t m0, uint32_t n0,
                                const Args& mc0, std::pair<uint32_t, uint32_t> s0, const Args& mc1,
                                std::pair<uint32_t, uint32_t> s1) {
                const CoreCoord core(x, y);
                Args ra = {m0, n0};
                ra.insert(ra.end(), mc0.begin(), mc0.end());
                ra.insert(ra.end(), {s0.first, s0.second});
                ra.insert(ra.end(), mc1.begin(), mc1.end());
                ra.insert(ra.end(), {s1.first, s1.second});
                ra.insert(ra.end(), none.begin(), none.end());
                ra.push_back(::ktt_gemm::kArgSentinel);
                const std::pair<bool, bool> role{a_sender, w_sender};
                reader_args[role].push_back({core, ra});
                reader_cores[role].emplace_back(core, core);
                // Dedicated NoC mode: the reader takes NoC0 when it multicasts, else NoC1; the
                // writer always takes the other one.
                const uint32_t reader_noc = (a_sender || w_sender) ? 0 : 1;
                const uint32_t writer_noc = 1 - reader_noc;
                writer_args[writer_noc].push_back(
                    {core, Args{m0, n0, std::min(m0 + p->pcm, mt), std::min(n0 + p->pcn, nt), ::ktt_gemm::kArgSentinel}}
                );
                writer_cores[writer_noc].emplace_back(core, core);
            };
            if (p->one_d) {
                uint32_t i = 0;
                for (uint32_t x = 0; x < cols; ++x) {
                    for (uint32_t y = 0; y < rows; ++y, ++i) {
                        const bool top = y == 0;
                        const Args mc1 = top && rows > 1 ? rect(x, 1, x, rows - 1, rows - 1) : none;
                        add_core(x, y, true, top, i * p->pcm, 0, none, {0, 0}, mc1, tr(x, 0));
                    }
                }
            } else {
                for (const auto& [x0, x1] : panels) {
                    for (uint32_t x = x0; x <= x1; ++x) {
                        for (uint32_t y = 0; y < rows; ++y) {
                            const Args mc0 = x1 > x0 ? rect(x0 + 1, y, x1, y, x1 - x0) : none;
                            const Args mc1 = rows > 1 ? rect(x, 1, x, rows - 1, rows - 1) : none;
                            add_core(x, y, x == x0, y == 0, y * p->pcm, x * p->pcn, mc0, tr(x0, y), mc1, tr(x, 0));
                        }
                    }
                }
            }
            program::mark(program::Phase::Args);

            std::vector<program::Slot> slots;
            auto a_ct = TensorAccessorArgs(*a.buffer()).get_compile_time_args();
            const auto w_ct = TensorAccessorArgs(*w.buffer()).get_compile_time_args();
            a_ct.insert(a_ct.end(), w_ct.begin(), w_ct.end());
            const std::vector<uint32_t> reader_common = {
                a.buffer()->address(), w.buffer()->address(), kt, nt, mt, p->kb, p->nk, p->bm, p->bn,
                p->nbh, p->nbw, 1, 0, 1, 2, 3, static_cast<uint32_t>(w_nk), ::ktt_gemm::kArgSentinel
            };
            for (const auto& [role, args] : reader_args) {
                KernelDescriptor::Defines defines;
                if (role.first) {
                    defines.emplace_back("IN0_SENDER", "1");
                }
                if (role.second) {
                    defines.emplace_back("IN1_SENDER", "1");
                }
                const auto index = static_cast<uint32_t>(desc.kernels.size());
                desc.kernels.push_back(KernelDescriptor{
                    .kernel_source = program::source("matmul/ktt_gemm/reader.cpp"),
                    .core_ranges = CoreRangeSet(reader_cores[role]).merge_ranges(),
                    .compile_time_args = a_ct,
                    .defines = defines,
                    .runtime_args = args,
                    .common_runtime_args = reader_common,
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = (role.first || role.second) ? NOC::NOC_0 : NOC::NOC_1,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                });
                slots.push_back({.kernel = index, .entry = program::Slot::kCommon, .position = 0, .tensor = 0});
                slots.push_back({.kernel = index, .entry = program::Slot::kCommon, .position = 1, .tensor = 1});
            }
            const auto out_ct = TensorAccessorArgs(*out.buffer()).get_compile_time_args();
            const std::vector<uint32_t> writer_common = {
                out.buffer()->address(), nt, p->bm, p->bn, p->nbh, p->nbw, p->sh, p->sw, 1, ::ktt_gemm::kArgSentinel
            };
            for (const auto& [noc, args] : writer_args) {
                const auto index = static_cast<uint32_t>(desc.kernels.size());
                desc.kernels.push_back(KernelDescriptor{
                    .kernel_source = program::source("matmul/ktt_gemm/writer.cpp"),
                    .core_ranges = CoreRangeSet(writer_cores[noc]).merge_ranges(),
                    .compile_time_args = out_ct,
                    .runtime_args = args,
                    .common_runtime_args = writer_common,
                    .config = DataMovementConfigDescriptor{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = noc == 1 ? NOC::NOC_1 : NOC::NOC_0,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC
                    }
                });
                slots.push_back({.kernel = index, .entry = program::Slot::kCommon, .position = 0, .tensor = 2});
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
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("matmul/ktt_gemm/compute.cpp"),
                .core_ranges = all_set,
                .common_runtime_args = {
                    p->bm, p->bn, p->kb, p->nk, kb_last, p->nbh, p->nbw, p->sh, p->sw, 1,
                    static_cast<uint32_t>(w_nk), ::ktt_gemm::kArgSentinel
                },
                .config = compute
            });
            return program::Built{std::move(desc), std::move(slots)};
        });
    }

}  // namespace ktt::bridge::ktt_gemm
