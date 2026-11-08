// Host side of ktt_conv: the 2-D split (column tile groups x output row tile ranges),
// circular buffers and runtime args of reader / compute / writer.
#include "aten/nn/ktt_conv/program.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_conv {
    namespace {

        constexpr uint32_t kTile = 2048;       // bf16 tile bytes
        constexpr uint32_t kRowsPerPage = 32;  // pixel rows per page of to_rows()
        // L1 bytes per core for the conv's circular buffers.
        constexpr uint32_t kBudget = 900 * 1024;

        // CB bytes of a core with nt column tiles.
        uint32_t cb_bytes(const Geometry& g, uint32_t nt) {
            const uint32_t ct = g.c / 32, k = g.kh * g.kw * ct;
            return (k * nt + 2 * ct + k + 2 * nt) * kTile;
        }

        // Bytes of the reader's two window sets (kh windows of 32 + (kw - 1) dw rows).
        uint32_t window_bytes(const Geometry& g) {
            return 2 * g.kh * (32 + (g.kw - 1) * g.dw) * g.c * 2;
        }

        // True unless KTT_CONV_WINDOW=0 (one read per tap row instead of shared
        // windows).
        bool window_enabled() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_CONV_WINDOW");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

    }  // namespace

    std::optional<uint32_t> columns_per_core(const Geometry& g) {
        const uint32_t ot = (g.o + 31) / 32;
        for (uint32_t nt : {4u, 2u, 1u}) {
            if (nt <= ot && cb_bytes(g, nt) <= kBudget) {
                return nt;
            }
        }
        return std::nullopt;
    }

    ttnn::Tensor run(
        const ttnn::Tensor& x_rows,
        const ttnn::Tensor& w,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config
    ) {
        using namespace tt::tt_metal;
        const uint32_t ct = g.c / 32, ot = (g.o + 31) / 32;
        const uint32_t nt = *columns_per_core(g);
        const uint32_t out_rows = g.n * g.oh * g.wp;
        const uint32_t mt = (out_rows + 31) / 32;
        const uint32_t in_rows = g.n * g.hp * g.wp;

        auto out = ttnn::empty(
            ttnn::Shape({out_rows, g.o}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );

        // Column groups of nt tiles; the cores of the grid split evenly among them,
        // each core a contiguous range of output row tiles.
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t cores = grid.x * grid.y;
        const uint32_t groups = (ot + nt - 1) / nt;
        const uint32_t per_group = (
            std::max<uint32_t>(
                1, std::min<uint32_t>(cores / std::max<uint32_t>(groups, 1), mt)
            )
        );

        struct Work {
            CoreCoord core;
            uint32_t m0, m1, n0, n;
        };

        std::vector<Work> work;
        for (uint32_t gi = 0; gi < groups; ++gi) {
            const uint32_t n0 = gi * nt, n = std::min(nt, ot - n0);
            for (uint32_t p = 0; p < per_group; ++p) {
                const uint32_t m0 = mt * p / per_group, m1 = mt * (p + 1) / per_group;
                if (m1 == m0) {
                    continue;
                }
                const uint32_t index = static_cast<uint32_t>(work.size());
                if (index >= cores) {
                    break;
                }
                work.push_back({
                    CoreCoord(index % grid.x, index / grid.x),
                    m0,
                    m1,
                    n0,
                    n
                });
            }
        }
        std::vector<CoreRange> ranges;
        for (const auto& wk : work) {
            ranges.emplace_back(wk.core, wk.core);
        }
        const CoreRangeSet range(ranges);

        ProgramDescriptor desc;
        const auto bf16 = tt::DataFormat::Float16_b;
        const uint32_t k = g.kh * g.kw * ct;
        const std::pair<uint8_t, uint32_t> cbs[] = {
            {0, 2 * ct}, {1, k * nt}, {2, k}, {16, 2 * nt}
        };
        for (auto [index, tiles] : cbs) {
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = tiles * kTile,
                    .core_ranges = range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index, .data_format = bf16, .page_size = kTile
                    }}
                }
            );
        }
        const uint32_t window =
            (window_enabled() && g.kw > 1 &&
             cb_bytes(g, nt) + window_bytes(g) <= kBudget);
        if (window) {
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = window_bytes(g),
                    .core_ranges = range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = 3,
                        .data_format = bf16,
                        .page_size = window_bytes(g)
                    }}
                }
            );
        }

        const uint32_t row_bytes = g.c * 2;
        const uint32_t per_page = x_rows.buffer()->page_size() / row_bytes;
        KernelDescriptor::RuntimeArgs reader_args, compute_args, writer_args;
        for (const auto& wk : work) {
            reader_args.push_back(
                {wk.core,
                 {x_rows.buffer()->address(),
                  w.buffer()->address(),
                  wk.m0,
                  wk.m1,
                  wk.n0,
                  wk.n,
                  ot,
                  ct,
                  g.kh,
                  g.kw,
                  g.dh,
                  g.dw,
                  g.wp,
                  g.hp * g.wp,
                  g.oh * g.wp,
                  in_rows,
                  out_rows,
                  row_bytes,
                  window,
                  per_page}}
            );
            compute_args.push_back({wk.core, {wk.m1 - wk.m0, wk.n, g.kh * g.kw, ct}});
            writer_args.push_back({
                wk.core,
                {out.buffer()->address(), wk.m0, wk.m1, wk.n0, wk.n, ot}
            });
        }

        auto reader_ct = TensorAccessorArgs(*x_rows.buffer()).get_compile_time_args();
        const auto w_ct = TensorAccessorArgs(*w.buffer()).get_compile_time_args();
        reader_ct.insert(reader_ct.end(), w_ct.begin(), w_ct.end());
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv/reader.cpp"),
                .core_ranges = range,
                .compile_time_args = reader_ct,
                .runtime_args = reader_args,
                .config = ReaderConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv/writer.cpp"),
                .core_ranges = range,
                .compile_time_args = TensorAccessorArgs(*out.buffer())
                                         .get_compile_time_args(),
                .runtime_args = writer_args,
                .config = WriterConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("nn/ktt_conv/compute.cpp"),
                .core_ranges = range,
                .runtime_args = compute_args,
                .config = ComputeConfigDescriptor{
                    .math_fidelity = config.math_fidelity,
                    .fp32_dest_acc_en = true,
                    .math_approx_mode = false
                }
            }
        );
        return ttnn::generic_op({x_rows, w, out}, desc);
    }

    namespace {

        // A single data-movement kernel `source` on the first `cores` cores of the grid
        // with circular buffers {index, bytes} and per-core args (core index, cores)
        // appended to `args`.
        ttnn::Tensor layout_op(
            const std::vector<ttnn::Tensor>& io,
            const std::string& source,
            const std::vector<std::pair<uint8_t, uint32_t>>& cbs,
            const std::vector<uint32_t>& head,
            const std::vector<uint32_t>& tail,
            uint32_t units
        ) {
            using namespace tt::tt_metal;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = (
                std::max<uint32_t>(1, std::min<uint32_t>(grid.x * grid.y, units))
            );
            std::vector<CoreRange> ranges;
            for (uint32_t c = 0; c < cores; ++c) {
                ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
            }
            const CoreRangeSet range(ranges);
            ProgramDescriptor desc;
            for (auto [index, bytes] : cbs) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = bytes,
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index,
                            .data_format = tt::DataFormat::Float16_b,
                            .page_size = bytes
                        }}
                    }
                );
            }
            KernelDescriptor::RuntimeArgs args;
            for (uint32_t c = 0; c < cores; ++c) {
                std::vector<uint32_t> a = head;
                a.push_back(c);
                a.push_back(cores);
                a.insert(a.end(), tail.begin(), tail.end());
                args.push_back({CoreCoord(c % grid.x, c / grid.x), a});
            }
            std::vector<uint32_t> ct;
            for (size_t i = 0; i + 1 < io.size(); ++i) {
                const auto more = (
                    TensorAccessorArgs(*io[i].buffer()).get_compile_time_args()
                );
                ct.insert(ct.end(), more.begin(), more.end());
            }
            const auto last = (
                TensorAccessorArgs(*io.back().buffer()).get_compile_time_args()
            );
            ct.insert(ct.end(), last.begin(), last.end());
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(source),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .runtime_args = args,
                    .config = ReaderConfigDescriptor{}
                }
            );
            return ttnn::generic_op(io, desc);
        }

    }  // namespace

    ttnn::Tensor to_rows(
        const ttnn::Tensor& x, const Geometry& g, uint32_t ph, uint32_t pw
    ) {
        using namespace tt::tt_metal;
        const uint32_t h = g.hp - 2 * ph, w = g.wp - 2 * pw;
        const uint32_t pixels = g.n * g.hp * g.wp;
        auto rows = ttnn::empty(
            ttnn::Shape({
                (pixels + kRowsPerPage - 1) / kRowsPerPage,
                kRowsPerPage * g.c
            }),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR,
            detail::device(),
            MemoryConfig{}
        );
        const uint32_t units = (
            std::max(
                g.n * ((h + 31) / 32) * ((w + 31) / 32) * (g.c / 32) * 4, g.n * g.hp
            )
        );
        const uint32_t zero_bytes = (g.c * 2 + 31) / 32 * 32;
        layout_op(
            {x, rows},
            "nn/ktt_conv/rows_in.cpp",
            {{0, 32 * kTile}, {1, 32 * kTile}, {2, zero_bytes}},
            {x.buffer()->address(), rows.buffer()->address()},
            {g.n, g.c, h, w, g.hp, g.wp, ph, pw, kRowsPerPage},
            units
        );
        return rows;
    }

    ttnn::Tensor to_nchw(
        const ttnn::Tensor& out,
        const Geometry& g,
        uint32_t ow,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        auto nchw = ttnn::empty(
            ttnn::Shape({g.n, g.o, g.oh, ow}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            memory
        );
        const uint32_t units =
            (g.n * ((g.oh + 31) / 32) * ((ow + 31) / 32) * ((g.o + 31) / 32) * 4);
        layout_op(
            {out, nchw},
            "nn/ktt_conv/rows_out.cpp",
            {{0, 16 * kTile}, {1, 32 * kTile}},
            {out.buffer()->address(), nchw.buffer()->address()},
            {g.n, g.o, g.oh, ow, g.wp},
            units
        );
        return nchw;
    }

}  // namespace ktt::bridge::ktt_conv
