// Host side of ktt_conv: the 2-D split (column tile groups x output row tile ranges),
// circular buffers and runtime args of reader / compute / writer.
#include "aten/nn/ktt_conv/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_conv {
    namespace {

        constexpr uint32_t kTile = 2048;  // bf16 tile bytes
        // L1 bytes per core for the conv's circular buffers.
        constexpr uint32_t kBudget = 900 * 1024;

        // CB bytes of a core with nt column tiles.
        uint32_t cb_bytes(const Geometry& g, uint32_t nt) {
            const uint32_t ct = g.c / 32, k = g.kh * g.kw * ct;
            return (k * nt + 2 * ct + k + 2 * nt) * kTile;
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

        const uint32_t row_bytes = x_rows.buffer()->page_size();
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
                  row_bytes}}
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

}  // namespace ktt::bridge::ktt_conv
