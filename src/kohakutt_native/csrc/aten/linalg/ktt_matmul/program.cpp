// Host side of ktt_matmul: column groups of nt tiles, the grid's cores split evenly
// among them (each a contiguous, possibly empty, range of row tiles: the same cores for
// every M), circular buffers and runtime args.
#include "aten/linalg/ktt_matmul/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_matmul {
    namespace {

        constexpr uint32_t kTile = 2048;          // bf16 tile bytes
        constexpr uint32_t kBudget = 900 * 1024;  // CB bytes per core

        // a tiles per CB push: a row's K tiles, at most 32.
        uint32_t chunk(uint32_t kt) {
            return std::min<uint32_t>(kt, 32);
        }

        uint32_t cb_bytes(uint32_t kt, uint32_t nt, uint32_t w_tile) {
            return kt * nt * w_tile + 2 * chunk(kt) * kTile + 2 * nt * kTile;
        }

    }  // namespace

    std::optional<uint32_t> columns_per_core(
        uint32_t kt, uint32_t nt, uint32_t w_tile
    ) {
        for (uint32_t cols : {4u, 2u, 1u}) {
            if (cols <= nt && cb_bytes(kt, cols, w_tile) <= kBudget) {
                return cols;
            }
        }
        return std::nullopt;
    }

    uint32_t l1_bytes(uint32_t kt, uint32_t nt, uint32_t w_tile) {
        return cb_bytes(kt, *columns_per_core(kt, nt, w_tile), w_tile);
    }

    ttnn::Tensor run(
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
        const uint32_t kt = (k + 31) / 32, ntot = (n + 31) / 32;
        uint32_t slices = 1;
        ttsl::SmallVector<uint32_t> dims;
        for (uint32_t i = 0; i + 2 < rank; ++i) {
            slices *= s[i];
            dims.push_back(s[i]);
        }
        dims.push_back(m);
        dims.push_back(n);
        const uint32_t mt = slices * ((m + 31) / 32);
        const bool bfp8 = w.dtype() == DataType::BFLOAT8_B;
        const uint32_t w_tile = bfp8 ? 1088 : kTile;
        const uint32_t nt = *columns_per_core(kt, ntot, w_tile);
        const uint32_t kc = chunk(kt);
        auto out = (
            ttnn::empty(
                ttnn::Shape(dims),
                DataType::BFLOAT16,
                Layout::TILE,
                detail::device(),
                memory
            )
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_matmul");
        key.add(a).add(w).add(out).add(w_nk).add(
            static_cast<int64_t>(config.math_fidelity)
        );
        return program::launch(key, {a, w, out}, [&] {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = grid.x * grid.y;
            const uint32_t groups = (ntot + nt - 1) / nt;
            const uint32_t per_group = std::max<uint32_t>(1, cores / groups);

            struct Work {
                CoreCoord core;
                uint32_t m0, m1, n0, n;
            };

            std::vector<Work> work;
            for (uint32_t gi = 0; gi < groups && work.size() < cores; ++gi) {
                const uint32_t n0 = gi * nt, cols = std::min(nt, ntot - n0);
                for (uint32_t p = 0; p < per_group && work.size() < cores; ++p) {
                    const auto index = static_cast<uint32_t>(work.size());
                    work.push_back({
                        CoreCoord(index % grid.x, index / grid.x),
                        mt * p / per_group,
                        mt * (p + 1) / per_group,
                        n0,
                        cols
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
            const auto wf = bfp8 ? tt::DataFormat::Bfp8_b : bf16;
            const std::tuple<uint8_t, uint32_t, tt::DataFormat, uint32_t> cbs[] = {
                {0, 2 * kc, bf16, kTile},
                {1, kt * nt, wf, w_tile},
                {16, 2 * nt, bf16, kTile}
            };
            for (auto [index, tiles, format, page] : cbs) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = tiles * page,
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index,
                            .data_format = format,
                            .page_size = page
                        }}
                    }
                );
            }

            // Per core: the block (m0, m1, n0, n) to the reader and writer, its row and
            // column tiles to compute.
            KernelDescriptor::RuntimeArgs block_args, compute_args;
            for (const auto& wk : work) {
                block_args.push_back({wk.core, {wk.m0, wk.m1, wk.n0, wk.n}});
                compute_args.push_back({wk.core, {wk.m1 - wk.m0, wk.n}});
            }
            program::mark(program::Phase::Args);
            const auto nk = static_cast<uint32_t>(w_nk);

            auto reader_ct = TensorAccessorArgs(*a.buffer()).get_compile_time_args();
            const auto w_ct = TensorAccessorArgs(*w.buffer()).get_compile_time_args();
            reader_ct.insert(reader_ct.end(), w_ct.begin(), w_ct.end());
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("linalg/ktt_matmul/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = reader_ct,
                    .runtime_args = block_args,
                    .common_runtime_args =
                        {
                            a.buffer()->address(),
                            w.buffer()->address(),
                            kt,
                            ntot,
                            kc,
                            nk
                        },
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv/writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*out.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = block_args,
                    .common_runtime_args = {out.buffer()->address(), ntot},
                    .config = WriterConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("linalg/ktt_matmul/compute.cpp"),
                    .core_ranges = range,
                    .runtime_args = compute_args,
                    .common_runtime_args = {kt, kc, nk},
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = config.math_fidelity,
                        .fp32_dest_acc_en = true,
                        .math_approx_mode = false
                    }
                }
            );
            return program::Built{
                std::move(desc),
                {{.kernel = 0, .entry = program::Slot::kCommon},
                 {
                     .kernel = 0,
                     .entry = program::Slot::kCommon,
                     .position = 1,
                     .tensor = 1
                 },
                 {.kernel = 1, .entry = program::Slot::kCommon, .tensor = 2}}
            };
        });
    }

}  // namespace ktt::bridge::ktt_matmul
