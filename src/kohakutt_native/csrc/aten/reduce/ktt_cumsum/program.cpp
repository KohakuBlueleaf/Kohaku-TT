// Host side of ktt_cumsum: row tiles split over the grid (strided), circular buffers,
// runtime args.
#include "aten/reduce/ktt_cumsum/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_cumsum {

    ttnn::Tensor run(const ttnn::Tensor& x, const tt::tt_metal::MemoryConfig& memory) {
        using namespace tt::tt_metal;
        const auto& s = x.logical_shape();
        const uint32_t rank = s.rank();
        uint32_t lead = 1;
        for (uint32_t d = 0; d + 2 < rank; ++d) {
            lead *= s[d];
        }
        const uint32_t rows = lead * ((s[rank - 2] + 31) / 32);
        const uint32_t wt = (s[rank - 1] + 31) / 32;
        const uint32_t elem = x.dtype() == DataType::FLOAT32 ? 4 : 2;
        auto out = ttnn::empty(s, x.dtype(), Layout::TILE, detail::device(), memory);

        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t cores = (
            std::max<uint32_t>(1, std::min<uint32_t>(grid.x * grid.y, rows))
        );
        std::vector<CoreRange> ranges;
        for (uint32_t c = 0; c < cores; ++c) {
            ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
        }
        const CoreRangeSet range(ranges);

        ProgramDescriptor desc;
        const auto df = elem == 4 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        const uint32_t page = 1024 * elem;
        for (
            auto [index, tiles] :
            {std::pair<uint8_t, uint32_t>{0, 2}, {4, 1}, {5, 1}, {16, 2}}
        ) {
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = tiles * page,
                    .core_ranges = range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index, .data_format = df, .page_size = page
                    }}
                }
            );
        }
        KernelDescriptor::RuntimeArgs reader_args, compute_args, writer_args;
        for (uint32_t c = 0; c < cores; ++c) {
            const CoreCoord core(c % grid.x, c / grid.x);
            reader_args.push_back({
                core,
                {x.buffer()->address(), c, cores, rows, wt, elem}
            });
            compute_args.push_back({core, {c, cores, rows, wt}});
            writer_args.push_back({
                core,
                {out.buffer()->address(), c, cores, rows, wt, elem}
            });
        }
        std::vector<UnpackToDestMode> unpack(
            NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
        );
        if (elem == 4) {
            unpack[0] = UnpackToDestMode::UnpackToDestFp32;
        }
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("reduce/ktt_cumsum/reader.cpp"),
                .core_ranges = range,
                .compile_time_args = TensorAccessorArgs(*x.buffer())
                                         .get_compile_time_args(),
                .runtime_args = reader_args,
                .config = ReaderConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("reduce/ktt_cumsum/writer.cpp"),
                .core_ranges = range,
                .compile_time_args = TensorAccessorArgs(*out.buffer())
                                         .get_compile_time_args(),
                .runtime_args = writer_args,
                .config = WriterConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("reduce/ktt_cumsum/compute.cpp"),
                .core_ranges = range,
                .runtime_args = compute_args,
                .config = ComputeConfigDescriptor{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true,
                    .unpack_to_dest_mode = unpack,
                    .math_approx_mode = false
                }
            }
        );
        return ttnn::generic_op({x, out}, desc);
    }

}  // namespace ktt::bridge::ktt_cumsum
