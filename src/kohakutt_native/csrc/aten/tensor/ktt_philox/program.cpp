// Host side of ktt_philox: output tiles split over the grid (strided), one writer
// kernel per core, sizes and global strides as common runtime args.
#include "aten/tensor/ktt_philox/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_philox {

    ttnn::Tensor run(
        const std::vector<int64_t>& local,
        const std::vector<int64_t>& global,
        tt::tt_metal::DataType dtype,
        const ttnn::Tensor& params,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto rank = static_cast<uint32_t>(local.size());
        const uint32_t elem = dtype == DataType::FLOAT32 ? 4 : 2;
        auto out = (
            ttnn::empty(
                detail::device_shape(local),
                dtype,
                Layout::TILE,
                detail::device(),
                memory
            )
        );
        program::mark(program::Phase::Alloc);
        const auto tiles = static_cast<uint32_t>(program::tile_count(local));
        program::Key key("ktt_philox");
        key.add(params).add(out);
        for (int64_t g : global) {
            key.add(g);
        }
        return program::launch(key, {params, out}, [&] {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = (
                std::max<uint32_t>(1, std::min<uint32_t>(grid.x * grid.y, tiles))
            );
            std::vector<CoreRange> ranges;
            for (uint32_t c = 0; c < cores; ++c) {
                ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
            }
            const CoreRangeSet range(ranges);
            ProgramDescriptor desc;
            const auto df = (
                elem == 4 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b
            );
            for (
                auto [index, page, format] :
                {
                    std::tuple<uint8_t,
                    uint32_t,
                    tt::DataFormat>{ 0, 4096, tt::DataFormat::Float32 },
                    {1, 1024 * elem, df}
                }
            ) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = page,
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index,
                            .data_format = format,
                            .page_size = page
                        }}
                    }
                );
            }
            const auto args = program::core_index_args(cores);
            program::mark(program::Phase::Args);
            std::vector<uint32_t> ct = (
                TensorAccessorArgs(*out.buffer()).get_compile_time_args()
            );
            const auto more = (
                TensorAccessorArgs(*params.buffer()).get_compile_time_args()
            );
            ct.insert(ct.end(), more.begin(), more.end());
            std::vector<uint32_t> common{
                out.buffer()->address(),
                params.buffer()->address(),
                cores,
                tiles,
                elem,
                rank
            };
            for (int64_t d : local) {
                common.push_back(static_cast<uint32_t>(d));
            }
            for (int64_t g : global) {
                common.push_back(static_cast<uint32_t>(g));
            }
            std::vector<uint64_t> stride(rank, 1);
            for (uint32_t k = rank; k-- > 1;) {
                stride[k - 1] = stride[k] * static_cast<uint64_t>(global[k]);
            }
            for (uint64_t s : stride) {
                common.push_back(static_cast<uint32_t>(s));
                common.push_back(static_cast<uint32_t>(s >> 32));
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("tensor/ktt_philox/writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .runtime_args = args,
                    .common_runtime_args = common,
                    .config = WriterConfigDescriptor{}
                }
            );
            constexpr auto common_slot = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0, .entry = common_slot, .position = 0, .tensor = 1},
                    {.kernel = 0, .entry = common_slot, .position = 1, .tensor = 0}
                }
            };
        });
    }

}  // namespace ktt::bridge::ktt_philox
