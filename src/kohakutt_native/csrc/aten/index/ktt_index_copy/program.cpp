// Host side of ktt_index_copy: units (batch x column tiles, or outer x plane tiles)
// split over the grid (strided), three scratch tiles, runtime args.
#include "aten/index/ktt_index_copy/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_index_copy {
    namespace {

        // Runtime arguments reader.cpp reads: common, then per core.
        constexpr uint32_t kCommon = 16, kPerCore = 1;

    }  // namespace

    ttnn::Tensor run(
        const ttnn::Tensor& dst,
        const ttnn::Tensor& src,
        const ttnn::Tensor& index,
        uint32_t n,
        const Plan& plan
    ) {
        using namespace tt::tt_metal;
        const uint32_t elem = dst.dtype() == DataType::BFLOAT16 ? 2 : 4;
        const uint32_t units = plan.batch * plan.dt;
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_index_copy");
        key.add(src).add(index).add(dst).add(n).add(plan.planes).add(plan.batch);
        key.add(plan.dt).add(plan.lt).add(plan.length).add(plan.tt).add(plan.src_wt);
        key.add(plan.src_coff).add(plan.src_hstride).add(plan.heads);
        return program::launch(key, {src, index, dst}, [&] {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = (std::max<uint32_t>(
                1, std::min<uint32_t>(grid.x * grid.y, units)
            ));
            std::vector<CoreRange> ranges;
            for (uint32_t c = 0; c < cores; ++c) {
                ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
            }
            const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();
            ProgramDescriptor desc;
            for (uint8_t index_cb : {0, 1, 2}) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = 4096,
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index_cb,
                            .data_format = tt::DataFormat::Float32,
                            .page_size = 4096
                        }}
                    }
                );
            }
            const auto args = program::core_index_args(cores);
            program::mark(program::Phase::Args);
            std::vector<uint32_t> ct;
            for (const ttnn::Tensor* t : {&src, &index, &dst}) {
                const auto more = (TensorAccessorArgs(*t->buffer())
                                       .get_compile_time_args());
                ct.insert(ct.end(), more.begin(), more.end());
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("index/ktt_index_copy/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .runtime_args = args,
                    .common_runtime_args =
                        {src.buffer()->address(),
                         index.buffer()->address(),
                         dst.buffer()->address(),
                         cores,
                         units,
                         n,
                         plan.planes,
                         plan.dt,
                         plan.lt,
                         plan.length,
                         plan.tt,
                         plan.src_wt,
                         plan.src_coff,
                         plan.src_hstride,
                         plan.heads,
                         elem},
                    .config = ReaderConfigDescriptor{}
                }
            );
            constexpr auto common = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {{.kernel = 0, .entry = common},
                 {.kernel = 0, .entry = common, .position = 1, .tensor = 1},
                 {.kernel = 0, .entry = common, .position = 2, .tensor = 2}},
                {{.kernel = 0, .per_core = kPerCore, .common = kCommon}},
                // Reserved once and written in place: the index tile, the dst tile and
                // the source tile.
                {{0, {}, 4096}, {1, {}, 1024 * elem}, {2, {}, 1024 * elem}}
            };
        });
    }

}  // namespace ktt::bridge::ktt_index_copy
