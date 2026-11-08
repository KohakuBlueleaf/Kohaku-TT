// Host side of ktt_embedding: units (slice, 32 output rows, one column tile) split over
// the grid (strided), two scratch buffers, runtime args.
#include "aten/index/ktt_embedding/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_embedding {

    ttnn::Tensor run(
        const ttnn::Tensor& table,
        const ttnn::Tensor& index,
        uint32_t n,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& s = table.logical_shape();
        const uint32_t rank = s.rank(), d = s[-1], v = s[-2];
        const uint32_t dt = (d + 31) / 32, vt = (v + 31) / 32;
        uint32_t batch = 1;
        ttsl::SmallVector<uint32_t> dims;
        for (uint32_t i = 0; i + 2 < rank; ++i) {
            batch *= s[i];
            dims.push_back(s[i]);
        }
        dims.push_back(n);
        dims.push_back(d);
        const uint32_t elem = table.dtype() == DataType::FLOAT32 ? 4 : 2;
        auto out = (
            ttnn::empty(
                ttnn::Shape(dims), table.dtype(), Layout::TILE, detail::device(), memory
            )
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_embedding");
        key.add(table).add(index).add(out).add(n);
        return program::launch(key, {table, index, out}, [&] {
            const uint32_t units = batch * ((n + 31) / 32) * dt;
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
            KernelDescriptor::RuntimeArgs args;
            for (uint32_t c = 0; c < cores; ++c) {
                args.push_back(
                    {CoreCoord(c % grid.x, c / grid.x),
                     {table.buffer()->address(),
                      index.buffer()->address(),
                      out.buffer()->address(),
                      c,
                      cores,
                      n,
                      dt,
                      batch,
                      vt,
                      elem}}
                );
            }
            program::mark(program::Phase::Args);
            std::vector<uint32_t> ct;
            for (
                const ttnn::Tensor* t :
                {&table, &index, static_cast<const ttnn::Tensor*>(&out)}
            ) {
                const auto more = (
                    TensorAccessorArgs(*t->buffer()).get_compile_time_args()
                );
                ct.insert(ct.end(), more.begin(), more.end());
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("index/ktt_embedding/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .runtime_args = args,
                    .config = ReaderConfigDescriptor{}
                }
            );
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0},
                    {.kernel = 0, .position = 1, .tensor = 1},
                    {.kernel = 0, .position = 2, .tensor = 2}
                }
            };
        });
    }

}  // namespace ktt::bridge::ktt_embedding
