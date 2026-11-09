// Host side of ktt_argmax: units (32 leading rows, one row tile) split over the grid
// (strided), circular buffers, runtime args.
#include "aten/reduce/ktt_argmax/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_argmax {

    ttnn::Tensor run(const ttnn::Tensor& x, const tt::tt_metal::MemoryConfig& memory) {
        using namespace tt::tt_metal;
        const auto& s = x.logical_shape();
        const uint32_t p = s[0], r = s[1], w = s[2];
        const uint32_t rt = (r + 31) / 32, wt = (w + 31) / 32;
        const uint32_t elem = x.dtype() == DataType::FLOAT32 ? 4 : 2;
        auto out = ttnn::empty(
            ttnn::Shape({p, r}), DataType::INT32, Layout::TILE, detail::device(), memory
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_argmax");
        key.add(x).add(out);
        return program::launch(key, {x, out}, [&] {
            const uint32_t units = (p + 31) / 32 * rt;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = (
                std::max<uint32_t>(1, std::min<uint32_t>(grid.x * grid.y, units))
            );
            std::vector<CoreRange> ranges;
            for (uint32_t c = 0; c < cores; ++c) {
                ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
            }
            const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();

            ProgramDescriptor desc;
            const auto xf = (
                elem == 4 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b
            );
            const auto f32 = tt::DataFormat::Float32;
            const std::tuple<uint8_t, uint32_t, tt::DataFormat> cbs[] = {
                {0, 2, xf}, {3, 1, f32}, {16, 1, f32}, {17, 1, f32}, {18, 1, f32}
            };
            for (auto [index, tiles, df] : cbs) {
                const uint32_t page = tt::tile_size(df);
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
            const auto unit_args = program::core_index_args(cores);
            program::mark(program::Phase::Args);
            std::vector<UnpackToDestMode> unpack(
                NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
            );
            unpack[3] = UnpackToDestMode::UnpackToDestFp32;
            if (elem == 4) {
                unpack[0] = UnpackToDestMode::UnpackToDestFp32;
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("reduce/ktt_argmax/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*x.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = unit_args,
                    .common_runtime_args =
                        {
                            x.buffer()->address(),
                            cores,
                            p,
                            rt,
                            wt,
                            w - (wt - 1) * 32,
                            elem
                        },
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("reduce/ktt_argmax/writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*out.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = unit_args,
                    .common_runtime_args = {out.buffer()->address(), cores, p, rt},
                    .config = WriterConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("reduce/ktt_argmax/compute.cpp"),
                    .core_ranges = range,
                    .runtime_args = unit_args,
                    .common_runtime_args = {cores, p, rt, wt},
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = true,
                        .unpack_to_dest_mode = unpack,
                        .math_approx_mode = false
                    }
                }
            );
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0, .entry = program::Slot::kCommon},
                    {.kernel = 1, .entry = program::Slot::kCommon, .tensor = 1}
                }
            };
        });
    }

}  // namespace ktt::bridge::ktt_argmax
