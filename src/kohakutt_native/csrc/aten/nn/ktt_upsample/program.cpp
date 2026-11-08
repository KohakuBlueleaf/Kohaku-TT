// Host side of ktt_upsample: output tiles split over the grid (strided), two scratch
// buffers, runtime args; nearest upsampling and pixel shuffle.
#include "aten/nn/ktt_upsample/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_upsample {
    namespace {

        // Runs data-movement kernel `source` over `units` output tiles of `out` from
        // `x`: scratch CB 0 of `in_tiles` tiles, CB 1 of one tile; per-core args
        // (x address, out address, core, cores, tail...).
        ttnn::Tensor launch(
            const std::string& source,
            const ttnn::Tensor& x,
            const ttnn::Tensor& out,
            uint32_t units,
            uint32_t in_tiles,
            const std::vector<uint32_t>& tail
        ) {
            using namespace tt::tt_metal;
            const uint32_t elem = x.dtype() == DataType::FLOAT32 ? 4 : 2;
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
            const auto df = (
                elem == 4 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b
            );
            for (
                auto [index, tiles] :
                {std::pair<uint8_t, uint32_t>{0, in_tiles}, {1, 1}}
            ) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = tiles * 1024 * elem,
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index,
                            .data_format = df,
                            .page_size = tiles * 1024 * elem
                        }}
                    }
                );
            }
            KernelDescriptor::RuntimeArgs args;
            for (uint32_t c = 0; c < cores; ++c) {
                std::vector<uint32_t> a{
                    x.buffer()->address(), out.buffer()->address(), c, cores
                };
                a.insert(a.end(), tail.begin(), tail.end());
                a.push_back(elem);
                args.push_back({CoreCoord(c % grid.x, c / grid.x), a});
            }
            auto ct = TensorAccessorArgs(*x.buffer()).get_compile_time_args();
            const auto more = TensorAccessorArgs(*out.buffer()).get_compile_time_args();
            ct.insert(ct.end(), more.begin(), more.end());
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(source),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .runtime_args = args,
                    .config = ReaderConfigDescriptor{}
                }
            );
            return ttnn::generic_op({x, out}, desc);
        }

    }  // namespace

    ttnn::Tensor run(
        const ttnn::Tensor& x,
        uint32_t kh,
        uint32_t kw,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& shape = x.logical_shape();
        const uint32_t rank = shape.rank();
        const uint32_t h = shape[rank - 2], w = shape[rank - 1];
        uint32_t planes = 1;
        ttsl::SmallVector<uint32_t> out_dims;
        for (uint32_t d = 0; d + 2 < rank; ++d) {
            planes *= shape[d];
            out_dims.push_back(shape[d]);
        }
        out_dims.push_back(h * kh);
        out_dims.push_back(w * kw);
        auto out = (
            ttnn::empty(
                ttnn::Shape(out_dims), x.dtype(), Layout::TILE, detail::device(), memory
            )
        );
        const uint32_t iht = (h + 31) / 32, iwt = (w + 31) / 32;
        const uint32_t oht = (h * kh + 31) / 32, owt = (w * kw + 31) / 32;
        return launch(
            "nn/ktt_upsample/reader.cpp",
            x,
            out,
            planes * oht * owt,
            1,
            {planes, iht, iwt, oht, owt, kh, kw}
        );
    }

    ttnn::Tensor shuffle(
        const ttnn::Tensor& x, uint32_t r, const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& shape = x.logical_shape();
        const uint32_t rank = shape.rank();
        const uint32_t c = shape[rank - 3] / (r * r);
        const uint32_t h = shape[rank - 2], w = shape[rank - 1];
        uint32_t prefix = 1;
        ttsl::SmallVector<uint32_t> out_dims;
        for (uint32_t d = 0; d + 3 < rank; ++d) {
            prefix *= shape[d];
            out_dims.push_back(shape[d]);
        }
        out_dims.push_back(c);
        out_dims.push_back(h * r);
        out_dims.push_back(w * r);
        auto out = (
            ttnn::empty(
                ttnn::Shape(out_dims), x.dtype(), Layout::TILE, detail::device(), memory
            )
        );
        const uint32_t iht = (h + 31) / 32, iwt = (w + 31) / 32;
        const uint32_t oht = (h * r + 31) / 32, owt = (w * r + 31) / 32;
        return launch(
            "nn/ktt_upsample/shuffle.cpp",
            x,
            out,
            prefix * c * oht * owt,
            r * r,
            {prefix * c, c, iht, iwt, oht, owt, r}
        );
    }

}  // namespace ktt::bridge::ktt_upsample
