// Host side of ktt_gather: output tiles dealt over both data-movement RISCs of the
// grid, per-RISC index, output and staging buffers, runtime args.
#include "aten/index/ktt_gather/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_gather {
    namespace {

        // Runtime arguments reader.cpp reads: common, then per core.
        constexpr uint32_t kCommon = 14, kPerCore = 1;
        constexpr uint32_t kSentinel = 0x6A710001u;
        // Per RISC: index tile, output tile, 1024 staging slots of 64 B (+ alignment).
        constexpr uint32_t kIndexBytes = 4096, kOutBytes = 4096,
                           kStageBytes = 65536 + 64;

    }  // namespace

    ttnn::Tensor run(
        const ttnn::Tensor& x,
        const ttnn::Tensor& index,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& xs = x.logical_shape();
        const auto& is = index.logical_shape();
        const uint32_t rank = is.rank();
        const uint32_t width = xs[-1], height = xs[-2];
        const uint32_t cols = is[-1], rows = is[-2];
        uint32_t batch = 1;
        for (uint32_t i = 0; i + 2 < rank; ++i) {
            batch *= is[i];
        }
        const uint32_t wt_out = (cols + 31) / 32, ht_out = (rows + 31) / 32;
        const uint32_t wt_in = (width + 31) / 32, ht_in = (height + 31) / 32;
        const uint32_t elem = x.dtype() == DataType::BFLOAT16 ? 2 : 4;
        auto out = ttnn::empty(is, x.dtype(), Layout::TILE, detail::device(), memory);
        program::mark(program::Phase::Alloc);
        const uint32_t units = batch * ht_out * wt_out;
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t cores = (std::max<uint32_t>(
            1, std::min<uint32_t>(grid.x * grid.y, (units + 1) / 2)
        ));
        program::Key key("ktt_gather");
        key.add(x).add(index).add(out);
        return program::launch(key, {x, index, out}, [&] {
            std::vector<CoreRange> ranges;
            for (uint32_t c = 0; c < cores; ++c) {
                ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
            }
            const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();
            ProgramDescriptor desc;
            std::vector<program::Ring> rings;
            for (uint8_t role = 0; role < 2; ++role) {
                for (uint32_t k = 0; k < 3; ++k) {
                    const uint8_t index_cb = role * 3 + k;
                    const uint32_t bytes = k == 0 ? kIndexBytes
                        : k == 1                  ? kOutBytes
                                                  : kStageBytes;
                    desc.cbs.push_back(
                        CBDescriptor{
                            .total_size = bytes,
                            .core_ranges = range,
                            .format_descriptors = {CBFormatDescriptor{
                                .buffer_index = index_cb,
                                .data_format = tt::DataFormat::Float32,
                                .page_size = bytes
                            }}
                        }
                    );
                    rings.push_back({index_cb, {}, bytes});
                }
            }
            const auto args = program::core_index_args(cores);
            program::mark(program::Phase::Args);
            std::vector<uint32_t> ct;
            for (
                const ttnn::Tensor* t :
                {&x, &index, static_cast<const ttnn::Tensor*>(&out)}
            ) {
                const auto more = (TensorAccessorArgs(*t->buffer())
                                       .get_compile_time_args());
                ct.insert(ct.end(), more.begin(), more.end());
            }
            const std::vector<uint32_t> common = {
                x.buffer()->address(),
                index.buffer()->address(),
                out.buffer()->address(),
                2 * cores,
                units,
                wt_out,
                ht_out,
                rows,
                cols,
                ht_in,
                wt_in,
                width,
                elem,
                kSentinel
            };
            // Kernel 0: the reader RISC (role 1); kernel 1: the writer RISC (role 0).
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("index/ktt_gather/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .defines = {{"ROLE", "1"}},
                    .runtime_args = args,
                    .common_runtime_args = common,
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("index/ktt_gather/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = ct,
                    .defines = {{"ROLE", "0"}},
                    .runtime_args = args,
                    .common_runtime_args = common,
                    .config = WriterConfigDescriptor{}
                }
            );
            constexpr auto entry = program::Slot::kCommon;
            std::vector<program::Slot> slots;
            for (uint32_t k = 0; k < 2; ++k) {
                for (uint32_t t = 0; t < 3; ++t) {
                    slots.push_back(
                        {.kernel = k, .entry = entry, .position = t, .tensor = t}
                    );
                }
            }
            return program::Built{
                std::move(desc),
                std::move(slots),
                {{.kernel = 0, .per_core = kPerCore, .common = kCommon},
                 {.kernel = 1, .per_core = kPerCore, .common = kCommon}},
                std::move(rings)
            };
        });
    }

}  // namespace ktt::bridge::ktt_gather
