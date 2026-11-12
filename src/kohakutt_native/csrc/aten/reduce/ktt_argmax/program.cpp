// Host side of ktt_argmax: items (unit, W chunk) over the grid, a unit's chunks
// combined on its first core; buffers and runtime args.
#include "aten/reduce/ktt_argmax/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_argmax {
    namespace {

        // Runtime arguments the kernel sources read: common per kernel, then per core.
        constexpr uint32_t kReaderCommon = 8, kWriterCommon = 7, kComputeCommon = 5;
        constexpr uint32_t kItemArgs = 1, kWriterArgs = 3;
        // W tiles a chunk scans at least, and the bytes of a unit's parts at most.
        constexpr uint32_t kChunkTiles = 16, kPartsBytes = 64 * 1024;

    }  // namespace

    ttnn::Tensor run(
        const ttnn::Tensor& x, const tt::tt_metal::MemoryConfig& memory, bool argmin
    ) {
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
        key.add(x).add(out).add(argmin ? 1 : 0);
        return program::launch(key, {x, out}, [&] {
            const uint32_t units = (p + 31) / 32 * rt;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t grid_cores = grid.x * grid.y;
            // Few units over a long W: each unit's W split into chunks on their own
            // cores (one item per core), else units strided over the grid.
            const uint32_t part_bytes = std::min<uint32_t>(p, 32) * 32 * 8;
            uint32_t split = std::min(
                {grid_cores / std::max<uint32_t>(units, 1),
                 wt / kChunkTiles,
                 kPartsBytes / part_bytes}
            );
            split = std::max<uint32_t>(split, 1);
            const uint32_t items = units * split;
            const uint32_t cores = (std::max<uint32_t>(
                1, std::min<uint32_t>(grid_cores, items)
            ));
            std::vector<CoreRange> ranges;
            for (uint32_t c = 0; c < cores; ++c) {
                ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
            }
            const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();

            ProgramDescriptor desc;
            const auto
                xf = (elem == 4 ? tt::DataFormat::Float32 : tt::DataFormat::UInt16);
            const auto f32 = tt::DataFormat::Float32;
            const uint32_t parts_bytes = split * part_bytes;
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
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = parts_bytes,
                    .core_ranges = range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = 19, .data_format = f32, .page_size = parts_bytes
                    }}
                }
            );
            // The chunk-0 cores count their unit's parts here.
            desc.semaphores.push_back(
                SemaphoreDescriptor{
                    .id = 0,
                    .core_type = tt::CoreType::WORKER,
                    .core_ranges = range,
                    .initial_value = 0
                }
            );
            const auto item_args = program::core_index_args(cores);
            KernelDescriptor::RuntimeArgs writer_args;
            for (uint32_t c = 0; c < cores; ++c) {
                const uint32_t head = c - c % split;
                const auto noc = detail::device()->worker_core_from_logical_core(
                    CoreCoord(head % grid.x, head / grid.x)
                );
                writer_args.push_back(
                    {CoreCoord(c % grid.x, c / grid.x),
                     {c, static_cast<uint32_t>(noc.x), static_cast<uint32_t>(noc.y)}}
                );
            }
            program::mark(program::Phase::Args);
            std::vector<UnpackToDestMode> unpack(
                NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
            );
            // x straight to DST as fp32; bf16 as its raw bits (UInt16, shifted in the
            // compute): unpacked as bf16 a NaN reads as infinity.
            unpack[0] = UnpackToDestMode::UnpackToDestFp32;
            unpack[3] = UnpackToDestMode::UnpackToDestFp32;
            KernelDescriptor::Defines defines;
            if (argmin) {
                defines.emplace_back("ARGMIN", "1");
            }
            auto compute_defines = defines;
            if (elem == 2) {
                compute_defines.emplace_back("X_BF16_BITS", "1");
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("reduce/ktt_argmax/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*x.buffer())
                                             .get_compile_time_args(),
                    .defines = defines,
                    .runtime_args = item_args,
                    .common_runtime_args =
                        {x.buffer()->address(),
                         cores,
                         p,
                         rt,
                         wt,
                         w - (wt - 1) * 32,
                         elem,
                         split},
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("reduce/ktt_argmax/writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*out.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = writer_args,
                    .common_runtime_args =
                        {out.buffer()->address(), cores, p, rt, split, 0, r},
                    .config = WriterConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("reduce/ktt_argmax/compute.cpp"),
                    .core_ranges = range,
                    .defines = compute_defines,
                    .runtime_args = item_args,
                    .common_runtime_args = {cores, p, rt, wt, split},
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = true,
                        .dst_full_sync_en = true,
                        .unpack_to_dest_mode = unpack,
                        .math_approx_mode = false
                    }
                }
            );
            return program::Built{
                std::move(desc),
                {{.kernel = 0, .entry = program::Slot::kCommon},
                 {.kernel = 1, .entry = program::Slot::kCommon, .tensor = 1}},
                {{.kernel = 0, .per_core = kItemArgs, .common = kReaderCommon},
                 {.kernel = 1, .per_core = kWriterArgs, .common = kWriterCommon},
                 {.kernel = 2, .per_core = kItemArgs, .common = kComputeCommon}},
                // One tile per push; the writer's staging tile and parts are written in
                // place.
                {{0, {1}, 0},
                 {3, {1}, 0},
                 {16, {1}, 0},
                 {17, {1}, 0},
                 {18, {}, 4096},
                 {19, {}, parts_bytes}}
            };
        });
    }

}  // namespace ktt::bridge::ktt_argmax
