// Host side of the ktt_adam program: circular buffers, work split and runtime args of
// reader.cpp / compute.cpp / writer.cpp.
#include "aten/optim/ktt_adam/program.hpp"

#include <cstring>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/program.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_adam {
    namespace {

        // Minimum tiles per core; smaller tensors use fewer cores.
        constexpr uint32_t kMinTilesPerCore = 4;

        // Tiles per ring transaction, and blocks per ring.
        constexpr uint32_t kBlock = 4, kRingBlocks = 2;

        // Kernel order in the descriptor.
        constexpr uint32_t kReader = 0, kWriter = 1, kCompute = 2;

        // Runtime arguments the kernel sources read: common, then per core.
        constexpr uint32_t kReaderCommon = 5, kWriterCommon = 4, kComputeCommon = 11;
        constexpr uint32_t kReaderArgs = 2, kWriterArgs = 2, kComputeArgs = 1;

        // The reader's common args 0 - 3 (p, g, m, v) and the writer's 0 - 2 (p', m',
        // v'): io's addresses in order.
        std::vector<program::Slot> address_slots() {
            std::vector<program::Slot> slots;
            for (uint32_t i = 0; i < 4; ++i) {
                slots.push_back({kReader, program::Slot::kCommon, i, i});
            }
            for (uint32_t i = 0; i < 3; ++i) {
                slots.push_back({kWriter, program::Slot::kCommon, i, 4 + i});
            }
            return slots;
        }

        uint32_t bits(float value) {
            uint32_t out;
            std::memcpy(&out, &value, sizeof(out));
            return out;
        }

        // The compute kernel's common args: block, then the coefficients.
        std::vector<uint32_t> compute_common(const Coefficients& c) {
            return {
                kBlock,
                bits(c.b1),
                bits(1.0f - c.b1),
                bits(c.b2),
                bits(1.0f - c.b2),
                bits(c.scale),
                bits(c.eps),
                bits(c.step),
                bits(c.decay),
                c.l2 == 0.0f ? 0u : bits(c.l2),
                c.negate ? 1u : 0u
            };
        }

    }  // namespace

    bool supports(const std::array<const ttnn::Tensor*, 4>& pgmv) {
        using namespace tt::tt_metal;
        const auto& p = *pgmv[0];
        if (!program::own_kernels_enabled() ||
            (p.dtype() != DataType::BFLOAT16 && p.dtype() != DataType::FLOAT32)) {
            return false;
        }
        for (const auto* t : pgmv) {
            if (t->dtype() != p.dtype() || t->layout() != Layout::TILE ||
                t->padded_shape() != p.padded_shape() ||
                t->memory_config().is_sharded()) {
                return false;
            }
        }
        return true;
    }

    void run(
        const std::array<const ttnn::Tensor*, 4>& pgmv,
        const std::array<const ttnn::Tensor*, 3>& out,
        const Coefficients& c
    ) {
        using namespace tt::tt_metal;
        const auto& p = *pgmv[0];
        const bool fp32 = p.dtype() == DataType::FLOAT32;
        const auto format = datatype_to_dataformat_converter(p.dtype());
        const uint32_t tile_bytes = tt::tile_size(format);
        const uint32_t tiles = p.physical_volume() / tt::constants::TILE_HW;

        std::vector<ttnn::Tensor> io;
        for (const auto* t : pgmv) {
            io.push_back(*t);
        }
        for (const auto* t : out) {
            io.push_back(*t);
        }
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_adam");
        for (const auto& t : io) {
            key.add(t);
        }
        const auto common = compute_common(c);
        std::vector<program::Value> values;
        for (uint32_t i = 0; i < common.size(); ++i) {
            values.push_back({kCompute, i, common[i]});
        }
        program::launch(
            key,
            io,
            [&] {
                const auto s = program::split_tiles(tiles, kMinTilesPerCore);

                ProgramDescriptor desc;
                for (uint8_t index : {0, 1, 2, 3, 16, 17, 18}) {
                    desc.cbs.push_back(
                        CBDescriptor{
                            .total_size = kRingBlocks * kBlock * tile_bytes,
                            .core_ranges = s.range,
                            .format_descriptors = {CBFormatDescriptor{
                                .buffer_index = index,
                                .data_format = format,
                                .page_size = tile_bytes
                            }}
                        }
                    );
                }

                KernelDescriptor::RuntimeArgs ranges, counts;
                uint32_t start = 0;
                for (size_t i = 0; i < s.cores.size(); ++i) {
                    const uint32_t n = s.count(i);
                    ranges.push_back({s.cores[i], {start, n}});
                    counts.push_back({s.cores[i], {n}});
                    start += n;
                }
                program::mark(program::Phase::Args);

                std::vector<uint32_t> reader_ct, writer_ct, reader_common,
                    writer_common;
                for (size_t i = 0; i < 4; ++i) {
                    const auto ct = TensorAccessorArgs(*io[i].buffer())
                                        .get_compile_time_args();
                    reader_ct.insert(reader_ct.end(), ct.begin(), ct.end());
                    reader_common.push_back(io[i].buffer()->address());
                }
                reader_common.push_back(kBlock);
                for (size_t i = 4; i < 7; ++i) {
                    const auto ct = TensorAccessorArgs(*io[i].buffer())
                                        .get_compile_time_args();
                    writer_ct.insert(writer_ct.end(), ct.begin(), ct.end());
                    writer_common.push_back(io[i].buffer()->address());
                }
                writer_common.push_back(kBlock);

                std::vector<UnpackToDestMode> unpack(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                );
                if (fp32) {
                    for (uint32_t cb = 0; cb < 4; ++cb) {
                        unpack[cb] = UnpackToDestMode::UnpackToDestFp32;
                    }
                }
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source("optim/ktt_adam/reader.cpp"),
                        .core_ranges = s.range,
                        .compile_time_args = reader_ct,
                        .runtime_args = ranges,
                        .common_runtime_args = reader_common,
                        .config = ReaderConfigDescriptor{}
                    }
                );
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source("optim/ktt_adam/writer.cpp"),
                        .core_ranges = s.range,
                        .compile_time_args = writer_ct,
                        .runtime_args = ranges,
                        .common_runtime_args = writer_common,
                        .config = WriterConfigDescriptor{}
                    }
                );
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source("optim/ktt_adam/compute.cpp"),
                        .core_ranges = s.range,
                        .runtime_args = counts,
                        .common_runtime_args = common,
                        .config = ComputeConfigDescriptor{
                            // SFPU only: no FPU multiply.
                            .math_fidelity = MathFidelity::LoFi,
                            .fp32_dest_acc_en = true,
                            .unpack_to_dest_mode = unpack,
                            .math_approx_mode = false
                        }
                    }
                );
                program::Built built{std::move(desc), address_slots()};
                built.args = {
                    {.kernel = kReader,
                     .per_core = kReaderArgs,
                     .common = kReaderCommon},
                    {.kernel = kWriter,
                     .per_core = kWriterArgs,
                     .common = kWriterCommon},
                    {.kernel = kCompute,
                     .per_core = kComputeArgs,
                     .common = kComputeCommon},
                };
                for (uint32_t cb : {0u, 1u, 2u, 3u, 16u, 17u, 18u}) {
                    built.rings.push_back(
                        {cb, {kBlock}, uint64_t{kRingBlocks} * kBlock * tile_bytes}
                    );
                }
                return built;
            },
            false,
            values
        );
    }

}  // namespace ktt::bridge::ktt_adam
