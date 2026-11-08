// Host side of the ktt_unary program: circular buffers, work split, runtime args and
// the op chain defines for reader.cpp / compute.cpp / writer.cpp.
#include "aten/eltwise/ktt_unary/program.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_unary {
    namespace {

        using ttnn::operations::unary::EltwiseUnaryWithParam;
        using ttnn::operations::unary::UnaryOpType;
        using ttnn::operations::unary::UnaryWithParam;

        // Smallest tensor (tiles) run here: below it TT-NN's ops are faster (measured
        // 96 tiles: 0.07 vs 0.045 ms; 3168 tiles: 1.7x ours).
        constexpr int64_t kMinTiles = 512;

        // Minimum tiles per core; smaller tensors use fewer cores.
        constexpr uint32_t kMinTilesPerCore = 4;

        // TT-NN's op chain for a unary op id (the parameters TT-NN's own entry points
        // use; fast-approximation parameters from `approx`); nullopt for ops without a
        // chain here.
        std::optional<std::vector<EltwiseUnaryWithParam>> chain_of(
            UnaryOp op, bool approx = false
        ) {
            const auto rc = static_cast<float>(ttnn::operations::unary::VecMode::RC);
            const float fast = approx ? 1.0f : 0.0f;
            switch (op) {
                case UnaryOp::Neg:
                    return {{UnaryWithParam(UnaryOpType::NEG)}};
                case UnaryOp::Abs:
                    return {{UnaryWithParam(UnaryOpType::ABS)}};
                case UnaryOp::Exp:
                    return {{UnaryWithParam(UnaryOpType::EXP, fast)}};
                case UnaryOp::Sqrt:
                    return {{UnaryWithParam(UnaryOpType::SQRT, fast)}};
                case UnaryOp::Rsqrt:
                    return {{UnaryWithParam(UnaryOpType::RSQRT, fast)}};
                case UnaryOp::Reciprocal:
                    return {{UnaryWithParam(UnaryOpType::RECIP)}};
                case UnaryOp::Tanh:
                    return {{UnaryWithParam(UnaryOpType::TANH, fast)}};
                case UnaryOp::Sigmoid:
                    return {{UnaryWithParam(UnaryOpType::SIGMOID, {rc, fast})}};
                case UnaryOp::Relu:
                    return {{UnaryWithParam(UnaryOpType::RELU)}};
                case UnaryOp::Gelu:
                    return {{UnaryWithParam(UnaryOpType::GELU, 0.0f)}};
                case UnaryOp::Silu:
                    return {{UnaryWithParam(UnaryOpType::SILU)}};
                case UnaryOp::Sin:
                    return {{UnaryWithParam(UnaryOpType::SIN)}};
                case UnaryOp::Cos:
                    return {{UnaryWithParam(UnaryOpType::COS)}};
                default:
                    return std::nullopt;
            }
        }

    }  // namespace

    bool supports(UnaryOp op, DType dtype, const std::vector<int64_t>& shape) {
        const bool float_dtype = dtype == DType::BFloat16 || dtype == DType::Float32;
        return (
            program::own_kernels_enabled() && float_dtype &&
            program::tile_count(shape) >= program::min_tiles(kMinTiles) &&
            chain_of(op).has_value()
        );
    }

    ttnn::Tensor run(
        UnaryOp op,
        const ttnn::Tensor& in,
        bool approx,
        const tt::tt_metal::MemoryConfig& out_config
    ) {
        using namespace tt::tt_metal;
        const auto chain = *chain_of(op, approx);
        const bool fp32 = in.dtype() == DataType::FLOAT32;
        const auto format = datatype_to_dataformat_converter(in.dtype());
        const uint32_t tile_bytes = tt::tile_size(format);
        const uint32_t tiles = in.physical_volume() / tt::constants::TILE_HW;
        const uint32_t block = fp32 ? 4 : 8;

        auto out = (
            ttnn::empty(
                in.logical_shape(),
                in.dtype(),
                Layout::TILE,
                detail::device(),
                out_config
            )
        );
        const auto s = program::split_tiles(tiles, kMinTilesPerCore);

        ProgramDescriptor desc;
        for (uint8_t index : {0, 16}) {
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = 4 * block * tile_bytes,
                    .core_ranges = s.range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index,
                        .data_format = format,
                        .page_size = tile_bytes
                    }}
                }
            );
        }

        KernelDescriptor::RuntimeArgs reader_args, writer_args, compute_args;
        uint32_t start = 0;
        for (size_t i = 0; i < s.cores.size(); ++i) {
            const uint32_t n = s.count(i);
            reader_args.push_back({
                s.cores[i],
                {in.buffer()->address(), start, n, block}
            });
            writer_args.push_back({
                s.cores[i],
                {out.buffer()->address(), start, n, block}
            });
            compute_args.push_back({s.cores[i], {n, block}});
            start += n;
        }

        KernelDescriptor::Defines defines;
        const auto chain_defines = (
            ttnn::operations::unary::utils::get_block_defines(
                chain, "0", "j", in.dtype()
            )
        );
        for (const auto& [k, v] : chain_defines) {
            defines.emplace_back(k, v);
        }
        std::vector<UnpackToDestMode> unpack(
            NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
        );
        if (fp32) {
            unpack[0] = UnpackToDestMode::UnpackToDestFp32;
        }

        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("eltwise/ktt_unary/reader.cpp"),
                .core_ranges = s.range,
                .compile_time_args = TensorAccessorArgs(*in.buffer())
                                         .get_compile_time_args(),
                .runtime_args = reader_args,
                .config = ReaderConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("eltwise/ktt_unary/writer.cpp"),
                .core_ranges = s.range,
                .compile_time_args = TensorAccessorArgs(*out.buffer())
                                         .get_compile_time_args(),
                .runtime_args = writer_args,
                .config = WriterConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("eltwise/ktt_unary/compute.cpp"),
                .core_ranges = s.range,
                .defines = defines,
                .runtime_args = compute_args,
                .config = ComputeConfigDescriptor{
                    .math_fidelity = MathFidelity::LoFi,  // SFPU only: no FPU multiply
                    .fp32_dest_acc_en = fp32,
                    .unpack_to_dest_mode = unpack,
                    .math_approx_mode = approx
                }
            }
        );
        return ttnn::generic_op({in, out}, desc);
    }

    ttnn::Tensor cast(
        const ttnn::Tensor& in,
        tt::tt_metal::DataType dtype,
        const tt::tt_metal::MemoryConfig& out_config
    ) {
        using namespace tt::tt_metal;
        const auto in_format = datatype_to_dataformat_converter(in.dtype());
        const auto out_format = datatype_to_dataformat_converter(dtype);
        const uint32_t tiles = in.physical_volume() / tt::constants::TILE_HW;
        constexpr uint32_t block = 4;
        auto out = (
            ttnn::empty(
                in.logical_shape(), dtype, Layout::TILE, detail::device(), out_config
            )
        );
        const auto s = program::split_tiles(tiles, kMinTilesPerCore);

        ProgramDescriptor desc;
        for (
            auto [index, format] : {std::pair{uint8_t{0}, in_format}, {16, out_format}}
        ) {
            const uint32_t tile_bytes = tt::tile_size(format);
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = 4 * block * tile_bytes,
                    .core_ranges = s.range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index,
                        .data_format = format,
                        .page_size = tile_bytes
                    }}
                }
            );
        }
        KernelDescriptor::RuntimeArgs reader_args, writer_args, compute_args;
        uint32_t start = 0;
        for (size_t i = 0; i < s.cores.size(); ++i) {
            const uint32_t n = s.count(i);
            reader_args.push_back({
                s.cores[i],
                {in.buffer()->address(), start, n, block}
            });
            writer_args.push_back({
                s.cores[i],
                {out.buffer()->address(), start, n, block}
            });
            compute_args.push_back({s.cores[i], {n, block}});
            start += n;
        }
        std::vector<UnpackToDestMode> unpack(
            NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
        );
        if (in.dtype() == DataType::FLOAT32) {
            unpack[0] = UnpackToDestMode::UnpackToDestFp32;
        }
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("eltwise/ktt_unary/reader.cpp"),
                .core_ranges = s.range,
                .compile_time_args = TensorAccessorArgs(*in.buffer())
                                         .get_compile_time_args(),
                .runtime_args = reader_args,
                .config = ReaderConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("eltwise/ktt_unary/writer.cpp"),
                .core_ranges = s.range,
                .compile_time_args = TensorAccessorArgs(*out.buffer())
                                         .get_compile_time_args(),
                .runtime_args = writer_args,
                .config = WriterConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = program::source("eltwise/ktt_unary/compute.cpp"),
                .core_ranges = s.range,
                .defines = {{"SFPU_OP_CHAIN_0", ""}},
                .runtime_args = compute_args,
                .config = ComputeConfigDescriptor{
                    .math_fidelity = MathFidelity::LoFi,
                    .fp32_dest_acc_en = true,
                    .unpack_to_dest_mode = unpack,
                    .math_approx_mode = false
                }
            }
        );
        return ttnn::generic_op({in, out}, desc);
    }

}  // namespace ktt::bridge::ktt_unary
