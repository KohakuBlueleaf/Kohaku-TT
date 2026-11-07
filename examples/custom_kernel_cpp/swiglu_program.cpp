// Device side of the SwiGLU extension: the program computing silu(a) * b, registered
// as "kohakutt_example_cpp::swiglu" (TT-NN / Metalium headers only).
#include <string>

#include <tt-metalium/tensor_accessor_args.hpp>

#include "ktt/extension/device.hpp"

namespace {

    using namespace tt::tt_metal;

    // KERNEL_DIR: examples/custom_kernel/kernels (set by CMakeLists.txt).
    std::string kernel(const char* name) {
        return std::string(KERNEL_DIR) + "/" + name;
    }

    // tensors: a, b, out (bf16 TILE, same shape); numerics: the caller's scope.
    ProgramDescriptor swiglu(
        const std::vector<ttnn::Tensor>& tensors,
        const ktt::extension::Numerics& numerics
    ) {
        const auto& a = tensors[0];
        const auto& b = tensors[1];
        const auto& out = tensors[2];
        const auto split = ktt::extension::split_tiles(ktt::extension::tile_count(a));

        ProgramDescriptor desc;
        const auto format = datatype_to_dataformat_converter(a.dtype());
        const uint32_t page = tt::tile_size(format);
        for (uint8_t index : {0, 1, 16}) {
            desc.cbs.push_back(
                CBDescriptor{
                    .total_size = 2 * page,
                    .core_ranges = split.range,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index, .data_format = format, .page_size = page
                    }}
                }
            );
        }

        KernelDescriptor::RuntimeArgs reader_args, writer_args, compute_args;
        for (size_t i = 0; i < split.cores.size(); ++i) {
            const auto core = split.cores[i];
            const uint32_t start = split.start(i), count = split.count(i);
            reader_args.push_back({
                core,
                {a.buffer()->address(), b.buffer()->address(), start, count}
            });
            writer_args.push_back({core, {out.buffer()->address(), start, count}});
            compute_args.push_back({core, {count}});
        }

        auto reader_ct = TensorAccessorArgs(*a.buffer()).get_compile_time_args();
        const auto b_ct = TensorAccessorArgs(*b.buffer()).get_compile_time_args();
        reader_ct.insert(reader_ct.end(), b_ct.begin(), b_ct.end());
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = kernel("reader.cpp"),
                .core_ranges = split.range,
                .compile_time_args = reader_ct,
                .runtime_args = reader_args,
                .config = ReaderConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = kernel("writer.cpp"),
                .core_ranges = split.range,
                .compile_time_args = TensorAccessorArgs(*out.buffer())
                                         .get_compile_time_args(),
                .runtime_args = writer_args,
                .config = WriterConfigDescriptor{}
            }
        );
        desc.kernels.push_back(
            KernelDescriptor{
                .kernel_source = kernel("compute.cpp"),
                .core_ranges = split.range,
                .runtime_args = compute_args,
                .config = ktt::extension::compute_config(numerics)
            }
        );
        return desc;
    }

    KTT_REGISTER_PROGRAM("kohakutt_example_cpp::swiglu", swiglu);

}  // namespace
