// User programs: a JSON program description from kohakutt.kernels, run as a
// ProgramDescriptor by ttnn::generic_op on the default stream.
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/program.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge {
    namespace {

        using nlohmann::json;
        using namespace tt::tt_metal;

        // Core ranges [[x0, y0, x1, y1], ...] as a CoreRangeSet.
        CoreRangeSet core_set(const json& ranges) {
            std::vector<CoreRange> out;
            for (const auto& r : ranges) {
                out.emplace_back(
                    CoreCoord(r.at(0).get<uint32_t>(), r.at(1).get<uint32_t>()),
                    CoreCoord(r.at(2).get<uint32_t>(), r.at(3).get<uint32_t>())
                );
            }
            return CoreRangeSet(out);
        }

        DataType data_type(const std::string& name) {
            if (name == "bfloat16") {
                return DataType::BFLOAT16;
            }
            if (name == "float32") {
                return DataType::FLOAT32;
            }
            if (name == "int32") {
                return DataType::INT32;
            }
            if (name == "bfloat8_b") {
                return DataType::BFLOAT8_B;
            }
            throw std::invalid_argument("kohakutt.kernels: unknown dtype " + name);
        }

        MathFidelity fidelity(const std::string& name) {
            if (name == "LoFi") {
                return MathFidelity::LoFi;
            }
            if (name == "HiFi2") {
                return MathFidelity::HiFi2;
            }
            if (name == "HiFi3") {
                return MathFidelity::HiFi3;
            }
            if (name == "HiFi4") {
                return MathFidelity::HiFi4;
            }
            throw std::invalid_argument("kohakutt.kernels: unknown fidelity " + name);
        }

        using program::Built;
        using program::Slot;

        // An argument list: ints, {"addr": t} (tensor t's buffer address, its position
        // and t added to `addrs`) and {"accessor": t} (tensor t's TensorAccessorArgs).
        std::vector<uint32_t> arguments(
            const json& items,
            const std::vector<ttnn::Tensor>& tensors,
            std::vector<std::pair<size_t, size_t>>* addrs = nullptr
        ) {
            std::vector<uint32_t> out;
            for (const auto& item : items) {
                if (item.is_number()) {
                    out.push_back(item.get<uint32_t>());
                } else if (item.contains("addr")) {
                    const auto t = item["addr"].get<size_t>();
                    if (addrs != nullptr) {
                        addrs->emplace_back(out.size(), t);
                    }
                    out.push_back(tensors.at(t).buffer()->address());
                } else if (item.contains("accessor")) {
                    const auto& t = tensors.at(item["accessor"].get<size_t>());
                    const auto args = (TensorAccessorArgs(*t.buffer())
                                           .get_compile_time_args());
                    out.insert(out.end(), args.begin(), args.end());
                } else {
                    throw std::invalid_argument(
                        "kohakutt.kernels: bad argument " + item.dump()
                    );
                }
            }
            return out;
        }

        // The program of `spec` over `tensors` (inputs, then outputs).
        Built build(const json& spec, const std::vector<ttnn::Tensor>& tensors) {
            Built built;
            auto& desc = built.desc;
            for (const auto& cb : spec.at("cbs")) {
                const auto dtype =
                    (cb.contains("tensor")
                         ? tensors.at(cb["tensor"].get<size_t>()).dtype()
                         : data_type(cb.at("dtype").get<std::string>()));
                const auto format = datatype_to_dataformat_converter(dtype);
                const uint32_t page = tt::tile_size(format);
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = cb.at("tiles").get<uint32_t>() * page,
                        .core_ranges = core_set(cb.at("cores")),
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = cb.at("index").get<uint8_t>(),
                            .data_format = format,
                            .page_size = page
                        }}
                    }
                );
            }
            for (const auto& k : spec.at("kernels")) {
                KernelDescriptor kernel{
                    .kernel_source = k.at("source").get<std::string>(),
                    .core_ranges = core_set(k.at("cores")),
                    .compile_time_args = arguments(
                        k.value("compile_args", json::array()), tensors
                    )
                };
                for (const auto& entry : k.value("runtime_args", json::array())) {
                    const auto& core = entry.at(0);
                    std::vector<std::pair<size_t, size_t>> addrs;
                    kernel.runtime_args.push_back(
                        {CoreCoord(
                             core.at(0).get<uint32_t>(), core.at(1).get<uint32_t>()
                         ),
                         arguments(entry.at(1), tensors, &addrs)}
                    );
                    for (const auto& [position, tensor] : addrs) {
                        built.slots.push_back(
                            {.kernel = static_cast<uint32_t>(desc.kernels.size()),
                             .entry = static_cast<int32_t>(
                                 kernel.runtime_args.size() - 1
                             ),
                             .position = static_cast<uint32_t>(position),
                             .tensor = static_cast<uint32_t>(tensor)}
                        );
                    }
                }
                // A named local: items() refers into the json it iterates.
                const json defines = k.value("defines", json::object());
                for (const auto& [name, value] : defines.items()) {
                    kernel.defines.emplace_back(name, value.get<std::string>());
                }
                const auto type = k.at("type").get<std::string>();
                if (type == "reader") {
                    kernel.config = ReaderConfigDescriptor{};
                } else if (type == "writer") {
                    kernel.config = WriterConfigDescriptor{};
                } else {
                    std::vector<UnpackToDestMode> unpack(
                        NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                    );
                    for (const auto& index : k.value("unpack_fp32", json::array())) {
                        unpack.at(
                            index.get<size_t>()
                        ) = UnpackToDestMode::UnpackToDestFp32;
                    }
                    kernel.config = ComputeConfigDescriptor{
                        .math_fidelity = fidelity(k.value("fidelity", "HiFi2")),
                        .fp32_dest_acc_en = k.value("fp32_dest", true),
                        .unpack_to_dest_mode = unpack,
                        .math_approx_mode = k.value("approx", false)
                    };
                }
                desc.kernels.push_back(std::move(kernel));
            }
            return built;
        }

    }  // namespace

    std::vector<HandlePtr> run_program(
        const std::string& spec,
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes
    ) {
        return program::run_on_stream(
            inputs, out_shapes, out_dtypes, [spec](const auto& tensors) {
                // The spec holds every value but the tensors' dtypes and buffer types.
                program::Key key("user_program");
                key.add(spec);
                for (const auto& t : tensors) {
                    key.add(static_cast<int64_t>(t.dtype()));
                    key.add(static_cast<int64_t>(t.buffer()->buffer_type()));
                }
                const bool always_cached = true;
                program::launch(
                    key,
                    tensors,
                    [&] {
                        return build(json::parse(spec), tensors);
                    },
                    always_cached
                );
            }
        );
    }

}  // namespace ktt::bridge
