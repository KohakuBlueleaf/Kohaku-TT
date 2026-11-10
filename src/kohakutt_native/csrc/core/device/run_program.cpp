// User programs: a JSON description from kohakutt.kernels or kohakutt.dsl.metal, run
// as a ProgramDescriptor by ttnn::generic_op on the default stream.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
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
            if (name == "uint32") {
                return DataType::UINT32;
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

        // `value` as a runtime argument word.
        uint32_t word(int64_t value) {
            if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument(
                    "kohakutt.kernels: value " + std::to_string(value) +
                    " does not fit a runtime argument"
                );
            }
            return static_cast<uint32_t>(value);
        }

        // An argument list: ints, {"addr": t} (t's address; `addrs` gets its position),
        // {"accessor": t}, {"value": i}, {"noc": [x, y, axis]} (a worker's NoC x / y).
        std::vector<uint32_t> arguments(
            const json& items,
            const std::vector<ttnn::Tensor>& tensors,
            const std::vector<int64_t>& values,
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
                } else if (item.contains("value")) {
                    out.push_back(word(values.at(item["value"].get<size_t>())));
                } else if (item.contains("noc")) {
                    const auto& at = item["noc"];
                    const auto x = at.at(0).get<uint32_t>(),
                               y = at.at(1).get<uint32_t>();
                    const auto axis = at.at(2).get<uint32_t>();
                    const auto grid = detail::device()
                                          ->compute_with_storage_grid_size();
                    if (axis > 1 || x >= grid.x || y >= grid.y) {
                        throw std::invalid_argument(
                            "kohakutt.kernels: bad core argument " + item.dump()
                        );
                    }
                    const auto noc = detail::device()->worker_core_from_logical_core(
                        CoreCoord(x, y)
                    );
                    out.push_back(static_cast<uint32_t>(axis == 0 ? noc.x : noc.y));
                } else {
                    throw std::invalid_argument(
                        "kohakutt.kernels: bad argument " + item.dump()
                    );
                }
            }
            return out;
        }

        // Buffer `cb`: "tiles" pages (of "page_bytes", else a tile) in a tensor's or a
        // dtype's format, or "bytes" of raw L1 as one page.
        CBDescriptor buffer(const json& cb, const std::vector<ttnn::Tensor>& tensors) {
            const auto index = cb.at("index").get<uint8_t>();
            if (cb.contains("page_bytes")) {
                const auto page = cb["page_bytes"].get<uint32_t>();
                const auto tiles = uint64_t{cb.at("tiles").get<uint32_t>()};
                if (page == 0 || page % 16 != 0 ||
                    tiles * page > std::numeric_limits<uint32_t>::max()) {
                    throw std::invalid_argument(
                        "kohakutt.kernels: bad buffer page size " + cb.dump()
                    );
                }
                const auto format = datatype_to_dataformat_converter(
                    data_type(cb.at("dtype").get<std::string>())
                );
                return CBDescriptor{
                    .total_size = cb.at("tiles").get<uint32_t>() * page,
                    .core_ranges = core_set(cb.at("cores")),
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index, .data_format = format, .page_size = page
                    }}
                };
            }
            if (cb.contains("bytes")) {
                const auto bytes = cb["bytes"].get<uint32_t>();
                return CBDescriptor{
                    .total_size = bytes,
                    .core_ranges = core_set(cb.at("cores")),
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index,
                        .data_format = tt::DataFormat::UInt8,
                        .page_size = bytes
                    }}
                };
            }
            const auto dtype =
                (cb.contains("tensor") ? tensors.at(cb["tensor"].get<size_t>()).dtype()
                                       : data_type(cb.at("dtype").get<std::string>()));
            const auto format = datatype_to_dataformat_converter(dtype);
            const uint32_t page = tt::tile_size(format);
            return CBDescriptor{
                .total_size = cb.at("tiles").get<uint32_t>() * page,
                .core_ranges = core_set(cb.at("cores")),
                .format_descriptors = {CBFormatDescriptor{
                    .buffer_index = index, .data_format = format, .page_size = page
                }}
            };
        }

        // The config of kernel entry `k`: reader / writer (pinned to "noc" when
        // given) or compute.
        KernelDescriptor::ConfigDescriptor config(const json& k) {
            const auto type = k.at("type").get<std::string>();
            if (type == "reader" || type == "writer") {
                if (!k.contains("noc")) {
                    if (type == "reader") {
                        return ReaderConfigDescriptor{};
                    }
                    return WriterConfigDescriptor{};
                }
                return DataMovementConfigDescriptor{
                    .processor =
                        (type == "reader" ? DataMovementProcessor::RISCV_1
                                          : DataMovementProcessor::RISCV_0),
                    .noc = k["noc"].get<int>() == 0 ? NOC::NOC_0 : NOC::NOC_1
                };
            }
            std::vector<UnpackToDestMode> unpack(
                NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
            );
            for (const auto& index : k.value("unpack_fp32", json::array())) {
                unpack.at(index.get<size_t>()) = UnpackToDestMode::UnpackToDestFp32;
            }
            return ComputeConfigDescriptor{
                .math_fidelity = fidelity(k.value("fidelity", "HiFi2")),
                .fp32_dest_acc_en = k.value("fp32_dest", true),
                .unpack_to_dest_mode = unpack,
                .math_approx_mode = k.value("approx", false)
            };
        }

        // The program of `spec` over `tensors` (inputs, then outputs) with `values`
        // in its {"value": i} arguments.
        Built build(
            const json& spec,
            const std::vector<ttnn::Tensor>& tensors,
            const std::vector<int64_t>& values
        ) {
            if (spec.value("interleaved", false)) {
                for (const auto& t : tensors) {
                    if (t.memory_config().is_sharded()) {
                        throw std::invalid_argument(
                            "kohakutt.dsl.metal: sharded tensors are not supported; "
                            "pass interleaved tensors"
                        );
                    }
                }
            }
            Built built;
            auto& desc = built.desc;
            for (const auto& cb : spec.at("cbs")) {
                desc.cbs.push_back(buffer(cb, tensors));
            }
            for (const auto& s : spec.value("semaphores", json::array())) {
                desc.semaphores.push_back(
                    SemaphoreDescriptor{
                        .id = s.at("id").get<uint32_t>(),
                        .core_ranges = core_set(s.at("cores")),
                        .initial_value = s.value("initial", 0u)
                    }
                );
            }
            for (const auto& k : spec.at("kernels")) {
                const auto index = static_cast<uint32_t>(desc.kernels.size());
                KernelDescriptor kernel{
                    .kernel_source = k.at("source").get<std::string>(),
                    .core_ranges = core_set(k.at("cores")),
                    .compile_time_args = arguments(
                        k.value("compile_args", json::array()), tensors, values
                    )
                };
                for (const auto& entry : k.value("runtime_args", json::array())) {
                    const auto& core = entry.at(0);
                    std::vector<std::pair<size_t, size_t>> addrs;
                    kernel.runtime_args.push_back(
                        {CoreCoord(
                             core.at(0).get<uint32_t>(), core.at(1).get<uint32_t>()
                         ),
                         arguments(entry.at(1), tensors, values, &addrs)}
                    );
                    for (const auto& [position, tensor] : addrs) {
                        built.slots.push_back(
                            {.kernel = index,
                             .entry = static_cast<int32_t>(
                                 kernel.runtime_args.size() - 1
                             ),
                             .position = static_cast<uint32_t>(position),
                             .tensor = static_cast<uint32_t>(tensor)}
                        );
                    }
                }
                std::vector<std::pair<size_t, size_t>> common;
                kernel.common_runtime_args = arguments(
                    k.value("common_args", json::array()), tensors, values, &common
                );
                for (const auto& [position, tensor] : common) {
                    built.slots.push_back(
                        {.kernel = index,
                         .entry = Slot::kCommon,
                         .position = static_cast<uint32_t>(position),
                         .tensor = static_cast<uint32_t>(tensor)}
                    );
                }
                // A named local: items() refers into the json it iterates.
                const json defines = k.value("defines", json::object());
                for (const auto& [name, value] : defines.items()) {
                    kernel.defines.emplace_back(name, value.get<std::string>());
                }
                kernel.config = config(k);
                if (k.contains("args")) {
                    built.args.push_back(
                        {.kernel = index,
                         .per_core = k["args"].at("per_core").get<uint32_t>(),
                         .common = k["args"].at("common").get<uint32_t>()}
                    );
                }
                desc.kernels.push_back(std::move(kernel));
            }
            for (const auto& r : spec.value("rings", json::array())) {
                built.rings.push_back(
                    {.cb = r.at("cb").get<uint32_t>(),
                     .pushes = r.value("pushes", std::vector<uint32_t>{}),
                     .written = r.value("written", uint64_t{0})}
                );
            }
            return built;
        }

    }  // namespace

    struct UserProgram {
        // Where a {"value": i} argument sits: (kernel, position, i), common or in
        // runtime_args entry `entry`.
        struct Slot {
            uint32_t kernel = 0, position = 0, index = 0;
            int32_t entry = program::Value::kCommon;
        };

        // A circular buffer: `bytes`, or (`tensor` >= 0) `tiles` tiles of that io
        // tensor's format; `group` numbers its distinct core ranges.
        struct Buffer {
            uint64_t bytes = 0, tiles = 0;
            int64_t tensor = -1;
            size_t group = 0;
        };

        std::string spec;
        std::vector<Slot> slots;
        // Names the spec in program keys.
        int64_t id = 0;
        // The spec's "name", for messages.
        std::string name;
        std::vector<Buffer> buffers;
        size_t groups = 0;
    };

    namespace {

        // The buffers of `cbs` (spec entries) as UserProgram::Buffer, numbering
        // their core ranges in `groups`.
        std::vector<UserProgram::Buffer> buffer_sizes(const json& cbs, size_t& groups) {
            std::vector<UserProgram::Buffer> out;
            std::unordered_map<std::string, size_t> ranges;
            for (const auto& cb : cbs) {
                UserProgram::Buffer b;
                const auto at = ranges.emplace(cb.at("cores").dump(), ranges.size());
                b.group = at.first->second;
                if (cb.contains("page_bytes")) {
                    b.bytes =
                        (cb.at("tiles").get<uint64_t>() *
                         cb["page_bytes"].get<uint64_t>());
                } else if (cb.contains("bytes")) {
                    b.bytes = cb["bytes"].get<uint64_t>();
                } else if (cb.contains("tensor")) {
                    b.tiles = cb.at("tiles").get<uint64_t>();
                    b.tensor = cb["tensor"].get<int64_t>();
                } else {
                    const auto format = tt::tt_metal::datatype_to_dataformat_converter(
                        data_type(cb.at("dtype").get<std::string>())
                    );
                    b.bytes = cb.at("tiles").get<uint64_t>() * tt::tile_size(format);
                }
                out.push_back(b);
            }
            groups = ranges.size();
            return out;
        }

        // Throws when the buffers of `user`, placed as TT-Metal places them, end above
        // the lowest live L1 buffer or the end of the L1 bank.
        void check_l1(
            const UserProgram& user, const std::vector<ttnn::Tensor>& tensors
        ) {
            using tt::tt_metal::datatype_to_dataformat_converter;
            if (user.buffers.empty()) {
                return;
            }
            const auto& device = detail::device();
            const auto& allocator = device->allocator();
            const uint64_t align = allocator->get_alignment(
                tt::tt_metal::BufferType::DRAM
            );
            std::vector<uint64_t> ends(user.groups, 0);
            for (const auto& b : user.buffers) {
                uint64_t bytes = b.bytes;
                if (b.tensor >= 0) {
                    const auto dtype = tensors.at(b.tensor).dtype();
                    bytes = b.tiles *
                        tt::tile_size(datatype_to_dataformat_converter(dtype));
                }
                auto& end = ends.at(b.group);
                end = (end + align - 1) / align * align + bytes;
            }
            const uint64_t need = *std::max_element(ends.begin(), ends.end());
            const uint64_t base = allocator->get_base_allocator_addr(
                tt::tt_metal::HalMemType::L1
            );
            // The L1 bank's end: TT-Metal checks against the end of L1, past L1_SMALL.
            uint64_t top = base +
                allocator->get_bank_size(tt::tt_metal::BufferType::L1);
            const auto lowest = device->lowest_occupied_compute_l1_address();
            if (lowest) {
                top = std::min<uint64_t>(top, *lowest);
            }
            const uint64_t free = top > base ? top - base : 0;
            if (need > free) {
                throw std::invalid_argument(
                    "user program " + user.name +
                    ": this launch's circular buffers and scratch take " +
                    std::to_string(need) + " bytes of L1 per core, " +
                    std::to_string(free) + " are free" +
                    (lowest ? " below the lowest live L1 buffer" : "") +
                    "; nothing was launched"
                );
            }
        }

    }  // namespace

    UserProgramPtr user_program(const std::string& spec) {
        static std::mutex mutex;
        static std::unordered_map<std::string, UserProgramPtr> cache;
        static int64_t next_id = 0;
        constexpr size_t kMaxCached = 4096;
        const std::lock_guard lock(mutex);
        if (const auto it = cache.find(spec); it != cache.end()) {
            return it->second;
        }
        auto program = std::make_shared<UserProgram>();
        program->spec = spec;
        program->id = next_id++;
        const json parsed = json::parse(spec);
        // Unnamed (kohakutt.kernels): named by its first kernel's source.
        const auto& kernels = parsed.at("kernels");
        program->name = parsed.value(
            "name",
            kernels.empty() ? std::string()
                            : kernels.at(0).value("source", std::string())
        );
        program->buffers = buffer_sizes(parsed.at("cbs"), program->groups);
        uint32_t kernel = 0;
        for (const auto& k : parsed.at("kernels")) {
            uint32_t position = 0;
            for (const auto& item : k.value("common_args", json::array())) {
                if (item.is_object() && item.contains("value")) {
                    program->slots.push_back(
                        {kernel, position, item["value"].get<uint32_t>()}
                    );
                }
                ++position;
            }
            int32_t entry = 0;
            for (const auto& e : k.value("runtime_args", json::array())) {
                uint32_t at = 0;
                for (const auto& item : e.at(1)) {
                    if (item.is_object() && item.contains("value")) {
                        program->slots.push_back(
                            {kernel, at, item["value"].get<uint32_t>(), entry}
                        );
                    }
                    ++at;
                }
                ++entry;
            }
            ++kernel;
        }
        if (cache.size() >= kMaxCached) {
            cache.clear();
        }
        cache.emplace(spec, program);
        return program;
    }

    std::vector<HandlePtr> run_program(
        const UserProgramPtr& user,
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        const std::vector<int64_t>& values
    ) {
        std::vector<program::Value> patches;
        if (!values.empty()) {
            patches.reserve(user->slots.size());
            for (const auto& slot : user->slots) {
                patches.push_back(
                    {slot.kernel,
                     slot.position,
                     word(values.at(slot.index)),
                     slot.entry}
                );
            }
        }
        return program::run_on_stream(
            inputs,
            out_shapes,
            out_dtypes,
            [user, values, patches = std::move(patches)](const auto& tensors) {
                // Every launch, a cached one and a replayed torch op included.
                check_l1(*user, tensors);
                // The spec holds every value but the tensors' dtypes and buffer types
                // and the per-call `values`.
                program::Key key("user_program");
                key.add(user->id);
                for (const auto& t : tensors) {
                    key.add(static_cast<int64_t>(t.dtype()));
                    key.add(static_cast<int64_t>(t.buffer()->buffer_type()));
                }
                const bool always_cached = true;
                program::launch(
                    key,
                    tensors,
                    [&] {
                        return build(json::parse(user->spec), tensors, values);
                    },
                    always_cached,
                    patches
                );
            }
        );
    }

    std::vector<HandlePtr> run_program(
        const std::string& spec,
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        const std::vector<int64_t>& values
    ) {
        return run_program(user_program(spec), inputs, out_shapes, out_dtypes, values);
    }

}  // namespace ktt::bridge
