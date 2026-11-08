// Host side of the ktt_binary program: the walk plan, circular buffers, whole groups
// per core, runtime args and the SFPU op defines for reader / compute / writer.
#include "aten/eltwise/ktt_binary/program.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_binary {
    namespace {

        // Smallest tensor (tiles) run here (as ktt_unary).
        constexpr int64_t kMinTiles = 512;

        // Minimum tiles per core; smaller tensors use fewer cores.
        constexpr uint32_t kMinTilesPerCore = 4;

        // Device dims right-aligned to rank `rank` (leading ones).
        std::vector<int64_t> aligned(const std::vector<int64_t>& s, size_t rank) {
            std::vector<int64_t> out(rank - s.size(), 1);
            out.insert(out.end(), s.begin(), s.end());
            return out;
        }

        int64_t product(const std::vector<int64_t>& s, size_t from, size_t to) {
            int64_t p = 1;
            for (size_t i = from; i < to; ++i) {
                p *= s[i];
            }
            return p;
        }

        // The walk for a [..., H, W] and a b broadcasting into it over a's leading
        // dims (head broadcast) or rows (row broadcast).
        std::optional<Plan> walk(
            const std::vector<int64_t>& a, const std::vector<int64_t>& b
        ) {
            const size_t rank = a.size();
            const int64_t ht = (a[rank - 2] + 31) / 32, wt = (a[rank - 1] + 31) / 32;
            const int64_t lead_a = product(a, 0, rank - 2);
            const int64_t lead_b = product(b, 0, rank - 2);
            const bool rows = b[rank - 2] == a[rank - 2];
            const bool cols = b[rank - 1] == a[rank - 1];
            Plan p;
            p.pairs = static_cast<uint32_t>(lead_a * ht * wt);
            if (rows && cols && lead_b == 1 && lead_a > 1) {
                // j over b's tiles, r over a's leading dims.
                p.nr = static_cast<uint32_t>(lead_a);
                p.sr = static_cast<uint32_t>(ht * wt);
                return p;
            }
            if (b[rank - 2] == 1 && cols && lead_b == 1) {
                // j over column tiles, r over every row tile of a; b's row 0 filled.
                p.nr = static_cast<uint32_t>(lead_a * ht);
                p.sr = static_cast<uint32_t>(wt);
                p.fill = 1;
                return p;
            }
            return std::nullopt;
        }

        // The SFPU binary op's init and call, as compute-kernel defines.
        std::optional<std::pair<std::string, std::string>> sfpu_binary(BinaryOp op) {
            switch (op) {
                case BinaryOp::Add:
                    return {{"add_binary_tile_init();", "add_binary_tile"}};
                case BinaryOp::Sub:
                    return {{"sub_binary_tile_init();", "sub_binary_tile"}};
                case BinaryOp::Mul:
                    return {{"mul_binary_tile_init();", "mul_binary_tile"}};
                case BinaryOp::Div:
                    return {{"div_binary_tile_init();", "div_binary_tile"}};
                default:
                    return std::nullopt;
            }
        }

    }  // namespace

    std::optional<Plan> plan(
        BinaryOp op,
        DType dtype,
        const std::vector<int64_t>& a,
        const std::vector<int64_t>& b
    ) {
        const bool float_dtype = dtype == DType::BFloat16 || dtype == DType::Float32;
        const size_t rank = std::max(a.size(), b.size());
        if (!program::own_kernels_enabled() || !float_dtype || !sfpu_binary(op) ||
            rank < 2) {
            return std::nullopt;
        }
        const auto ra = aligned(a, rank), rb = aligned(b, rank);
        if (program::tile_count(ra) < program::min_tiles(kMinTiles)) {
            return std::nullopt;
        }
        // Measured faster than TT-NN with equal error statistics at the DiT shapes:
        // head and row broadcasts, except bf16 division over heads.
        auto p = walk(ra, rb);
        const bool heads = p && p->fill == 0;
        if (!p || (heads && op == BinaryOp::Div && dtype == DType::BFloat16)) {
            return std::nullopt;
        }
        return p;
    }

    ttnn::Tensor run(
        BinaryOp op,
        const ttnn::Tensor& a,
        const ttnn::Tensor& b,
        const Plan& plan,
        const tt::tt_metal::MemoryConfig& out_config
    ) {
        using namespace tt::tt_metal;
        const bool fp32 = a.dtype() == DataType::FLOAT32;
        const auto format = datatype_to_dataformat_converter(a.dtype());
        const uint32_t tile_bytes = tt::tile_size(format);

        // a tiles per block (beside b in a 4-tile fp32 DST half), dividing the group
        // length: every CB transaction has one size, so none crosses the buffer end.
        const uint32_t block = plan.nr % 3 == 0 ? 3 : plan.nr % 2 == 0 ? 2 : 1;

        auto out = (
            ttnn::empty(
                a.logical_shape(), a.dtype(), Layout::TILE, detail::device(), out_config
            )
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_binary");
        key.add(a).add(b).add(out).add(static_cast<int64_t>(op));
        for (const int64_t v : {plan.pairs, plan.nr, plan.sr, plan.sj, plan.fill}) {
            key.add(v);
        }
        return program::launch(key, {a, b, out}, [&] {
            // Whole groups per core (a group: one b tile and its nr repeats).
            const uint32_t groups = plan.pairs / plan.nr;
            const uint32_t min_groups = (
                std::max<uint32_t>(1, kMinTilesPerCore / plan.nr)
            );
            const auto s = program::split_tiles(groups, min_groups);

            ProgramDescriptor desc;
            const std::pair<uint8_t, uint32_t> cbs[] = {
                {0, 4 * block}, {1, 2}, {16, 4 * block}
            };
            for (auto [index, tiles] : cbs) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = tiles * tile_bytes,
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
            const uint32_t elem = fp32 ? 4 : 2;
            const uint32_t a_addr = a.buffer()->address();
            const uint32_t b_addr = b.buffer()->address();
            const uint32_t o_addr = out.buffer()->address();
            const auto& p = plan;
            uint32_t start = 0;
            for (size_t i = 0; i < s.cores.size(); ++i) {
                const uint32_t n = s.count(i) * p.nr;
                const auto core = s.cores[i];
                reader_args.push_back({
                    core,
                    {a_addr, b_addr, start, n, p.nr, p.sr, p.sj, p.fill, block, elem}
                });
                writer_args.push_back({
                    core,
                    {o_addr, start, n, p.nr, p.sr, p.sj, block}
                });
                compute_args.push_back({core, {start, n, p.nr, block}});
                start += n;
            }
            program::mark(program::Phase::Args);

            const auto [init, call] = *sfpu_binary(op);
            std::vector<UnpackToDestMode> unpack(
                NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
            );
            if (fp32) {
                unpack[0] = unpack[1] = UnpackToDestMode::UnpackToDestFp32;
            }
            auto reader_ct = TensorAccessorArgs(*a.buffer()).get_compile_time_args();
            const auto b_ct = TensorAccessorArgs(*b.buffer()).get_compile_time_args();
            reader_ct.insert(reader_ct.end(), b_ct.begin(), b_ct.end());

            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("eltwise/ktt_binary/reader.cpp"),
                    .core_ranges = s.range,
                    .compile_time_args = reader_ct,
                    .runtime_args = reader_args,
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("eltwise/ktt_binary/writer.cpp"),
                    .core_ranges = s.range,
                    .compile_time_args = TensorAccessorArgs(*out.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = writer_args,
                    .config = WriterConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("eltwise/ktt_binary/compute.cpp"),
                    .core_ranges = s.range,
                    .defines = {{"BINARY_OP_INIT", init}, {"BINARY_OP", call}},
                    .runtime_args = compute_args,
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = MathFidelity::LoFi,  // SFPU only: no FPU
                                                              // multiply
                        .fp32_dest_acc_en = true,
                        .unpack_to_dest_mode = unpack,
                        .math_approx_mode = false  // exact: division's reciprocal
                    }
                }
            );
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0},
                    {.kernel = 0, .position = 1, .tensor = 1},
                    {.kernel = 1, .tensor = 2}
                }
            };
        });
    }

}  // namespace ktt::bridge::ktt_binary
