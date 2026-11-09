// Host side of ktt_attention: units (batch-head, q chunk) split over the grid in
// contiguous ranges, chunk sizes within the L1 budget, circular buffers, runtime args.
#include "aten/attention/ktt_attention/program.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_attention {
    namespace {

        constexpr uint32_t kTile = 2048;  // bf16 tile bytes

        // L1 bytes of the circular buffers for chunks (qc, kc) with intermediates of
        // `im` bytes per tile.
        uint32_t cb_bytes(
            uint32_t qc, uint32_t kc, uint32_t dt, uint32_t dvt, uint32_t im
        ) {
            const uint32_t inputs = (qc * dt + 2 * kc * dt + 2 * kc * dvt + 4) * kTile;
            const uint32_t scores = qc * kc * (im + kTile);
            const uint32_t stats = 6 * qc * im + 2 * qc * dvt * im;
            return inputs + scores + stats + 2 * dvt * kTile;
        }

    }  // namespace

    ttnn::Tensor run(
        const ttnn::Tensor& q,
        const ttnn::Tensor& k,
        const ttnn::Tensor& v,
        float scale,
        bool causal,
        const ttnn::ComputeKernelConfig& config,
        bool fp32_dest
    ) {
        using namespace tt::tt_metal;
        const auto& qs = q.logical_shape();
        const auto& ks = k.logical_shape();
        const auto& vs = v.logical_shape();
        const uint32_t b = qs[0], heads = qs[1], sq = qs[2], d = qs[3];
        const uint32_t kv_heads = ks[1], sk = ks[2], dv = vs[3];
        const uint32_t sqt = (sq + 31) / 32, skt = (sk + 31) / 32;
        const uint32_t dt = (d + 31) / 32, dvt = (dv + 31) / 32;
        const uint32_t im = fp32_dest ? 4096 : kTile;

        // Chunks: q 4 / 2 / 1 tiles, k 8 / 4 / 2 / 1 tiles, the largest within budget.
        const uint32_t budget = static_cast<uint32_t>(
            std::clamp<int64_t>(detail::l1_free_per_bank() - 96 * 1024, 0, 1100 * 1024)
        );
        uint32_t qc = std::min<uint32_t>(4, sqt), kc = std::min<uint32_t>(8, skt);
        while (cb_bytes(qc, kc, dt, dvt, im) > budget && (qc > 1 || kc > 1)) {
            if (kc >= 2 * qc || qc == 1) {
                kc = std::max<uint32_t>(1, kc / 2);
            } else {
                qc = std::max<uint32_t>(1, qc / 2);
            }
        }
        if (cb_bytes(qc, kc, dt, dvt, im) > budget) {
            throw std::runtime_error(
                "kohakutt_native: ktt_attention: the smallest chunk does not fit the free L1 span (" +
                std::to_string(budget) + " bytes per core)"
            );
        }

        auto out = ttnn::empty(
            ttnn::Shape({b, heads, sq, dv}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_attention");
        key.add(q).add(k).add(v).add(out).add(qc).add(kc).add(causal).add(fp32_dest);
        key.add(std::bit_cast<uint32_t>(scale))
            .add(static_cast<int64_t>(config.math_fidelity));
        return program::launch(key, {q, k, v, out}, [&] {
            const uint32_t q_chunks = (sqt + qc - 1) / qc;
            const uint32_t units = b * heads * q_chunks;
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = std::min<uint32_t>(grid.x * grid.y, units);
            std::vector<CoreRange> ranges;
            for (uint32_t c = 0; c < cores; ++c) {
                const CoreCoord core(c % grid.x, c / grid.x);
                ranges.emplace_back(core, core);
            }
            const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            const auto im_df = fp32_dest ? tt::DataFormat::Float32 : bf16;

            struct Cb {
                uint8_t index;
                uint32_t tiles;
                tt::DataFormat df;
            };

            const Cb cbs[] = {
                {0, qc * dt, bf16},
                {1, 2 * kc * dt, bf16},
                {2, 2 * kc * dvt, bf16},
                {3, 3, bf16},
                {4, 1, bf16},
                {5, qc * kc, im_df},
                {6, qc * kc, bf16},
                {7, qc, im_df},
                {8, qc, im_df},
                {9, qc, im_df},
                {10, qc, im_df},
                {11, qc, im_df},
                {12, qc * dvt, im_df},
                {13, qc * dvt, im_df},
                {14, qc, im_df},
                {16, 2 * dvt, bf16},
            };
            for (const auto& cb : cbs) {
                const uint32_t page = cb.df == tt::DataFormat::Float32 ? 4096 : kTile;
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = cb.tiles * page,
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = cb.index,
                            .data_format = cb.df,
                            .page_size = page
                        }}
                    }
                );
            }

            const uint32_t q_valid = sq - (sqt - 1) * 32, k_valid = sk - (skt - 1) * 32;
            const uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
            // Per core (every kernel): the first unit; core c runs units c, c +
            // cores, c + 2 cores, ...
            KernelDescriptor::RuntimeArgs unit_args;
            for (uint32_t c = 0; c < cores; ++c) {
                unit_args.push_back({CoreCoord(c % grid.x, c / grid.x), {c}});
            }
            program::mark(program::Phase::Args);
            const uint32_t causal_arg = causal ? 1u : 0u;

            auto reader_ct = TensorAccessorArgs(*q.buffer()).get_compile_time_args();
            for (const auto* t : {&k, &v}) {
                const auto more = (
                    TensorAccessorArgs(*t->buffer()).get_compile_time_args()
                );
                reader_ct.insert(reader_ct.end(), more.begin(), more.end());
            }
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "attention/ktt_attention/reader.cpp"
                    ),
                    .core_ranges = range,
                    .compile_time_args = reader_ct,
                    .runtime_args = unit_args,
                    .common_runtime_args =
                        {
                            q.buffer()->address(),
                            k.buffer()->address(),
                            v.buffer()->address(),
                            cores,
                            units,
                            heads,
                            heads / kv_heads,
                            sqt,
                            skt,
                            dt,
                            dvt,
                            qc,
                            kc,
                            q_valid,
                            k_valid,
                            causal_arg
                        },
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "attention/ktt_attention/writer.cpp"
                    ),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*out.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = unit_args,
                    .common_runtime_args =
                        {out.buffer()->address(), cores, units, sqt, dvt, qc},
                    .config = WriterConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source(
                        "attention/ktt_attention/compute.cpp"
                    ),
                    .core_ranges = range,
                    .runtime_args = unit_args,
                    .common_runtime_args =
                        {
                            cores,
                            units,
                            sqt,
                            skt,
                            dt,
                            dvt,
                            qc,
                            kc,
                            k_valid,
                            scale_bits,
                            causal_arg
                        },
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = config.math_fidelity,
                        .fp32_dest_acc_en = fp32_dest,
                        .math_approx_mode = false
                    }
                }
            );
            constexpr auto common = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {
                    {.kernel = 0, .entry = common},
                    {.kernel = 0, .entry = common, .position = 1, .tensor = 1},
                    {.kernel = 0, .entry = common, .position = 2, .tensor = 2},
                    {.kernel = 1, .entry = common, .tensor = 3}
                }
            };
        });
    }

}  // namespace ktt::bridge::ktt_attention
