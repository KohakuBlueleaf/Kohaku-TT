// Bridge matmul ops: mm and linear with fp32 accumulation, per-shape program configs.
#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>

#include <tt-metalium/allocator.hpp>

#include "aten/linalg/ktt_matmul/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ktt::bridge {
    namespace {

        // True unless KTT_MATMUL_EXPLICIT=0 (TT-NN's default program configs only).
        bool explicit_configs() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_MATMUL_EXPLICIT");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // KTT_MINIMAL_MATMUL=0: [K, N] weights go to ttnn::matmul instead.
        bool minimal_matmuls() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_MINIMAL_MATMUL");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // True unless KTT_OWN_MATMUL=0 (TT-NN's matmuls for every size).
        bool own_matmuls() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_OWN_MATMUL");
                return !(v && std::string(v) == "0");
            }();
            return on && program::own_kernels_enabled();
        }

        // L1 bytes per core of Kohaku-TT's matmul for x @ w (bf16 w).
        int64_t own_l1(const HandlePtr& x, const HandlePtr& w, bool w_nk) {
            const auto kt = static_cast<uint32_t>((x->shape.back() + 31) / 32);
            const int64_t n = w_nk ? w->shape[0] : w->shape[1];
            return ktt_matmul::l1_bytes(kt, static_cast<uint32_t>((n + 31) / 32), 2048);
        }

        bool tiled(const HandlePtr& h) {
            return h->tensor.layout() == tt::tt_metal::Layout::TILE;
        }

        // Kohaku-TT's matmul for a bf16 [..., K] @ a 2-D bf16 weight: K and N up to
        // 1024, at least 4096 rows.
        // KTT_OWN_MATMUL_MIN_ROWS when set: Kohaku-TT's matmul from that many rows
        // (M) on; unset: 0 (the measured-speed rule).
        int64_t own_min_rows() {
            static const int64_t rows = [] {
                const char* v = std::getenv("KTT_OWN_MATMUL_MIN_ROWS");
                return v ? std::max<int64_t>(1, std::atoll(v)) : int64_t{0};
            }();
            return rows;
        }

        // Kohaku-TT's matmul is faster warm than TT-NN's (measured, K and N up to
        // 1024): M from 16384 rows with an output of 12.5M to 30M elements.
        bool own_faster(int64_t m, int64_t n) {
            const int64_t elements = m * n;
            return m >= 16384 && elements >= 12'500'000 && elements < 30'000'000;
        }

        bool own_fits(const HandlePtr& x, const HandlePtr& w, bool w_nk) {
            // K a multiple of 32: tile padding along K (not zero in every tensor) never
            // enters the sums.
            if (!own_matmuls() || x->dtype != DType::BFloat16 || w->shape.size() != 2 ||
                x->shape.back() % 32 != 0 || w->dtype != DType::BFloat16) {
                return false;
            }
            const auto kt = static_cast<uint32_t>((x->shape.back() + 31) / 32);
            const int64_t n = w_nk ? w->shape[0] : w->shape[1];
            const auto nt = static_cast<uint32_t>((n + 31) / 32);
            const uint32_t w_tile = w->dtype == DType::BFloat8B ? 1088 : 2048;
            // K, N up to 32 tiles, and own_faster (or the KTT_OWN_MATMUL_MIN_ROWS
            // rows).
            int64_t m = 1;
            for (size_t d = 0; d + 1 < x->shape.size(); ++d) {
                m *= x->shape[d];
            }
            return (
                kt <= 32 && nt <= 32 &&
                (own_min_rows() > 0 ? m >= own_min_rows() : own_faster(m, n)) &&
                ktt_matmul::columns_per_core(kt, nt, w_tile).has_value()
            );
        }

    }  // namespace

    ttnn::ComputeKernelConfig detail::matmul_config(int64_t k, bool fast) {
        auto n = numerics::effective();
        if (fast) {
            n.fidelity = tt::tt_metal::MathFidelity::LoFi;
        }
        return numerics::compute_config(n, k >= kPackerL1AccMinK);
    }

    namespace {
        using detail::matmul_config;

        // a @ b's device shape: a's leading dims, rows of (a or a^T), columns of (b or
        // b^T).
        std::vector<int64_t> mm_shape(
            const HandlePtr& a, const HandlePtr& b, bool transpose_a, bool transpose_b
        ) {
            auto out = a->shape;
            const size_t r = out.size();
            const auto& bs = b->shape;
            out[r - 2] = transpose_a ? a->shape[r - 1] : a->shape[r - 2];
            out[r - 1] = transpose_b ? bs[bs.size() - 2] : bs[bs.size() - 1];
            return out;
        }

        // x @ weight^T's device shape: x with its last dim replaced by weight's rows.
        std::vector<int64_t> linear_shape(const HandlePtr& x, const HandlePtr& weight) {
            auto out = x->shape;
            out.back() = weight->shape[weight->shape.size() - 2];
            return out;
        }

        // The result dtype of a matmul with first operand `a`: requested, else a's
        // (bf16 for a block-float a).
        DType result_dtype(const HandlePtr& a, std::optional<DType> out) {
            if (out) {
                return *out;
            }
            return is_block(a->dtype) ? DType::BFloat16 : a->dtype;
        }

        // A full-grid 2-D multicast config for [m, k] @ [k, n] (largest K block within
        // L1,
        // <= 4-tile sub-blocks); nullopt (TT-NN's default) when it does not cover the
        // grid.
        std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_for(
            int64_t m, int64_t k, int64_t n, bool bias, int64_t budget
        ) {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const int64_t gx = grid.x, gy = grid.y;
            const int64_t mt = (m + 31) / 32, kt = (k + 31) / 32, nt = (n + 31) / 32;
            if (!explicit_configs() || mt < gy || nt < gx || k % 32 != 0) {
                return std::nullopt;
            }
            const int64_t pm = (mt + gy - 1) / gy, pn = (nt + gx - 1) / gx;

            // L1 per core: double-buffered in0 / in1 blocks (2 KiB tiles), fp32 output
            // and intermediate (4 KiB tiles), bias row; within `budget` and 1000 KiB.
            const int64_t kBudget = std::min<int64_t>(budget, 1000 * 1024);
            int64_t kb = 0;
            // K blocks of 1 tile measured slower than TT-NN's default: not used.
            for (int64_t cand : {8, 4, 2}) {
                const int64_t bytes =
                    (2 * (pm + pn) * cand * 2048 + pm * pn * 4096 +
                     (bias ? pn * 2048 : 0));
                if (kt % cand == 0 && bytes <= kBudget) {
                    kb = cand;
                    break;
                }
            }
            if (kb == 0) {
                return std::nullopt;
            }
            int64_t sh = 1, sw = 1;
            for (int64_t h : {4, 2, 1}) {
                if (pm % h == 0) {
                    sh = h;
                    break;
                }
            }
            for (int64_t w : {4, 2, 1}) {
                if (pn % w == 0 && sh * w <= 4) {
                    sw = w;
                    break;
                }
            }
            return ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = grid,
                .in0_block_w = static_cast<std::size_t>(kb),
                .out_subblock_h = static_cast<std::size_t>(sh),
                .out_subblock_w = static_cast<std::size_t>(sw),
                .out_block_h = static_cast<std::size_t>(pm),
                .out_block_w = static_cast<std::size_t>(pn),
                .per_core_M = static_cast<std::size_t>(pm),
                .per_core_N = static_cast<std::size_t>(pn),
                .transpose_mcast = false,
                .fused_activation = std::nullopt,
                .fuse_batch = true
            };
        }

        // A matmul output's memory config (the activation policy) and the L1 per core
        // left for its program's circular buffers (on the stream worker).
        std::pair<tt::tt_metal::MemoryConfig, int64_t> output_placement(
            const std::vector<int64_t>& shape, DType dtype
        ) {
            const int64_t bytes = detail::padded_bytes(shape, dtype);
            auto mc = detail::activation_config(bytes);
            int64_t free = detail::l1_free_per_bank();
            if (mc.buffer_type() == tt::tt_metal::BufferType::L1) {
                const auto banks = (
                    detail::device()->allocator()->get_num_banks(
                        tt::tt_metal::BufferType::L1
                    )
                );
                free -= bytes / static_cast<int64_t>(banks) + 4096;
            }
            constexpr int64_t kMargin = 64 * 1024;  // TT-NN's other per-core buffers
            return {mc, free - kMargin};
        }

        // Rows of a as a matmul's M (all leading dims folded).
        int64_t rows(const HandlePtr& a) {
            int64_t r = 1;
            for (size_t d = 0; d + 1 < a->shape.size(); ++d) {
                r *= a->shape[d];
            }
            return r;
        }

        // True when b has no batch (all leading dims 1): a plain weight matrix.
        bool unbatched(const HandlePtr& b) {
            for (size_t d = 0; d + 2 < b->shape.size(); ++d) {
                if (b->shape[d] != 1) {
                    return false;
                }
            }
            return true;
        }

        // TT-NN's minimal matmul: x [..., K] @ w [K, N] (+ bias [N]).
        ttnn::Tensor minimal(
            const ttnn::Tensor& x,
            const ttnn::Tensor& w,
            const HandlePtr& bias,
            const tt::tt_metal::MemoryConfig& mc,
            DType dtype,
            const ttnn::ComputeKernelConfig& config
        ) {
            std::optional<ttnn::Tensor> b;
            if (bias) {
                b.emplace(bias->tensor);
            }
            return (
                ttnn::experimental::minimal_matmul(
                    x,
                    w,
                    b,
                    std::nullopt,
                    std::nullopt,
                    mc,
                    detail::to_ttnn(dtype),
                    config
                )
            );
        }

    }  // namespace

    HandlePtr mm(
        const HandlePtr& a,
        const HandlePtr& b,
        bool transpose_a,
        bool transpose_b,
        std::optional<DType> out,
        bool fast
    ) {
        const DType dtype = result_dtype(a, out);
        const int64_t k = transpose_a ? a->shape[a->shape.size() - 2] : a->shape.back();
        const auto shape = mm_shape(a, b, transpose_a, transpose_b);
        // Zero-size operands: an empty result, or zeros when only K is 0.
        int64_t elements = 1;
        for (int64_t s : shape) {
            elements *= s;
        }
        if (elements == 0) {
            return empty(shape, dtype);
        }
        if (k == 0) {
            return full(shape, 0.0, dtype);
        }
        const bool plain = !transpose_a && unbatched(b);
        const int64_t m = rows(a);
        const auto config = matmul_config(k, fast);
        const bool own =
            (plain && !transpose_a && dtype == DType::BFloat16 &&
             b->shape.size() == 2 && own_fits(a, b, transpose_b));
        return detail::timed_op(dtype, shape, [=] {
            const auto [mc, budget] = output_placement(shape, dtype);
            if (own && tiled(a) && tiled(b) && own_l1(a, b, transpose_b) <= budget) {
                return ktt_matmul::run(a->tensor, b->tensor, transpose_b, mc, config);
            }
            const bool low =
                (a->dtype != DType::Float32 && b->dtype != DType::Float32 &&
                 dtype != DType::Float32);
            if (plain && !transpose_b && low && minimal_matmuls()) {
                return minimal(a->tensor, b->tensor, nullptr, mc, dtype, config);
            }
            const auto program =
                (plain ? program_for(m, k, shape.back(), false, budget) : std::nullopt);
            return (
                ttnn::matmul(
                    a->tensor,
                    b->tensor,
                    transpose_a,
                    transpose_b,
                    mc,
                    detail::to_ttnn(dtype),
                    program,
                    std::nullopt,
                    config
                )
            );
        });
    }

    HandlePtr linear(
        const HandlePtr& x, const HandlePtr& weight, const HandlePtr& bias, bool kn
    ) {
        const DType dtype = result_dtype(x, std::nullopt);
        auto shape = x->shape;
        shape.back() = kn ? weight->shape.back()
                          : weight->shape[weight->shape.size() - 2];
        const int64_t m = rows(x), k = x->shape.back();
        const auto config = matmul_config(k, false);
        const bool own = dtype == DType::BFloat16 && own_fits(x, weight, !kn);
        return detail::timed_op(dtype, shape, [=] {
            const auto [mc, budget] = output_placement(shape, dtype);
            if (own && tiled(x) && tiled(weight) && own_l1(x, weight, !kn) <= budget) {
                auto out = ktt_matmul::run(x->tensor, weight->tensor, !kn, mc, config);
                return bias ? ttnn::add(out, bias->tensor, std::nullopt, mc) : out;
            }
            if (kn && minimal_matmuls()) {
                return minimal(x->tensor, weight->tensor, bias, mc, dtype, config);
            }
            const auto program = program_for(m, k, shape.back(), !!bias, budget);
            std::optional<const ttnn::Tensor> b;
            if (bias) {
                b.emplace(bias->tensor);
            }
            // TT-NN's default config misbroadcasts a bias (rank > 2 input, or K not a
            // multiple of the tile): the bias is added after the matmul.
            const bool separate = !program && bias;
            auto out = ttnn::linear(
                x->tensor,
                weight->tensor,
                separate ? std::nullopt : b,
                false,  // transpose_a
                !kn,    // transpose_b
                mc,
                detail::to_ttnn(dtype),
                program,
                std::nullopt,
                config
            );
            if (!separate) {
                return out;
            }
            return ttnn::add(out, bias->tensor, std::nullopt, mc);
        });
    }

}  // namespace ktt::bridge
