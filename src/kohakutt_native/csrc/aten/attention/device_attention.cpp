// Bridge attention: Kohaku-TT's flash attention (ktt_attention) for unmasked
// non-causal bf16 calls, TT-NN's fused scaled dot-product attention otherwise.
#include <cmath>

#include "aten/attention/ktt_attention/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "metal/ops/sdpa_bw/sdpa_bw.hpp"
#include "metal/ops/sdpa_fw/sdpa_fw.hpp"

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"

namespace ktt::bridge {
    namespace {

        // Unmasked bf16 attention on ktt_attention: causal or grouped-head calls unless
        // KTT_OWN_ATTENTION=0; every call with KTT_OWN_ATTENTION=all.
        bool own_attention(bool causal, bool grouped) {
            static const std::string mode = [] {
                const char* v = std::getenv("KTT_OWN_ATTENTION");
                return std::string(v ? v : "1");
            }();
            return mode == "all" || (mode != "0" && (causal || grouped));
        }

        // True unless KTT_ATTENTION_FP32=0 (ktt_attention with bf16 DST and
        // intermediates).
        bool own_attention_fp32() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_ATTENTION_FP32");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // The most (batch, head) pairs per TT-NN SDPA call; past it, outputs mix
        // across pairs (docs/reference/ttnn-issues.md). KTT_SDPA_MAX_UNITS overrides.
        const int64_t kMaxUnits = [] {
            const char* v = std::getenv("KTT_SDPA_MAX_UNITS");
            return v ? std::atoll(v) : int64_t{48};
        }();

        // The largest chunk in {128, 64, 32} dividing sequence length n padded to
        // tiles.
        std::size_t chunk(int64_t n) {
            const int64_t padded = (n + 31) / 32 * 32;
            for (std::size_t c : {128, 64}) {
                if (padded % static_cast<int64_t>(c) == 0) {
                    return c;
                }
            }
            return 32;
        }

        // The power-of-two chunk (32 .. limit) covering sequence length n padded to
        // tiles; TT-NN pads the last chunk.
        std::size_t covering_chunk(int64_t n, std::size_t limit) {
            const auto padded = static_cast<std::size_t>((n + 31) / 32 * 32);
            std::size_t c = 32;
            while (c < padded && c < limit) {
                c *= 2;
            }
            return c;
        }

        // L1 bytes the SDPA circular buffers take for chunks (qc, kc) at head dim d
        // (an upper estimate: double-buffered bf16 q / k / v, fp32 scores and output).
        int64_t sdpa_l1_bytes(int64_t qc, int64_t kc, int64_t d) {
            return (2 * qc * d + 4 * kc * d) * 2 + (qc * kc + 2 * qc * d) * 4;
        }

        // Full grid. Unmasked non-causal attention: q / k chunks up to 256 (measured
        // best at DiT shapes); masked / causal: chunks dividing the
        // lengths. Chunks are halved (k first while larger) until sdpa_l1_bytes fits
        // the L1 free per core (less room for TT-NN's other buffers, at most 1200
        // KiB; on the stream worker). exp approximation from `approx`.
        ttnn::operations::transformer::SDPAProgramConfig program_config(
            int64_t sq, int64_t sk, int64_t d, bool plain, bool approx
        ) {
            const int64_t budget = (
                std::min<int64_t>(1200 * 1024, detail::l1_free_per_bank() - 160 * 1024)
            );
            std::size_t qc = plain ? covering_chunk(sq, 256) : chunk(sq);
            std::size_t kc = plain ? covering_chunk(sk, 256) : chunk(sk);
            // KTT_SDPA_CHUNKS=q,k: fixed chunk sizes (experiments).
            if (const char* fixed = std::getenv("KTT_SDPA_CHUNKS")) {
                std::sscanf(fixed, "%zu,%zu", &qc, &kc);
            }
            while (sdpa_l1_bytes(qc, kc, d) > budget && (qc > 32 || kc > 32)) {
                if (kc >= qc && kc > 32) {
                    kc /= 2;
                } else {
                    qc /= 2;
                }
            }
            return ttnn::operations::transformer::SDPAProgramConfig{
                .compute_with_storage_grid_size =
                    detail::device()->compute_with_storage_grid_size(),
                .q_chunk_size = qc,
                .k_chunk_size = kc,
                .exp_approx_mode = approx
            };
        }

    }  // namespace

    HandlePtr sdpa(
        const HandlePtr& q,
        const HandlePtr& k,
        const HandlePtr& v,
        const HandlePtr& mask,
        bool is_causal,
        std::optional<double> scale
    ) {
        auto shape = q->shape;
        shape.back() = v->shape.back();
        const auto n = numerics::effective();
        // Unmasked non-causal: streaming compute (bf16 dest), which masks the keys of a
        // ragged last tile; causal and masked calls mask them with fp32 dest.
        auto config = numerics::compute_config(n);
        static const bool fp32_dest = [] {
            const char* v = std::getenv("KTT_SDPA_FP32_DEST");
            return v && std::string(v) == "1";
        }();
        config.fp32_dest_acc_en = fp32_dest || mask || is_causal;
        const auto exact_sum = numerics::compute_config(
            numerics::effective({.floor = tt::tt_metal::MathFidelity::HiFi4})
        );
        const int64_t sq = q->shape[2], sk = k->shape[2];
        const int64_t d = q->shape.back();
        const bool plain = !mask && !is_causal;
        using numerics::Attention;
        const auto mode = n.attention;
        return detail::timed_op(q->dtype, shape, [=] {
            const auto program = program_config(sq, sk, d, plain, n.approx);
            std::optional<ttnn::Tensor> m;
            if (mask) {
                m.emplace(mask->tensor);
            }
            std::optional<float> s;
            if (scale) {
                s = static_cast<float>(*scale);
            }
            // One TT-NN SDPA call.
            auto row = [&](ttnn::Tensor qt,
                           ttnn::Tensor kt,
                           ttnn::Tensor vt,
                           const std::optional<ttnn::Tensor>& mr) {
                // K centered over the sequence: each query's scores shift by one
                // constant, which softmax removes; Q / K then fit block floats with
                // less error.
                if (mode != Attention::Exact) {
                    const ttsl::SmallVector<int> seq{2};
                    const auto mean = (
                        ttnn::mean(kt, seq, true, std::nullopt, exact_sum)
                    );
                    kt = ttnn::subtract(kt, mean);
                }
                if (mode == Attention::CenteredBfp8) {
                    qt = ttnn::typecast(qt, tt::tt_metal::DataType::BFLOAT8_B);
                    kt = ttnn::typecast(kt, tt::tt_metal::DataType::BFLOAT8_B);
                }
                auto out = (
                    ttnn::transformer::scaled_dot_product_attention(
                        qt,
                        kt,
                        vt,
                        mr,
                        is_causal,
                        s,
                        std::nullopt,
                        std::nullopt,
                        program,
                        config
                    )
                );
                // The output takes Q's dtype: back to the handle's.
                return (
                    out.dtype() == detail::to_ttnn(q->dtype)
                        ? out
                        : ttnn::typecast(out, detail::to_ttnn(q->dtype))
                );
            };
            const int64_t batch = q->shape[0], heads = q->shape[1];
            const int64_t kv_heads = k->shape[1], group = heads / kv_heads;
            const auto bf16 = tt::tt_metal::DataType::BFLOAT16;
            if (own_attention(is_causal, heads != kv_heads) && !mask &&
                mode != Attention::CenteredBfp8 && q->tensor.dtype() == bf16 &&
                k->tensor.dtype() == bf16 && v->tensor.dtype() == bf16 &&
                heads % kv_heads == 0) {
                ttnn::Tensor kt = k->tensor;
                if (mode != Attention::Exact) {
                    const ttsl::SmallVector<int> seq{2};
                    kt = (
                        ttnn::subtract(
                            kt, ttnn::mean(kt, seq, true, std::nullopt, exact_sum)
                        )
                    );
                }
                const float sc = (
                    static_cast<float>(
                        scale ? *scale : 1.0 / std::sqrt(static_cast<double>(d))
                    )
                );
                auto out = (
                    ktt_attention::run(
                        q->tensor,
                        kt,
                        v->tensor,
                        sc,
                        is_causal,
                        config,
                        own_attention_fp32()
                    )
                );
                return (
                    out.dtype() == detail::to_ttnn(q->dtype)
                        ? out
                        : ttnn::typecast(out, detail::to_ttnn(q->dtype))
                );
            }
            if (batch * heads <= kMaxUnits) {
                return row(q->tensor, k->tensor, v->tensor, m);
            }
            // t[b0:b1, h0:h1] (dims of size 1 kept whole).
            auto pick = [](const ttnn::Tensor& t,
                           int64_t b0,
                           int64_t b1,
                           int64_t h0,
                           int64_t h1) {
                const auto& shape = t.logical_shape();
                const auto rank = shape.rank();
                ttsl::SmallVector<int32_t> begin(rank, 0), end(rank), step(rank, 1);
                for (size_t i = 0; i < rank; ++i) {
                    end[i] = static_cast<int32_t>(shape[i]);
                }
                if (shape[0] > 1) {
                    begin[0] = static_cast<int32_t>(b0);
                    end[0] = static_cast<int32_t>(b1);
                }
                if (shape[1] > 1) {
                    begin[1] = static_cast<int32_t>(h0);
                    end[1] = static_cast<int32_t>(h1);
                }
                return ttnn::slice(t, begin, end, step);
            };
            // Calls of whole rows (heads <= kMaxUnits) or of head ranges, each at most
            // kMaxUnits (batch, head) pairs; head ranges keep whole K / V groups.
            const int64_t rows_per = std::max<int64_t>(1, kMaxUnits / heads);
            const int64_t heads_per = (
                std::max<int64_t>(
                    group, heads <= kMaxUnits ? heads : kMaxUnits / group * group
                )
            );
            std::vector<ttnn::Tensor> rows;
            for (int64_t b = 0; b < batch; b += rows_per) {
                const int64_t b1 = std::min(batch, b + rows_per);
                std::vector<ttnn::Tensor> parts;
                for (int64_t h = 0; h < heads; h += heads_per) {
                    const int64_t h1 = std::min(heads, h + heads_per);
                    std::optional<ttnn::Tensor> mr;
                    if (m) {
                        mr = pick(*m, b, b1, h, h1);
                    }
                    parts.push_back(
                        row(pick(q->tensor, b, b1, h, h1),
                            pick(k->tensor, b, b1, h / group, h1 / group),
                            pick(v->tensor, b, b1, h / group, h1 / group),
                            mr)
                    );
                }
                rows.push_back(parts.size() == 1 ? parts[0] : ttnn::concat(parts, 1));
            }
            return rows.size() == 1 ? rows[0] : ttnn::concat(rows, 0);
        });
    }

    TrainAttention sdpa_train(
        const HandlePtr& q, const HandlePtr& k, const HandlePtr& v, bool is_causal
    ) {
        auto shape = q->shape;
        shape.back() = v->shape.back();
        auto stats_shape = q->shape;
        stats_shape.back() = 32;
        TrainAttention out{
            detail::make_handle(shape, q->dtype),
            detail::make_handle(stats_shape, DType::Float32)
        };
        detail::counters().ops++;
        detail::submit([out, q, k, v, is_causal] {
            auto r = ttml::metal::sdpa_fw(
                q->tensor,
                k->tensor,
                v->tensor,
                is_causal ? ttml::metal::AttentionMaskType::Causal
                          : ttml::metal::AttentionMaskType::None,
                std::nullopt,
                0.0F,
                true  // return_intermediates
            );
            out.out->tensor = *r[0];
            out.stats->tensor = *r[1];
            detail::check_shape(*out.out);
            detail::check_shape(*out.stats);
        });
        return out;
    }

    std::vector<HandlePtr> sdpa_train_backward(
        const HandlePtr& grad,
        const HandlePtr& out,
        const HandlePtr& q,
        const HandlePtr& k,
        const HandlePtr& v,
        const HandlePtr& stats,
        bool is_causal
    ) {
        std::vector<HandlePtr> grads{
            detail::make_handle(q->shape, q->dtype),
            detail::make_handle(k->shape, k->dtype),
            detail::make_handle(v->shape, v->dtype)
        };
        detail::counters().ops++;
        detail::submit([grads, grad, out, q, k, v, stats, is_causal] {
            auto [gq, gk, gv] = (
                ttml::metal::sdpa_bw(
                    grad->tensor,
                    out->tensor,
                    q->tensor,
                    k->tensor,
                    v->tensor,
                    stats->tensor,
                    is_causal ? ttml::metal::AttentionMaskType::Causal
                              : ttml::metal::AttentionMaskType::None,
                    std::nullopt,
                    0.0F
                )
            );
            grads[0]->tensor = gq;
            grads[1]->tensor = gk;
            grads[2]->tensor = gv;
            for (const auto& g : grads) {
                detail::check_shape(*g);
            }
        });
        return grads;
    }

    std::vector<HandlePtr> split_qkv_heads(const HandlePtr& qkv, int64_t heads) {
        const auto& s = qkv->shape;
        const int64_t b = s[0], t = s[1], d = s[2] / (3 * heads);
        std::vector<HandlePtr> out;
        for (int i = 0; i < 3; ++i) {
            out.push_back(detail::make_handle({b, heads, t, d}, qkv->dtype));
        }
        detail::counters().ops++;
        detail::submit([out, qkv, b, t, heads] {
            const ttnn::Shape shape4({
                static_cast<uint32_t>(b),
                1,
                static_cast<uint32_t>(t),
                static_cast<uint32_t>(qkv->shape[2])
            });
            auto fused = ttnn::reshape(qkv->tensor, shape4);
            auto create = [&](const ttnn::Tensor& x) {
                return ttnn::experimental::nlp_create_qkv_heads(
                    x,
                    std::nullopt,
                    static_cast<uint32_t>(heads),
                    static_cast<uint32_t>(heads),
                    false  // transpose_k_heads
                );
            };
            std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> heads_out;
            try {
                heads_out = create(fused);
            } catch (const std::exception& e) {
                if (!detail::is_l1_clash(e)) {
                    throw;
                }
                // Again with L1 activations spilled and the heads in DRAM.
                fused = ttnn::Tensor();
                detail::spill_l1();
                const detail::DramOnly dram;
                fused = ttnn::reshape(qkv->tensor, shape4);
                heads_out = create(fused);
            }
            auto [q, k, v] = heads_out;
            out[0]->tensor = q;
            out[1]->tensor = k;
            out[2]->tensor = v;
            for (const auto& h : out) {
                detail::check_shape(*h);
            }
        });
        return out;
    }

    HandlePtr split_heads(const HandlePtr& x, int64_t heads) {
        const auto& s = x->shape;
        const int64_t b = s[0], t = s[1], d = s[2] / heads;
        auto out = detail::make_handle({b, heads, t, d}, x->dtype);
        detail::counters().ops++;
        detail::submit([out, x, b, t, heads] {
            const ttnn::Shape shape4({
                static_cast<uint32_t>(b),
                1,
                static_cast<uint32_t>(t),
                static_cast<uint32_t>(x->shape[2])
            });
            auto create = [&](const ttnn::Tensor& in) {
                return std::get<0>(ttnn::experimental::nlp_create_qkv_heads(
                    in,
                    std::nullopt,
                    static_cast<uint32_t>(heads),
                    0,     // num_kv_heads: q heads only
                    false  // transpose_k_heads
                ));
            };
            auto in = ttnn::reshape(x->tensor, shape4);
            try {
                out->tensor = create(in);
            } catch (const std::exception& e) {
                if (!detail::is_l1_clash(e)) {
                    throw;
                }
                // Again with L1 activations spilled and the heads in DRAM.
                in = ttnn::Tensor();
                detail::spill_l1();
                const detail::DramOnly dram;
                in = ttnn::reshape(x->tensor, shape4);
                out->tensor = create(in);
            }
            detail::check_shape(*out);
        });
        return out;
    }

    HandlePtr rotary(const HandlePtr& x, const HandlePtr& cos, const HandlePtr& sin) {
        const int64_t bytes = x->bytes;
        return detail::timed_op(x->dtype, x->shape, [=] {
            auto out = (
                ttnn::experimental::rotary_embedding(
                    x->tensor,
                    cos->tensor,
                    sin->tensor,
                    std::nullopt,
                    detail::activation_config(bytes)
                )
            );
            // The result's sequence is padded to the tile: back to x's.
            if (out.logical_shape() == x->tensor.logical_shape()) {
                return out;
            }
            return ttnn::reshape(out, x->tensor.logical_shape(), out.padded_shape());
        });
    }

    HandlePtr concat_heads(const HandlePtr& x) {
        const auto& s = x->shape;
        const int64_t b = s[0], heads = s[1], t = s[2], d = s[3];
        return detail::timed_op(x->dtype, {b, t, heads * d}, [=] {
            auto merged = ttnn::experimental::nlp_concat_heads(x->tensor);
            return ttnn::reshape(
                merged,
                ttnn::Shape({
                    static_cast<uint32_t>(b),
                    static_cast<uint32_t>(t),
                    static_cast<uint32_t>(heads * d)
                })
            );
        });
    }

}  // namespace ktt::bridge
