// aten scaled dot-product attention for tt: TT-NN flash attention via the overrideable
// SDPA backend when the shapes allow it, else the math composite.

#include <ATen/SDPBackend.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <map>
#include <memory>

#include "core/torch/contract.hpp"
#include "core/torch/constants.hpp"

namespace ktt {
    namespace {

        constexpr double kNegInf = -std::numeric_limits<double>::infinity();

        // True for bf16 [B, H, S, D] q / k / v (equal heads, D % 32 == 0), no dropout,
        // causal only with Sq == Sk and no mask, mask broadcastable to [B, H, Sq, Sk].
        bool fused_ok(
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const std::optional<at::Tensor>& mask,
            double dropout_p,
            bool is_causal
        ) {
            for (const auto* t : {&q, &k, &v}) {
                if (t->dim() != 4 || t->scalar_type() != at::kBFloat16 ||
                    t->sym_numel() == 0) {
                    return false;
                }
            }

            // Symbolic sizes (fake tensors under torch.compile) compare via guards.
            const c10::SymInt d = q.sym_size(3);
            if (dropout_p != 0.0 || d % 32 != 0 || k.sym_size(3) != d ||
                v.sym_size(3) != d) {
                return false;
            }
            if (k.sym_size(0) != q.sym_size(0) || v.sym_size(0) != q.sym_size(0) ||
                k.sym_size(1) != q.sym_size(1) || v.sym_size(1) != q.sym_size(1) ||
                v.sym_size(2) != k.sym_size(2)) {
                return false;
            }
            if (is_causal && (mask.has_value() || q.sym_size(2) != k.sym_size(2))) {
                return false;
            }
            if (mask.has_value() && mask->defined()) {
                const auto& m = *mask;
                if (m.dim() > 4) {
                    return false;
                }
                const std::vector<c10::SymInt> full{
                    q.sym_size(0), q.sym_size(1), q.sym_size(2), k.sym_size(2)
                };
                for (int64_t i = 0; i < m.dim(); ++i) {
                    const c10::SymInt s = m.sym_size(m.dim() - 1 - i);
                    if (s != 1 && s != full[3 - i]) {
                        return false;
                    }
                }
            }
            return true;
        }

        int64_t tt_fused_sdp_choice(
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const std::optional<at::Tensor>& mask,
            double dropout_p,
            bool is_causal,
            std::optional<double> /*scale*/,
            bool /*enable_gqa*/
        ) {
            // torch.nn.attention.sdpa_kernel can disable the overrideable backend.
            const bool ok =
                (at::globalContext().userEnabledOverrideableSDP() &&
                 fused_ok(q, k, v, mask, dropout_p, is_causal));
            return (
                static_cast<int64_t>(
                    ok ? at::SDPBackend::overrideable : at::SDPBackend::math
                )
            );
        }

        // An additive bf16 mask [B or 1, H or 1, Sq, Sk] from a bool (keep) or float
        // mask.
        at::Tensor additive_mask(
            const at::Tensor& mask, const at::Tensor& q, int64_t sk
        ) {
            auto m = mask;
            if (m.scalar_type() == at::kBool) {
                m = (
                    at::zeros(m.sizes(), q.options())
                        .masked_fill(m.logical_not(), kNegInf)
                );
            } else if (m.scalar_type() != at::kBFloat16) {
                m = m.to(at::kBFloat16);
            }
            while (m.dim() < 4) {
                m = m.unsqueeze(0);
            }
            return m.expand({m.size(0), m.size(1), q.size(2), sk}).contiguous();
        }

        // The cached additive causal mask [Sq, Sk]: 0 where col <= row, -inf above.
        at::Tensor causal_mask(int64_t sq, int64_t sk) {
            return device_constant(c10::str("sdpa_causal|", sq, "|", sk), [&] {
                auto r = at::arange(sq).reshape({sq, 1});
                auto c = at::arange(sk).reshape({1, sk});
                return at::zeros({sq, sk})
                    .masked_fill(c > r, kNegInf)
                    .to(at::kBFloat16);
            });
        }

        double scale_of(const at::Tensor& q, std::optional<double> scale) {
            return (
                scale.has_value() ? *scale
                                  : 1.0 / std::sqrt(static_cast<double>(q.size(3)))
            );
        }

        // True for tt-train's fused training attention: equal q / k / v shapes with S %
        // 32 == 0, the default scale, no mask beyond causal.
        bool train_ok(
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const std::optional<at::Tensor>& bias,
            std::optional<double> scale
        ) {
            const bool grads = (
                q.requires_grad() || k.requires_grad() || v.requires_grad()
            );
            const double default_scale = (
                1.0 / std::sqrt(static_cast<double>(q.size(3)))
            );
            const bool scaled = !scale || std::abs(*scale - default_scale) < 1e-12;
            return (
                grads && !bias && scaled && q.sizes() == k.sizes() &&
                q.sizes() == v.sizes() && q.size(2) % 32 == 0
            );
        }

        // Row statistics of training attention outputs, by the output's device handle.
        struct Stats {
            std::weak_ptr<bridge::Handle> out;
            bridge::HandlePtr stats;
        };

        std::map<const bridge::Handle*, Stats>& stats_table() {
            static auto* table = new std::map<const bridge::Handle*, Stats>();
            return *table;
        }

        void remember_stats(
            const bridge::HandlePtr& out, const bridge::HandlePtr& stats
        ) {
            auto& table = stats_table();
            std::erase_if(table, [](const auto& e) {
                return e.second.out.expired();
            });
            table[out.get()] = {out, stats};
        }

        bridge::HandlePtr stats_of(const bridge::HandlePtr& out) {
            auto& table = stats_table();
            auto it = table.find(out.get());
            if (it == table.end() || it->second.out.lock() != out) {
                return nullptr;
            }
            return it->second.stats;
        }

        std::tuple<
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            c10::SymInt,
            c10::SymInt,
            at::Tensor,
            at::Tensor,
            at::Tensor>
        tt_sdpa_overrideable(
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const std::optional<at::Tensor>& attn_bias,
            double dropout_p,
            bool is_causal,
            bool return_debug_mask,
            std::optional<double> scale
        ) {
            const char* op = "_scaled_dot_product_fused_attention_overrideable";
            std::optional<at::Tensor> bias;
            if (attn_bias.has_value() && attn_bias->defined()) {
                bias = attn_bias;
            }
            KTT_REQUIRE(
                fused_ok(q, k, v, bias, dropout_p, is_causal), op, "unsupported shapes"
            );
            KTT_REQUIRE(!return_debug_mask, op, "return_debug_mask is not supported");

            at::Tensor out;
            if (train_ok(q, k, v, bias, scale)) {
                auto r = (
                    bridge::sdpa_train(
                        dense(op, q), dense(op, k), dense(op, v), is_causal
                    )
                );
                remember_stats(r.out, r.stats);
                out = wrap_output(r.out, to_vector(q.sizes()), q.scalar_type());
            } else {
                bridge::HandlePtr mask;
                if (bias) {
                    mask = dense(op, additive_mask(*bias, q, k.size(2)));
                }
                auto h = (
                    bridge::sdpa(
                        dense(op, q),
                        dense(op, k),
                        dense(op, v),
                        mask,
                        is_causal,
                        scale_of(q, scale)
                    )
                );
                out = wrap_output(h, to_vector(q.sizes()), q.scalar_type());
            }

            // The backward recomputes the probabilities, so the log-sum-exp is never
            // read.
            const int64_t b = q.size(0);
            const int64_t heads = q.size(1);
            const int64_t sq = q.size(2);
            const int64_t sk = k.size(2);
            auto lse = at::empty({b, heads, sq}, q.options().dtype(at::kFloat));
            auto seed = at::empty({}, at::dtype(at::kLong));
            auto offset = at::empty({}, at::dtype(at::kLong));
            return {
                out, lse, at::Tensor(), at::Tensor(), sq, sk, seed, offset, at::Tensor()
            };
        }

        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
        tt_sdpa_overrideable_backward(
            const at::Tensor& grad,
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const at::Tensor& attn_bias,
            std::array<bool, 4> grad_input_mask,
            const at::Tensor& out,
            const at::Tensor& /*logsumexp*/,
            const at::Tensor& /*cum_seq_q*/,
            const at::Tensor& /*cum_seq_k*/,
            c10::SymInt /*max_q*/,
            c10::SymInt /*max_k*/,
            double /*dropout_p*/,
            bool is_causal,
            const at::Tensor& /*philox_seed*/,
            const at::Tensor& /*philox_offset*/,
            std::optional<double> scale
        ) {
            const char* op = (
                "_scaled_dot_product_fused_attention_overrideable_backward"
            );

            // Forward ran tt-train's training attention: its fused backward.
            if (!attn_bias.defined()) {
                auto out_handle = dense(op, out);
                if (auto stats = stats_of(out_handle)) {
                    auto g = (
                        bridge::sdpa_train_backward(
                            dense_as(op, grad, q.scalar_type()),
                            out_handle,
                            dense(op, q),
                            dense(op, k),
                            dense(op, v),
                            stats,
                            is_causal
                        )
                    );
                    const auto dtype = q.scalar_type();
                    return {
                        grad_input_mask[0]
                            ? wrap_output(g[0], to_vector(q.sizes()), dtype)
                            : at::Tensor(),
                        grad_input_mask[1]
                            ? wrap_output(g[1], to_vector(k.sizes()), dtype)
                            : at::Tensor(),
                        grad_input_mask[2]
                            ? wrap_output(g[2], to_vector(v.sizes()), dtype)
                            : at::Tensor(),
                        at::Tensor()
                    };
                }
            }
            const double s = scale_of(q, scale);

            // p = softmax(q k^T * s + mask), recomputed.
            auto scores = at::matmul(q, k.transpose(-2, -1)) * s;
            if (attn_bias.defined()) {
                scores = scores + additive_mask(attn_bias, q, k.size(2));
            }
            if (is_causal) {
                scores = scores + causal_mask(q.size(2), k.size(2));
            }
            auto p = at::softmax(scores, -1);

            // ds: softmax backward of dp = g v^T (the math path's kernel).
            auto dp = at::matmul(grad, v.transpose(-2, -1));
            auto ds = at::_softmax_backward_data(dp, p, -1, p.scalar_type());

            at::Tensor gq, gk, gv, gbias;
            if (grad_input_mask[0]) {
                gq = at::matmul(ds, k) * s;
            }
            if (grad_input_mask[1]) {
                gk = at::matmul(ds.transpose(-2, -1), q) * s;
            }
            if (grad_input_mask[2]) {
                gv = at::matmul(p.transpose(-2, -1), grad);
            }
            if (grad_input_mask[3] && attn_bias.defined()) {
                gbias = at::sum_to(ds, attn_bias.sizes()).to(attn_bias.scalar_type());
            }
            return {gq, gk, gv, gbias};
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("_fused_sdp_choice", &tt_fused_sdp_choice);
        m.impl(
            "_scaled_dot_product_fused_attention_overrideable", &tt_sdpa_overrideable
        );
        m.impl(
            "_scaled_dot_product_fused_attention_overrideable_backward",
            &tt_sdpa_overrideable_backward
        );
    }

}  // namespace ktt

namespace at::native {
    REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &ktt::tt_fused_sdp_choice);
}  // namespace at::native
