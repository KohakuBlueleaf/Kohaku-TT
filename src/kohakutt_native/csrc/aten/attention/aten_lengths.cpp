// ktt ops of attention with per-batch key lengths (kohakutt.sdpa(..., key_lengths=)):
// the forward on q [B, H, Sq, D], k / v [B, Hk, Sk, D / Dv] and its backward.
#include <torch/library.h>

#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <tuple>

#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"

namespace ktt {
    namespace {

        // Row statistics of the forwards run with `stats`, by the output's handle.
        struct Stats {
            std::weak_ptr<bridge::Handle> out;
            bridge::HandlePtr stats;
        };

        std::map<const bridge::Handle*, Stats>& stats_table() {
            static auto* table = new std::map<const bridge::Handle*, Stats>();
            return *table;
        }

        void check(
            const char* op,
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const at::Tensor& lengths
        ) {
            for (const auto* t : {&q, &k, &v}) {
                KTT_REQUIRE(
                    t->dim() == 4 && t->scalar_type() == at::kBFloat16 &&
                        t->numel() > 0,
                    op,
                    "needs non-empty bf16 [batch, heads, sequence, head dim] tensors"
                );
                KTT_REQUIRE(
                    !mesh::placed_base(*t),
                    op,
                    "key lengths do not take mesh-placed q / k / v yet"
                );
            }
            KTT_REQUIRE(
                k.size(0) == q.size(0) && v.size(0) == q.size(0) &&
                    k.size(1) == v.size(1) && q.size(1) % k.size(1) == 0 &&
                    k.size(2) == v.size(2) && k.size(3) == q.size(3),
                op,
                "q / k / v shapes do not match"
            );
            KTT_REQUIRE(
                lengths.dim() == 1 && lengths.size(0) == q.size(0) &&
                    lengths.scalar_type() == at::kInt && !mesh::placed_base(lengths),
                op,
                "key_lengths must be int32 [batch] on the device, not mesh-placed"
            );
        }

        double scale_of(const at::Tensor& q, std::optional<double> scale) {
            return scale ? *scale : 1.0 / std::sqrt(static_cast<double>(q.size(3)));
        }

        // softmax(q k^T scale) v, batch b's queries seeing its keys below lengths[b];
        // `stats`: keep the row statistics for _sdpa_lengths_backward.
        at::Tensor sdpa_lengths(
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const at::Tensor& lengths,
            bool is_causal,
            std::optional<double> scale,
            bool stats
        ) {
            const char* op = "_sdpa_lengths";
            check(op, q, k, v, lengths);
            auto r = bridge::sdpa_lengths(
                dense(op, q),
                dense(op, k),
                dense(op, v),
                dense(op, lengths),
                is_causal,
                scale_of(q, scale),
                stats
            );
            if (stats) {
                auto& table = stats_table();
                std::erase_if(table, [](const auto& e) {
                    return e.second.out.expired();
                });
                table[r.out.get()] = {r.out, r.stats};
            }
            auto sizes = to_vector(q.sizes());
            sizes.back() = v.size(3);
            return wrap_output(r.out, sizes, q.scalar_type());
        }

        // Gradients (dq, dk, dv) of _sdpa_lengths with `stats` from its output `out`
        // and the output gradient `grad`.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_lengths_backward(
            const at::Tensor& grad,
            const at::Tensor& out,
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const at::Tensor& lengths,
            bool is_causal,
            std::optional<double> scale
        ) {
            const char* op = "_sdpa_lengths_backward";
            check(op, q, k, v, lengths);
            // The forward takes causal with Sq != Sk (top-left); the backward does not.
            KTT_REQUIRE(
                !is_causal || q.size(2) == k.size(2), op, "causal needs Sq == Sk"
            );
            KTT_REQUIRE(
                grad.sizes() == out.sizes() && grad.scalar_type() == at::kBFloat16,
                op,
                "grad must be bf16 shaped as the output"
            );
            const auto out_handle = dense(op, out);
            auto& table = stats_table();
            const auto it = table.find(out_handle.get());
            KTT_REQUIRE(
                it != table.end() && it->second.out.lock() == out_handle,
                op,
                "out is not an output of _sdpa_lengths with statistics"
            );
            const auto g = bridge::sdpa_lengths_backward(
                dense(op, grad),
                out_handle,
                dense(op, q),
                dense(op, k),
                dense(op, v),
                it->second.stats,
                dense(op, lengths),
                is_causal,
                scale_of(q, scale)
            );
            const auto dtype = q.scalar_type();
            return {
                wrap_output(g[0], to_vector(q.sizes()), dtype),
                wrap_output(g[1], to_vector(k.sizes()), dtype),
                wrap_output(g[2], to_vector(v.sizes()), dtype)
            };
        }

        // True when _sdpa_lengths(q, k, v, ..., is_causal, ..., stats) runs on
        // ktt_flash (with `stats`: also its backward); asked on the stream once per
        // shapes.
        bool sdpa_lengths_supported(
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            bool is_causal,
            bool stats
        ) {
            const char* op = "_sdpa_lengths_supported";
            return bridge::sdpa_flash_supported(
                dense(op, q),
                dense(op, k),
                dense(op, v),
                nullptr,
                is_causal,
                stats,
                true
            );
        }

    }  // namespace

    TORCH_LIBRARY_FRAGMENT(ktt, m) {
        m.def(
            "_sdpa_lengths_supported(Tensor q, Tensor k, Tensor v, bool is_causal, "
            "bool "
            "stats) -> bool"
        );
        m.def(
            "_sdpa_lengths(Tensor q, Tensor k, Tensor v, Tensor lengths, bool "
            "is_causal, float? scale, bool stats) -> Tensor"
        );
        m.def(
            "_sdpa_lengths_backward(Tensor grad, Tensor out, Tensor q, Tensor k, "
            "Tensor v, Tensor lengths, bool is_causal, float? scale) -> (Tensor, "
            "Tensor, Tensor)"
        );
    }

    TORCH_LIBRARY_IMPL(ktt, PrivateUse1, m) {
        m.impl("_sdpa_lengths", &sdpa_lengths);
        m.impl("_sdpa_lengths_supported", &sdpa_lengths_supported);
        m.impl("_sdpa_lengths_backward", &sdpa_lengths_backward);
    }

}  // namespace ktt
