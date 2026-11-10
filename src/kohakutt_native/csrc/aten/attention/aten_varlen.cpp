// ktt ops of packed variable-length attention (kohakutt.sdpa_varlen and torch's
// varlen_attn on tt): the host plans of a set of sequence offsets, the forward on the
// packed tensors and its backward.
#include <torch/library.h>

#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "aten/attention/ktt_flash/flash.hpp"
#include "aten/attention/ktt_flash/varlen_plan.hpp"
#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        namespace varlen = bridge::ktt_flash::varlen;

        std::vector<int64_t> offsets(at::IntArrayRef cu) {
            return std::vector<int64_t>(cu.begin(), cu.end());
        }

        // The plan of offsets `cu_q` / `cu_k` for softmax scale `scale` as CPU tensors:
        // the records as int32 [pages x 32, 32] (each 32 x 32 block one tile whose
        // words in device order are the records, flash.hpp) and the mask tiles as bf16
        // [tiles x 32, 32] (one zero tile when the plan has none). `key_side`: the plan
        // by key tile row (varlen::key_plan, the backward's dK / dV program) instead of
        // by query tile row.
        std::tuple<at::Tensor, at::Tensor> varlen_plan(
            at::IntArrayRef cu_q,
            at::IntArrayRef cu_k,
            bool is_causal,
            double scale,
            bool key_side
        ) {
            KTT_REQUIRE(scale > 0, "_varlen_plan", "the scale must be positive");
            const float masked = static_cast<float>(varlen::kMaskedLogit / scale);
            const auto plan = key_side
                ? varlen::key_plan(offsets(cu_q), offsets(cu_k), is_causal, masked)
                : varlen::plan(offsets(cu_q), offsets(cu_k), is_causal, masked);
            const int64_t per = ::ktt_flash::kRecordsPerPage,
                          words = ::ktt_flash::kRecordWords;
            const int64_t pages = std::max<int64_t>(
                1, (static_cast<int64_t>(plan.rows.size()) + per - 1) / per
            );
            auto records = at::zeros({pages * 32, 32}, at::kInt);
            auto* rec = records.data_ptr<int32_t>();
            for (size_t i = 0; i < plan.rows.size(); ++i) {
                const auto& row = plan.rows[i];
                const uint32_t values[5] = {
                    row.kfirst, row.klast, row.a, row.b, row.u0
                };
                for (int64_t f = 0; f < 5; ++f) {
                    // Word w of a page lies in face w / 256 (row-major 16 x 16 faces).
                    const int64_t page = static_cast<int64_t>(i) / per;
                    const int64_t w = static_cast<int64_t>(i) % per * words + f;
                    const int64_t face = w / 256, fr = w % 256 / 16, fc = w % 16;
                    const int64_t r = page * 32 + 16 * (face / 2) + fr,
                                  c = 16 * (face % 2) + fc;
                    rec[r * 32 + c] = static_cast<int32_t>(values[f]);
                }
            }
            const int64_t tiles = std::max<int64_t>(1, plan.mask_tiles());
            auto masks = at::zeros({tiles * 32, 32}, at::kFloat);
            std::copy(plan.masks.begin(), plan.masks.end(), masks.data_ptr<float>());
            return {records, masks.to(at::kBFloat16)};
        }

        // Row statistics of the forward's outputs, by the output's device handle.
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

        void check_packed(
            const char* op,
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            at::IntArrayRef cu_q,
            at::IntArrayRef cu_k
        ) {
            for (const auto* t : {&q, &k, &v}) {
                KTT_REQUIRE(
                    t->dim() == 3 && t->scalar_type() == at::kBFloat16 &&
                        t->numel() > 0,
                    op,
                    "needs non-empty bf16 [tokens, heads, head dim] tensors"
                );
            }
            KTT_REQUIRE(
                k.size(0) == v.size(0) && k.size(1) == v.size(1) &&
                    q.size(1) % k.size(1) == 0 && q.size(2) == k.size(2),
                op,
                "q / k / v shapes do not match"
            );
            KTT_REQUIRE(
                !cu_q.empty() && cu_q.back() == q.size(0) && !cu_k.empty() &&
                    cu_k.back() == k.size(0),
                op,
                "the offsets must end at the token counts"
            );
        }

        // The kernels take a token's heads side by side: [T, H D], which is the storage
        // of a [T, H, D] view of a [T, H D] tensor (no device work then).
        bridge::HandlePtr merged(const char* op, const at::Tensor& t) {
            return dense(op, t.reshape({t.size(0), t.size(1) * t.size(2)}));
        }

        // The device handle [1, Tk] of key mask `mask` (bf16 [Tk]: added to each key's
        // scores, already divided by the scale, finite), or null.
        bridge::HandlePtr key_mask_handle(
            const char* op, const std::optional<at::Tensor>& mask, const at::Tensor& k
        ) {
            if (!mask) {
                return nullptr;
            }
            KTT_REQUIRE(
                mask->dim() == 1 && mask->size(0) == k.size(0) &&
                    mask->scalar_type() == at::kBFloat16,
                op,
                "the key mask must be bf16 [key tokens]"
            );
            return dense(op, mask->reshape({1, mask->size(0)}));
        }

        // Packed q [Tq, H, D], k [Tk, Hk, D], v [Tk, Hk, Dv] (bf16) with the offsets
        // and their plan tensors on the device: out [Tq, H, Dv] and, with `stats`, the
        // query rows' log-sum-exp [H, Tq] fp32 (else empty). `key_mask`:
        // key_mask_handle's.
        std::tuple<at::Tensor, at::Tensor> sdpa_varlen(
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const at::Tensor& plan,
            const at::Tensor& masks,
            at::IntArrayRef cu_q,
            at::IntArrayRef cu_k,
            bool is_causal,
            std::optional<double> scale,
            bool stats,
            const std::optional<at::Tensor>& key_mask
        ) {
            const char* op = "_sdpa_varlen";
            check_packed(op, q, k, v, cu_q, cu_k);
            const double s = scale ? *scale
                                   : 1.0 / std::sqrt(static_cast<double>(q.size(2)));
            const int64_t tq = q.size(0), h = q.size(1), hk = k.size(1), dv = v.size(2);
            auto r = bridge::sdpa_varlen(
                merged(op, q),
                merged(op, k),
                merged(op, v),
                h,
                hk,
                dense(op, plan),
                dense(op, masks),
                offsets(cu_q),
                offsets(cu_k),
                is_causal,
                s,
                stats,
                key_mask_handle(op, key_mask, k)
            );
            auto out = wrap_output(r.out, {tq, h * dv}, q.scalar_type())
                           .view({tq, h, dv});
            at::Tensor lse;
            if (stats) {
                remember_stats(r.out, r.stats);
                // Head h's value of token t: column 32 h of row t.
                lse = wrap_output(r.stats, {tq, h * 32}, at::kFloat)
                          .view({tq, h, 32})
                          .select(2, 0)
                          .t();
            } else {
                lse = at::empty({0}, q.options().dtype(at::kFloat));
            }
            return {out, lse};
        }

        // Gradients (dq, dk, dv), packed as q, k, v, of _sdpa_varlen with `stats` from
        // its output `out` and the output gradient `grad` [Tq, H, Dv]; `plan` / `masks`
        // and `key_plan` / `key_masks`: _varlen_plan of the offsets, by query and by
        // key rows.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_varlen_backward(
            const at::Tensor& grad,
            const at::Tensor& out,
            const at::Tensor& q,
            const at::Tensor& k,
            const at::Tensor& v,
            const at::Tensor& plan,
            const at::Tensor& masks,
            const at::Tensor& key_plan,
            const at::Tensor& key_masks,
            at::IntArrayRef cu_q,
            at::IntArrayRef cu_k,
            bool is_causal,
            std::optional<double> scale,
            const std::optional<at::Tensor>& key_mask
        ) {
            const char* op = "_sdpa_varlen_backward";
            check_packed(op, q, k, v, cu_q, cu_k);
            KTT_REQUIRE(
                grad.sizes() == out.sizes() && grad.scalar_type() == at::kBFloat16 &&
                    out.dim() == 3 && out.size(0) == q.size(0) &&
                    out.size(1) == q.size(1) && out.size(2) == v.size(2),
                op,
                "grad and out must be the forward's bf16 [query tokens, heads, value "
                "dim]"
            );
            const double s = scale ? *scale
                                   : 1.0 / std::sqrt(static_cast<double>(q.size(2)));
            const auto out_handle = merged(op, out);
            const auto stats = stats_of(out_handle);
            KTT_REQUIRE(
                stats != nullptr,
                op,
                "out is not an output of _sdpa_varlen with statistics"
            );
            const auto g = bridge::sdpa_varlen_backward(
                merged(op, grad),
                out_handle,
                merged(op, q),
                merged(op, k),
                merged(op, v),
                stats,
                q.size(1),
                k.size(1),
                dense(op, plan),
                dense(op, masks),
                dense(op, key_plan),
                dense(op, key_masks),
                offsets(cu_q),
                offsets(cu_k),
                is_causal,
                s,
                key_mask_handle(op, key_mask, k)
            );
            auto packed = [&](const bridge::HandlePtr& handle, const at::Tensor& like) {
                return wrap_output(
                           handle,
                           {like.size(0), like.size(1) * like.size(2)},
                           like.scalar_type()
                )
                    .view(like.sizes());
            };
            return {packed(g[0], q), packed(g[1], k), packed(g[2], v)};
        }

    }  // namespace

    TORCH_LIBRARY_FRAGMENT(ktt, m) {
        m.def(
            "_varlen_plan(int[] cu_q, int[] cu_k, bool is_causal, float scale, bool "
            "key_side) -> "
            "(Tensor, Tensor)",
            &varlen_plan
        );
        m.def(
            "_sdpa_varlen(Tensor q, Tensor k, Tensor v, Tensor plan, Tensor masks, "
            "int[] cu_q, "
            "int[] cu_k, bool is_causal, float? scale, bool stats, Tensor? "
            "key_mask=None) -> "
            "(Tensor, Tensor)"
        );
        m.def(
            "_sdpa_varlen_backward(Tensor grad, Tensor out, Tensor q, Tensor k, Tensor "
            "v, Tensor plan, "
            "Tensor masks, Tensor key_plan, Tensor key_masks, int[] cu_q, int[] cu_k, "
            "bool is_causal, "
            "float? scale, Tensor? key_mask=None) -> (Tensor, Tensor, Tensor)"
        );
    }

    TORCH_LIBRARY_IMPL(ktt, PrivateUse1, m) {
        m.impl("_sdpa_varlen", &sdpa_varlen);
        m.impl("_sdpa_varlen_backward", &sdpa_varlen_backward);
    }

}  // namespace ktt
