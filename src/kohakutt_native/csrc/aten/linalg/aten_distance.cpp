// aten distance kernels for tt: cdist (p-norm between row sets) and pdist (condensed
// pairwise distances), from broadcast differences in fp32.
#include <ATen/ExpandUtils.h>
#include <c10/util/accumulate.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // p-norm over the last dim of `diff`: p = 0 (count nonzero), 1, 2, inf, or
        // general.
        at::Tensor p_norm_last(const at::Tensor& diff, double p) {
            auto a = at::abs(diff);
            if (p == 0.0) {
                return at::ne(a, 0).to(at::kFloat).sum(-1);
            }
            if (p == 1.0) {
                return a.sum(-1);
            }
            if (p == 2.0) {
                return at::sqrt((a * a).sum(-1));
            }
            if (std::isinf(p)) {
                return a.amax(-1);
            }
            return at::pow(at::pow(a, p).sum(-1), 1.0 / p);
        }

        // x1 [..., P, M], x2 [..., R, M] -> [..., P, R] distances.
        at::Tensor tt_cdist_forward(
            const at::Tensor& x1,
            const at::Tensor& x2,
            double p,
            std::optional<int64_t> compute_mode
        ) {
            const char* op = "_cdist_forward";
            KTT_REQUIRE(
                x1.dim() >= 2 && x2.dim() >= 2, op, "inputs must be at least 2-D"
            );
            KTT_REQUIRE(
                x1.size(-1) == x2.size(-1), op, "inputs must have the same last dim"
            );
            auto out_sizes = (
                at::infer_size(
                    x1.sizes().slice(0, x1.dim() - 2), x2.sizes().slice(0, x2.dim() - 2)
                )
            );
            out_sizes.push_back(x1.size(-2));
            out_sizes.push_back(x2.size(-2));
            if (c10::multiply_integers(out_sizes) == 0) {
                return at::empty(out_sizes, x1.options());
            }
            if (x1.size(-1) == 0) {
                return at::zeros(out_sizes, x1.options());
            }
            auto a = x1.to(at::kFloat).unsqueeze(-2);  // [..., P, 1, M]
            auto b = x2.to(at::kFloat).unsqueeze(-3);  // [..., 1, R, M]
            return p_norm_last(a - b, p).to(x1.scalar_type());
        }

        // Euclidean cdist from direct differences (PyTorch's composite uses |a|^2 +
        // |b|^2 - 2 a.b, which cancels badly in bf16: 14x the floor on OpInfo samples).
        at::Tensor tt_euclidean_dist(const at::Tensor& x1, const at::Tensor& x2) {
            return tt_cdist_forward(x1, x2, 2.0, std::nullopt);
        }

        // x [N, M] -> distances of all pairs i < j, in row-major order (length
        // N(N-1)/2).
        at::Tensor tt_pdist_forward(const at::Tensor& self, double p) {
            const char* op = "_pdist_forward";
            KTT_REQUIRE(self.dim() == 2, op, "input must be 2-D");
            const int64_t n = self.size(0);
            if (n < 2) {
                return at::empty({0}, self.options());
            }
            auto pairs = device_constant(c10::str("pdist|", n), [&] {
                std::vector<int64_t> flat;
                flat.reserve(n * (n - 1) / 2);
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = i + 1; j < n; ++j) {
                        flat.push_back(i * n + j);
                    }
                }
                return at::tensor(flat);
            });
            auto all = tt_cdist_forward(self, self, p, std::nullopt).reshape({n * n});
            return at::index_select(all, 0, pairs);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("_cdist_forward", &tt_cdist_forward);
        m.impl("_pdist_forward", &tt_pdist_forward);
        m.impl("_euclidean_dist", &tt_euclidean_dist);
    }

}  // namespace ktt
