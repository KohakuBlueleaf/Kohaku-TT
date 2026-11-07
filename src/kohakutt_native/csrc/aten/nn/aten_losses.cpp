// aten loss kernels for tt (fp32, + backward): nll (1-D / 2-D / N-d), mse, binary
// cross entropy, huber, smooth L1, multi-margin, multilabel margin.

#include <torch/library.h>

#include <algorithm>
#include <limits>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // General nll_loss inputs: rows [M, C] fp32 (class dim last, via `perm`), safe
        // target [M] (ignored -> 0) and per-row weight [M] (0 when ignored).
        struct NllRows {
            at::Tensor rows, safe, weight;
            std::vector<int64_t> perm;
        };

        NllRows nll_rows(
            const at::Tensor& self,
            const at::Tensor& target,
            const std::optional<at::Tensor>& weight,
            int64_t ignore_index
        ) {
            auto logp = self.dim() == 1 ? self.unsqueeze(0) : self;
            const int64_t c = logp.size(1);
            std::vector<int64_t> perm{0};
            for (int64_t d = 2; d < logp.dim(); ++d) {
                perm.push_back(d);
            }
            perm.push_back(1);
            auto rows = logp.permute(perm).reshape({-1, c}).to(at::kFloat);
            auto t = target.reshape({-1}).to(at::kLong);
            auto valid = at::ne(t, ignore_index);
            auto safe = at::where(valid, t, at::zeros_like(t));
            auto w = valid.to(at::kFloat);
            if (weight && weight->defined()) {
                w = w * at::index_select(weight->to(at::kFloat), 0, safe);
            }
            return {rows, safe, w, perm};
        }

        // [M, C] bool: true at each row's target class (one elementwise pass, no
        // gather).
        at::Tensor target_mask(const NllRows& r) {
            auto classes = at::arange(r.rows.size(1), r.safe.options()).unsqueeze(0);
            return at::eq(classes, r.safe.unsqueeze(1));
        }

        std::tuple<at::Tensor, at::Tensor> nll_general_forward(
            const at::Tensor& self,
            const at::Tensor& target,
            const std::optional<at::Tensor>& weight,
            int64_t reduction,
            int64_t ignore_index
        ) {
            auto dtype = self.scalar_type();
            if (target.numel() == 0) {
                auto total = at::zeros({}, self.options());
                if (reduction == at::Reduction::None) {
                    return {at::empty(target.sizes(), self.options()), total};
                }
                const double v =
                    (reduction == at::Reduction::Mean
                         ? std::numeric_limits<double>::quiet_NaN()
                         : 0.0);
                return {at::full({}, v, self.options()), total};
            }
            auto r = nll_rows(self, target, weight, ignore_index);
            auto picked = at::where(target_mask(r), r.rows, 0.0).sum(1);
            auto per = -(picked * r.weight);
            auto total = r.weight.sum();
            if (reduction == at::Reduction::None) {
                return {per.reshape(target.sizes()).to(dtype), total.to(dtype)};
            }
            auto loss = per.sum();
            if (reduction == at::Reduction::Mean) {
                loss = loss / total;
            }
            return {loss.to(dtype), total.to(dtype)};
        }

        at::Tensor nll_general_backward(
            const at::Tensor& grad_output,
            const at::Tensor& self,
            const at::Tensor& target,
            const std::optional<at::Tensor>& weight,
            int64_t reduction,
            int64_t ignore_index,
            const at::Tensor& total_weight
        ) {
            if (target.numel() == 0) {
                return at::zeros(self.sizes(), self.options());
            }
            auto r = nll_rows(self, target, weight, ignore_index);
            const int64_t m = r.rows.size(0);
            auto g = reduction == at::Reduction::None
                ? grad_output.reshape({m}).to(at::kFloat)
                : grad_output.to(at::kFloat).reshape({1}).expand({m});
            if (reduction == at::Reduction::Mean) {
                g = g / total_weight.to(at::kFloat);
            }
            auto values = -(r.weight * g);
            auto grad_rows = at::where(target_mask(r), values.unsqueeze(1), 0.0);
            // Back to the input layout: undo the flatten, then the permutation.
            auto logp_sizes = self.dim() == 1 ? std::vector<int64_t>{1, self.size(0)}
                                              : self.sizes().vec();
            std::vector<int64_t> permuted;
            for (auto d : r.perm) {
                permuted.push_back(logp_sizes[d]);
            }
            std::vector<int64_t> inverse(r.perm.size());
            for (size_t k = 0; k < r.perm.size(); ++k) {
                inverse[r.perm[k]] = static_cast<int64_t>(k);
            }
            auto out = grad_rows.reshape(permuted).permute(inverse).contiguous();
            return out.reshape(self.sizes()).to(self.scalar_type());
        }

        // TT-NN's fused nll covers bf16 [N, C] log-probabilities (moreh rejects fp32),
        // no class weights, mean / sum.
        bool nll_fused(
            const at::Tensor& self,
            const at::Tensor& target,
            const std::optional<at::Tensor>& weight,
            int64_t reduction
        ) {
            return (
                self.dim() == 2 && target.dim() == 1 &&
                self.scalar_type() == at::kBFloat16 &&
                (!weight || !weight->defined()) && reduction != at::Reduction::None
            );
        }

        std::tuple<at::Tensor, at::Tensor> tt_nll_loss_forward(
            const at::Tensor& self,
            const at::Tensor& target,
            const std::optional<at::Tensor>& weight,
            int64_t reduction,
            int64_t ignore_index
        ) {
            if (!nll_fused(self, target, weight, reduction)) {
                return (
                    nll_general_forward(self, target, weight, reduction, ignore_index)
                );
            }
            const char* op = "nll_loss_forward";
            const bool mean = reduction == at::Reduction::Mean;
            auto r = (
                bridge::nll_loss(
                    dense(op, self), dense_as(op, target, at::kInt), mean, ignore_index
                )
            );
            const auto dtype = self.scalar_type();
            auto loss = wrap_output(r.loss, {}, dtype);
            // total_weight: the mean's divisor, or the count of non-ignored targets for
            // sum.
            auto total = mean ? wrap_output(r.total, {}, dtype)
                              : at::ne(target, ignore_index).sum().to(dtype);
            return {loss, total};
        }

        // -grad / total_weight (mean) at each row's target class, 0 elsewhere.
        at::Tensor tt_nll_loss_backward(
            const at::Tensor& grad_output,
            const at::Tensor& self,
            const at::Tensor& target,
            const std::optional<at::Tensor>& weight,
            int64_t reduction,
            int64_t ignore_index,
            const at::Tensor& total_weight
        ) {
            if (!nll_fused(self, target, weight, reduction) ||
                grad_output.scalar_type() != self.scalar_type()) {
                return (
                    nll_general_backward(
                        grad_output,
                        self,
                        target,
                        weight,
                        reduction,
                        ignore_index,
                        total_weight
                    )
                );
            }
            const char* op = "nll_loss_backward";
            const bool mean = reduction == at::Reduction::Mean;
            auto h = (
                bridge::nll_loss_backward(
                    dense(op, self),
                    dense_as(op, target, at::kInt),
                    dense(op, grad_output),
                    mean ? dense_as(op, total_weight, self.scalar_type())
                         : bridge::HandlePtr(),
                    ignore_index
                )
            );
            return wrap_output(h, self.sizes(), self.scalar_type());
        }

        // Per-element loss reduced as PyTorch's `reduction` (0 none, 1 mean, 2 sum).
        at::Tensor reduce(
            const at::Tensor& loss, int64_t reduction, at::ScalarType dtype
        ) {
            if (reduction == at::Reduction::Mean) {
                return loss.mean().to(dtype);
            }
            if (reduction == at::Reduction::Sum) {
                return loss.sum().to(dtype);
            }
            return loss.to(dtype);
        }

        // The upstream gradient per element: grad as is (none), / numel (mean),
        // broadcast.
        at::Tensor per_element(
            const at::Tensor& grad, const at::Tensor& self, int64_t reduction
        ) {
            auto g = grad.to(at::kFloat);
            if (reduction == at::Reduction::Mean) {
                g = g / static_cast<double>(std::max<int64_t>(self.numel(), 1));
            }
            return g.expand(self.sizes());
        }

        at::Tensor tt_mse_loss(
            const at::Tensor& self, const at::Tensor& target, int64_t r
        ) {
            auto d = self.to(at::kFloat) - target.to(at::kFloat);
            return reduce(d * d, r, self.scalar_type());
        }

        at::Tensor& tt_mse_loss_out(
            const at::Tensor& self, const at::Tensor& target, int64_t r, at::Tensor& out
        ) {
            return into("mse_loss.out", out, tt_mse_loss(self, target, r));
        }

        at::Tensor tt_mse_loss_backward(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Tensor& target,
            int64_t r
        ) {
            auto d = self.to(at::kFloat) - target.to(at::kFloat);
            return (per_element(grad, self, r) * d * 2.0).to(self.scalar_type());
        }

        // -w * (y log x + (1 - y) log(1 - x)), logs clamped at -100 as PyTorch does.
        at::Tensor tt_bce(
            const at::Tensor& self,
            const at::Tensor& target,
            const std::optional<at::Tensor>& weight,
            int64_t r
        ) {
            auto x = self.to(at::kFloat), y = target.to(at::kFloat);
            auto log_x = at::clamp(at::log(x), -100.0, std::nullopt);
            auto log_1mx = at::clamp(at::log(1.0 - x), -100.0, std::nullopt);
            auto loss = -(y * log_x + (1.0 - y) * log_1mx);
            if (weight && weight->defined()) {
                loss = loss * weight->to(at::kFloat);
            }
            return reduce(loss, r, self.scalar_type());
        }

        at::Tensor tt_bce_backward(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Tensor& target,
            const std::optional<at::Tensor>& weight,
            int64_t r
        ) {
            auto x = self.to(at::kFloat), y = target.to(at::kFloat);
            // d/dx = (x - y) / max(x (1 - x), eps), eps = 1e-12 as PyTorch.
            auto g = (x - y) / at::clamp(x * (1.0 - x), 1e-12, std::nullopt);
            g = g * per_element(grad, self, r);
            if (weight && weight->defined()) {
                g = g * weight->to(at::kFloat);
            }
            return g.to(self.scalar_type());
        }

        // 0.5 d^2 for |d| < delta, else delta (|d| - 0.5 delta).
        at::Tensor tt_huber_loss(
            const at::Tensor& self, const at::Tensor& target, int64_t r, double delta
        ) {
            auto d = at::abs(self.to(at::kFloat) - target.to(at::kFloat));
            auto loss = (
                at::where(at::lt(d, delta), d * d * 0.5, (d - 0.5 * delta) * delta)
            );
            return reduce(loss, r, self.scalar_type());
        }

        // grad * d for |d| <= delta, else grad * delta * sign(d).
        at::Tensor& tt_huber_loss_backward_out(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Tensor& target,
            int64_t r,
            double delta,
            at::Tensor& grad_input
        ) {
            auto d = self.to(at::kFloat) - target.to(at::kFloat);
            auto slope = at::where(at::le(at::abs(d), delta), d, at::sign(d) * delta);
            auto result = per_element(grad, self, r) * slope;
            return into("huber_loss_backward.out", grad_input, result);
        }

        // 0.5 d^2 / beta for |d| < beta, else |d| - 0.5 beta (beta = 0: L1).
        at::Tensor smooth_l1(
            const at::Tensor& self, const at::Tensor& target, int64_t r, double beta
        ) {
            auto d = at::abs(self.to(at::kFloat) - target.to(at::kFloat));
            auto loss =
                (beta == 0.0 ? d
                             : at::where(
                                   at::lt(d, beta), d * d * (0.5 / beta), d - 0.5 * beta
                               ));
            return reduce(loss, r, self.scalar_type());
        }

        at::Tensor& tt_smooth_l1_out(
            const at::Tensor& self,
            const at::Tensor& target,
            int64_t r,
            double beta,
            at::Tensor& out
        ) {
            return into("smooth_l1_loss.out", out, smooth_l1(self, target, r, beta));
        }

        // grad * d / beta for |d| < beta, else grad * sign(d).
        at::Tensor& tt_smooth_l1_backward_out(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Tensor& target,
            int64_t r,
            double beta,
            at::Tensor& grad_input
        ) {
            auto d = self.to(at::kFloat) - target.to(at::kFloat);
            auto slope =
                (beta == 0.0
                     ? at::sign(d)
                     : at::where(at::lt(at::abs(d), beta), d / beta, at::sign(d)));
            auto result = per_element(grad, self, r) * slope;
            return into("smooth_l1_loss_backward.grad_input", grad_input, result);
        }

        // multi_margin_loss parts for x [N, C] (1-D: N = 1): hinge z_i = margin - x_y +
        // x_i off the target class, the per-sample class weight w_y, and the target
        // one-hot.
        struct MultiMargin {
            at::Tensor z, hinge, w, onehot;
        };

        MultiMargin multi_margin(
            const at::Tensor& self,
            const at::Tensor& target,
            double margin,
            const std::optional<at::Tensor>& weight
        ) {
            auto x = self.to(at::kFloat).reshape({-1, self.size(-1)});
            const int64_t c = x.size(1);
            auto y = target.reshape({-1, 1}).to(at::kLong);
            auto classes = at::arange(c, y.options()).reshape({1, c});
            auto onehot = at::eq(classes, y);
            const bool keepdim = true;
            auto xy = (x * onehot).sum(1, keepdim);
            auto z = at::where(onehot, at::zeros_like(x), x - xy + margin);
            auto w = weight && weight->defined()
                ? at::index_select(weight->to(at::kFloat), 0, y.reshape({-1}))
                      .reshape({-1, 1})
                : at::ones({1, 1}, x.options());
            return {z, at::clamp(z, 0.0, std::nullopt), w, onehot};
        }

        // sum over i != y of w_y max(0, margin - x_y + x_i)^p / C, p in {1, 2}.
        at::Tensor tt_multi_margin_loss(
            const at::Tensor& self,
            const at::Tensor& target,
            const at::Scalar& p,
            const at::Scalar& margin,
            const std::optional<at::Tensor>& weight,
            int64_t r
        ) {
            const int64_t power = p.toLong();
            KTT_REQUIRE(
                power == 1 || power == 2, "multi_margin_loss", "p must be 1 or 2"
            );
            auto parts = multi_margin(self, target, margin.toDouble(), weight);
            auto term = power == 1 ? parts.hinge : parts.hinge * parts.hinge;
            auto loss = (term * parts.w).sum(1) / static_cast<double>(self.size(-1));
            if (self.dim() <= 1) {
                loss = loss.reshape({});
            }
            return reduce(loss, r, self.scalar_type());
        }

        at::Tensor tt_multi_margin_loss_backward(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Tensor& target,
            const at::Scalar& p,
            const at::Scalar& margin,
            const std::optional<at::Tensor>& weight,
            int64_t r
        ) {
            const int64_t power = p.toLong();
            auto parts = multi_margin(self, target, margin.toDouble(), weight);
            auto active = at::gt(parts.z, 0.0).to(at::kFloat);
            auto slope = power == 1 ? active : parts.hinge * 2.0;
            auto g_i = slope * parts.w / static_cast<double>(self.size(-1));
            const bool keepdim = true;
            auto g_y = -g_i.sum(1, keepdim);
            auto g = g_i + parts.onehot.to(at::kFloat) * g_y;
            const int64_t n = g.size(0);
            auto upstream = grad.to(at::kFloat);
            if (r == at::Reduction::None) {
                upstream = upstream.reshape({-1, 1});
            } else if (r == at::Reduction::Mean) {
                upstream = upstream / static_cast<double>(n);
            }
            return (g * upstream).reshape(self.sizes()).to(self.scalar_type());
        }

        // multilabel_margin_loss parts for x [N, C]: targets are listed until the first
        // -1; hinge[n, k, i] = max(0, 1 - x[y_k] + x_i) for valid k and non-target i.
        struct MultiLabel {
            at::Tensor hinge, active, target_hot, is_target;
        };

        MultiLabel multilabel(const at::Tensor& self, const at::Tensor& target) {
            auto x = self.to(at::kFloat).reshape({-1, self.size(-1)});
            const int64_t c = x.size(1);
            auto t = target.reshape({-1, c}).to(at::kLong);
            auto valid = at::cumprod(at::ne(t, -1).to(at::kFloat), 1);  // [N, K]
            auto classes = at::arange(c, t.options()).reshape({1, 1, c});
            auto target_hot = at::eq(t.unsqueeze(-1), classes).to(at::kFloat) *
                valid.unsqueeze(-1);  // [N, K, C]
            auto is_target = at::clamp(target_hot.sum(1), std::nullopt, 1.0);  // [N, C]
            auto xt = at::gather(x, 1, at::clamp(t, 0, std::nullopt));         // [N, K]
            auto margin = 1.0 - xt.unsqueeze(-1) + x.unsqueeze(1);  // [N, K, C]
            auto keep = valid.unsqueeze(-1) * (1.0 - is_target.unsqueeze(1));
            auto hinge = at::clamp(margin, 0.0, std::nullopt) * keep;
            auto active = at::gt(margin, 0.0).to(at::kFloat) * keep;
            return {hinge, active, target_hot, is_target};
        }

        std::tuple<at::Tensor, at::Tensor> tt_multilabel_margin_loss_forward(
            const at::Tensor& self, const at::Tensor& target, int64_t r
        ) {
            auto parts = multilabel(self, target);
            auto loss = parts.hinge.sum({1, 2}) / static_cast<double>(self.size(-1));
            if (self.dim() <= 1) {
                loss = loss.reshape({});
            }
            auto flag = parts.is_target.reshape(self.sizes()).to(self.scalar_type());
            return {reduce(loss, r, self.scalar_type()), flag};
        }

        at::Tensor tt_multilabel_margin_loss_backward(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Tensor& target,
            int64_t r,
            const at::Tensor& is_target
        ) {
            auto parts = multilabel(self, target);
            const int64_t c = self.size(-1);
            // d/dx_i: active pairs over k; d/dx_{y_k}: minus the active count of pair
            // row k.
            auto g_i = parts.active.sum(1);
            const bool keepdim = true;
            auto per_k = parts.active.sum(2, keepdim);  // [N, K, 1]
            auto g_y = (parts.target_hot * per_k).sum(1);
            auto g = (g_i - g_y) / static_cast<double>(c);
            auto upstream = grad.to(at::kFloat);
            if (r == at::Reduction::None) {
                upstream = upstream.reshape({-1, 1});
            } else if (r == at::Reduction::Mean) {
                upstream = upstream / static_cast<double>(g.size(0));
            }
            return (g * upstream).reshape(self.sizes()).to(self.scalar_type());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("nll_loss2d_forward", &nll_general_forward);
        m.impl("nll_loss2d_backward", &nll_general_backward);
        m.impl("nll_loss_forward", &tt_nll_loss_forward);
        m.impl("nll_loss_backward", &tt_nll_loss_backward);
        m.impl("mse_loss", &tt_mse_loss);
        m.impl("mse_loss.out", &tt_mse_loss_out);
        m.impl("mse_loss_backward", &tt_mse_loss_backward);
        m.impl("binary_cross_entropy", &tt_bce);
        m.impl("binary_cross_entropy_backward", &tt_bce_backward);
        m.impl("huber_loss", &tt_huber_loss);
        m.impl("huber_loss_backward.out", &tt_huber_loss_backward_out);
        m.impl("smooth_l1_loss.out", &tt_smooth_l1_out);
        m.impl("smooth_l1_loss_backward.grad_input", &tt_smooth_l1_backward_out);
        m.impl("multi_margin_loss", &tt_multi_margin_loss);
        m.impl("multi_margin_loss_backward", &tt_multi_margin_loss_backward);
        m.impl("multilabel_margin_loss_forward", &tt_multilabel_margin_loss_forward);
        m.impl("multilabel_margin_loss_backward", &tt_multilabel_margin_loss_backward);
    }

}  // namespace ktt
