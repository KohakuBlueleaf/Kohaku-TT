// aten activation kernels for tt (+ backward): leaky_relu, elu, hardtanh, gelu,
// silu, tanh, threshold / relu, softplus, glu, hard / soft shrink, prelu, logit.

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <cmath>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        using B = bridge::BinaryOp;
        using U = bridge::UnaryOp;

        at::Tensor tt_leaky_relu(const at::Tensor& self, const at::Scalar& slope) {
            return at::where(at::gt(self, 0), self, self * slope);
        }

        at::Tensor tt_elu(
            const at::Tensor& self,
            const at::Scalar& alpha,
            const at::Scalar& scale,
            const at::Scalar& input_scale
        ) {
            auto negative = (at::exp(self * input_scale) - 1.0) * alpha;
            return at::where(at::gt(self, 0), self, negative) * scale;
        }

        at::Tensor tt_hardtanh(
            const at::Tensor& self, const at::Scalar& lo, const at::Scalar& hi
        ) {
            return at::clamp(self, lo, hi);
        }

        // grad * (Phi(x) + x * phi(x)) for exact GELU.
        at::Tensor tt_gelu_backward(
            const at::Tensor& grad, const at::Tensor& self, c10::string_view approximate
        ) {
            if (approximate == "tanh") {
                // d/dx of 0.5 x (1 + t), t = tanh(k (x + c x^3)).
                constexpr double k = 0.7978845608028654, c = 0.044715;
                auto x = self.to(at::kFloat);
                auto t = at::tanh((x + x * x * x * c) * k);
                auto inner = (x * x * (3 * c) + 1) * k;
                auto d = (t + 1) * 0.5 + x * 0.5 * (-t * t + 1) * inner;
                return (grad.to(at::kFloat) * d).to(grad.scalar_type());
            }
            KTT_REQUIRE(
                approximate == "none",
                "gelu_backward",
                "approximate '",
                approximate,
                "'"
            );
            // Same-shape bf16 operands: TT-NN's fused GELU backward (one program).
            if (grad.scalar_type() == at::kBFloat16 &&
                self.scalar_type() == at::kBFloat16 && grad.sizes() == self.sizes()) {
                auto placed = placed_pointwise(
                    "gelu_backward",
                    {grad, self},
                    self.sizes(),
                    at::kBFloat16,
                    [](const auto& in) {
                        return bridge::gelu_backward(in[0], in[1]);
                    }
                );
                if (placed) {
                    return *placed;
                }
                auto h = (
                    bridge::gelu_backward(
                        dense("gelu_backward", grad), dense("gelu_backward", self)
                    )
                );
                return wrap_output(h, self.sizes(), at::kBFloat16);
            }
            auto x = self.to(at::kFloat);
            auto cdf = ((x * M_SQRT1_2).erf() + 1.0) * 0.5;
            auto pdf = (x * x * -0.5).exp() * (0.5 * M_2_SQRTPI * M_SQRT1_2);
            return (grad.to(at::kFloat) * (cdf + x * pdf)).to(grad.scalar_type());
        }

        at::Tensor& tt_gelu_backward_out(
            const at::Tensor& grad,
            const at::Tensor& self,
            c10::string_view approximate,
            at::Tensor& grad_input
        ) {
            return (
                write_into(
                    "gelu_backward.grad_input",
                    grad_input,
                    tt_gelu_backward(grad, self, approximate)
                )
            );
        }

        // grad * sigmoid(x) * (1 + x * (1 - sigmoid(x))).
        at::Tensor tt_silu_backward(const at::Tensor& grad, const at::Tensor& self) {
            const char* op = "silu_backward";
            auto shape = broadcast_shape(grad, self);
            auto x = dense(op, self.expand(shape));
            auto s = bridge::unary(U::Sigmoid, x);
            auto one_minus = (
                bridge::binary_scalar(
                    B::Add, bridge::binary_scalar(B::Mul, s, -1.0), 1.0
                )
            );
            auto inner = (
                bridge::binary_scalar(B::Add, bridge::binary(B::Mul, x, one_minus), 1.0)
            );
            auto h = (
                bridge::binary(
                    B::Mul,
                    dense(op, grad.expand(shape)),
                    bridge::binary(B::Mul, s, inner)
                )
            );
            return wrap_output(h, shape, grad.scalar_type());
        }

        // grad * (1 - y^2) for y = tanh(x), and the grad_input= form.
        at::Tensor tt_tanh_backward(const at::Tensor& grad, const at::Tensor& output) {
            return grad * (1.0 - output * output);
        }

        at::Tensor& tt_tanh_backward_out(
            const at::Tensor& grad, const at::Tensor& output, at::Tensor& grad_input
        ) {
            return (
                write_into(
                    "tanh_backward.grad_input", grad_input, tt_tanh_backward(grad, output)
                )
            );
        }

        // grad where self > threshold, else 0 (relu backward).
        at::Tensor tt_threshold_backward(
            const at::Tensor& grad, const at::Tensor& self, const at::Scalar& threshold
        ) {
            const char* op = "threshold_backward";
            auto shape = broadcast_shape(grad, self);
            auto mask = (
                bridge::binary_scalar(
                    B::Gt, dense(op, self.expand(shape)), threshold.toDouble()
                )
            );
            auto g = dense(op, grad.expand(shape));
            auto h = (
                bridge::binary(B::Mul, g, bridge::typecast(mask, bridge::dtype(g)))
            );
            return wrap_output(h, shape, grad.scalar_type());
        }

        // x where x > threshold, else value.
        at::Tensor& tt_threshold_out(
            const at::Tensor& self,
            const at::Scalar& threshold,
            const at::Scalar& value,
            at::Tensor& out
        ) {
            auto filler = at::full_like(self, value);
            auto result = at::where(at::gt(self, threshold), self, filler);
            return into("threshold.out", out, result);
        }

        // log(1 + exp(beta x)) / beta, linear above threshold.
        at::Tensor& tt_softplus_out(
            const at::Tensor& self,
            const at::Scalar& beta,
            const at::Scalar& threshold,
            at::Tensor& out
        ) {
            const double b = beta.toDouble(), t = threshold.toDouble();
            auto x = self.to(at::kFloat);
            auto soft = at::log1p(at::exp(x * b)) / b;
            return into("softplus.out", out, at::where(at::gt(x * b, t), x, soft));
        }

        at::Tensor& tt_softplus_backward_out(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Scalar& beta,
            const at::Scalar& threshold,
            at::Tensor& grad_input
        ) {
            const double b = beta.toDouble(), t = threshold.toDouble();
            auto x = self.to(at::kFloat);
            auto slope = (
                at::where(at::gt(x * b, t), at::ones_like(x), at::sigmoid(x * b))
            );
            auto result = grad.to(at::kFloat) * slope;
            return into("softplus_backward.grad_input", grad_input, result);
        }

        // a * sigmoid(b) for the two halves (a, b) of `dim`.
        at::Tensor& tt_glu_out(const at::Tensor& self, int64_t dim, at::Tensor& out) {
            dim = c10::maybe_wrap_dim(dim, self.dim());
            const int64_t half = self.size(dim) / 2;
            auto x = self.to(at::kFloat);
            auto result =
                (x.narrow(dim, 0, half) * at::sigmoid(x.narrow(dim, half, half)));
            return into("glu.out", out, result);
        }

        at::Tensor tt_glu_backward(
            const at::Tensor& grad, const at::Tensor& self, int64_t dim
        ) {
            dim = c10::maybe_wrap_dim(dim, self.dim());
            const int64_t half = self.size(dim) / 2;
            auto x = self.to(at::kFloat);
            auto a = x.narrow(dim, 0, half), s = at::sigmoid(x.narrow(dim, half, half));
            auto g = grad.to(at::kFloat);
            auto da = g * s, db = g * a * s * (1.0 - s);
            return at::cat({da, db}, dim).to(self.scalar_type());
        }

        // x where |x| > lambda, else 0 (hard); x -+ lambda outside, 0 inside (soft).
        at::Tensor& tt_hardshrink_out(
            const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out
        ) {
            const double l = lambd.toDouble();
            auto keep = at::gt(at::abs(self), l);
            return (
                into("hardshrink.out", out, at::where(keep, self, at::zeros_like(self)))
            );
        }

        at::Tensor& tt_hardshrink_backward_out(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Scalar& lambd,
            at::Tensor& grad_input
        ) {
            auto keep = at::gt(at::abs(self), lambd.toDouble());
            auto result = at::where(keep, grad, at::zeros_like(grad));
            return into("hardshrink_backward.grad_input", grad_input, result);
        }

        at::Tensor& tt_softshrink_out(
            const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out
        ) {
            const double l = lambd.toDouble();
            auto x = self.to(at::kFloat);
            auto shrunk = x - at::sign(x) * l;
            auto result = at::where(at::gt(at::abs(x), l), shrunk, at::zeros_like(x));
            return into("softshrink.out", out, result);
        }

        at::Tensor& tt_softshrink_backward_out(
            const at::Tensor& grad,
            const at::Tensor& self,
            const at::Scalar& lambd,
            at::Tensor& grad_input
        ) {
            auto keep = at::gt(at::abs(self), lambd.toDouble());
            auto result = at::where(keep, grad, at::zeros_like(grad));
            return into("softshrink_backward.grad_input", grad_input, result);
        }

        // x where x > 0, else weight * x (weight per channel, dim 1).
        at::Tensor tt_prelu(const at::Tensor& self, const at::Tensor& weight) {
            auto w = weight;
            if (self.dim() >= 2 && weight.numel() > 1) {
                std::vector<int64_t> shape(self.dim(), 1);
                shape[1] = weight.numel();
                w = weight.reshape(shape);
            }
            return at::where(at::gt(self, 0), self, self * w);
        }

        std::tuple<at::Tensor, at::Tensor> tt_prelu_backward(
            const at::Tensor& grad, const at::Tensor& self, const at::Tensor& weight
        ) {
            auto w = weight;
            std::vector<int64_t> dims;
            if (self.dim() >= 2 && weight.numel() > 1) {
                std::vector<int64_t> shape(self.dim(), 1);
                shape[1] = weight.numel();
                w = weight.reshape(shape);
                for (int64_t d = 0; d < self.dim(); ++d) {
                    if (d != 1) {
                        dims.push_back(d);
                    }
                }
            }
            auto positive = at::gt(self, 0);
            auto grad_input = at::where(positive, grad, grad * w);
            auto contribution = at::where(positive, at::zeros_like(grad), grad * self);
            auto grad_weight =
                (dims.empty() && weight.numel() == 1
                     ? contribution.to(at::kFloat).sum().reshape(weight.sizes())
                     : contribution.to(at::kFloat).sum(dims).reshape(weight.sizes()));
            return {grad_input, grad_weight.to(weight.scalar_type())};
        }

        // log(x / (1 - x)), x clamped to [eps, 1 - eps] when eps is given.
        at::Tensor tt_logit(const at::Tensor& self, std::optional<double> eps) {
            auto x = self.to(at::kFloat);
            if (eps) {
                // x < eps ? eps : (x > 1 - eps ? 1 - eps : x), as PyTorch (eps > 0.5
                // too).
                auto hi = (
                    at::where(at::gt(x, 1.0 - *eps), at::full_like(x, 1.0 - *eps), x)
                );
                x = at::where(at::lt(x, *eps), at::full_like(x, *eps), hi);
            }
            auto dtype =
                (at::isFloatingType(self.scalar_type()) ? self.scalar_type()
                                                        : at::kFloat);
            return at::log(x / (1.0 - x)).to(dtype);
        }

        at::Tensor tt_logit_backward(
            const at::Tensor& grad, const at::Tensor& self, std::optional<double> eps
        ) {
            auto x = self.to(at::kFloat);
            auto g = grad.to(at::kFloat) / (x * (1.0 - x));
            if (eps) {
                auto inside = at::logical_and(at::ge(x, *eps), at::le(x, 1.0 - *eps));
                g = at::where(inside, g, at::zeros_like(g));
            }
            return g.to(grad.scalar_type());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("leaky_relu", &tt_leaky_relu);
        m.impl("elu", &tt_elu);
        m.impl("hardtanh", &tt_hardtanh);
        m.impl("gelu_backward", &tt_gelu_backward);
        m.impl("gelu_backward.grad_input", &tt_gelu_backward_out);
        m.impl("silu_backward", &tt_silu_backward);
        m.impl("tanh_backward", &tt_tanh_backward);
        m.impl("tanh_backward.grad_input", &tt_tanh_backward_out);
        m.impl("threshold_backward", &tt_threshold_backward);
        m.impl("threshold.out", &tt_threshold_out);
        m.impl("softplus.out", &tt_softplus_out);
        m.impl("softplus_backward.grad_input", &tt_softplus_backward_out);
        m.impl("glu.out", &tt_glu_out);
        m.impl("glu_backward", &tt_glu_backward);
        m.impl("hardshrink.out", &tt_hardshrink_out);
        m.impl("hardshrink_backward.grad_input", &tt_hardshrink_backward_out);
        m.impl("softshrink.out", &tt_softshrink_out);
        m.impl("softshrink_backward.grad_input", &tt_softshrink_backward_out);
        m.impl("_prelu_kernel", &tt_prelu);
        m.impl("_prelu_kernel_backward", &tt_prelu_backward);
        m.impl("logit", &tt_logit);
        m.impl("logit_backward", &tt_logit_backward);
    }

}  // namespace ktt
