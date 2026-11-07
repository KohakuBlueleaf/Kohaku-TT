// aten pointwise kernels for tt (fp32 where rounding matters): rsub, div modes,
// logaddexp(2), xlogy, fmax / fmin, copysign, heaviside, sinc, lerp, hypot, pow.

#include <ATen/Dispatch.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <utility>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"

namespace ktt {
    namespace {

        at::Tensor tt_rsub(
            const at::Tensor& self, const at::Tensor& other, const at::Scalar& a
        ) {
            return a.toDouble() == 1.0 ? other - self : other - self * a;
        }

        // a / b with rounding None (true division), "trunc" or "floor", computed in
        // fp32.
        at::Tensor tt_div_mode(
            const at::Tensor& a,
            const at::Tensor& b,
            std::optional<c10::string_view> mode
        ) {
            if (!mode) {
                return at::div(a, b);
            }
            auto dtype = at::result_type(a, b);
            auto q = a.to(at::kFloat) / b.to(at::kFloat);
            KTT_REQUIRE(
                *mode == "trunc" || *mode == "floor", "div.Tensor_mode", "rounding_mode"
            );
            q = *mode == "trunc" ? at::trunc(q) : at::floor(q);
            return q.to(dtype);
        }

        at::Tensor& tt_div_out_mode(
            const at::Tensor& a,
            const at::Tensor& b,
            std::optional<c10::string_view> mode,
            at::Tensor& out
        ) {
            return into("div.out_mode", out, tt_div_mode(a, b, mode));
        }

        at::Tensor tt_floor_divide(const at::Tensor& a, const at::Tensor& b) {
            return tt_div_mode(a, b, c10::string_view("floor"));
        }

        // log(exp(a) + exp(b)) = m + log(exp(a - m) + exp(b - m)), m = max(a, b), in
        // fp32.
        at::Tensor& tt_logaddexp_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            auto x = a.to(at::kFloat), y = b.to(at::kFloat);
            auto m = at::maximum(x, y);
            auto result = m + at::log(at::exp(x - m) + at::exp(y - m));
            // Both -inf: m - m is NaN; the answer is -inf.
            const double neg_inf = -std::numeric_limits<double>::infinity();
            result = at::where(at::eq(m, neg_inf), m, result);
            return into("logaddexp.out", out, result);
        }

        // x * log(y), 0 where x == 0 (unless y is NaN).
        at::Tensor& tt_xlogy_out(
            const at::Tensor& x, const at::Tensor& y, at::Tensor& out
        ) {
            auto xf = x.to(at::kFloat), yf = y.to(at::kFloat);
            auto result = xf * at::log(yf);
            auto zero = at::logical_and(at::eq(xf, 0), at::logical_not(at::isnan(yf)));
            auto masked = at::where(zero, at::zeros_like(result), result);
            return into("xlogy.OutTensor", out, masked);
        }

        // Maximum / minimum that ignore a NaN operand.
        at::Tensor nan_aware(const at::Tensor& a, const at::Tensor& b, bool max) {
            auto best = max ? at::maximum(a, b) : at::minimum(a, b);
            return at::where(at::isnan(a), b, at::where(at::isnan(b), a, best));
        }

        at::Tensor& tt_fmax_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            return into("fmax.out", out, nan_aware(a, b, true));
        }

        at::Tensor& tt_fmin_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            return into("fmin.out", out, nan_aware(a, b, false));
        }

        // |a| with the sign of b (including -0.0, detected as 1 / b < 0).
        at::Tensor& tt_copysign_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            auto y = b.to(
                a.device(), at::kFloat
            );  // a Python scalar arrives as a CPU tensor
            auto negative = at::logical_or(at::lt(y, 0), at::lt(at::reciprocal(y), 0));
            auto magnitude = at::abs(a.to(at::kFloat));
            return (
                into("copysign.out", out, at::where(negative, -magnitude, magnitude))
            );
        }

        // 0 for x < 0, values for x == 0, 1 for x > 0.
        at::Tensor& tt_heaviside_out(
            const at::Tensor& self, const at::Tensor& values, at::Tensor& out
        ) {
            auto dtype = at::result_type(self, values);
            auto step = at::gt(self, 0).to(dtype);
            auto v = values.to(self.device(), dtype);
            return into("heaviside.out", out, at::where(at::eq(self, 0), v, step));
        }

        // Real input: pi for negative values, 0 otherwise (NaN stays NaN).
        at::Tensor tt_angle(const at::Tensor& self) {
            if (holds_complex(self)) {
                auto [re, im] = parts(self);
                return at::atan2(im, re);
            }
            auto x = (
                self.to(
                    at::isFloatingType(self.scalar_type()) ? self.scalar_type() : at::kFloat
                )
            );
            auto angle = at::where(at::lt(x, 0), M_PI, 0.0).to(x.scalar_type());
            return at::where(at::isnan(x), x, angle);
        }

        at::Tensor& tt_isposinf_out(const at::Tensor& self, at::Tensor& out) {
            auto pos = at::logical_and(at::isinf(self), at::gt(self, 0));
            return into("isposinf.out", out, pos);
        }

        // NaN -> nan (default 0), +inf -> posinf, -inf -> neginf (defaults: the dtype's
        // largest / lowest finite values).
        at::Tensor& tt_nan_to_num_out(
            const at::Tensor& self,
            std::optional<double> nan,
            std::optional<double> posinf,
            std::optional<double> neginf,
            at::Tensor& out
        ) {
            if (!at::isFloatingType(self.scalar_type())) {
                return into("nan_to_num.out", out, self.clone());
            }
            const auto [lowest, largest] = AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf, self.scalar_type(), "nan_to_num", [&] {
                    return std::pair<double, double>{
                        static_cast<double>(std::numeric_limits<scalar_t>::lowest()),
                        static_cast<double>(std::numeric_limits<scalar_t>::max())
                    };
                }
            );
            auto inf = at::isinf(self);
            auto pos = at::logical_and(inf, at::gt(self, 0));
            auto neg = at::logical_and(inf, at::lt(self, 0));
            auto result = at::where(at::isnan(self), nan.value_or(0.0), self);
            result = at::where(pos, posinf.value_or(largest), result);
            result = at::where(neg, neginf.value_or(lowest), result);
            return into("nan_to_num.out", out, result);
        }

        // log(sigmoid(x)) = min(x, 0) - log1p(exp(-|x|)); the buffer is unused (empty).
        std::tuple<at::Tensor, at::Tensor> tt_log_sigmoid_forward(
            const at::Tensor& self
        ) {
            auto x = self.to(at::kFloat);
            auto out = at::clamp(x, std::nullopt, 0) - at::log1p(at::exp(-at::abs(x)));
            return {out.to(self.scalar_type()), at::empty({0}, self.options())};
        }

        // d log(sigmoid(x)) / dx = sigmoid(-x).
        at::Tensor tt_log_sigmoid_backward(
            const at::Tensor& grad, const at::Tensor& self, const at::Tensor& buffer
        ) {
            return grad * at::sigmoid(-self);
        }

        // sin(pi x) / (pi x), 1 at x = 0.
        at::Tensor& tt_sinc_out(const at::Tensor& self, at::Tensor& out) {
            auto px = self.to(at::kFloat) * M_PI;
            auto zero = at::eq(px, 0.0);
            auto safe = at::where(zero, at::ones_like(px), px);
            auto result = at::where(zero, at::ones_like(px), at::sin(safe) / safe);
            return into("sinc.out", out, result);
        }

        at::Tensor& tt_sgn_out(const at::Tensor& self, at::Tensor& out) {
            return into("sgn.out", out, at::sign(self));
        }

        // x < 0, plus -0.0 (1 / x = -inf); NaN sign bits are not visible after device
        // ops.
        at::Tensor& tt_signbit_out(const at::Tensor& self, at::Tensor& out) {
            auto x = self.to(at::kFloat);
            auto negative_zero = at::logical_and(at::eq(x, 0.0), at::lt(1.0 / x, 0.0));
            return (
                into("signbit.out", out, at::logical_or(at::lt(x, 0.0), negative_zero))
            );
        }

        // log2(2^a + 2^b), through logaddexp in natural log units.
        at::Tensor& tt_logaddexp2_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            const double ln2 = std::log(2.0);
            auto x = a.to(at::kFloat) * ln2, y = b.to(at::kFloat) * ln2;
            auto m = at::maximum(x, y);
            auto result = (m + at::log(at::exp(x - m) + at::exp(y - m))) / ln2;
            const double neg_inf = -std::numeric_limits<double>::infinity();
            result = at::where(at::eq(m, neg_inf), m, result);
            return into("logaddexp2.out", out, result);
        }

        // base ^ e for a scalar base: exp(e ln base); base 1 gives 1, base 0 gives 0^e.
        at::Tensor& tt_pow_scalar_base_out(
            const at::Scalar& base, const at::Tensor& e, at::Tensor& out
        ) {
            const double b = base.toDouble();
            auto x = e.to(at::kFloat);
            at::Tensor result;
            if (b == 1.0) {
                result = at::ones_like(x);
            } else if (b == 0.0) {
                result = at::where(at::eq(x, 0.0), at::ones_like(x), at::zeros_like(x));
                result = at::where(at::lt(x, 0.0), at::full_like(x, INFINITY), result);
            } else {
                KTT_REQUIRE(b > 0.0, "pow.Scalar_out", "negative scalar base");
                result = at::exp(x * std::log(b));
            }
            return into("pow.Scalar_out", out, result);
        }

        at::Tensor& tt_logical_not_out(const at::Tensor& self, at::Tensor& out) {
            return into("logical_not.out", out, at::eq(self, 0));
        }

        // sqrt(a^2 + b^2) as m sqrt(1 + (n / m)^2), m = max(|a|, |b|), n = min (0 when
        // m = 0).
        at::Tensor& tt_hypot_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            auto x = at::abs(a.to(at::kFloat)), y = at::abs(b.to(at::kFloat));
            auto m = at::maximum(x, y), n = at::minimum(x, y);
            auto safe = at::where(at::eq(m, 0.0), at::ones_like(m), m);
            auto ratio = n / safe;
            auto result = (
                at::where(at::eq(m, 0.0), m, m * at::sqrt(1.0 + ratio * ratio))
            );
            return into("hypot.out", out, result);
        }

        // a + w (b - a) with a tensor weight.
        at::Tensor& tt_lerp_tensor_out(
            const at::Tensor& a,
            const at::Tensor& b,
            const at::Tensor& w,
            at::Tensor& out
        ) {
            auto x = a.to(at::kFloat);
            auto result = x + w.to(at::kFloat) * (b.to(at::kFloat) - x);
            return into("lerp.Tensor_out", out, result);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("rsub.Tensor", &tt_rsub);
        m.impl("div.Tensor_mode", &tt_div_mode);
        m.impl("div.out_mode", &tt_div_out_mode);
        m.impl("floor_divide", &tt_floor_divide);
        m.impl("logaddexp.out", &tt_logaddexp_out);
        m.impl("xlogy.OutTensor", &tt_xlogy_out);
        m.impl("fmax.out", &tt_fmax_out);
        m.impl("fmin.out", &tt_fmin_out);
        m.impl("copysign.out", &tt_copysign_out);
        m.impl("heaviside.out", &tt_heaviside_out);
        m.impl("angle", &tt_angle);
        m.impl("isposinf.out", &tt_isposinf_out);
        m.impl("nan_to_num.out", &tt_nan_to_num_out);
        m.impl("log_sigmoid_forward", &tt_log_sigmoid_forward);
        m.impl("log_sigmoid_backward", &tt_log_sigmoid_backward);
        m.impl("sinc.out", &tt_sinc_out);
        m.impl("sgn.out", &tt_sgn_out);
        m.impl("signbit.out", &tt_signbit_out);
        m.impl("logaddexp2.out", &tt_logaddexp2_out);
        m.impl("lerp.Tensor_out", &tt_lerp_tensor_out);
        m.impl("pow.Scalar_out", &tt_pow_scalar_base_out);
        m.impl("logical_not.out", &tt_logical_not_out);
        m.impl("hypot.out", &tt_hypot_out);
    }

}  // namespace ktt
