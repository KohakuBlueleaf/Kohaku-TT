// aten rounding kernels for tt (fp32): remainder / fmod (Tensor and Scalar),
// round (ties to even, decimals).

#include <torch/library.h>

#include <cmath>
#include <limits>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // a - round(a / b) * b in fp32 for floating inputs (round = floor: remainder,
        // sign of b; round = trunc: fmod, sign of a), cast back to the result dtype.
        at::Tensor modulo(const at::Tensor& a, const at::Tensor& b, bool floor) {
            auto dtype = at::result_type(a, b);
            auto compute = at::isFloatingType(dtype) ? at::kFloat : dtype;
            auto x = a.to(compute), y = b.to(compute);
            if (!at::isFloatingType(dtype)) {
                // Integers: the integer quotient (floor / trunc division).
                auto q = at::div(x, y, floor ? "floor" : "trunc");
                return (x - q * y).to(dtype);
            }
            auto q = x / y;
            q = floor ? at::floor(q) : at::trunc(q);
            return (x - q * y).to(dtype);
        }

        at::Tensor tt_remainder(const at::Tensor& a, const at::Tensor& b) {
            return modulo(a, b, true);
        }

        at::Tensor tt_fmod(const at::Tensor& a, const at::Tensor& b) {
            return modulo(a, b, false);
        }

        at::Tensor modulo_scalar(const at::Tensor& a, const at::Scalar& b, bool floor) {
            auto dtype = at::result_type(a, b);
            auto x = a.to(at::isFloatingType(dtype) ? at::kFloat : dtype);
            auto q = x / b;
            q = floor ? at::floor(q) : at::trunc(q);
            return (x - q * b).to(dtype);
        }

        at::Tensor tt_remainder_scalar(const at::Tensor& a, const at::Scalar& b) {
            return modulo_scalar(a, b, true);
        }

        at::Tensor tt_fmod_scalar(const at::Tensor& a, const at::Scalar& b) {
            return modulo_scalar(a, b, false);
        }

        // x rounded to the nearest integer, ties to even, in fp32.
        at::Tensor round_even(const at::Tensor& x) {
            auto r = at::floor(x + 0.5);
            auto tie = at::eq(r - x, 0.5);
            auto odd = at::ne(r - at::floor(r * 0.5) * 2.0, 0);
            return at::where(at::logical_and(tie, odd), r - 1.0, r);
        }

        // self rounded to `decimals` places (ties to even); integers unchanged.
        at::Tensor tt_round_decimals(const at::Tensor& self, int64_t decimals) {
            if (!at::isFloatingType(self.scalar_type())) {
                return self.clone();
            }
            auto x = self.to(at::kFloat);
            if (decimals == 0) {
                return round_even(x).to(self.scalar_type());
            }
            const double scale = std::pow(10.0, static_cast<double>(decimals));
            return (round_even(x * scale) / scale).to(self.scalar_type());
        }

        at::Tensor tt_round(const at::Tensor& self) {
            return tt_round_decimals(self, 0);
        }

        at::Tensor& tt_round_out(const at::Tensor& self, at::Tensor& out) {
            return into("round.out", out, tt_round(self));
        }

        at::Tensor& tt_round_decimals_out(
            const at::Tensor& self, int64_t d, at::Tensor& out
        ) {
            return into("round.decimals_out", out, tt_round_decimals(self, d));
        }

        // The int32 bits of fp32 `x` (tt) and back.
        at::Tensor to_bits(const char* op, const at::Tensor& x) {
            auto h = bridge::bitcast(dense(op, x.to(at::kFloat)), bridge::DType::Int32);
            return wrap_output(h, x.sizes(), at::kInt);
        }

        at::Tensor from_bits(const char* op, const at::Tensor& b) {
            auto h = bridge::bitcast(dense(op, b), bridge::DType::Float32);
            return wrap_output(h, b.sizes(), at::kFloat);
        }

        at::Tensor int_like(const at::Tensor& t, int64_t v) {
            return at::full(t.sizes(), v, t.options().dtype(at::kInt));
        }

        // (mantissa in [0.5, 1) with x's sign, exponent) with bf16 x = m 2^e from its
        // fp32 bits; 0, inf and NaN keep x with exponent 0.
        std::tuple<at::Tensor&, at::Tensor&> tt_frexp_out(
            const at::Tensor& self, at::Tensor& mantissa, at::Tensor& exponent
        ) {
            const char* op = "frexp.Tensor_out";
            KTT_REQUIRE(self.scalar_type() == at::kBFloat16, op, "bf16 input only");
            auto x = self.to(at::kFloat);
            auto b = to_bits(op, x);
            auto field = (
                at::bitwise_and(
                    at::bitwise_right_shift(b, int_like(b, 23)), int_like(b, 0xFF)
                )
            );
            auto e = field - int_like(b, 126);
            auto keep = (
                at::bitwise_and(b, int_like(b, static_cast<int32_t>(0x807FFFFF)))
            );
            auto m = from_bits(op, at::bitwise_or(keep, int_like(b, 126 << 23)));
            auto special = (
                at::logical_or(at::eq(x, 0.0), at::logical_not(at::isfinite(x)))
            );
            m = at::where(special, x, m);
            e = at::where(special, int_like(b, 0), e);
            into(op, mantissa, m.to(self.scalar_type()));
            into(op, exponent, e.to(at::kInt));
            return {mantissa, exponent};
        }

        // The next bf16 value after x towards y (2^16 units of its fp32 bits); y when
        // equal, NaN for NaN.
        at::Tensor& tt_nextafter_out(
            const at::Tensor& self, const at::Tensor& other, at::Tensor& out
        ) {
            const char* op = "nextafter.out";
            const auto dtype = at::result_type(self, other);
            KTT_REQUIRE(dtype == at::kBFloat16, op, "bf16 operands only");
            const int64_t unit = 1 << 16;
            auto shape = broadcast_shape(self, other);
            auto x = self.to(dtype).to(at::kFloat).expand(shape).contiguous();
            auto y = other.to(dtype).to(at::kFloat).expand(shape).contiguous();
            auto b = to_bits(op, x);
            auto grow = (
                at::logical_or(
                    at::logical_and(at::gt(x, 0.0), at::gt(y, x)),
                    at::logical_and(at::lt(x, 0.0), at::lt(y, x))
                )
            );
            auto step = at::where(grow, b + int_like(b, unit), b - int_like(b, unit));
            auto smallest = (
                at::where(
                    at::gt(y, 0.0),
                    int_like(b, unit),
                    int_like(b, static_cast<int32_t>(0x80000000) + unit)
                )
            );
            step = at::where(at::eq(x, 0.0), smallest, step);
            auto r = from_bits(op, step);
            r = at::where(at::eq(x, y), y, r);
            r = (
                at::where(
                    at::logical_or(at::isnan(x), at::isnan(y)),
                    std::numeric_limits<double>::quiet_NaN(),
                    r
                )
            );
            return into(op, out, r.to(dtype));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("remainder.Tensor", &tt_remainder);
        m.impl("fmod.Tensor", &tt_fmod);
        m.impl("remainder.Scalar", &tt_remainder_scalar);
        m.impl("fmod.Scalar", &tt_fmod_scalar);
        m.impl("round.decimals", &tt_round_decimals);
        m.impl("round", &tt_round);
        m.impl("round.out", &tt_round_out);
        m.impl("round.decimals_out", &tt_round_decimals_out);
        m.impl("frexp.Tensor_out", &tt_frexp_out);
        m.impl("nextafter.out", &tt_nextafter_out);
    }

}  // namespace ktt
