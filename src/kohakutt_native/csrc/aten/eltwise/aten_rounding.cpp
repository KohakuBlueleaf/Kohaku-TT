// aten rounding kernels for tt (fp32): remainder / fmod (Tensor and Scalar),
// round (ties to even, decimals).

#include <torch/library.h>

#include <cmath>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // a - round(a / b) * b in fp32 for floating inputs (round = floor: remainder,
        // sign of b; round = trunc: fmod, sign of a), cast back to the result dtype.
        at::Tensor modulo(const at::Tensor& a, const at::Tensor& b, bool floor) {
            auto dtype = at::result_type(a, b);
            auto compute = at::isFloatingType(dtype) ? at::kFloat : dtype;
            auto x = a.to(compute), y = b.to(compute);
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
    }

}  // namespace ktt
