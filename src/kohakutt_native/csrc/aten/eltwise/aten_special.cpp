// aten special-function kernels for tt, composed in fp32: digamma / polygamma (shift
// recurrence + asymptotic series), entr, xlog1py, log_ndtr, ndtri, i0e, i1, i1e, erfcx.

#include <torch/library.h>

#include <cmath>
#include <limits>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        constexpr double kInf = std::numeric_limits<double>::infinity();
        constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

        at::Tensor f32(const at::Tensor& x) {
            return x.to(at::kFloat);
        }

        // Steps of the shift recurrence: the series runs at x + kShift (accurate for x
        // > -18).
        constexpr int kShift = 24;

        // t^n for a positive integer n by repeated products (pow fails on negative
        // bases).
        at::Tensor int_pow(const at::Tensor& t, int64_t n) {
            auto result = t;
            for (int64_t i = 1; i < n; ++i) {
                result = result * t;
            }
            return result;
        }

        // psi(x) for x > 0: psi(x) = psi(x + N) - sum 1 / (x + k), then the asymptotic
        // series ln z - 1/(2z) - sum B_2k / (2k z^2k) at z = x + N.
        at::Tensor digamma_positive(const at::Tensor& x) {
            at::Tensor shift = at::zeros_like(x);
            for (int k = 0; k < kShift; ++k) {
                shift = shift + 1.0 / (x + k);
            }
            auto z = x + kShift;
            auto r = 1.0 / (z * z);
            auto series =
                (r * (1.0 / 12 - r * (1.0 / 120 - r * (1.0 / 252 - r * (1.0 / 240)))));
            return at::log(z) - 0.5 / z - series - shift;
        }

        // psi(x), with reflection psi(x) = psi(1 - x) - pi / tan(pi x) for x < 0.5;
        // poles (0, negative integers) give -inf at 0 and NaN at negative integers, as
        // PyTorch.
        at::Tensor tt_digamma(const at::Tensor& self) {
            auto x = f32(self);
            auto reflect = at::lt(x, 0.5);
            auto base = at::where(reflect, 1.0 - x, x);
            auto value = digamma_positive(base);
            auto px = x * M_PI;
            auto reflected = value - M_PI * at::cos(px) / at::sin(px);
            auto result = at::where(reflect, reflected, value);
            auto pole = at::logical_and(at::le(x, 0.0), at::eq(x, at::floor(x)));
            result = at::where(pole, at::full_like(x, kNan), result);
            return at::where(at::eq(x, 0.0), at::full_like(x, -kInf), result);
        }

        // psi^(n)(x), n >= 1: (-1)^(n+1) n! [sum_k 1 / (x + k)^(n+1)] plus the
        // asymptotic tail at z = x + N: (n-1)!/z^n + n!/(2 z^(n+1)) + sum B_2k
        // (2k+n-1)!/((2k)! z^(2k+n)).
        at::Tensor tt_polygamma(int64_t n, const at::Tensor& self) {
            auto x = f32(self);
            const double nf = static_cast<double>(n);
            const double fact_n = std::tgamma(nf + 1.0);
            at::Tensor shift = at::zeros_like(x);
            for (int k = 0; k < kShift; ++k) {
                shift = shift + int_pow(1.0 / (x + k), n + 1);
            }
            auto rz = 1.0 / (x + kShift);
            auto rz_n = int_pow(rz, n);
            auto tail = rz_n * (std::tgamma(nf) + fact_n * 0.5 * rz);
            const double bernoulli[] = {
                1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66
            };
            auto rz2 = rz * rz, term = rz_n;
            for (int k = 1; k <= 5; ++k) {
                const double c =
                    (bernoulli[k - 1] * std::tgamma(2.0 * k + nf) /
                     std::tgamma(2.0 * k + 1.0));
                term = term * rz2;
                tail = tail + term * c;
            }
            const double sign = n % 2 == 1 ? 1.0 : -1.0;
            auto result = (shift * fact_n + tail) * sign;
            auto pole = at::logical_and(at::le(x, 0.0), at::eq(x, at::floor(x)));
            return at::where(pole, at::full_like(x, kNan), result);
        }

        at::Tensor& tt_digamma_out(const at::Tensor& self, at::Tensor& out) {
            return into("digamma.out", out, tt_digamma(self));
        }

        at::Tensor& tt_polygamma_out(
            int64_t n, const at::Tensor& self, at::Tensor& out
        ) {
            KTT_REQUIRE(n >= 0, "polygamma.out", "n must be non-negative");
            return (into(
                "polygamma.out", out, n == 0 ? tt_digamma(self) : tt_polygamma(n, self)
            ));
        }

        // -x ln x (x > 0), 0 at 0, -inf for x < 0.
        at::Tensor& tt_entr_out(const at::Tensor& self, at::Tensor& out) {
            auto x = f32(self);
            auto safe = at::where(at::gt(x, 0.0), x, at::ones_like(x));
            auto result = (at::where(
                at::gt(x, 0.0), -x * at::log(safe), at::zeros_like(x)
            ));
            result = at::where(at::lt(x, 0.0), at::full_like(x, -kInf), result);
            return into("special_entr.out", out, result);
        }

        // x log1p(y), 0 where x == 0.
        at::Tensor& tt_xlog1py_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            auto x = f32(a), y = f32(b);
            auto product = x * at::log1p(y);
            auto result = at::where(at::eq(x, 0.0), at::zeros_like(product), product);
            return into("special_xlog1py.out", out, result);
        }

        // log Phi(x): log(erfc(-x / sqrt 2) / 2), and for x < -5 the asymptotic
        // -x^2/2 - log(-x) - log(2 pi)/2 + log(1 - 1/x^2 + 3/x^4 - 15/x^6).
        at::Tensor& tt_log_ndtr_out(const at::Tensor& self, at::Tensor& out) {
            auto x = f32(self);
            auto direct = at::log(at::erfc(-x * M_SQRT1_2) * 0.5);
            auto far = at::where(at::lt(x, -5.0), x, at::full_like(x, -5.0));
            auto r = 1.0 / (far * far);
            auto tail =
                (-0.5 * far * far - at::log(-far) - 0.5 * std::log(2.0 * M_PI) +
                 at::log1p(r * (-1.0 + r * (3.0 - 15.0 * r))));
            return (into(
                "special_log_ndtr.out", out, at::where(at::lt(x, -5.0), tail, direct)
            ));
        }

        // Phi^-1(p) = sqrt 2 erfinv(2p - 1).
        at::Tensor& tt_ndtri_out(const at::Tensor& self, at::Tensor& out) {
            auto p = f32(self);
            return into("special_ndtri.out", out, at::erfinv(p * 2.0 - 1.0) * M_SQRT2);
        }

        // Above this |x| the large-argument expansions are used for the Bessel
        // functions.
        constexpr double kBesselFar = 8.0;

        // exp(-|x|) I0(x): device i0 near 0, else (1 + 1/(8x) + 9/(128x^2) + ...) /
        // sqrt(2 pi x).
        at::Tensor i0e(const at::Tensor& x) {
            auto ax = at::abs(x);
            auto near_x = at::where(at::le(ax, kBesselFar), x, at::zeros_like(x));
            auto near = at::i0(near_x) * at::exp(-at::abs(near_x));
            auto far_x = (at::where(
                at::gt(ax, kBesselFar), ax, at::full_like(ax, kBesselFar)
            ));
            auto r = 1.0 / far_x;
            auto poly =
                (1.0 +
                 r *
                     (1.0 / 8 +
                      r * (9.0 / 128 + r * (225.0 / 3072 + r * (11025.0 / 98304)))));
            auto far = poly / at::sqrt(far_x * (2.0 * M_PI));
            return at::where(at::le(ax, kBesselFar), near, far);
        }

        // exp(-|x|) I1(x): series x/2 sum (x^2/4)^k / (k! (k+1)!) near 0 (times
        // exp(-|x|)), else sign(x) (1 - 3/(8x) - 15/(128x^2) - 315/(3072x^3) - ...) /
        // sqrt(2 pi |x|).
        at::Tensor i1e(const at::Tensor& x) {
            auto ax = at::abs(x);
            auto near_x = at::where(at::le(ax, kBesselFar), x, at::zeros_like(x));
            auto q = near_x * near_x * 0.25;
            auto term = near_x * 0.5, sum = term;
            for (int k = 1; k < 30; ++k) {
                term = term * q / static_cast<double>(k * (k + 1));
                sum = sum + term;
            }
            auto near = sum * at::exp(-at::abs(near_x));
            auto far_x = (at::where(
                at::gt(ax, kBesselFar), ax, at::full_like(ax, kBesselFar)
            ));
            auto r = 1.0 / far_x;
            auto poly =
                (1.0 -
                 r *
                     (3.0 / 8 +
                      r * (15.0 / 128 + r * (315.0 / 3072 + r * (14175.0 / 98304)))));
            auto far = at::sign(x) * poly / at::sqrt(far_x * (2.0 * M_PI));
            return at::where(at::le(ax, kBesselFar), near, far);
        }

        at::Tensor& tt_i0e_out(const at::Tensor& self, at::Tensor& out) {
            return into("special_i0e.out", out, i0e(f32(self)));
        }

        at::Tensor& tt_i1e_out(const at::Tensor& self, at::Tensor& out) {
            return into("special_i1e.out", out, i1e(f32(self)));
        }

        at::Tensor& tt_i1_out(const at::Tensor& self, at::Tensor& out) {
            auto x = f32(self);
            return into("special_i1.out", out, i1e(x) * at::exp(at::abs(x)));
        }

        // exp(x^2) erfc(x): direct for x < 3, else the expansion
        // (1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6)) / (x sqrt pi).
        at::Tensor& tt_erfcx_out(const at::Tensor& self, at::Tensor& out) {
            auto x = f32(self);
            auto near_x = at::where(at::lt(x, 3.0), x, at::zeros_like(x));
            auto near = at::exp(near_x * near_x) * at::erfc(near_x);
            auto far_x = at::where(at::ge(x, 3.0), x, at::full_like(x, 3.0));
            auto r = 1.0 / (far_x * far_x);
            auto poly =
                (1.0 + r * (-0.5 + r * (0.75 + r * (-15.0 / 8 + r * (105.0 / 16)))));
            auto far = poly / (far_x * std::sqrt(M_PI));
            return into("special_erfcx.out", out, at::where(at::lt(x, 3.0), near, far));
        }

        // a^-x for x an integer where a < 0 (|a|^-x with the sign (-1)^x).
        at::Tensor power_neg(const at::Tensor& a, const at::Tensor& x) {
            auto magnitude = at::pow(at::abs(a), -x);
            auto odd = at::ne(at::remainder(x, 2.0), 0.0);
            auto negative = at::logical_and(at::lt(a, 0.0), odd);
            return at::where(negative, -magnitude, magnitude);
        }

        // Hurwitz zeta(x, q) = sum_k (q + k)^-x: the first terms summed until q + k > 9
        // (at least 9), then the Euler-Maclaurin tail with 12 Bernoulli terms. inf at x
        // = 1 or q a non-positive integer; NaN for x < 1 or q <= 0 with x non-integer.
        at::Tensor& tt_zeta_out(
            const at::Tensor& self, const at::Tensor& other, at::Tensor& out
        ) {
            static const double kA[] = {
                12.0,
                -720.0,
                30240.0,
                -1209600.0,
                47900160.0,
                -1.8924375803183791606e9,
                7.47242496e10,
                -2.950130727918164224e12,
                1.1646782814350067249e14,
                -4.5979787224074726105e15,
                1.8152105401943546773e17,
                -7.1661652561756670113e18
            };
            auto shape = broadcast_shape(self, other);
            auto x = f32(self).expand(shape), q = f32(other).expand(shape);
            if (c10::multiply_integers(shape) == 0) {
                return into("special_zeta.out", out, at::empty(shape, x.options()));
            }
            const double q_min = q.min().item<double>();
            const int64_t steps = (std::min<int64_t>(
                100000,
                std::max<int64_t>(9, static_cast<int64_t>(std::ceil(9.0 - q_min)) + 1)
            ));
            auto s = power_neg(q, x);
            auto a = q, b = at::zeros_like(q);
            for (int64_t i = 0; i < steps; ++i) {
                a = a + 1.0;
                b = power_neg(a, x);
                s = s + b;
            }
            auto w = a;
            s = s + b * w / (x - 1.0) - b * 0.5;
            auto c = b;
            double k = 0.0;
            for (double coefficient : kA) {
                c = c * (x + k) / w;
                s = s + c / coefficient;
                k += 1.0;
                c = c * (x + k) / w;
                k += 1.0;
            }
            auto integer_q = at::eq(q, at::floor(q));
            auto integer_x = at::eq(x, at::floor(x));
            auto nonpositive = at::le(q, 0.0);
            s = at::where(at::logical_and(nonpositive, integer_q), kInf, s);
            s = (at::where(
                at::logical_and(
                    at::logical_and(nonpositive, at::logical_not(integer_q)),
                    at::logical_not(integer_x)
                ),
                std::numeric_limits<double>::quiet_NaN(),
                s
            ));
            s = at::where(at::lt(x, 1.0), std::numeric_limits<double>::quiet_NaN(), s);
            s = at::where(at::eq(x, 1.0), kInf, s);
            return into("special_zeta.out", out, s);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("special_zeta.out", &tt_zeta_out);
        m.impl("digamma.out", &tt_digamma_out);
        m.impl("polygamma.out", &tt_polygamma_out);
        m.impl("special_entr.out", &tt_entr_out);
        m.impl("special_xlog1py.out", &tt_xlog1py_out);
        m.impl("special_log_ndtr.out", &tt_log_ndtr_out);
        m.impl("special_ndtri.out", &tt_ndtri_out);
        m.impl("special_i0e.out", &tt_i0e_out);
        m.impl("special_i1.out", &tt_i1_out);
        m.impl("special_i1e.out", &tt_i1e_out);
        m.impl("special_erfcx.out", &tt_erfcx_out);
    }

}  // namespace ktt
