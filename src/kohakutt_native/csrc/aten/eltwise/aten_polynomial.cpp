// aten orthogonal-polynomial kernels for tt, composed in fp32 by three-term recurrence:
// special_{chebyshev, shifted_chebyshev}_polynomial_{t, u, v, w},
// legendre_polynomial_p, laguerre_polynomial_l, hermite_polynomial_{h, he}.

#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // Largest degree computed.
        constexpr int64_t kMaxDegree = 4096;

        enum class Family { T, U, V, W, P, L, H, He };

        // p_{k + 1} from p_k (`cur`), p_{k - 1} (`prev`) and x.
        at::Tensor next(
            Family f,
            const at::Tensor& x,
            const at::Tensor& cur,
            const at::Tensor& prev,
            int64_t k
        ) {
            const double kd = static_cast<double>(k);
            switch (f) {
                case Family::P:
                    return (x * cur * (2 * kd + 1) - prev * kd) / (kd + 1);
                case Family::L:
                    return ((-x + (2 * kd + 1)) * cur - prev * kd) / (kd + 1);
                case Family::H:
                    return x * cur * 2 - prev * (2 * kd);
                case Family::He:
                    return x * cur - prev * kd;
                default:
                    return x * cur * 2 - prev;
            }
        }

        // p_n(x) elementwise (x, n broadcast; negative n: 0), x shifted to 2 x - 1 when
        // `shifted`; in the floating result dtype of x and n.
        at::Tensor polynomial(
            const char* op,
            const at::Tensor& x,
            const at::Tensor& n,
            Family f,
            bool shifted
        ) {
            auto dtype = at::result_type(x, n);
            if (!at::isFloatingType(dtype)) {
                dtype = at::kFloat;
            }
            auto shape = at::infer_size(x.sizes(), n.sizes());
            auto xf = x.to(at::kFloat).expand(shape);
            if (shifted) {
                xf = xf * 2 - 1;
            }
            // The degree truncated to an integer, as in PyTorch.
            auto nf = at::trunc(n.to(at::kFloat)).expand(shape);
            if (c10::multiply_integers(shape) == 0) {
                return at::empty(shape, x.options().dtype(dtype));
            }
            const auto top = static_cast<int64_t>(nf.max().item<float>());
            KTT_REQUIRE(top <= kMaxDegree, op, "degree above ", kMaxDegree);
            auto prev = at::ones(shape, xf.options());
            at::Tensor cur;
            switch (f) {
                case Family::T:
                case Family::P:
                case Family::He:
                    cur = xf;
                    break;
                case Family::U:
                case Family::H:
                    cur = xf * 2;
                    break;
                case Family::V:
                    cur = xf * 2 - 1;
                    break;
                case Family::W:
                    cur = xf * 2 + 1;
                    break;
                case Family::L:
                    cur = -xf + 1;
                    break;
            }
            auto out = at::where(at::eq(nf, 0), prev, at::zeros_like(prev));
            if (top >= 1) {
                out = at::where(at::eq(nf, 1), cur, out);
            }
            for (int64_t k = 1; k < top; ++k) {
                auto following = next(f, xf, cur, prev, k);
                out = at::where(at::eq(nf, static_cast<double>(k + 1)), following, out);
                prev = cur;
                cur = following;
            }
            return out.to(dtype);
        }

        template <Family F, bool Shifted>
        at::Tensor& polynomial_out(
            const at::Tensor& x, const at::Tensor& n, at::Tensor& out
        ) {
            const char* op = "special_polynomial.out";
            return into(op, out, polynomial(op, x, n, F, Shifted));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("special_chebyshev_polynomial_t.out", &polynomial_out<Family::T, false>);
        m.impl("special_chebyshev_polynomial_u.out", &polynomial_out<Family::U, false>);
        m.impl("special_chebyshev_polynomial_v.out", &polynomial_out<Family::V, false>);
        m.impl("special_chebyshev_polynomial_w.out", &polynomial_out<Family::W, false>);
        m.impl(
            "special_shifted_chebyshev_polynomial_t.out",
            &polynomial_out<Family::T, true>
        );
        m.impl(
            "special_shifted_chebyshev_polynomial_u.out",
            &polynomial_out<Family::U, true>
        );
        m.impl(
            "special_shifted_chebyshev_polynomial_v.out",
            &polynomial_out<Family::V, true>
        );
        m.impl(
            "special_shifted_chebyshev_polynomial_w.out",
            &polynomial_out<Family::W, true>
        );
        m.impl("special_legendre_polynomial_p.out", &polynomial_out<Family::P, false>);
        m.impl("special_laguerre_polynomial_l.out", &polynomial_out<Family::L, false>);
        m.impl("special_hermite_polynomial_h.out", &polynomial_out<Family::H, false>);
        m.impl("special_hermite_polynomial_he.out", &polynomial_out<Family::He, false>);
    }

}  // namespace ktt
