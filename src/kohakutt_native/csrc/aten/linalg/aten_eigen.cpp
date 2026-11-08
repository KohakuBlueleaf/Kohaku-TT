// aten symmetric eigendecomposition and SVD for tt: _linalg_eigh (+ out), _linalg_svd
// (+ out). Parallel cyclic Jacobi rotations in fp32 (round-robin pairs, n / 2
// rotations per round as row / column gathers): two-sided for eigh, one-sided
// (Hestenes) for the SVD.
#include <torch/library.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        constexpr int kMaxSweeps = 30;

        // Round-robin schedule over `n` (even) indices: for each of the n - 1 rounds,
        // every index's partner and whether it is the first (p) of its pair.
        struct Schedule {
            std::vector<at::Tensor> partner;  // [n] int64 per round
            std::vector<at::Tensor> sign;     // [n] fp32 per round: -1 for p, +1 for q
        };

        Schedule schedule(int64_t n, const at::TensorOptions& options) {
            Schedule s;
            std::vector<int64_t> ring(n);
            for (int64_t i = 0; i < n; ++i) {
                ring[i] = i;
            }
            for (int64_t round = 0; round < n - 1; ++round) {
                std::vector<int64_t> partner(n);
                std::vector<float> sign(n);
                for (int64_t i = 0; i < n / 2; ++i) {
                    const int64_t a = ring[i], b = ring[n - 1 - i];
                    const int64_t p = std::min(a, b), q = std::max(a, b);
                    partner[p] = q;
                    partner[q] = p;
                    sign[p] = -1.0f;
                    sign[q] = 1.0f;
                }
                s.partner.push_back(
                    at::tensor(partner, at::kLong).to(options.device())
                );
                s.sign.push_back(at::tensor(sign, at::kFloat).to(options.device()));
                // Index 0 fixed; the others rotate by one.
                const int64_t last = ring[n - 1];
                for (int64_t i = n - 1; i > 1; --i) {
                    ring[i] = ring[i - 1];
                }
                ring[1] = last;
            }
            return s;
        }

        // Jacobi rotation (c, s) per index zeroing the (p, q) entry of a symmetric
        // 2 x 2 problem [[app, apq], [apq, aqq]]; identity where apq is 0.
        std::pair<at::Tensor, at::Tensor> rotation(
            const at::Tensor& app, const at::Tensor& aqq, const at::Tensor& apq
        ) {
            auto zero = at::eq(apq, 0.0);
            auto safe = at::where(zero, 1.0, apq);
            auto theta = (aqq - app) / (2.0 * safe);
            auto t =
                (at::where(at::ge(theta, 0.0), 1.0, -1.0) /
                 (at::abs(theta) + at::sqrt(theta * theta + 1.0)));
            t = at::where(zero, 0.0, at::where(at::isinf(theta), 0.0, t));
            auto one_t2 = t * t + 1.0;
            auto c = 1.0 / at::sqrt(one_t2);
            // Newton steps for c = (1 + t^2)^-1/2 (multiplies only), so c^2 + s^2 = 1
            // to fp32 rounding.
            for (int i = 0; i < 2; ++i) {
                c = c * (1.5 - 0.5 * one_t2 * c * c);
            }
            return {c, t * c};
        }

        // Pads [B, m, n] matrices with zero rows / columns to `rows` x `cols`.
        at::Tensor pad_to(const at::Tensor& x, int64_t rows, int64_t cols) {
            if (x.size(1) == rows && x.size(2) == cols) {
                return x;
            }
            return at::constant_pad_nd(
                x, {0, cols - x.size(2), 0, rows - x.size(1)}, 0.0
            );
        }

        // a @ b for [B, m, k] x [B, k, n] as elementwise products summed in fp32.
        at::Tensor product(const at::Tensor& a, const at::Tensor& b) {
            return (a.unsqueeze(3) * b.unsqueeze(1)).sum(2);
        }

        // One Newton-Schulz step towards the nearest orthogonal matrix:
        // V (1.5 I - 0.5 V^T V).
        at::Tensor reorthogonalize(const at::Tensor& v) {
            const int64_t n = v.size(-1);
            auto gram = product(v.transpose(1, 2), v);
            auto eye = at::eye(n, v.options());
            return product(v, eye * 1.5 - gram * 0.5);
        }

        // (eigenvalues [B, n] ascending, eigenvectors [B, n, n]) of symmetric `a`
        // [B, n, n] fp32: Jacobi sweeps until the off-diagonal mass stops halving,
        // then V re-orthogonalized and the eigenvalues taken as v^T A v.
        std::pair<at::Tensor, at::Tensor> jacobi_eigh(const at::Tensor& a) {
            const int64_t n = a.size(-1), b = a.size(0), even = n + (n % 2);
            auto w = pad_to(a, even, even);
            auto v = at::eye(even, w.options()).expand({b, even, even}).contiguous();
            auto sched = schedule(even, w.options());
            auto offdiag = 1.0 - at::eye(even, w.options());
            const double total = (w * w).sum().item<double>();
            double previous = std::numeric_limits<double>::infinity();
            for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
                for (size_t r = 0; r < sched.partner.size(); ++r) {
                    const auto& partner = sched.partner[r];
                    const auto& sign = sched.sign[r];
                    auto diag = w.diagonal(0, 1, 2);
                    auto other = diag.index_select(1, partner);
                    auto cross = w.gather(
                                      2, partner.view({1, even, 1}).expand({b, even, 1})
                    )
                                     .squeeze(2);
                    auto is_p = at::lt(sign, 0.0);
                    auto [c, s] = (
                        rotation(
                            at::where(is_p, diag, other),
                            at::where(is_p, other, diag),
                            cross
                        )
                    );
                    auto ss = s * sign;
                    w =
                        (w * c.unsqueeze(2) +
                         w.index_select(1, partner) * ss.unsqueeze(2));
                    w =
                        (w * c.unsqueeze(1) +
                         w.index_select(2, partner) * ss.unsqueeze(1));
                    v =
                        (v * c.unsqueeze(1) +
                         v.index_select(2, partner) * ss.unsqueeze(1));
                }
                const double off = (w * w * offdiag).sum().item<double>();
                if (off <= 1e-14 * total || off > 0.5 * previous) {
                    break;
                }
                previous = off;
            }
            auto vectors = reorthogonalize(v.narrow(1, 0, n).narrow(2, 0, n));
            auto values = (vectors * product(a, vectors)).sum(1);
            auto [sorted, order] = values.sort(1);
            vectors = vectors.gather(2, order.unsqueeze(1).expand({b, n, n}));
            return {sorted, vectors};
        }

        std::tuple<at::Tensor, at::Tensor> tt_eigh(
            const at::Tensor& a, c10::string_view uplo, bool compute_v
        ) {
            const char* op = "_linalg_eigh";
            KTT_REQUIRE(
                a.dim() >= 2 && a.size(-1) == a.size(-2), op, "needs square matrices"
            );
            const int64_t n = a.size(-1);
            auto batch = a.sizes().slice(0, a.dim() - 2).vec();
            auto value_sizes = batch;
            value_sizes.push_back(n);
            if (a.numel() == 0) {
                return {
                    at::empty(value_sizes, a.options()),
                    compute_v ? at::empty(a.sizes(), a.options())
                              : at::empty({0}, a.options())
                };
            }
            auto f = a.reshape({-1, n, n}).to(at::kFloat);
            auto tri = uplo == "U" ? f.triu() : f.tril();
            auto sym = tri + (uplo == "U" ? f.triu(1) : f.tril(-1)).transpose(1, 2);
            auto [values, vectors] = jacobi_eigh(sym);
            return {
                values.reshape(value_sizes).to(a.scalar_type()),
                compute_v ? vectors.reshape(a.sizes()).to(a.scalar_type())
                          : at::empty({0}, a.options())
            };
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_eigh_out(
            const at::Tensor& a,
            c10::string_view uplo,
            bool compute_v,
            at::Tensor& values,
            at::Tensor& vectors
        ) {
            auto [w, v] = tt_eigh(a, uplo, compute_v);
            into("_linalg_eigh.eigenvalues", values, w);
            if (compute_v) {
                into("_linalg_eigh.eigenvalues", vectors, v);
            }
            return {values, vectors};
        }

        // (U [B, m, n], S [B, n] descending, V [B, n, n]) of `a` [B, m, n] fp32, m >=
        // n, by one-sided Jacobi on the columns.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_svd(const at::Tensor& a) {
            const int64_t b = a.size(0), m = a.size(1), n = a.size(2);
            const int64_t even = n + (n % 2);
            auto u = pad_to(a, m, even);
            auto v = at::eye(even, u.options()).expand({b, even, even}).contiguous();
            auto sched = schedule(even, u.options());
            double previous = std::numeric_limits<double>::infinity();
            for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
                double worst = 0.0;
                for (size_t r = 0; r < sched.partner.size(); ++r) {
                    const auto& partner = sched.partner[r];
                    const auto& sign = sched.sign[r];
                    auto norms = (u * u).sum(1);
                    auto other = norms.index_select(1, partner);
                    auto cross = (u * u.index_select(2, partner)).sum(1);
                    auto is_p = at::lt(sign, 0.0);
                    auto [c, s] = (
                        rotation(
                            at::where(is_p, norms, other),
                            at::where(is_p, other, norms),
                            cross
                        )
                    );
                    auto ss = s * sign;
                    u =
                        (u * c.unsqueeze(1) +
                         u.index_select(2, partner) * ss.unsqueeze(1));
                    v =
                        (v * c.unsqueeze(1) +
                         v.index_select(2, partner) * ss.unsqueeze(1));
                    if (r + 1 == sched.partner.size()) {
                        auto scale = at::sqrt(norms * other);
                        auto rel =
                            (at::abs(cross) /
                             at::where(at::gt(scale, 0.0), scale, 1.0));
                        worst = rel.max().item<double>();
                    }
                }
                if (worst <= 1e-7 || (worst < 1e-5 && worst > 0.5 * previous)) {
                    break;
                }
                previous = worst;
            }
            v = reorthogonalize(v.narrow(1, 0, n).narrow(2, 0, n));
            u = product(a, v);
            auto norm2 = (u * u).sum(1);
            auto s = at::sqrt(norm2);
            s = (
                at::where(
                    at::gt(s, 0.0),
                    0.5 * (s + norm2 / at::where(at::gt(s, 0.0), s, 1.0)),
                    s
                )
            );
            auto [sorted, order] = s.sort(1, true);
            auto safe = at::where(at::gt(sorted, 0.0), sorted, 1.0);
            u = u.gather(2, order.unsqueeze(1).expand({b, m, n}));
            u = u / safe.unsqueeze(1);
            v = v.gather(2, order.unsqueeze(1).expand({b, n, n}));
            return {u, sorted, v};
        }

        // (U, S, Vh) of `a` [..., m, n]: reduced (U [..., m, k], Vh [..., k, n]) or
        // full; only S when !compute_uv (U, Vh empty).
        std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_svd(
            const at::Tensor& a,
            bool full_matrices,
            bool compute_uv,
            std::optional<c10::string_view> driver
        ) {
            const char* op = "_linalg_svd";
            KTT_REQUIRE(a.dim() >= 2, op, "needs matrices [..., m, n]");
            const int64_t m = a.size(-2), n = a.size(-1), k = std::min(m, n);
            KTT_REQUIRE(
                !compute_uv || !full_matrices || m == n || k == 0,
                op,
                "full_matrices for non-square inputs"
            );
            auto batch = a.sizes().slice(0, a.dim() - 2).vec();
            auto s_sizes = batch, u_sizes = batch, vh_sizes = batch;
            s_sizes.push_back(k);
            u_sizes.insert(u_sizes.end(), {m, full_matrices ? m : k});
            vh_sizes.insert(vh_sizes.end(), {full_matrices ? n : k, n});
            if (a.numel() == 0) {
                auto eye_or_zero = [&](const std::vector<int64_t>& sizes) {
                    auto z = at::zeros(sizes, a.options());
                    if (c10::multiply_integers(sizes) > 0) {
                        z = z + at::eye(sizes.end()[-2], sizes.back(), a.options());
                    }
                    return z;
                };
                return {
                    compute_uv ? eye_or_zero(u_sizes) : at::empty({0}, a.options()),
                    at::zeros(s_sizes, a.options()),
                    compute_uv ? eye_or_zero(vh_sizes) : at::empty({0}, a.options())
                };
            }
            auto f = a.reshape({-1, m, n}).to(at::kFloat);
            const bool wide = m < n;
            auto [u, s, v] = jacobi_svd(wide ? f.transpose(1, 2).contiguous() : f);
            if (wide) {
                std::swap(u, v);
            }
            auto dtype = a.scalar_type();
            if (!compute_uv) {
                return {
                    at::empty({0}, a.options()),
                    s.reshape(s_sizes).to(dtype),
                    at::empty({0}, a.options())
                };
            }
            return {
                u.reshape(u_sizes).to(dtype),
                s.reshape(s_sizes).to(dtype),
                v.transpose(1, 2).reshape(vh_sizes).to(dtype)
            };
        }

        std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> tt_svd_out(
            const at::Tensor& a,
            bool full_matrices,
            bool compute_uv,
            std::optional<c10::string_view> driver,
            at::Tensor& u,
            at::Tensor& s,
            at::Tensor& vh
        ) {
            auto [uu, ss, vv] = tt_svd(a, full_matrices, compute_uv, driver);
            into("_linalg_svd.U", s, ss);
            if (compute_uv) {
                into("_linalg_svd.U", u, uu);
                into("_linalg_svd.U", vh, vv);
            }
            return {u, s, vh};
        }

        // Tolerance per matrix [..., 1]: max(atol, rtol * largest) with PyTorch's
        // defaults (rtol = eps * max(m, n) when atol is unset or 0, else 0).
        at::Tensor tolerance(
            const at::Tensor& largest,
            const std::optional<at::Tensor>& atol_t,
            const std::optional<at::Tensor>& rtol_t,
            std::optional<double> atol,
            std::optional<double> rtol,
            int64_t m,
            int64_t n
        ) {
            const bool has_atol = (atol_t && atol_t->defined()) || (atol && *atol > 0);
            auto a =
                (atol_t && atol_t->defined()
                     ? atol_t->to(at::kFloat).unsqueeze(-1)
                     : at::full_like(largest, atol.value_or(0.0)));
            at::Tensor r;
            if (rtol_t && rtol_t->defined()) {
                r = rtol_t->to(at::kFloat).unsqueeze(-1);
            } else {
                const double eps = std::numeric_limits<float>::epsilon();
                r = (
                    at::full_like(
                        largest, rtol.value_or(has_atol ? 0.0 : eps * std::max(m, n))
                    )
                );
            }
            return at::maximum(a, r * largest);
        }

        // Singular values (|eigenvalues| when `hermitian`) of `a` [..., m, n], fp32.
        at::Tensor spectrum(const at::Tensor& a, bool hermitian) {
            if (hermitian) {
                auto [w, v] = tt_eigh(a.to(at::kFloat), "L", false);
                return at::abs(w);
            }
            auto [u, s, vh] = tt_svd(a.to(at::kFloat), false, false, std::nullopt);
            return s;
        }

        at::Tensor matrix_rank(
            const at::Tensor& a,
            const std::optional<at::Tensor>& atol_t,
            const std::optional<at::Tensor>& rtol_t,
            std::optional<double> atol,
            std::optional<double> rtol,
            bool hermitian
        ) {
            const int64_t m = a.size(-2), n = a.size(-1);
            auto batch = a.sizes().slice(0, a.dim() - 2).vec();
            if (a.numel() == 0) {
                return at::zeros(batch, a.options().dtype(at::kLong));
            }
            auto s = spectrum(a, hermitian);
            auto largest = s.amax(-1, true);
            auto tol = tolerance(largest, atol_t, rtol_t, atol, rtol, m, n);
            return at::gt(s, tol).to(at::kLong).sum(-1);
        }

        at::Tensor tt_rank_tensor(
            const at::Tensor& a,
            const std::optional<at::Tensor>& atol,
            const std::optional<at::Tensor>& rtol,
            bool hermitian
        ) {
            return matrix_rank(a, atol, rtol, std::nullopt, std::nullopt, hermitian);
        }

        at::Tensor tt_rank_float(
            const at::Tensor& a,
            std::optional<double> atol,
            std::optional<double> rtol,
            bool hermitian
        ) {
            return matrix_rank(a, std::nullopt, std::nullopt, atol, rtol, hermitian);
        }

        at::Tensor tt_rank_tol(const at::Tensor& a, double tol, bool hermitian) {
            return matrix_rank(a, std::nullopt, std::nullopt, tol, 0.0, hermitian);
        }

        at::Tensor tt_rank_tol_tensor(
            const at::Tensor& a, const at::Tensor& tol, bool hermitian
        ) {
            return matrix_rank(a, tol, std::nullopt, std::nullopt, 0.0, hermitian);
        }

        // Pseudo-inverse V diag(1/s) U^T over singular values above the tolerance
        // (V diag(1/w) V^T over eigenvalues when `hermitian`).
        at::Tensor pinv(
            const at::Tensor& a,
            const std::optional<at::Tensor>& atol_t,
            const std::optional<at::Tensor>& rtol_t,
            std::optional<double> atol,
            std::optional<double> rtol,
            bool hermitian
        ) {
            const int64_t m = a.size(-2), n = a.size(-1);
            auto out_sizes = a.sizes().vec();
            std::swap(out_sizes.end()[-1], out_sizes.end()[-2]);
            if (a.numel() == 0) {
                return at::zeros(out_sizes, a.options());
            }
            auto f = a.to(at::kFloat);
            at::Tensor left, values, right;  // result = left diag(1/values) right
            if (hermitian) {
                auto [w, v] = tt_eigh(f, "L", true);
                left = v;
                values = w;
                right = v.transpose(-2, -1);
            } else {
                auto [u, s, vh] = tt_svd(f, false, true, std::nullopt);
                left = vh.transpose(-2, -1);
                values = s;
                right = u.transpose(-2, -1);
            }
            auto largest = at::abs(values).amax(-1, true);
            auto tol = tolerance(largest, atol_t, rtol_t, atol, rtol, m, n);
            auto keep = at::gt(at::abs(values), tol);
            auto inv = at::where(keep, 1.0 / at::where(keep, values, 1.0), 0.0);
            auto scaled = left * inv.unsqueeze(-2);
            auto result = (scaled.unsqueeze(-1) * right.unsqueeze(-3)).sum(-2);
            return result.to(a.scalar_type());
        }

        at::Tensor tt_pinv_tensor(
            const at::Tensor& a,
            const std::optional<at::Tensor>& atol,
            const std::optional<at::Tensor>& rtol,
            bool hermitian
        ) {
            return pinv(a, atol, rtol, std::nullopt, std::nullopt, hermitian);
        }

        at::Tensor tt_pinv_float(
            const at::Tensor& a,
            std::optional<double> atol,
            std::optional<double> rtol,
            bool hermitian
        ) {
            return pinv(a, std::nullopt, std::nullopt, atol, rtol, hermitian);
        }

        at::Tensor tt_pinv_rcond(const at::Tensor& a, double rcond, bool hermitian) {
            return pinv(a, std::nullopt, std::nullopt, 0.0, rcond, hermitian);
        }

        at::Tensor tt_pinv_rcond_tensor(
            const at::Tensor& a, const at::Tensor& rcond, bool hermitian
        ) {
            return pinv(a, std::nullopt, rcond, 0.0, std::nullopt, hermitian);
        }

        // Square root of symmetric `a` [..., n, n]: V diag(sqrt(w)) V^T from eigh.
        at::Tensor tt_matrix_sqrth(const at::Tensor& a) {
            KTT_REQUIRE(
                a.dim() >= 2 && a.size(-1) == a.size(-2),
                "linalg_matrix_sqrth",
                "needs square matrices"
            );
            if (a.numel() == 0) {
                return at::empty(a.sizes(), a.options());
            }
            const int64_t n = a.size(-1);
            auto [w, v] = tt_eigh(a.to(at::kFloat), "L", true);
            auto vb = v.reshape({-1, n, n});
            auto scaled = vb * at::sqrt(w.reshape({-1, n})).unsqueeze(1);
            auto out = product(scaled, vb.transpose(1, 2));
            return out.reshape(a.sizes()).to(a.scalar_type());
        }

        at::Tensor& tt_matrix_sqrth_out(const at::Tensor& a, at::Tensor& out) {
            return into("linalg_matrix_sqrth.out", out, tt_matrix_sqrth(a));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("linalg_matrix_sqrth", &tt_matrix_sqrth);
        m.impl("linalg_matrix_sqrth.out", &tt_matrix_sqrth_out);
        m.impl("linalg_matrix_rank.atol_rtol_tensor", &tt_rank_tensor);
        m.impl("linalg_matrix_rank.atol_rtol_float", &tt_rank_float);
        m.impl("linalg_matrix_rank", &tt_rank_tol);
        m.impl("linalg_matrix_rank.tol_tensor", &tt_rank_tol_tensor);
        m.impl("linalg_pinv.atol_rtol_tensor", &tt_pinv_tensor);
        m.impl("linalg_pinv.atol_rtol_float", &tt_pinv_float);
        m.impl("linalg_pinv", &tt_pinv_rcond);
        m.impl("linalg_pinv.rcond_tensor", &tt_pinv_rcond_tensor);
        m.impl("_linalg_eigh", &tt_eigh);
        m.impl("_linalg_eigh.eigenvalues", &tt_eigh_out);
        m.impl("_linalg_svd", &tt_svd);
        m.impl("_linalg_svd.U", &tt_svd_out);
    }

}  // namespace ktt
