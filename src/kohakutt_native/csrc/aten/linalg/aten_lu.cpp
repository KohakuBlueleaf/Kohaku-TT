// aten LU-based linear algebra for tt: linalg_lu_factor_ex, _linalg_det,
// _linalg_slogdet, linalg_inv_ex (+ out forms). LU with partial pivoting by row
// elimination in fp32; the inverse by Gauss-Jordan elimination.
#include <torch/library.h>

#include <algorithm>
#include <limits>
#include <string>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // Packed LU factorization of a [B, n, n] batch: P A = L U.
        struct Lu {
            at::Tensor
                lu;  // [B, n, n] fp32: L below the diagonal (unit), U on and above
            at::Tensor
                pivots;       // [B, n] int32: 1-based row swapped with row k at step k
            at::Tensor sign;  // [B] fp32: determinant of P
        };

        // LU of `a` [..., n, n] (n >= 1) flattened to [B, n, n]; `pivot` false keeps
        // the row order.
        Lu factor(const at::Tensor& a, bool pivot) {
            const int64_t m = a.size(-2), n = a.size(-1);
            auto lu = a.reshape({-1, m, n}).to(at::kFloat).clone();
            const int64_t b = lu.size(0);
            auto rows = at::arange(m, lu.options().dtype(at::kLong));
            auto cols = at::arange(n, lu.options().dtype(at::kLong));
            auto grid = rows.unsqueeze(0).expand({b, m}).contiguous();
            auto sign = at::ones({b}, lu.options());
            std::vector<at::Tensor> pivots;
            for (int64_t k = 0; k < std::min(m, n); ++k) {
                at::Tensor p;
                if (pivot) {
                    auto col = at::where(rows < k, -1.0, lu.select(2, k).abs());
                    p = at::argmax(col, 1, true);
                    auto swap = at::where(grid == k, p, at::where(grid == p, k, grid));
                    lu = lu.gather(1, swap.unsqueeze(2).expand({b, m, n}));
                    sign = at::where(p.squeeze(1) != k, -sign, sign);
                } else {
                    p = at::full({b, 1}, k, grid.options());
                }
                pivots.push_back(p);
                auto head = lu.select(1, k).select(1, k);
                auto safe = at::where(head == 0, 1.0, head);
                auto below = (rows > k).to(at::kFloat);
                auto right = (cols > k).to(at::kFloat);
                auto column = (cols == k).to(at::kFloat);
                auto l = lu.select(2, k) / safe.unsqueeze(1) * below;
                auto u = lu.select(1, k) * right;
                lu = lu - l.unsqueeze(2) * u.unsqueeze(1);
                auto mask = below.unsqueeze(1) * column.unsqueeze(0);
                lu = lu * (1 - mask) + l.unsqueeze(2) * column.unsqueeze(0);
            }
            return {lu, (at::cat(pivots, 1) + 1).to(at::kInt), sign};
        }

        // (X = A^-1 R [B, n, r] fp32, LAPACK info [B] int32) for `a` [B, n, n] (n >= 1)
        // and `rhs` [B, n, r] by Gauss-Jordan elimination with partial pivoting on
        // [A | R].
        std::pair<at::Tensor, at::Tensor> gauss_jordan(
            const at::Tensor& a, const at::Tensor& rhs
        ) {
            const int64_t n = a.size(-1), r = rhs.size(-1);
            auto f = a.to(at::kFloat);
            const int64_t b = f.size(0);
            auto m = at::cat({f, rhs.to(at::kFloat)}, -1);
            auto rows = at::arange(n, f.options().dtype(at::kLong));
            auto grid = rows.unsqueeze(0).expand({b, n}).contiguous();
            std::vector<at::Tensor> zeros;
            for (int64_t k = 0; k < n; ++k) {
                auto col = at::where(rows < k, -1.0, m.select(2, k).abs());
                auto p = at::argmax(col, 1, true);
                auto swap = at::where(grid == k, p, at::where(grid == p, k, grid));
                m = m.gather(1, swap.unsqueeze(2).expand({b, n, n + r}));
                auto head = m.select(1, k).select(1, k);
                zeros.push_back((head == 0).to(at::kFloat).unsqueeze(1));
                auto safe = at::where(head == 0, 1.0, head);
                auto column = (rows == k).to(at::kFloat);
                auto pivot_row = m.select(1, k) / safe.unsqueeze(1);
                auto factors = m.select(2, k) * (1 - column);
                m = m - factors.unsqueeze(2) * pivot_row.unsqueeze(1);
                auto mask = column.unsqueeze(1);
                m = m * (1 - mask) + pivot_row.unsqueeze(1) * mask;
            }
            auto zero = at::cat(zeros, 1);
            auto info = at::where(zero.amax(-1) > 0, at::argmax(zero, -1) + 1, 0);
            return {m.narrow(-1, n, r), info.to(at::kInt)};
        }

        // A^-1 [B, n, n] and info of `a` [..., n, n] (n >= 1), flattened to a batch.
        std::pair<at::Tensor, at::Tensor> invert(const at::Tensor& a) {
            const int64_t n = a.size(-1);
            auto f = a.reshape({-1, n, n});
            auto eye = at::eye(n, f.options().dtype(at::kFloat));
            return gauss_jordan(f, eye.expand({f.size(0), n, n}));
        }

        // Row k of `y` [B, n, r] replaced by `row` [B, r].
        at::Tensor with_row(const at::Tensor& y, int64_t k, const at::Tensor& row) {
            auto rows = at::arange(y.size(1), y.options().dtype(at::kLong));
            auto mask = (rows == k).to(at::kFloat).unsqueeze(1);
            return y * (1 - mask) + row.unsqueeze(1) * mask;
        }

        // Y with T Y = R for `t` [B, n, n] triangular (lower: rows eliminated top
        // down; upper: bottom up), reading only that triangle; unit diagonal when
        // `unit`.
        at::Tensor substitute(
            const at::Tensor& t, at::Tensor y, bool lower, bool unit
        ) {
            const int64_t n = t.size(-1);
            auto rows = at::arange(n, y.options().dtype(at::kLong));
            for (int64_t s = 0; s < n; ++s) {
                const int64_t k = lower ? s : n - 1 - s;
                auto yk = y.select(1, k);
                if (!unit) {
                    yk = yk / t.select(1, k).select(1, k).unsqueeze(1);
                    y = with_row(y, k, yk);
                }
                auto rest = (lower ? rows > k : rows < k).to(at::kFloat);
                auto factors = t.select(2, k) * rest;
                y = y - factors.unsqueeze(2) * yk.unsqueeze(1);
            }
            return y;
        }

        // Rows of `y` [B, n, r] swapped by LAPACK `pivots` [B, n] (1-based), in step
        // order, or reversed (P^T) when `reverse`.
        at::Tensor apply_pivots(at::Tensor y, const at::Tensor& pivots, bool reverse) {
            const int64_t b = y.size(0), n = y.size(1), r = y.size(2);
            auto rows = at::arange(n, y.options().dtype(at::kLong));
            auto grid = rows.unsqueeze(0).expand({b, n}).contiguous();
            auto p_all = pivots.to(at::kLong) - 1;
            const int64_t steps = pivots.size(1);
            for (int64_t s = 0; s < steps; ++s) {
                const int64_t k = reverse ? steps - 1 - s : s;
                auto p = p_all.select(1, k).unsqueeze(1);
                auto swap = at::where(grid == k, p, at::where(grid == p, k, grid));
                y = y.gather(1, swap.unsqueeze(2).expand({b, n, r}));
            }
            return y;
        }

        // X with A X = R (A^H X = R when `adjoint`) from A's packed `lu` [B, n, n] and
        // `pivots` [B, n]; `rhs` [B, n, r]. Elementwise fp32 substitution.
        at::Tensor lu_solve_rows(
            const at::Tensor& lu,
            const at::Tensor& pivots,
            const at::Tensor& rhs,
            bool adjoint
        ) {
            auto f = lu.to(at::kFloat);
            auto y = rhs.to(at::kFloat);
            if (!adjoint) {
                y = apply_pivots(y, pivots, false);
                y = substitute(f, y, true, true);
                return substitute(f, y, false, false);
            }
            auto ft = f.transpose(1, 2);
            y = substitute(ft, y, true, false);
            y = substitute(ft, y, false, true);
            return apply_pivots(y, pivots, true);
        }

        // Batch sizes of `a` [..., n, n] followed by `tail`.
        std::vector<int64_t> batch_sizes(
            const at::Tensor& a, std::initializer_list<int64_t> tail
        ) {
            auto sizes = a.sizes().slice(0, a.dim() - 2).vec();
            sizes.insert(sizes.end(), tail);
            return sizes;
        }

        // LAPACK info: 1-based index of the first zero on U's diagonal, else 0.
        at::Tensor zero_pivot_info(const at::Tensor& lu) {
            auto zero = (lu.diagonal(0, -2, -1) == 0).to(at::kFloat);
            auto first = at::argmax(zero, -1) + 1;
            return at::where(zero.amax(-1) > 0, first, 0).to(at::kInt);
        }

        void require_square(const char* op, const at::Tensor& a) {
            KTT_REQUIRE(
                a.dim() >= 2 && a.size(-1) == a.size(-2),
                op,
                "needs square matrices [..., n, n]"
            );
        }

        std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_lu_factor_ex(
            const at::Tensor& a, bool pivot, bool check_errors
        ) {
            const char* op = "linalg_lu_factor_ex";
            KTT_REQUIRE(a.dim() >= 2, op, "needs matrices [..., m, n]");
            const int64_t n = std::min(a.size(-2), a.size(-1));
            if (a.numel() == 0) {
                return {
                    at::empty(a.sizes(), a.options()),
                    at::empty(batch_sizes(a, {n}), a.options().dtype(at::kInt)),
                    at::zeros(batch_sizes(a, {}), a.options().dtype(at::kInt))
                };
            }
            auto f = factor(a, pivot);
            auto info = zero_pivot_info(f.lu).reshape(batch_sizes(a, {}));
            if (check_errors) {
                TORCH_CHECK(
                    !info.ne(0).any().item<bool>(),
                    "linalg.lu_factor: U has a zero on its diagonal (singular input)"
                );
            }
            return {
                f.lu.reshape(a.sizes()).to(a.scalar_type()),
                f.pivots.reshape(batch_sizes(a, {n})),
                info
            };
        }

        std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> tt_lu_factor_ex_out(
            const at::Tensor& a,
            bool pivot,
            bool check_errors,
            at::Tensor& lu,
            at::Tensor& pivots,
            at::Tensor& info
        ) {
            auto [l, p, i] = tt_lu_factor_ex(a, pivot, check_errors);
            into("linalg_lu_factor_ex.out", lu, l);
            into("linalg_lu_factor_ex.out", pivots, p);
            into("linalg_lu_factor_ex.out", info, i);
            return {lu, pivots, info};
        }

        // (sign, log |det|, LU, pivots); a singular matrix gives (0, -inf).
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tt_slogdet(
            const at::Tensor& a
        ) {
            require_square("_linalg_slogdet", a);
            const int64_t n = a.size(-1);
            auto dtype = a.scalar_type();
            if (a.numel() == 0) {
                return {
                    at::ones(batch_sizes(a, {}), a.options()),
                    at::zeros(batch_sizes(a, {}), a.options()),
                    at::empty(a.sizes(), a.options()),
                    at::empty(batch_sizes(a, {n}), a.options().dtype(at::kInt))
                };
            }
            auto f = factor(a, true);
            auto diag = f.lu.diagonal(0, -2, -1);
            auto sign = f.sign * at::sign(diag).prod(-1);
            auto logabs = at::log(diag.abs()).sum(-1);
            return {
                sign.reshape(batch_sizes(a, {})).to(dtype),
                logabs.reshape(batch_sizes(a, {})).to(dtype),
                f.lu.reshape(a.sizes()).to(dtype),
                f.pivots.reshape(batch_sizes(a, {n}))
            };
        }

        std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> tt_slogdet_out(
            const at::Tensor& a,
            at::Tensor& sign,
            at::Tensor& logabs,
            at::Tensor& lu,
            at::Tensor& pivots
        ) {
            auto [s, l, f, p] = tt_slogdet(a);
            into("_linalg_slogdet.sign", sign, s);
            into("_linalg_slogdet.sign", logabs, l);
            into("_linalg_slogdet.sign", lu, f);
            into("_linalg_slogdet.sign", pivots, p);
            return {sign, logabs, lu, pivots};
        }

        // (det, LU, pivots): det = sign(P) * prod(diag U).
        std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_det(const at::Tensor& a) {
            require_square("_linalg_det", a);
            const int64_t n = a.size(-1);
            if (a.numel() == 0) {
                return {
                    at::ones(batch_sizes(a, {}), a.options()),
                    at::empty(a.sizes(), a.options()),
                    at::empty(batch_sizes(a, {n}), a.options().dtype(at::kInt))
                };
            }
            auto f = factor(a, true);
            auto det = f.sign * f.lu.diagonal(0, -2, -1).prod(-1);
            return {
                det.reshape(batch_sizes(a, {})).to(a.scalar_type()),
                f.lu.reshape(a.sizes()).to(a.scalar_type()),
                f.pivots.reshape(batch_sizes(a, {n}))
            };
        }

        std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> tt_det_out(
            const at::Tensor& a, at::Tensor& result, at::Tensor& lu, at::Tensor& pivots
        ) {
            auto [d, l, p] = tt_det(a);
            into("_linalg_det.result", result, d);
            into("_linalg_det.result", lu, l);
            into("_linalg_det.result", pivots, p);
            return {result, lu, pivots};
        }

        // (A^-1, info): A^-1 = U^-1 L^-1 P with P A = L U.
        std::tuple<at::Tensor, at::Tensor> tt_inv_ex(
            const at::Tensor& a, bool check_errors
        ) {
            require_square("linalg_inv_ex", a);
            const int64_t n = a.size(-1);
            if (a.numel() == 0) {
                return {
                    at::empty(a.sizes(), a.options()),
                    at::zeros(batch_sizes(a, {}), a.options().dtype(at::kInt))
                };
            }
            auto [inv, zero] = invert(a);
            auto info = zero.reshape(batch_sizes(a, {}));
            if (check_errors) {
                TORCH_CHECK(
                    !info.ne(0).any().item<bool>(),
                    "linalg.inv: the input is singular (U has a zero on its diagonal)"
                );
            }
            return {inv.reshape(a.sizes()).to(a.scalar_type()), info};
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_inv_ex_out(
            const at::Tensor& a,
            bool check_errors,
            at::Tensor& inverse,
            at::Tensor& info
        ) {
            auto [x, i] = tt_inv_ex(a, check_errors);
            into("linalg_inv_ex.inverse", inverse, x);
            into("linalg_inv_ex.inverse", info, i);
            return {inverse, info};
        }

        // X with A X = B (X A = B when !left; A^H for `adjoint`) from A's `lu` [..., n,
        // n] and `pivots`; B [..., n, r] (left) or [..., r, n], batch dims broadcast.
        at::Tensor solve_with_lu(
            const at::Tensor& lu,
            const at::Tensor& pivots,
            const at::Tensor& b,
            bool left,
            bool adjoint
        ) {
            const int64_t n = lu.size(-1);
            auto rhs = left ? b : b.transpose(-2, -1);
            if (!left) {
                adjoint = !adjoint;  // X A = B  <=>  A^T X^T = B^T (real)
            }
            auto batch = (at::infer_size(
                lu.sizes().slice(0, lu.dim() - 2), rhs.sizes().slice(0, rhs.dim() - 2)
            ));
            const int64_t r = rhs.size(-1);
            auto lu_sizes = batch, rhs_sizes = batch, piv_sizes = batch;
            lu_sizes.insert(lu_sizes.end(), {n, n});
            rhs_sizes.insert(rhs_sizes.end(), {n, r});
            piv_sizes.push_back(n);
            auto x = lu_solve_rows(
                lu.expand(lu_sizes).reshape({-1, n, n}),
                pivots.expand(piv_sizes).reshape({-1, n}),
                rhs.expand(rhs_sizes).reshape({-1, n, r}),
                adjoint
            );
            x = x.reshape(rhs_sizes);
            return left ? x : x.transpose(-2, -1);
        }

        at::Tensor tt_lu_solve(
            const at::Tensor& lu,
            const at::Tensor& pivots,
            const at::Tensor& b,
            bool left,
            bool adjoint
        ) {
            const char* op = "linalg_lu_solve";
            require_square(op, lu);
            KTT_REQUIRE(b.dim() >= 2, op, "B must be at least 2-D");
            auto dtype = at::result_type(lu, b);
            if (lu.numel() == 0 || b.numel() == 0) {
                return at::zeros_like(b, b.options().dtype(dtype));
            }
            return solve_with_lu(lu, pivots, b, left, adjoint).to(dtype);
        }

        at::Tensor& tt_lu_solve_out(
            const at::Tensor& lu,
            const at::Tensor& pivots,
            const at::Tensor& b,
            bool left,
            bool adjoint,
            at::Tensor& out
        ) {
            return (into(
                "linalg_lu_solve.out", out, tt_lu_solve(lu, pivots, b, left, adjoint)
            ));
        }

        // (X, LU, pivots, info) with A X = B (X A = B when !left); B a vector or a
        // batch of vectors under PyTorch's rule.
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tt_solve_ex(
            const at::Tensor& a, const at::Tensor& b, bool left, bool check_errors
        ) {
            const char* op = "_linalg_solve_ex";
            require_square(op, a);
            const int64_t n = a.size(-1);
            const bool vector =
                (b.dim() == 1 ||
                 (b.dim() == a.dim() - 1 &&
                  b.sizes() == a.sizes().slice(0, a.dim() - 1)));
            auto rhs = vector ? b.unsqueeze(left ? -1 : -2) : b;
            auto dtype = at::result_type(a, b);
            auto [lu, pivots, info] = tt_lu_factor_ex(a, true, false);
            if (check_errors && a.numel() > 0) {
                TORCH_CHECK(
                    !info.ne(0).any().item<bool>(),
                    "linalg.solve: the input is singular (U has a zero on its diagonal)"
                );
            }
            at::Tensor x;
            if (a.numel() == 0 || rhs.numel() == 0) {
                auto batch = (at::infer_size(
                    a.sizes().slice(0, a.dim() - 2), rhs.sizes().slice(0, rhs.dim() - 2)
                ));
                batch.push_back(rhs.size(-2));
                batch.push_back(rhs.size(-1));
                x = at::zeros(batch, b.options().dtype(dtype));
            } else {
                x = solve_with_lu(lu, pivots, rhs, left, false).to(dtype);
            }
            if (vector) {
                x = x.squeeze(left ? -1 : -2);
            }
            return {x, lu, pivots, info};
        }

        std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> tt_solve_ex_out(
            const at::Tensor& a,
            const at::Tensor& b,
            bool left,
            bool check_errors,
            at::Tensor& result,
            at::Tensor& lu,
            at::Tensor& pivots,
            at::Tensor& info
        ) {
            auto [x, l, p, i] = tt_solve_ex(a, b, left, check_errors);
            into("_linalg_solve_ex.result", result, x);
            into("_linalg_solve_ex.result", lu, l);
            into("_linalg_solve_ex.result", pivots, p);
            into("_linalg_solve_ex.result", info, i);
            return {result, lu, pivots, info};
        }

        // (P, L, U) with A = P L U from packed `lu` [..., m, n] and `pivots` [..., k],
        // k = min(m, n): L [..., m, k] unit lower, U [..., k, n]; empty tensors for
        // the parts not unpacked.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_lu_unpack(
            const at::Tensor& lu,
            const at::Tensor& pivots,
            bool unpack_data,
            bool unpack_pivots
        ) {
            const int64_t m = lu.size(-2), n = lu.size(-1), k = std::min(m, n);
            at::Tensor p, l, u;
            if (unpack_data) {
                l = lu.narrow(-1, 0, k).tril(-1) + at::eye(m, k, lu.options());
                u = lu.narrow(-2, 0, k).triu();
            } else {
                l = at::empty({0}, lu.options());
                u = at::empty({0}, lu.options());
            }
            if (unpack_pivots && lu.numel() > 0) {
                auto batch = lu.sizes().slice(0, lu.dim() - 2).vec();
                auto eye = at::eye(m, lu.options().dtype(at::kFloat));
                auto flat = pivots.reshape({-1, k});
                auto perm = apply_pivots(
                    eye.expand({flat.size(0), m, m}).contiguous(), flat, false
                );
                batch.push_back(m);
                batch.push_back(m);
                p = perm.transpose(1, 2).reshape(batch).to(lu.scalar_type());
            } else if (unpack_pivots) {
                auto batch = lu.sizes().slice(0, lu.dim() - 2).vec();
                batch.push_back(m);
                batch.push_back(m);
                p = at::zeros(batch, lu.options());
                if (m > 0 && c10::multiply_integers(batch) > 0) {
                    p = p + at::eye(m, lu.options());
                }
            } else {
                p = at::empty({0}, lu.options());
            }
            return {p, l, u};
        }

        std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> tt_lu_unpack_out(
            const at::Tensor& lu,
            const at::Tensor& pivots,
            bool unpack_data,
            bool unpack_pivots,
            at::Tensor& p,
            at::Tensor& l,
            at::Tensor& u
        ) {
            auto [pp, ll, uu] = tt_lu_unpack(lu, pivots, unpack_data, unpack_pivots);
            into("lu_unpack.out", p, pp);
            into("lu_unpack.out", l, ll);
            into("lu_unpack.out", u, uu);
            return {p, l, u};
        }

        // (P, L, U) with A = P L U for `a` [..., m, n]; P empty when !pivot.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_linalg_lu(
            const at::Tensor& a, bool pivot
        ) {
            auto [lu, pivots, info] = tt_lu_factor_ex(a, pivot, false);
            auto [p, l, u] = tt_lu_unpack(lu, pivots, true, pivot);
            return {p, l, u};
        }

        std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> tt_linalg_lu_out(
            const at::Tensor& a, bool pivot, at::Tensor& p, at::Tensor& l, at::Tensor& u
        ) {
            auto [pp, ll, uu] = tt_linalg_lu(a, pivot);
            into("linalg_lu.out", p, pp);
            into("linalg_lu.out", l, ll);
            into("linalg_lu.out", u, uu);
            return {p, l, u};
        }

        // (L, info) with A = L L^T for `a` [..., n, n] by right-looking elimination in
        // fp32 (L^T when `upper`); info is the 1-based first non-positive pivot.
        std::tuple<at::Tensor, at::Tensor> tt_cholesky_ex(
            const at::Tensor& a, bool upper, bool check_errors
        ) {
            const char* op = "linalg_cholesky_ex";
            require_square(op, a);
            const int64_t n = a.size(-1);
            if (a.numel() == 0) {
                return {
                    at::empty(a.sizes(), a.options()),
                    at::zeros(batch_sizes(a, {}), a.options().dtype(at::kInt))
                };
            }
            auto w = a.reshape({-1, n, n}).to(at::kFloat);
            auto rows = at::arange(n, w.options().dtype(at::kLong));
            auto l = at::zeros_like(w);
            std::vector<at::Tensor> bad;
            for (int64_t j = 0; j < n; ++j) {
                auto head = w.select(1, j).select(1, j);
                bad.push_back(at::le(head, 0.0).to(at::kFloat).unsqueeze(1));
                auto d = at::sqrt(at::where(at::gt(head, 0.0), head, 1.0));
                auto below = (rows > j).to(at::kFloat);
                auto column = (rows == j).to(at::kFloat);
                auto c =
                    (w.select(2, j) / d.unsqueeze(1) * below + column * d.unsqueeze(1));
                l = l + c.unsqueeze(2) * column;
                auto rest = c * below;
                w = w - rest.unsqueeze(2) * rest.unsqueeze(1);
            }
            auto flags = at::cat(bad, 1);
            auto info = at::where(flags.amax(-1) > 0, at::argmax(flags, -1) + 1, 0)
                            .to(at::kInt)
                            .reshape(batch_sizes(a, {}));
            if (check_errors) {
                TORCH_CHECK(
                    !info.ne(0).any().item<bool>(),
                    "linalg.cholesky: the input is not positive-definite"
                );
            }
            auto out = upper ? l.transpose(1, 2) : l;
            return {out.reshape(a.sizes()).to(a.scalar_type()), info};
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_cholesky_ex_out(
            const at::Tensor& a,
            bool upper,
            bool check_errors,
            at::Tensor& l,
            at::Tensor& info
        ) {
            auto [ll, ii] = tt_cholesky_ex(a, upper, check_errors);
            into("linalg_cholesky_ex.L", l, ll);
            into("linalg_cholesky_ex.L", info, ii);
            return {l, info};
        }

        // X with A X = B, A = L L^T from its factor `factor` (L, or U = L^T when
        // `upper`); batch dims already broadcast.
        at::Tensor tt_cholesky_solve_helper(
            const at::Tensor& b, const at::Tensor& factor, bool upper
        ) {
            if (b.numel() == 0 || factor.numel() == 0) {
                return at::zeros_like(b);
            }
            const int64_t n = factor.size(-1), r = b.size(-1);
            auto f = factor.reshape({-1, n, n}).to(at::kFloat);
            auto l = upper ? f.transpose(1, 2) : f;
            auto y = b.reshape({-1, n, r}).to(at::kFloat);
            y = substitute(l, y, true, false);
            y = substitute(l.transpose(1, 2), y, false, false);
            return y.reshape(b.sizes()).to(b.scalar_type());
        }

        // (L L^T)^-1 (or (U^T U)^-1 when `upper`) from the Cholesky factor.
        at::Tensor tt_cholesky_inverse(const at::Tensor& factor, bool upper) {
            require_square("cholesky_inverse", factor);
            if (factor.numel() == 0) {
                return at::empty(factor.sizes(), factor.options());
            }
            const int64_t n = factor.size(-1);
            auto eye = at::eye(n, factor.options().dtype(at::kFloat));
            auto b = eye.expand(factor.sizes()).contiguous().to(factor.scalar_type());
            return tt_cholesky_solve_helper(b, factor, upper);
        }

        at::Tensor& tt_cholesky_inverse_out(
            const at::Tensor& factor, bool upper, at::Tensor& out
        ) {
            return (
                into("cholesky_inverse.out", out, tt_cholesky_inverse(factor, upper))
            );
        }

        // (Q [B, m, m], R [B, m, n]) with A = Q R for `a` [B, m, n] by Householder
        // reflections in fp32 (LAPACK's signs: no reflection for a column already
        // zero below the diagonal).
        std::pair<at::Tensor, at::Tensor> householder(const at::Tensor& a) {
            const int64_t b = a.size(0), m = a.size(1), n = a.size(2);
            auto r = a.to(at::kFloat);
            auto q = at::eye(m, r.options()).expand({b, m, m}).contiguous();
            auto rows = at::arange(m, r.options().dtype(at::kLong));
            for (int64_t j = 0; j < std::min(m, n); ++j) {
                auto column = r.select(2, j);
                auto below = column * (rows > j).to(at::kFloat);
                auto below2 = (below * below).sum(1);
                auto alpha = column.select(1, j);
                auto norm = at::sqrt(below2 + alpha * alpha);
                auto sign = at::where(at::ge(alpha, 0.0), 1.0, -1.0);
                auto unit = (rows == j).to(at::kFloat);
                auto v = below + (alpha + sign * norm).unsqueeze(1) * unit;
                auto vnorm = at::sqrt((v * v).sum(1));
                auto skip = at::eq(below2, 0.0);
                auto scale = at::where(skip, 0.0, 1.0 / at::where(skip, 1.0, vnorm));
                v = v * scale.unsqueeze(1);
                auto vr = (v.unsqueeze(2) * r).sum(1, true);
                r = r - 2.0 * v.unsqueeze(2) * vr;
                auto qv = (q * v.unsqueeze(1)).sum(2, true);
                q = q - 2.0 * qv * v.unsqueeze(1);
            }
            return {q, r.triu()};
        }

        // LAPACK geqrf of `a` [B, m, n] fp32: (packed [B, m, n] with R on and above the
        // diagonal and each reflector's v below it (v[j] = 1 implied), tau [B, k]);
        // H_j = I - tau_j v_j v_j^T, beta = -sign(alpha) ||x||; tau 0 for a column
        // already zero below the diagonal.
        std::pair<at::Tensor, at::Tensor> geqrf_rows(const at::Tensor& a) {
            const int64_t m = a.size(1), n = a.size(2), k = std::min(m, n);
            auto r = a.to(at::kFloat);
            auto rows = at::arange(m, r.options().dtype(at::kLong));
            auto cols = at::arange(n, r.options().dtype(at::kLong));
            std::vector<at::Tensor> taus;
            for (int64_t j = 0; j < k; ++j) {
                auto column = r.select(2, j);
                auto below_mask = (rows > j).to(at::kFloat);
                auto below = column * below_mask;
                auto below2 = (below * below).sum(1);
                auto alpha = column.select(1, j);
                auto norm = at::sqrt(below2 + alpha * alpha);
                auto beta = at::where(at::ge(alpha, 0.0), -norm, norm);
                auto skip = at::eq(below2, 0.0);
                auto safe_beta = at::where(skip, 1.0, beta);
                auto denom = at::where(skip, 1.0, alpha - beta);
                auto tau = at::where(skip, 0.0, (beta - alpha) / safe_beta);
                auto unit = (rows == j).to(at::kFloat);
                auto v = below / denom.unsqueeze(1) + unit;
                // Reflect the columns right of j; column j becomes beta on the
                // diagonal and v below it.
                auto right = (cols > j).to(at::kFloat);
                auto vr = (v.unsqueeze(2) * r).sum(1, true) * right;
                r = r - tau.unsqueeze(1).unsqueeze(2) * v.unsqueeze(2) * vr;
                auto column_mask = (cols == j).to(at::kFloat);
                auto store = (rows >= j).to(at::kFloat).unsqueeze(1) * column_mask;
                auto packed_column = beta.unsqueeze(1) * unit + v * below_mask;
                packed_column = at::where(skip.unsqueeze(1), column, packed_column);
                r = r * (1 - store) + packed_column.unsqueeze(2) * store;
                taus.push_back(tau.unsqueeze(1));
            }
            auto tau = k > 0 ? at::cat(taus, 1)
                             : at::zeros({a.size(0), 0}, r.options());
            return {r, tau};
        }

        // Reflector j of packed `packed` [B, m, *]: e_j + the entries below the
        // diagonal.
        at::Tensor reflector(const at::Tensor& packed, int64_t j) {
            const int64_t m = packed.size(1);
            auto rows = at::arange(m, packed.options().dtype(at::kLong));
            auto below = (rows > j).to(at::kFloat);
            return (
                packed.select(2, j).to(at::kFloat) * below + (rows == j).to(at::kFloat)
            );
        }

        // H y = y - tau v (v^T y) on the rows of `y` [B, m, r] (left) or
        // y H = y - tau (y v) v^T on its columns [B, r, m] (right).
        at::Tensor reflect(
            const at::Tensor& y, const at::Tensor& v, const at::Tensor& tau, bool left
        ) {
            auto t = tau.unsqueeze(1).unsqueeze(2);
            if (left) {
                auto vy = (v.unsqueeze(2) * y).sum(1, true);
                return y - t * v.unsqueeze(2) * vy;
            }
            auto yv = (y * v.unsqueeze(1)).sum(2, true);
            return y - t * yv * v.unsqueeze(1);
        }

        std::tuple<at::Tensor, at::Tensor> tt_geqrf(const at::Tensor& a) {
            KTT_REQUIRE(a.dim() >= 2, "geqrf", "needs matrices [..., m, n]");
            const int64_t m = a.size(-2), n = a.size(-1), k = std::min(m, n);
            auto tau_sizes = a.sizes().slice(0, a.dim() - 2).vec();
            tau_sizes.push_back(k);
            if (a.numel() == 0) {
                return {a.clone(), at::zeros(tau_sizes, a.options())};
            }
            auto [packed, tau] = geqrf_rows(a.reshape({-1, m, n}));
            return {
                packed.reshape(a.sizes()).to(a.scalar_type()),
                tau.reshape(tau_sizes).to(a.scalar_type())
            };
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_geqrf_out(
            const at::Tensor& a, at::Tensor& packed, at::Tensor& tau
        ) {
            auto [p, t] = tt_geqrf(a);
            into("geqrf.a", packed, p);
            into("geqrf.a", tau, t);
            return {packed, tau};
        }

        // The first n columns of H_0 ... H_{k-1} from packed reflectors `a` [..., m, n]
        // and `tau` [..., k].
        at::Tensor tt_householder_product(const at::Tensor& a, const at::Tensor& tau) {
            const char* op = "linalg_householder_product";
            KTT_REQUIRE(a.dim() >= 2, op, "needs matrices [..., m, n]");
            const int64_t m = a.size(-2), n = a.size(-1), k = tau.size(-1);
            if (a.numel() == 0) {
                return at::zeros(a.sizes(), a.options());
            }
            auto packed = a.reshape({-1, m, n});
            auto q = at::eye(m, n, packed.options().dtype(at::kFloat))
                         .expand({packed.size(0), m, n})
                         .contiguous();
            if (k == 0) {
                return q.reshape(a.sizes()).to(a.scalar_type());
            }
            auto taus = tau.reshape({packed.size(0), k}).to(at::kFloat);
            for (int64_t j = k - 1; j >= 0; --j) {
                q = reflect(q, reflector(packed, j), taus.select(1, j), true);
            }
            return q.reshape(a.sizes()).to(a.scalar_type());
        }

        at::Tensor& tt_householder_product_out(
            const at::Tensor& a, const at::Tensor& tau, at::Tensor& out
        ) {
            return (into(
                "linalg_householder_product.out", out, tt_householder_product(a, tau)
            ));
        }

        // Q C, Q^T C, C Q or C Q^T (`left`, `transpose`) with Q = H_0 ... H_{k-1} from
        // packed reflectors `a` [..., m, k'] and `tau` [..., k]; C [..., m, r] (left)
        // or [..., r, m].
        at::Tensor tt_ormqr(
            const at::Tensor& a,
            const at::Tensor& tau,
            const at::Tensor& c,
            bool left,
            bool transpose
        ) {
            const char* op = "ormqr";
            KTT_REQUIRE(a.dim() >= 2 && c.dim() >= 2, op, "needs matrices");
            const int64_t k = tau.size(-1);
            if (c.numel() == 0 || k == 0 || a.numel() == 0) {
                return c.clone();
            }
            const int64_t m = a.size(-2), cols = a.size(-1);
            auto packed = a.reshape({-1, m, cols});
            auto taus = tau.reshape({packed.size(0), k}).to(at::kFloat);
            auto y = c.reshape({-1, c.size(-2), c.size(-1)}).to(at::kFloat);
            // Left: Q C applies H_{k-1} first; Q^T C applies H_0 first. Right: C Q
            // applies H_0 first; C Q^T applies H_{k-1} first.
            const bool ascending = left == transpose;
            for (int64_t s = 0; s < k; ++s) {
                const int64_t j = ascending ? s : k - 1 - s;
                y = reflect(y, reflector(packed, j), taus.select(1, j), left);
            }
            return y.reshape(c.sizes()).to(c.scalar_type());
        }

        at::Tensor& tt_ormqr_out(
            const at::Tensor& a,
            const at::Tensor& tau,
            const at::Tensor& c,
            bool left,
            bool transpose,
            at::Tensor& out
        ) {
            return into("ormqr.out", out, tt_ormqr(a, tau, c, left, transpose));
        }

        // (Q, R) of `a` [..., m, n]: 'reduced' (Q [..., m, k], R [..., k, n]),
        // 'complete' (Q [..., m, m], R [..., m, n]) or 'r' (Q empty, R reduced).
        std::tuple<at::Tensor, at::Tensor> tt_qr(
            const at::Tensor& a, c10::string_view mode
        ) {
            const char* op = "linalg_qr";
            KTT_REQUIRE(a.dim() >= 2, op, "needs matrices [..., m, n]");
            KTT_REQUIRE(
                mode == "reduced" || mode == "complete" || mode == "r",
                op,
                "mode '",
                mode,
                "'"
            );
            const int64_t m = a.size(-2), n = a.size(-1), k = std::min(m, n);
            const bool complete = mode == "complete";
            auto batch = a.sizes().slice(0, a.dim() - 2).vec();
            auto q_sizes = batch, r_sizes = batch;
            q_sizes.insert(q_sizes.end(), {m, complete ? m : k});
            r_sizes.insert(r_sizes.end(), {complete ? m : k, n});
            at::Tensor q, r;
            if (a.numel() == 0) {
                q = at::zeros(q_sizes, a.options());
                if (c10::multiply_integers(q_sizes) > 0) {
                    q = q + at::eye(q_sizes.end()[-2], q_sizes.back(), a.options());
                }
                r = at::zeros(r_sizes, a.options());
            } else {
                auto [qq, rr] = householder(a.reshape({-1, m, n}));
                q = (qq.narrow(2, 0, q_sizes.back())
                         .reshape(q_sizes)
                         .to(a.scalar_type()));
                r = (rr.narrow(1, 0, r_sizes.end()[-2])
                         .reshape(r_sizes)
                         .to(a.scalar_type()));
            }
            if (mode == "r") {
                q = at::empty({0}, a.options());
            }
            return {q, r};
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_qr_out(
            const at::Tensor& a, c10::string_view mode, at::Tensor& q, at::Tensor& r
        ) {
            auto [qq, rr] = tt_qr(a, mode);
            into("linalg_qr.out", q, qq);
            into("linalg_qr.out", r, rr);
            return {q, r};
        }

        // Least squares for full-rank `a` [..., m, n], `b` [..., m, k] (or [..., m]):
        // m >= n: X = R^-1 Q^T B (reduced QR of A); m < n: the minimum-norm X = Q R^-T
        // B (QR of A^T). residuals for m > n and a driver other than gelsy, rank for
        // any driver but gels (singular values above rcond * max), singular values for
        // gelsd / gelss; empty otherwise.
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tt_lstsq(
            const at::Tensor& a,
            const at::Tensor& b,
            std::optional<double> rcond,
            std::optional<c10::string_view> driver_opt
        ) {
            const char* op = "linalg_lstsq";
            KTT_REQUIRE(a.dim() >= 2, op, "needs matrices [..., m, n]");
            const std::string driver = driver_opt ? std::string(*driver_opt) : "gelsy";
            const int64_t m = a.size(-2), n = a.size(-1);
            const bool vector = b.dim() == a.dim() - 1;
            auto rhs = vector ? b.unsqueeze(-1) : b;
            auto batch = (at::infer_size(
                a.sizes().slice(0, a.dim() - 2), rhs.sizes().slice(0, rhs.dim() - 2)
            ));
            const int64_t k = rhs.size(-1);
            auto a_sizes = batch, b_sizes = batch, x_sizes = batch;
            a_sizes.insert(a_sizes.end(), {m, n});
            b_sizes.insert(b_sizes.end(), {m, k});
            x_sizes.insert(x_sizes.end(), {n, k});
            auto dtype = at::result_type(a, b);
            auto fa = a.expand(a_sizes).reshape({-1, m, n}).to(at::kFloat);
            auto fb = rhs.expand(b_sizes).reshape({-1, m, k}).to(at::kFloat);
            at::Tensor x;
            if (fa.numel() == 0 || fb.numel() == 0) {
                x = at::zeros({fa.size(0), n, k}, fa.options());
            } else if (m >= n) {
                auto [q, r] = householder(fa);
                auto q1 = q.narrow(2, 0, n);
                auto qtb = (q1.unsqueeze(3) * fb.unsqueeze(2)).sum(1);
                x = substitute(r.narrow(1, 0, n), qtb, false, false);
            } else {
                auto [q, r] = householder(fa.transpose(1, 2).contiguous());
                auto r1 = r.narrow(1, 0, m);
                auto y = substitute(r1.transpose(1, 2), fb, true, false);
                x = (q.narrow(2, 0, m).unsqueeze(3) * y.unsqueeze(1)).sum(2);
            }
            auto solution = x.reshape(x_sizes);
            at::Tensor residuals = at::empty({0}, a.options().dtype(dtype));
            if (m > n && driver != "gelsy" && fa.numel() > 0) {
                auto ax = (fa.unsqueeze(3) * x.unsqueeze(1)).sum(2);
                auto res_sizes = batch;
                res_sizes.push_back(k);
                residuals = ((ax - fb) * (ax - fb)).sum(1).reshape(res_sizes).to(dtype);
            }
            at::Tensor rank = at::empty({0}, a.options().dtype(at::kLong));
            at::Tensor singular = at::empty({0}, a.options().dtype(dtype));
            if (driver != "gels") {
                auto s = at::linalg_svdvals(a.expand(a_sizes).to(at::kFloat));
                const double eps = std::numeric_limits<float>::epsilon();
                const double tol = rcond && *rcond > 0 ? *rcond : eps * std::max(m, n);
                auto top = s.numel() > 0 ? s.amax(-1, true) : s;
                rank =
                    (s.numel() > 0 ? at::gt(s, top * tol).to(at::kLong).sum(-1)
                                   : at::zeros(batch, a.options().dtype(at::kLong)));
                if (driver == "gelsd" || driver == "gelss") {
                    singular = s.to(dtype);
                }
            }
            if (vector) {
                solution = solution.squeeze(-1);
            }
            return {solution.to(dtype), residuals, rank, singular};
        }

        std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> tt_lstsq_out(
            const at::Tensor& a,
            const at::Tensor& b,
            std::optional<double> rcond,
            std::optional<c10::string_view> driver,
            at::Tensor& solution,
            at::Tensor& residuals,
            at::Tensor& rank,
            at::Tensor& singular
        ) {
            auto [x, r, k, s] = tt_lstsq(a, b, rcond, driver);
            into("linalg_lstsq.out", solution, x);
            into("linalg_lstsq.out", residuals, r);
            into("linalg_lstsq.out", rank, k);
            into("linalg_lstsq.out", singular, s);
            return {solution, residuals, rank, singular};
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("linalg_lstsq", &tt_lstsq);
        m.impl("linalg_lstsq.out", &tt_lstsq_out);
        m.impl("geqrf", &tt_geqrf);
        m.impl("geqrf.a", &tt_geqrf_out);
        m.impl("linalg_householder_product", &tt_householder_product);
        m.impl("linalg_householder_product.out", &tt_householder_product_out);
        m.impl("ormqr", &tt_ormqr);
        m.impl("ormqr.out", &tt_ormqr_out);
        m.impl("linalg_qr", &tt_qr);
        m.impl("linalg_qr.out", &tt_qr_out);
        m.impl("linalg_lu", &tt_linalg_lu);
        m.impl("linalg_lu.out", &tt_linalg_lu_out);
        m.impl("lu_unpack", &tt_lu_unpack);
        m.impl("lu_unpack.out", &tt_lu_unpack_out);
        m.impl("linalg_cholesky_ex", &tt_cholesky_ex);
        m.impl("linalg_cholesky_ex.L", &tt_cholesky_ex_out);
        m.impl("_cholesky_solve_helper", &tt_cholesky_solve_helper);
        m.impl("cholesky_inverse", &tt_cholesky_inverse);
        m.impl("cholesky_inverse.out", &tt_cholesky_inverse_out);
        m.impl("linalg_lu_solve", &tt_lu_solve);
        m.impl("linalg_lu_solve.out", &tt_lu_solve_out);
        m.impl("_linalg_solve_ex", &tt_solve_ex);
        m.impl("_linalg_solve_ex.result", &tt_solve_ex_out);
        m.impl("linalg_lu_factor_ex", &tt_lu_factor_ex);
        m.impl("linalg_lu_factor_ex.out", &tt_lu_factor_ex_out);
        m.impl("_linalg_slogdet", &tt_slogdet);
        m.impl("_linalg_slogdet.sign", &tt_slogdet_out);
        m.impl("_linalg_det", &tt_det);
        m.impl("_linalg_det.result", &tt_det_out);
        m.impl("linalg_inv_ex", &tt_inv_ex);
        m.impl("linalg_inv_ex.inverse", &tt_inv_ex_out);
    }

}  // namespace ktt
