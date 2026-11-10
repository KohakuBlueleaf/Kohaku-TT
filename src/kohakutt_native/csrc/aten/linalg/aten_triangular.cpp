// aten triangular solves for tt: linalg_solve_triangular (+ out), triangular_solve.
// The triangular inverse is built from matmuls only: 2 x 2 block recursion down to 4 x
// 4, and inside a block T = D (I + N) with N strictly triangular (nilpotent), so
// (I + N)^-1 = (I - N)(I + N^2). Computed in fp32.

#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // Block size of the series inverse. The series multiplies powers of N up to
        // N^(kBlock - 1); at 4 they stay near N's own size (64 x 64 blocks' powers
        // grow by orders of magnitude and lose the inverse to cancellation in fp32).
        constexpr int64_t kBlock = 4;

        // Largest matrix inverted whole (one tile); larger ones split at tile bounds.
        constexpr int64_t kLeaf = 32;

        // [n, n] mask (l's dtype and device) of the diagonal blocks of size s.
        at::Tensor block_mask(int64_t n, int64_t s, const at::Tensor& l) {
            auto block = at::arange(n, at::kLong).div(s, "floor");
            return ((block.unsqueeze(-1) == block.unsqueeze(0))
                        .to(l.scalar_type())
                        .to(l.device()));
        }

        // Inverse of a lower-triangular [..., n, n] fp32 matrix, n <= kLeaf, on whole
        // tiles: with N the strictly lower part of the unit-diagonal form, the inverse
        // of its kBlock diagonal blocks is (I - N_4)(I + N_4^2) (N_4^4 = 0); each
        // doubling of the block size s updates X -= X C X, C = N within the 2s blocks
        // and outside the s blocks.
        at::Tensor lower_block_inverse(const at::Tensor& l, bool unit) {
            const int64_t n = l.size(-1);
            auto eye = at::eye(n, l.options());
            auto diag =
                (unit ? at::ones_like(l.diagonal(0, -2, -1)) : l.diagonal(0, -2, -1));
            auto nil = (unit ? l : l / diag.unsqueeze(-1)).tril(-1);
            auto leaf = nil * block_mask(n, kBlock, l);
            auto inv = at::matmul(eye - leaf, eye + at::matmul(leaf, leaf));
            for (int64_t s = kBlock; s < n; s *= 2) {
                auto c = nil * (block_mask(n, 2 * s, l) - block_mask(n, s, l));
                inv = inv - at::matmul(at::matmul(inv, c), inv);
            }
            return unit ? inv : inv / diag.unsqueeze(-2);
        }

        // Inverse of a lower-triangular [..., n, n] fp32 matrix:
        // [[A, 0], [C, B]]^-1 = [[A^-1, 0], [-B^-1 C A^-1, B^-1]].
        at::Tensor lower_inverse(const at::Tensor& l, bool unit) {
            const int64_t n = l.size(-1);
            if (n <= kLeaf) {
                return lower_block_inverse(l, unit);
            }
            const int64_t h = (n / 2 + kLeaf - 1) / kLeaf * kLeaf;
            auto a = lower_inverse(l.narrow(-2, 0, h).narrow(-1, 0, h), unit);
            auto b = lower_inverse(l.narrow(-2, h, n - h).narrow(-1, h, n - h), unit);
            auto c = l.narrow(-2, h, n - h).narrow(-1, 0, h);
            auto low = -at::matmul(at::matmul(b, c), a);
            auto top = at::cat({a, at::zeros_like(c.transpose(-2, -1))}, -1);
            return at::cat({top, at::cat({low, b}, -1)}, -2);
        }

        // Inverse of the triangle of `a` (fp32): an upper one through its transpose.
        at::Tensor triangular_inverse(const at::Tensor& a, bool upper, bool unit) {
            auto f = a.to(at::kFloat);
            if (upper) {
                return (
                    lower_inverse(f.transpose(-2, -1).tril(), unit).transpose(-2, -1)
                );
            }
            return lower_inverse(f.tril(), unit);
        }

        // X with A X = B (left) or X A = B (right), A triangular [..., n, n].
        at::Tensor tt_linalg_solve_triangular(
            const at::Tensor& a, const at::Tensor& b, bool upper, bool left, bool unit
        ) {
            const char* op = "linalg_solve_triangular";
            const FidelityFloor exact(3);  // series inverse: errors compound
            KTT_REQUIRE(
                a.dim() >= 2 && a.size(-1) == a.size(-2), op, "A must be square"
            );
            KTT_REQUIRE(b.dim() >= 2, op, "B must be at least 2-D");
            const auto dtype = at::result_type(a, b);
            if (a.numel() == 0 || b.numel() == 0) {
                // Batch dims broadcast; B's matrix dims.
                auto sizes = (at::infer_size(
                    a.sizes().slice(0, a.dim() - 2), b.sizes().slice(0, b.dim() - 2)
                ));
                sizes.push_back(b.size(-2));
                sizes.push_back(b.size(-1));
                return at::zeros(sizes, b.options().dtype(dtype));
            }
            auto inv = triangular_inverse(a, upper, unit);
            auto bf = b.to(at::kFloat);
            auto x = left ? at::matmul(inv, bf) : at::matmul(bf, inv);
            return x.to(dtype);
        }

        at::Tensor& tt_linalg_solve_triangular_out(
            const at::Tensor& a,
            const at::Tensor& b,
            bool upper,
            bool left,
            bool unit,
            at::Tensor& out
        ) {
            return (into(
                "linalg_solve_triangular.out",
                out,
                tt_linalg_solve_triangular(a, b, upper, left, unit)
            ));
        }

        // Legacy (solution, cloned A) with A X = B, A optionally transposed.
        std::tuple<at::Tensor, at::Tensor> tt_triangular_solve(
            const at::Tensor& b,
            const at::Tensor& a,
            bool upper,
            bool transpose,
            bool unit
        ) {
            auto coef = transpose ? a.transpose(-2, -1) : a;
            const bool tri_upper = transpose ? !upper : upper;
            const bool left = true;
            auto x = tt_linalg_solve_triangular(coef, b, tri_upper, left, unit);
            return {x, a.clone()};
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("linalg_solve_triangular", &tt_linalg_solve_triangular);
        m.impl("linalg_solve_triangular.out", &tt_linalg_solve_triangular_out);
        m.impl("triangular_solve", &tt_triangular_solve);
    }

}  // namespace ktt
