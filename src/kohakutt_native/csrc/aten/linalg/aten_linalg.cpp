// aten linear-algebra kernels for tt: addmm, dot / vdot, mv / addmv, baddbmm /
// addbmm, norms (vector norm included), cross.

#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <cmath>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // beta * self + alpha * (mat1 @ mat2).
        at::Tensor tt_addmm(
            const at::Tensor& self,
            const at::Tensor& mat1,
            const at::Tensor& mat2,
            const at::Scalar& beta,
            const at::Scalar& alpha
        ) {
            auto prod = at::mm(mat1, mat2);
            if (alpha.toDouble() != 1.0) {
                prod = prod * alpha;
            }
            if (beta.toDouble() == 0.0) {
                return prod;
            }
            return beta.toDouble() == 1.0 ? prod + self : prod + self * beta;
        }

        at::Tensor& tt_addmm_out(
            const at::Tensor& self,
            const at::Tensor& mat1,
            const at::Tensor& mat2,
            const at::Scalar& beta,
            const at::Scalar& alpha,
            at::Tensor& out
        ) {
            return into("addmm.out", out, tt_addmm(self, mat1, mat2, beta, alpha));
        }

        // beta * self + alpha * product (beta == 0 ignores self, as PyTorch does).
        at::Tensor scaled_add(
            const at::Tensor& self,
            const at::Tensor& product,
            const at::Scalar& beta,
            const at::Scalar& alpha
        ) {
            auto out = alpha.toDouble() == 1.0 ? product : product * alpha;
            if (beta.toDouble() == 0.0) {
                return out;
            }
            return beta.toDouble() == 1.0 ? out + self : out + self * beta;
        }

        // Sum of a * b in fp32.
        at::Tensor tt_dot(const at::Tensor& a, const at::Tensor& b) {
            auto dtype = at::result_type(a, b);
            return (a.to(at::kFloat) * b.to(at::kFloat)).sum().to(dtype);
        }

        // Real tensors: vdot is dot (no conjugation needed).
        at::Tensor tt_vdot(const at::Tensor& a, const at::Tensor& b) {
            return tt_dot(a, b);
        }

        // a x b along `dim` (size 3), broadcasting the other dims.
        at::Tensor& tt_cross_out(
            const at::Tensor& a, const at::Tensor& b, int64_t dim, at::Tensor& out
        ) {
            const char* op = "linalg_cross.out";
            auto shape = at::infer_size(a.sizes(), b.sizes());
            auto x = a.expand(shape).to(at::kFloat), y = b.expand(shape).to(at::kFloat);
            dim = c10::maybe_wrap_dim(dim, static_cast<int64_t>(shape.size()));
            KTT_REQUIRE(shape[dim] == 3, op, "the cross dim must have size 3");
            auto c = [&](const at::Tensor& t, int64_t i) {
                return t.narrow(dim, i, 1);
            };
            auto result = at::cat(
                {c(x, 1) * c(y, 2) - c(x, 2) * c(y, 1),
                 c(x, 2) * c(y, 0) - c(x, 0) * c(y, 2),
                 c(x, 0) * c(y, 1) - c(x, 1) * c(y, 0)},
                dim
            );
            return into(op, out, result.to(at::result_type(a, b)));
        }

        at::Tensor tt_mv(const at::Tensor& mat, const at::Tensor& vec) {
            return at::mm(mat, vec.unsqueeze(1)).squeeze(1);
        }

        at::Tensor& tt_addmv_out(
            const at::Tensor& self,
            const at::Tensor& mat,
            const at::Tensor& vec,
            const at::Scalar& beta,
            const at::Scalar& alpha,
            at::Tensor& out
        ) {
            return (
                into("addmv.out", out, scaled_add(self, tt_mv(mat, vec), beta, alpha))
            );
        }

        at::Tensor& tt_mm_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            return into("mm.out", out, at::mm(a, b));
        }

        at::Tensor& tt_baddbmm_out(
            const at::Tensor& self,
            const at::Tensor& b1,
            const at::Tensor& b2,
            const at::Scalar& beta,
            const at::Scalar& alpha,
            at::Tensor& out
        ) {
            auto result = scaled_add(self, at::bmm(b1, b2), beta, alpha);
            return into("baddbmm.out", out, result);
        }

        // beta * self + alpha * sum over the batch of b1[i] @ b2[i].
        at::Tensor tt_addbmm(
            const at::Tensor& self,
            const at::Tensor& b1,
            const at::Tensor& b2,
            const at::Scalar& beta,
            const at::Scalar& alpha
        ) {
            auto product = at::bmm(b1, b2).to(at::kFloat).sum(0).to(self.scalar_type());
            return scaled_add(self, product, beta, alpha);
        }

        // ||x||_p over `dim` in fp32: p = 0 (nonzeros), inf / -inf (max / min |x|),
        // else (sum |x|^p)^(1/p); p absent means 2.
        at::Tensor tt_norm(
            const at::Tensor& self,
            const std::optional<at::Scalar>& p,
            at::IntArrayRef dim,
            bool keepdim
        ) {
            const double ord = p ? p->toDouble() : 2.0;
            auto dims = dim.vec();
            auto a = at::abs(self.to(at::kFloat));
            at::Tensor out;
            if (ord == 0.0) {
                out = at::ne(a, 0).to(at::kFloat).sum(dims, keepdim);
            } else if (std::isinf(ord)) {
                out = ord > 0 ? at::amax(a, dims, keepdim) : at::amin(a, dims, keepdim);
            } else if (ord == 1.0) {
                out = a.sum(dims, keepdim);
            } else if (ord == 2.0) {
                out = at::sqrt((a * a).sum(dims, keepdim));
            } else {
                out = at::pow(at::pow(a, ord).sum(dims, keepdim), 1.0 / ord);
            }
            return out.to(self.scalar_type());
        }

        at::Tensor& tt_norm_out(
            const at::Tensor& self,
            const std::optional<at::Scalar>& p,
            at::IntArrayRef dim,
            bool keepdim,
            at::Tensor& out
        ) {
            return into("norm.out", out, tt_norm(self, p, dim, keepdim));
        }

        // ||x||_ord over dims (all when none): ord 2 as sqrt(sum(x^2)), ord inf as max
        // |x|.
        at::Tensor tt_vector_norm(
            const at::Tensor& self,
            const at::Scalar& ord,
            at::OptionalIntArrayRef dim,
            bool keepdim,
            std::optional<at::ScalarType> dtype
        ) {
            const double p = ord.toDouble();
            auto x = dtype ? self.to(*dtype) : self;
            std::vector<int64_t> dims = dim ? to_vector(*dim) : std::vector<int64_t>{};
            if (std::isinf(p) && p > 0) {
                return x.abs().amax(dims, keepdim);
            }
            if (p == 2.0) {
                return (x * x).sum(dims, keepdim).sqrt();
            }
            // Other orders in fp32: 0 (nonzeros), 1, -inf (min |x|), else (sum
            // |x|^p)^(1/p).
            auto a = x.abs().to(at::kFloat);
            at::Tensor out;
            if (p == 0.0) {
                out = at::ne(a, 0).to(at::kFloat).sum(dims, keepdim);
            } else if (std::isinf(p)) {
                out = at::amin(a, dims, keepdim);
            } else if (p == 1.0) {
                out = a.sum(dims, keepdim);
            } else {
                out = at::pow(at::pow(a, p).sum(dims, keepdim), 1.0 / p);
            }
            return out.to(x.scalar_type());
        }

        at::Tensor& tt_vector_norm_out(
            const at::Tensor& self,
            const at::Scalar& ord,
            at::OptionalIntArrayRef dim,
            bool keepdim,
            std::optional<at::ScalarType> dtype,
            at::Tensor& out
        ) {
            auto result = tt_vector_norm(self, ord, dim, keepdim, dtype);
            if (out.sizes() != result.sizes()) {
                out.resize_(result.sizes());
            }
            return write_into("linalg_vector_norm.out", out, result);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("addmm", &tt_addmm);
        m.impl("addmm.out", &tt_addmm_out);
        m.impl("dot", &tt_dot);
        m.impl("vdot", &tt_vdot);
        m.impl("linalg_cross.out", &tt_cross_out);
        m.impl("mv", &tt_mv);
        m.impl("addmv.out", &tt_addmv_out);
        m.impl("mm.out", &tt_mm_out);
        m.impl("baddbmm.out", &tt_baddbmm_out);
        m.impl("addbmm", &tt_addbmm);
        m.impl("norm.ScalarOpt_dim", &tt_norm);
        m.impl("norm.out", &tt_norm_out);
        m.impl("linalg_vector_norm", &tt_vector_norm);
        m.impl("linalg_vector_norm.out", &tt_vector_norm_out);
    }

}  // namespace ktt
