// aten scan kernels for tt: cumsum, cumprod, cummax / cummin (doubling scan),
// logcumsumexp.

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        at::Tensor& tt_cumsum_out(
            const at::Tensor& self,
            int64_t dim,
            std::optional<at::ScalarType> dtype,
            at::Tensor& out
        ) {
            const char* op = "cumsum.out";
            if (self.dim() == 0 || self.numel() == 0) {
                return into(op, out, self.clone());
            }
            auto target = dtype.value_or(out.scalar_type());
            const int64_t d = (c10::maybe_wrap_dim(
                dim, std::max<int64_t>(self.dim(), 1)
            ));
            auto device_dim = d + std::max<int64_t>(0, 2 - self.dim());
            // bf16 -> bf16 along the last dim stays bf16 (Kohaku-TT's kernel, fp32
            // accumulation).
            auto compute =
                (at::isIntegralType(target, true) ? at::kInt
                     : (self.scalar_type() == at::kBFloat16 &&
                        target == at::kBFloat16 && d == self.dim() - 1 &&
                        self.dim() >= 2)
                     ? at::kBFloat16
                     : at::kFloat);
            auto h = bridge::cumsum(dense_as(op, self, compute), device_dim);
            return into(op, out, wrap_output(h, self.sizes(), compute));
        }

        // Running product along `dim`: TT-NN cumprod in fp32 for floating dtypes.
        at::Tensor& tt_cumprod_out(
            const at::Tensor& self,
            int64_t dim,
            std::optional<at::ScalarType> dtype,
            at::Tensor& out
        ) {
            const char* op = "cumprod.out";
            if (self.dim() == 0 || self.numel() == 0) {
                return into(op, out, self.clone());
            }
            auto target = dtype.value_or(out.scalar_type());
            const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
            // Integers: a log-depth doubling scan of integer multiplies in `target`.
            if (!at::isFloatingType(target)) {
                auto x = self.to(target);
                const int64_t n = x.size(d);
                for (int64_t shift = 1; shift < n; shift *= 2) {
                    auto later =
                        x.narrow(d, shift, n - shift) * x.narrow(d, 0, n - shift);
                    x = at::cat({x.narrow(d, 0, shift), later}, d);
                }
                return into(op, out, x);
            }
            auto device_dim = d + std::max<int64_t>(0, 2 - self.dim());
            auto h = bridge::cumprod(dense_as(op, self, at::kFloat), device_dim);
            return into(op, out, wrap_output(h, self.sizes(), at::kFloat));
        }

        // Running max (or min) along `dim` with the index of the latest extreme
        // (PyTorch's tie rule): a log-depth doubling scan over (value, index) pairs.
        void cumulative_extreme(
            const at::Tensor& self,
            at::Tensor& values,
            at::Tensor& indices,
            int64_t dim,
            bool max
        ) {
            const char* op = max ? "_cummax_helper" : "_cummin_helper";
            if (self.numel() == 0) {
                return;
            }
            auto x = self.dim() == 0 ? self.reshape({1}) : self;
            dim = c10::maybe_wrap_dim(dim, x.dim());
            const int64_t n = x.size(dim);
            std::vector<int64_t> view(x.dim(), 1);
            view[dim] = n;
            auto v = x.to(at::kFloat);
            auto positions = at::arange(n, x.options().dtype(at::kLong));
            auto idx = positions.reshape(view).expand(x.sizes());
            for (int64_t shift = 1; shift < n; shift *= 2) {
                // Each position combines with the one `shift` earlier (if any).
                auto earlier_v = v.narrow(dim, 0, n - shift);
                auto earlier_i = idx.narrow(dim, 0, n - shift);
                auto later_v = v.narrow(dim, shift, n - shift);
                auto later_i = idx.narrow(dim, shift, n - shift);
                auto keep_later =
                    (max ? at::ge(later_v, earlier_v) : at::le(later_v, earlier_v));
                auto best_v = at::where(keep_later, later_v, earlier_v);
                auto best_i = at::where(keep_later, later_i, earlier_i);
                v = at::cat({v.narrow(dim, 0, shift), best_v}, dim);
                idx = at::cat({idx.narrow(dim, 0, shift), best_i}, dim);
            }
            write_into(op, values, v.reshape(self.sizes()).to(values.scalar_type()));
            write_into(
                op, indices, idx.reshape(self.sizes()).to(indices.scalar_type())
            );
        }

        void tt_cummax_helper(
            const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim
        ) {
            cumulative_extreme(self, values, indices, dim, true);
        }

        void tt_cummin_helper(
            const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim
        ) {
            cumulative_extreme(self, values, indices, dim, false);
        }

        // log(cumsum(exp(x - m))) + m along dim, m = the row max (fp32; -inf rows stay
        // -inf).
        at::Tensor tt_logcumsumexp(const at::Tensor& self, int64_t dim) {
            if (self.numel() == 0) {
                return at::empty_like(self);
            }
            auto x = self.to(at::kFloat);
            dim = c10::maybe_wrap_dim(dim, std::max<int64_t>(x.dim(), 1));
            const bool keepdim = true;
            auto m = at::amax(x, {dim}, keepdim);
            const double neg_inf = -std::numeric_limits<double>::infinity();
            m = at::where(at::eq(m, neg_inf), at::zeros_like(m), m);
            auto result = at::log(at::cumsum(at::exp(x - m), dim)) + m;
            return result.to(self.scalar_type());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("cumsum.out", &tt_cumsum_out);
        m.impl("cumprod.out", &tt_cumprod_out);
        m.impl("_cummax_helper", &tt_cummax_helper);
        m.impl("_cummin_helper", &tt_cummin_helper);
        m.impl("_logcumsumexp", &tt_logcumsumexp);
    }

}  // namespace ktt
