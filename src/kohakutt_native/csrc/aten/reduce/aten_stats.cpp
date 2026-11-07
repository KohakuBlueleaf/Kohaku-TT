// aten statistics kernels for tt: var / std (+ mean), nansum, sum dtype out,
// count_nonzero, median / mode, bincount, histc, trace, renorm, equal.

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // sum((x - mean)^2) / (N - correction) over dims (all when none).
        at::Tensor tt_var(
            const at::Tensor& self,
            at::OptionalIntArrayRef dim,
            const std::optional<at::Scalar>& correction,
            bool keepdim
        ) {
            std::vector<int64_t> dims = dim ? dim->vec() : std::vector<int64_t>{};
            int64_t count = 1;
            if (dims.empty()) {
                count = self.numel();
            } else {
                for (auto d : dims) {
                    count *= self.size(c10::maybe_wrap_dim(d, self.dim()));
                }
            }
            // Empty input: PyTorch's result is NaN of the reduced shape.
            if (self.numel() == 0) {
                auto meta = at::empty(self.sizes(), self.options().device(at::kMeta));
                auto sizes = at::var(meta, dim, correction, keepdim).sizes().vec();
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return at::full(sizes, nan, self.options());
            }
            auto x = self.to(at::kFloat);
            const bool keep_mean_dims = true;
            auto centered = x - x.mean(dims, keep_mean_dims);
            auto c = correction ? correction->toDouble() : 1.0;
            auto var =
                ((centered * centered).sum(dims, keepdim) / std::max(count - c, 0.0));
            return var.to(self.scalar_type());
        }

        std::tuple<at::Tensor, at::Tensor> tt_var_mean(
            const at::Tensor& self,
            at::OptionalIntArrayRef dim,
            const std::optional<at::Scalar>& correction,
            bool keepdim
        ) {
            auto dims = dim ? dim->vec() : std::vector<int64_t>{};
            return {tt_var(self, dim, correction, keepdim), self.mean(dims, keepdim)};
        }

        at::Tensor tt_std(
            const at::Tensor& self,
            at::OptionalIntArrayRef dim,
            const std::optional<at::Scalar>& correction,
            bool keepdim
        ) {
            // TT-NN sqrt maps NaN to inf; NaN variances (empty input) stay NaN.
            auto v = at::var(self, dim, correction, keepdim);
            return at::where(at::isnan(v), v, at::sqrt(v));
        }

        std::tuple<at::Tensor, at::Tensor> tt_std_mean(
            const at::Tensor& self,
            at::OptionalIntArrayRef dim,
            const std::optional<at::Scalar>& correction,
            bool keepdim
        ) {
            auto dims = dim ? dim->vec() : std::vector<int64_t>{};
            return {tt_std(self, dim, correction, keepdim), self.mean(dims, keepdim)};
        }

        // sum over dims, cast to `dtype` first when given (logsumexp, masked ops).
        at::Tensor& tt_sum_out(
            const at::Tensor& self,
            at::OptionalIntArrayRef dim,
            bool keepdim,
            std::optional<at::ScalarType> dtype,
            at::Tensor& out
        ) {
            auto x = dtype ? self.to(*dtype) : self;
            auto dims = dim ? dim->vec() : std::vector<int64_t>{};
            return into("sum.IntList_out", out, x.sum(dims, keepdim));
        }

        at::Tensor tt_count_nonzero(const at::Tensor& self, at::IntArrayRef dim) {
            return at::ne(self, 0).to(at::kFloat).sum(dim.vec()).to(at::kLong);
        }

        // The lower median along `dim` (NaNs ignored for nanmedian) and its index, by
        // sort.
        std::tuple<at::Tensor, at::Tensor> median_dim(
            const at::Tensor& self, int64_t dim, bool keepdim, bool ignore_nan
        ) {
            auto x = self.dim() == 0 ? self.reshape({1}) : self;
            dim = c10::maybe_wrap_dim(dim, x.dim());
            auto [sorted, order] = at::sort(x, dim);
            const int64_t n = x.size(dim);
            at::Tensor pick;
            if (ignore_nan) {
                // NaNs sort last: the middle of the non-NaN prefix.
                auto count = at::logical_not(at::isnan(x)).to(at::kLong).sum(dim, true);
                auto half = at::floor((count - 1).to(at::kFloat) * 0.5).to(at::kLong);
                pick = at::clamp(half, 0, std::nullopt);
            } else {
                auto sizes = x.sizes().vec();
                sizes[dim] = 1;
                pick = at::full(sizes, (n - 1) / 2, x.options().dtype(at::kLong));
            }
            auto value = at::gather(sorted, dim, pick);
            auto index = at::gather(order, dim, pick);
            if (!keepdim) {
                value = value.squeeze(dim);
                index = index.squeeze(dim);
            }
            if (self.dim() == 0) {
                value = value.reshape({});
                index = index.reshape({});
            }
            return {value, index};
        }

        at::Tensor tt_median(const at::Tensor& self) {
            KTT_REQUIRE(self.numel() > 0, "median", "median of an empty tensor");
            return std::get<0>(median_dim(self.reshape({-1}), 0, false, false));
        }

        at::Tensor tt_nanmedian(const at::Tensor& self) {
            KTT_REQUIRE(self.numel() > 0, "nanmedian", "median of an empty tensor");
            return std::get<0>(median_dim(self.reshape({-1}), 0, false, true));
        }

        std::tuple<at::Tensor&, at::Tensor&> median_out(
            const char* op,
            const at::Tensor& self,
            int64_t dim,
            bool keepdim,
            bool ignore_nan,
            at::Tensor& values,
            at::Tensor& indices
        ) {
            auto [v, i] = median_dim(self, dim, keepdim, ignore_nan);
            into(op, values, v);
            into(op, indices, i);
            return {values, indices};
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_median_dim(
            const at::Tensor& self,
            int64_t dim,
            bool keepdim,
            at::Tensor& values,
            at::Tensor& indices
        ) {
            return (
                median_out(
                    "median.dim_values", self, dim, keepdim, false, values, indices
                )
            );
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_nanmedian_dim(
            const at::Tensor& self,
            int64_t dim,
            bool keepdim,
            at::Tensor& values,
            at::Tensor& indices
        ) {
            const char* op = "nanmedian.dim_values";
            return median_out(op, self, dim, keepdim, true, values, indices);
        }

        // Most frequent value along dim and an index of it, on host (PyTorch's tie
        // rules).
        std::tuple<at::Tensor, at::Tensor> tt_mode(
            const at::Tensor& self, int64_t dim, bool keepdim
        ) {
            auto [values, indices] = at::mode(self.cpu(), dim, keepdim);
            return {values.to(self.device()), indices.to(self.device())};
        }

        // Counts (or weight sums) of each non-negative integer; the length is
        // data-dependent, so the maximum is read on host.
        at::Tensor tt_bincount(
            const at::Tensor& self,
            const std::optional<at::Tensor>& weights,
            int64_t minlength
        ) {
            const char* op = "bincount";
            KTT_REQUIRE(self.dim() == 1, op, "input must be 1-D");
            const bool weighted = weights && weights->defined();
            auto dtype = at::kLong;
            if (weighted) {
                const auto w = weights->scalar_type();
                dtype = w == at::kDouble ? at::kFloat : w;
            }
            const int64_t length = (
                std::max<int64_t>(
                    minlength, self.numel() > 0 ? self.max().item<int64_t>() + 1 : 0
                )
            );
            auto counts = at::zeros({length}, self.options().dtype(at::kFloat));
            if (self.numel() == 0) {
                return counts.to(dtype);
            }
            auto add = weighted ? weights->to(self.device(), at::kFloat)
                                : at::ones({self.numel()}, counts.options());
            return at::scatter_add(counts, 0, self.to(at::kLong), add).to(dtype);
        }

        // Histogram of `bins` equal bins over [min, max] (data range when both are 0).
        at::Tensor tt_histc(
            const at::Tensor& self,
            int64_t bins,
            const at::Scalar& min,
            const at::Scalar& max
        ) {
            double lo = min.toDouble(), hi = max.toDouble();
            auto x = self.reshape({-1}).to(at::kFloat);
            if (lo == 0.0 && hi == 0.0 && x.numel() > 0) {
                lo = x.min().item<double>();
                hi = x.max().item<double>();
            }
            if (lo == hi) {
                lo -= 1.0;
                hi += 1.0;
            }
            auto counts = at::zeros({bins}, x.options());
            if (x.numel() == 0) {
                return counts.to(self.scalar_type());
            }
            auto bin = at::floor((x - lo) * (static_cast<double>(bins) / (hi - lo)));
            bin = at::clamp(bin, 0, bins - 1);  // x == hi lands in the last bin
            auto inside = at::logical_and(at::ge(x, lo), at::le(x, hi)).to(at::kFloat);
            auto result = at::scatter_add(counts, 0, bin.to(at::kLong), inside);
            return result.to(self.scalar_type());
        }

        at::Tensor& tt_histc_out(
            const at::Tensor& self,
            int64_t bins,
            const at::Scalar& min,
            const at::Scalar& max,
            at::Tensor& out
        ) {
            return into("histc.out", out, tt_histc(self, bins, min, max));
        }

        // Each slice along dim with p-norm above maxnorm scaled to maxnorm / (norm +
        // 1e-7).
        at::Tensor& tt_renorm_out(
            const at::Tensor& self,
            const at::Scalar& p,
            int64_t dim,
            const at::Scalar& maxnorm,
            at::Tensor& out
        ) {
            auto x = self.to(at::kFloat);
            dim = c10::maybe_wrap_dim(dim, x.dim());
            std::vector<int64_t> others;
            for (int64_t d = 0; d < x.dim(); ++d) {
                if (d != dim) {
                    others.push_back(d);
                }
            }
            const bool keepdim = true;
            auto norm = at::linalg_vector_norm(x, p, others, keepdim);
            const double cap = maxnorm.toDouble();
            auto factor = (
                at::where(at::gt(norm, cap), cap / (norm + 1e-7), at::ones_like(norm))
            );
            return into("renorm.out", out, x * factor);
        }

        // Same sizes and every element equal (one host read of the result).
        bool tt_equal(const at::Tensor& a, const at::Tensor& b) {
            if (a.sizes() != b.sizes()) {
                return false;
            }
            if (a.numel() == 0) {
                return true;
            }
            return at::all(at::eq(a, b)).item<bool>();
        }

        // Sum treating NaN as 0.
        at::Tensor tt_nansum(
            const at::Tensor& self,
            at::OptionalIntArrayRef dim,
            bool keepdim,
            std::optional<at::ScalarType> dtype
        ) {
            auto x = dtype ? self.to(*dtype) : self;
            auto clean =
                (at::isFloatingType(x.scalar_type())
                     ? at::where(at::isnan(x), at::zeros_like(x), x)
                     : x);
            auto dims = dim ? dim->vec() : std::vector<int64_t>{};
            return clean.sum(dims, keepdim);
        }

        // Sum of the main diagonal of a 2-D tensor (integers sum as int64).
        at::Tensor tt_trace(const at::Tensor& self) {
            KTT_REQUIRE(self.dim() == 2, "trace", "expected a 2-D tensor");
            auto diag = at::diagonal(self);
            auto dtype =
                (at::isIntegralType(self.scalar_type(), true) ? at::kLong
                                                              : self.scalar_type());
            return diag.to(at::kFloat).sum().to(dtype);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("var.correction", &tt_var);
        m.impl("var_mean.correction", &tt_var_mean);
        m.impl("std.correction", &tt_std);
        m.impl("std_mean.correction", &tt_std_mean);
        m.impl("sum.IntList_out", &tt_sum_out);
        m.impl("count_nonzero.dim_IntList", &tt_count_nonzero);
        m.impl("median", &tt_median);
        m.impl("nanmedian", &tt_nanmedian);
        m.impl("median.dim_values", &tt_median_dim);
        m.impl("nanmedian.dim_values", &tt_nanmedian_dim);
        m.impl("mode", &tt_mode);
        m.impl("bincount", &tt_bincount);
        m.impl("histc", &tt_histc);
        m.impl("histc.out", &tt_histc_out);
        m.impl("renorm.out", &tt_renorm_out);
        m.impl("equal", &tt_equal);
        m.impl("nansum", &tt_nansum);
        m.impl("trace", &tt_trace);
    }

}  // namespace ktt
