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
            // nanmedian: NaNs as +inf sort last.
            const auto nan = at::isnan(x);
            auto keys =
                (ignore_nan
                     ? x.masked_fill(nan, std::numeric_limits<double>::infinity())
                     : x);
            auto [sorted, order] = at::sort(keys, dim);
            const int64_t n = x.size(dim);
            at::Tensor pick, count;
            if (ignore_nan) {
                // The middle of the non-NaN prefix.
                count = at::logical_not(nan).to(at::kLong).sum(dim, true);
                auto half = at::floor((count - 1).to(at::kFloat) * 0.5).to(at::kLong);
                pick = at::clamp(half, 0, std::nullopt);
            } else {
                auto sizes = x.sizes().vec();
                sizes[dim] = 1;
                pick = at::full(sizes, (n - 1) / 2, x.options().dtype(at::kLong));
            }
            auto value = at::gather(sorted, dim, pick);
            auto index = at::gather(order, dim, pick);
            if (ignore_nan) {
                // All-NaN slices: NaN.
                value = (
                    value.masked_fill(
                        count == 0, std::numeric_limits<double>::quiet_NaN()
                    )
                );
            }
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

        // (hist, edges) of flattened `self` over sorted `edges` [bins + 1] (on device,
        // self's dtype): bin = last edge <= x (`uniform`: the linear estimate corrected
        // by one step against the edges), x on the last edge in the last bin, x outside
        // [edges[0], edges[bins]] dropped; weighted sums; density divides by
        // sum * width.
        // (bin [L] int64, inside [L] bool) of fp32 values `x` [L] over sorted fp32
        // `e` [bins + 1] as histogram_over describes.
        std::pair<at::Tensor, at::Tensor> bin_of(
            const at::Tensor& x, const at::Tensor& e, bool uniform
        ) {
            const int64_t bins = e.numel() - 1;
            const double lo = e[0].item<double>(), hi = e[bins].item<double>();
            at::Tensor bin;
            if (uniform) {
                bin = at::floor((x - lo) * (static_cast<double>(bins) / (hi - lo)));
                bin = at::clamp(bin, 0, bins - 1).to(at::kLong);
                auto below = e.index_select(0, bin);
                auto above = e.index_select(0, bin + 1);
                bin = at::where(at::lt(x, below), bin - 1, bin);
                bin = (
                    at::where(
                        at::logical_and(at::ge(x, above), at::lt(bin, bins - 1)),
                        bin + 1,
                        bin
                    )
                );
            } else {
                auto at_or_above = (
                    at::ge(x.unsqueeze(1), e.unsqueeze(0)).to(at::kFloat)
                );
                bin = at_or_above.sum(1).to(at::kLong) - 1;
            }
            bin = at::clamp(bin, 0, bins - 1);
            return {bin, at::logical_and(at::ge(x, lo), at::le(x, hi))};
        }

        std::tuple<at::Tensor, at::Tensor> histogram_over(
            const at::Tensor& self,
            const at::Tensor& edges,
            const std::optional<at::Tensor>& weight,
            bool density,
            bool uniform
        ) {
            const int64_t bins = edges.numel() - 1;
            auto x = self.reshape({-1}).to(at::kFloat);
            auto e = edges.to(at::kFloat);
            auto hist = at::zeros({bins}, x.options());
            if (x.numel() > 0 && bins > 0) {
                auto [bin, inside] = bin_of(x, e, uniform);
                auto w = weight && weight->defined()
                    ? weight->reshape({-1}).to(at::kFloat)
                    : at::ones_like(x);
                hist = at::scatter_add(hist, 0, bin, at::where(inside, w, 0.0));
            }
            if (density) {
                auto widths = e.narrow(0, 1, bins) - e.narrow(0, 0, bins);
                hist = hist / (hist.sum() * widths);
            }
            return {hist.to(self.scalar_type()), edges.to(self.scalar_type())};
        }

        // Equal-width edges [bins + 1] (self's dtype, on its device) over [lo, hi]
        // (`range` pair `d`, else the data's min / max; widened by 0.5 each way when
        // equal; [0, 1] for no data) of fp32 values `x`.
        at::Tensor uniform_edges(
            const at::Tensor& x,
            int64_t bins,
            std::optional<at::ArrayRef<double>> range,
            int64_t d,
            const at::Tensor& self
        ) {
            double lo = 0.0, hi = 0.0;
            if (range && static_cast<int64_t>(range->size()) >= 2 * d + 2) {
                lo = (*range)[2 * d];
                hi = (*range)[2 * d + 1];
            } else if (x.numel() > 0) {
                lo = x.min().item<double>();
                hi = x.max().item<double>();
            } else {
                hi = 1.0;
            }
            if (lo == hi) {
                lo -= 0.5;
                hi += 0.5;
            }
            auto cpu = at::TensorOptions().dtype(self.scalar_type());
            return at::linspace(lo, hi, bins + 1, cpu).to(self.device());
        }

        // Per-dim edges of the D-dim points `self` [..., D] with `bins` [D] equal bins.
        std::vector<at::Tensor> tt_histogramdd_edges(
            const at::Tensor& self,
            at::IntArrayRef bins,
            std::optional<at::ArrayRef<double>> range,
            const std::optional<at::Tensor>& weight,
            bool density
        ) {
            const int64_t dims = self.size(-1);
            auto points = self.reshape({-1, dims}).to(at::kFloat);
            std::vector<at::Tensor> edges;
            for (int64_t d = 0; d < dims; ++d) {
                edges.push_back(
                    uniform_edges(points.select(1, d), bins[d], range, d, self)
                );
            }
            return edges;
        }

        // D-dim histogram over per-dim `edges`: flat bin = sum of per-dim bins times
        // strides, points outside any dim dropped; density divides by sum * volume.
        at::Tensor histogramdd_over(
            const at::Tensor& self,
            const std::vector<at::Tensor>& edges,
            const std::optional<at::Tensor>& weight,
            bool density,
            bool uniform
        ) {
            const int64_t dims = self.size(-1);
            auto points = self.reshape({-1, dims}).to(at::kFloat);
            std::vector<int64_t> sizes;
            int64_t total = 1;
            for (const auto& e : edges) {
                sizes.push_back(e.numel() - 1);
                total *= e.numel() - 1;
            }
            auto hist = at::zeros({total}, points.options());
            if (points.size(0) > 0 && total > 0) {
                at::Tensor flat, inside;
                int64_t stride = total;
                for (int64_t d = 0; d < dims; ++d) {
                    stride /= sizes[d];
                    auto [bin, in_d] = (
                        bin_of(points.select(1, d), edges[d].to(at::kFloat), uniform)
                    );
                    flat = flat.defined() ? flat + bin * stride : bin * stride;
                    inside = inside.defined() ? at::logical_and(inside, in_d) : in_d;
                }
                auto w = weight && weight->defined()
                    ? weight->reshape({-1}).to(at::kFloat)
                    : at::ones({points.size(0)}, points.options());
                hist = at::scatter_add(hist, 0, flat, at::where(inside, w, 0.0));
            }
            hist = hist.reshape(sizes);
            if (density) {
                auto volume = at::ones(sizes, points.options());
                for (int64_t d = 0; d < dims; ++d) {
                    auto e = edges[d].to(at::kFloat);
                    std::vector<int64_t> view(dims, 1);
                    view[d] = sizes[d];
                    volume =
                        (volume *
                         (e.narrow(0, 1, sizes[d]) - e.narrow(0, 0, sizes[d]))
                             .reshape(view));
                }
                hist = hist / (hist.sum() * volume);
            }
            return hist.to(self.scalar_type());
        }

        at::Tensor tt_histogramdd_counts(
            const at::Tensor& self,
            at::IntArrayRef bins,
            std::optional<at::ArrayRef<double>> range,
            const std::optional<at::Tensor>& weight,
            bool density
        ) {
            auto edges = tt_histogramdd_edges(self, bins, range, weight, density);
            return histogramdd_over(self, edges, weight, density, true);
        }

        at::Tensor tt_histogramdd_tensors(
            const at::Tensor& self,
            at::TensorList bins,
            const std::optional<at::Tensor>& weight,
            bool density
        ) {
            std::vector<at::Tensor> edges(bins.begin(), bins.end());
            return histogramdd_over(self, edges, weight, density, false);
        }

        // histogram with `bins` equal bins over `range` (the data's min / max when
        // unset; widened by 0.5 each way when equal; [0, 1] for an empty input).
        std::tuple<at::Tensor, at::Tensor> tt_histogram_bins(
            const at::Tensor& self,
            int64_t bins,
            std::optional<at::ArrayRef<double>> range,
            const std::optional<at::Tensor>& weight,
            bool density
        ) {
            double lo = 0.0, hi = 0.0;
            if (range && range->size() == 2) {
                lo = (*range)[0];
                hi = (*range)[1];
            } else if (self.numel() > 0) {
                auto x = self.to(at::kFloat);
                lo = x.min().item<double>();
                hi = x.max().item<double>();
            } else {
                hi = 1.0;
            }
            if (lo == hi) {
                lo -= 0.5;
                hi += 0.5;
            }
            auto cpu = at::TensorOptions().dtype(self.scalar_type());
            auto edges = at::linspace(lo, hi, bins + 1, cpu).to(self.device());
            return histogram_over(self, edges, weight, density, true);
        }

        std::tuple<at::Tensor, at::Tensor> tt_histogram_edges(
            const at::Tensor& self,
            const at::Tensor& bins,
            const std::optional<at::Tensor>& weight,
            bool density
        ) {
            return histogram_over(self, bins.reshape({-1}), weight, density, false);
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_histogram_bins_out(
            const at::Tensor& self,
            int64_t bins,
            std::optional<at::ArrayRef<double>> range,
            const std::optional<at::Tensor>& weight,
            bool density,
            at::Tensor& hist,
            at::Tensor& edges
        ) {
            auto [h, e] = tt_histogram_bins(self, bins, range, weight, density);
            into("histogram.bin_ct_out", hist, h);
            into("histogram.bin_ct_out", edges, e);
            return {hist, edges};
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_histogram_edges_out(
            const at::Tensor& self,
            const at::Tensor& bins,
            const std::optional<at::Tensor>& weight,
            bool density,
            at::Tensor& hist,
            at::Tensor& edges
        ) {
            auto [h, e] = tt_histogram_edges(self, bins, weight, density);
            into("histogram.bins_tensor_out", hist, h);
            into("histogram.bins_tensor_out", edges, e);
            return {hist, edges};
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

        // Quantiles `q` (0-d or 1-D, in [0, 1]) of `self` along `dim` (flattened when
        // unset) from the sorted values: rank = q (n - 1) over the n non-NaN values
        // (`ignore_nan`) or over all of them (NaN when the slice has one), then
        // linear / lower / higher / midpoint / nearest. q's dim comes first.
        at::Tensor quantile_impl(
            const char* op,
            const at::Tensor& self,
            const at::Tensor& q_in,
            std::optional<int64_t> dim,
            bool keepdim,
            c10::string_view interpolation,
            bool ignore_nan
        ) {
            KTT_REQUIRE(q_in.dim() <= 1, op, "q must be a scalar or 1-D");
            auto dtype = self.scalar_type();
            auto x = self.to(at::kFloat);
            std::vector<int64_t> out_sizes;
            if (dim) {
                const int64_t d = (
                    c10::maybe_wrap_dim(*dim, std::max<int64_t>(x.dim(), 1))
                );
                if (x.dim() == 0) {
                    x = x.reshape({1});
                }
                out_sizes = x.sizes().vec();
                if (keepdim) {
                    out_sizes[d] = 1;
                } else {
                    out_sizes.erase(out_sizes.begin() + d);
                }
                x = x.movedim(d, -1);
                if (self.dim() == 0) {
                    out_sizes.clear();
                }
            } else {
                if (keepdim) {
                    out_sizes.assign(x.dim(), 1);
                }
                x = x.reshape({-1});
            }
            const int64_t n = x.size(-1);
            const int64_t nq = q_in.numel();
            auto rows = x.reshape({-1, n});
            auto sorted = std::get<0>(rows.sort(-1));
            auto nan = at::isnan(rows);
            auto valid = ignore_nan
                ? (n - nan.to(at::kFloat).sum(-1, true))
                : at::full({rows.size(0), 1}, static_cast<double>(n), rows.options());
            // Ranks q (n - 1) in float64 on the host (as PyTorch), then indices and
            // weights on device.
            auto rank_cpu = q_in.cpu().to(at::kDouble).reshape({1, -1}) *
                at::clamp_min(valid.cpu().to(at::kDouble) - 1.0, 0.0);  // [R, Q]
            const std::string mode(interpolation);
            auto lower_cpu = at::floor(rank_cpu), upper_cpu = at::ceil(rank_cpu);
            if (mode == "nearest") {
                lower_cpu = upper_cpu = at::round(rank_cpu);
            }
            auto last = static_cast<double>(std::max<int64_t>(n - 1, 0));
            auto weight = (rank_cpu - lower_cpu).to(at::kFloat).to(x.device());
            auto lo = (
                sorted.gather(
                    -1, at::clamp(lower_cpu, 0.0, last).to(at::kLong).to(x.device())
                )
            );
            auto hi = (
                sorted.gather(
                    -1, at::clamp(upper_cpu, 0.0, last).to(at::kLong).to(x.device())
                )
            );
            at::Tensor value;
            if (mode == "linear") {
                value = lo + (hi - lo) * weight;
            } else if (mode == "lower" || mode == "nearest") {
                value = lo;
            } else if (mode == "higher") {
                value = hi;
            } else {
                KTT_REQUIRE(mode == "midpoint", op, "interpolation '", mode, "'");
                value = (lo + hi) * 0.5;
            }
            const double kNan = std::numeric_limits<double>::quiet_NaN();
            if (ignore_nan) {
                value = at::where(at::eq(valid, 0.0), kNan, value);
            } else {
                value = at::where(nan.any(-1, true), kNan, value);
            }
            auto sizes = out_sizes;
            sizes.push_back(nq);
            value = value.reshape(sizes).movedim(-1, 0);
            if (q_in.dim() == 0) {
                value = value.squeeze(0);
            }
            return value.to(dtype);
        }

        at::Tensor tt_quantile(
            const at::Tensor& self,
            const at::Tensor& q,
            std::optional<int64_t> dim,
            bool keepdim,
            c10::string_view interpolation
        ) {
            return (
                quantile_impl("quantile", self, q, dim, keepdim, interpolation, false)
            );
        }

        at::Tensor tt_quantile_scalar(
            const at::Tensor& self,
            double q,
            std::optional<int64_t> dim,
            bool keepdim,
            c10::string_view interpolation
        ) {
            return (
                quantile_impl(
                    "quantile",
                    self,
                    at::scalar_tensor(q),
                    dim,
                    keepdim,
                    interpolation,
                    false
                )
            );
        }

        at::Tensor tt_nanquantile(
            const at::Tensor& self,
            const at::Tensor& q,
            std::optional<int64_t> dim,
            bool keepdim,
            c10::string_view interpolation
        ) {
            return (
                quantile_impl("nanquantile", self, q, dim, keepdim, interpolation, true)
            );
        }

        at::Tensor tt_nanquantile_scalar(
            const at::Tensor& self,
            double q,
            std::optional<int64_t> dim,
            bool keepdim,
            c10::string_view interpolation
        ) {
            return (
                quantile_impl(
                    "nanquantile",
                    self,
                    at::scalar_tensor(q),
                    dim,
                    keepdim,
                    interpolation,
                    true
                )
            );
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("quantile", &tt_quantile);
        m.impl("quantile.scalar", &tt_quantile_scalar);
        m.impl("nanquantile", &tt_nanquantile);
        m.impl("nanquantile.scalar", &tt_nanquantile_scalar);
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
        m.impl("_histogramdd_bin_edges", &tt_histogramdd_edges);
        m.impl("_histogramdd_from_bin_cts", &tt_histogramdd_counts);
        m.impl("_histogramdd_from_bin_tensors", &tt_histogramdd_tensors);
        m.impl("histogram.bin_ct", &tt_histogram_bins);
        m.impl("histogram.bin_ct_out", &tt_histogram_bins_out);
        m.impl("histogram.bins_tensor", &tt_histogram_edges);
        m.impl("histogram.bins_tensor_out", &tt_histogram_edges_out);
        m.impl("renorm.out", &tt_renorm_out);
        m.impl("equal", &tt_equal);
        m.impl("nansum", &tt_nansum);
        m.impl("trace", &tt_trace);
    }

}  // namespace ktt
