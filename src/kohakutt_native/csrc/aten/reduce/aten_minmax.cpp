// aten min / max kernels for tt: amin, argmin / argmax, max / min (all and per dim),
// aminmax.

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        at::Tensor tt_amin(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
            return -at::amax(-self, dim, keepdim);
        }

        at::Tensor argmax_values(
            const at::Tensor& self, std::optional<int64_t> dim, bool keepdim
        );
        at::Tensor nan_first(
            const at::Tensor& self,
            const at::Tensor& values,
            std::optional<int64_t> dim,
            bool keepdim
        );

        at::Tensor arg_extreme(
            const at::Tensor& self,
            std::optional<int64_t> dim,
            bool keepdim,
            bool argmin
        );

        // argmin over `dim` (or all elements) as int64; NaN first, as in PyTorch.
        at::Tensor tt_argmin(
            const at::Tensor& self, std::optional<int64_t> dim, bool keepdim
        ) {
            return arg_extreme(self, dim, keepdim, true);
        }

        std::tuple<at::Tensor, at::Tensor> tt_max_dim(
            const at::Tensor& self, int64_t dim, bool keepdim
        ) {
            return {at::amax(self, {dim}, keepdim), at::argmax(self, dim, keepdim)};
        }

        std::tuple<at::Tensor, at::Tensor> tt_min_dim(
            const at::Tensor& self, int64_t dim, bool keepdim
        ) {
            return {tt_amin(self, {dim}, keepdim), tt_argmin(self, dim, keepdim)};
        }

        // argmax over `dim` (or all elements) as int64, NaN not considered.
        at::Tensor argmax_values(
            const at::Tensor& self, std::optional<int64_t> dim, bool keepdim
        ) {
            const char* op = "argmax";
            KTT_REQUIRE(self.numel() > 0, op, "argmax of an empty tensor");
            if (self.dim() == 0) {
                return at::zeros({}, self.options().dtype(at::kLong));
            }
            if (!dim) {
                auto flat = self.reshape({1, -1});
                auto h = bridge::argmax_last(
                    dense_as(op, flat, at::kFloat),
                    keepdim ? std::vector<int64_t>(self.dim(), 1)
                            : std::vector<int64_t>{}
                );
                return wrap_output(
                    h,
                    keepdim ? std::vector<int64_t>(self.dim(), 1)
                            : std::vector<int64_t>{},
                    at::kLong
                );
            }
            const int64_t d = c10::maybe_wrap_dim(*dim, self.dim());
            auto moved = self.movedim(d, -1).contiguous();
            auto sizes = to_vector(moved.sizes());
            sizes.pop_back();
            auto out_sizes = sizes;
            auto h = bridge::argmax_last(dense_as(op, moved, at::kFloat), out_sizes);
            auto result = wrap_output(h, out_sizes, at::kLong);
            return keepdim ? result.unsqueeze(d) : result;
        }

        // `values` (argmax / argmin of self over `dim`) with the first NaN's index on
        // slices holding a NaN, as in PyTorch.
        at::Tensor nan_first(
            const at::Tensor& self,
            const at::Tensor& values,
            std::optional<int64_t> dim,
            bool keepdim
        ) {
            if (!at::isFloatingType(self.scalar_type()) || self.dim() == 0) {
                return values;
            }
            auto nan = at::isnan(self);
            auto first_nan = argmax_values(nan.to(at::kFloat), dim, keepdim);
            auto has_nan = dim ? nan.any(*dim, keepdim) : nan.any();
            if (!dim && keepdim) {
                has_nan = has_nan.reshape(values.sizes());
            }
            return at::where(has_nan, first_nan, values);
        }

        // argmax (argmin) over `dim` or all elements as int64, NaN first as in PyTorch:
        // bf16 / fp32 in one own program, else argmax_values (of -self) and nan_first.
        at::Tensor arg_extreme(
            const at::Tensor& self,
            std::optional<int64_t> dim,
            bool keepdim,
            bool argmin
        ) {
            const auto type = self.scalar_type();
            if (self.dim() > 0 && self.numel() > 0 &&
                (type == at::kBFloat16 || type == at::kFloat)) {
                const int64_t d = dim ? c10::maybe_wrap_dim(*dim, self.dim()) : 0;
                auto rows = dim ? self.movedim(d, -1).contiguous()
                                : self.reshape({1, -1});
                auto sizes = to_vector(rows.sizes());
                sizes.pop_back();
                const auto out_sizes = dim ? sizes : std::vector<int64_t>{};
                if (bridge::argmax_own(to_bridge(type), out_sizes.size())) {
                    const char* op = argmin ? "argmin" : "argmax";
                    auto h = bridge::argmax_last(dense(op, rows), out_sizes, argmin);
                    auto result = wrap_output(h, out_sizes, at::kLong);
                    if (!keepdim) {
                        return result;
                    }
                    return dim ? result.unsqueeze(d)
                               : result.reshape(std::vector<int64_t>(self.dim(), 1));
                }
            }
            auto values = argmax_values(argmin ? -self : self, dim, keepdim);
            return nan_first(self, values, dim, keepdim);
        }

        // argmax over `dim` (or all elements) as int64; NaN first, as in PyTorch.
        at::Tensor tt_argmax(
            const at::Tensor& self, std::optional<int64_t> dim, bool keepdim
        ) {
            return arg_extreme(self, dim, keepdim, false);
        }

        at::Tensor& tt_argmax_out(
            const at::Tensor& self,
            std::optional<int64_t> dim,
            bool keepdim,
            at::Tensor& out
        ) {
            return into("argmax.out", out, tt_argmax(self, dim, keepdim));
        }

        at::Tensor tt_max(const at::Tensor& self) {
            return self.amax(std::vector<int64_t>{});
        }

        at::Tensor tt_min(const at::Tensor& self) {
            KTT_REQUIRE(self.numel() > 0, "min", "min of an empty tensor");
            return at::amin(self, {});
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_aminmax_out(
            const at::Tensor& self,
            std::optional<int64_t> dim,
            bool keepdim,
            at::Tensor& min,
            at::Tensor& max
        ) {
            auto dims = dim ? std::vector<int64_t>{*dim} : std::vector<int64_t>{};
            into("aminmax.out", min, at::amin(self, dims, keepdim));
            into("aminmax.out", max, at::amax(self, dims, keepdim));
            return {min, max};
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("amin", &tt_amin);
        m.impl("argmin", &tt_argmin);
        m.impl("max.dim", &tt_max_dim);
        m.impl("min.dim", &tt_min_dim);
        m.impl("argmax", &tt_argmax);
        m.impl("argmax.out", &tt_argmax_out);
        m.impl("max", &tt_max);
        m.impl("min", &tt_min);
        m.impl("aminmax.out", &tt_aminmax_out);
    }

}  // namespace ktt
