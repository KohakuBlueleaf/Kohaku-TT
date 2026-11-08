// aten reduction kernels for tt: sum, mean, amax, prod (all dims or a dim list).
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <limits>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"

namespace ktt {
    namespace {

        enum class Reduce { Sum, Mean, Max };

        // `kind` of `self` over logical `dims` (all when empty), as a tt tensor of
        // PyTorch's shape.
        at::Tensor reduce(
            const char* op,
            Reduce kind,
            const at::Tensor& self,
            std::vector<int64_t> dims,
            bool keep
        ) {
            const int64_t rank = self.dim();
            if (rank == 0) {
                return self.clone();
            }
            if (dims.empty()) {
                for (int64_t d = 0; d < rank; ++d) {
                    dims.push_back(d);
                }
            }
            std::vector<bool> reduced(rank, false);
            for (auto& d : dims) {
                d = c10::maybe_wrap_dim(d, rank);
                reduced[d] = true;
            }

            // Output sizes, the device dims (rank < 2 tensors are padded) and the
            // count.
            std::vector<int64_t> out, device_dims;
            int64_t count = 1;
            for (int64_t d = 0; d < rank; ++d) {
                if (!reduced[d]) {
                    out.push_back(self.size(d));
                } else if (keep) {
                    out.push_back(1);
                }
                if (reduced[d]) {
                    device_dims.push_back(d + std::max<int64_t>(0, 2 - rank));
                    count *= self.size(d);
                }
            }
            if (device_dims.empty()) {
                return self.clone();
            }

            // Empty inputs: sum 0, mean NaN (PyTorch's values); max over nothing is an
            // error.
            if (self.numel() == 0) {
                KTT_REQUIRE(kind != Reduce::Max, op, "max over an empty dimension");
                const double fill =
                    (kind == Reduce::Sum ? 0.0
                                         : std::numeric_limits<double>::quiet_NaN());
                return at::full(out, fill, self.options());
            }

            // Mean of integers is an error in PyTorch; sums of bool/int stay integral.
            auto dtype = self.scalar_type();
            auto x = dense(op, self);
            if (kind == Reduce::Max) {
                return wrap_output(bridge::amax(x, device_dims, out), out, dtype);
            }
            auto h = bridge::sum(x, device_dims, out);
            if (kind == Reduce::Mean) {
                h = (
                    bridge::binary_scalar(
                        bridge::BinaryOp::Div, h, static_cast<double>(count)
                    )
                );
            }
            return wrap_output(h, out, dtype);
        }

        std::vector<int64_t> dim_list(at::OptionalIntArrayRef dim) {
            return dim ? to_vector(*dim) : std::vector<int64_t>{};
        }

        // Complex sum / mean over `dims` (all when empty) of the fp32 pairs.
        at::Tensor complex_reduce(
            const at::Tensor& self, std::vector<int64_t> dims, bool keepdim, bool mean
        ) {
            if (self.dim() == 0) {
                return self.clone();
            }
            if (dims.empty()) {
                for (int64_t d = 0; d < self.dim(); ++d) {
                    dims.push_back(d);
                }
            }
            for (auto& d : dims) {
                d = c10::maybe_wrap_dim(d, self.dim());
            }
            auto pairs = as_real(self);
            return (
                as_complex(mean ? pairs.mean(dims, keepdim) : pairs.sum(dims, keepdim))
            );
        }

        at::Tensor sum(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
            if (holds_complex(self)) {
                return complex_reduce(self, {}, false, false);
            }
            KTT_REQUIRE(
                !dtype || *dtype == self.scalar_type(),
                "sum",
                "dtype cast not supported yet"
            );
            return reduce("sum", Reduce::Sum, self, {}, false);
        }

        at::Tensor sum_dim(
            const at::Tensor& self,
            at::OptionalIntArrayRef dim,
            bool keepdim,
            std::optional<at::ScalarType> dtype
        ) {
            if (holds_complex(self)) {
                return complex_reduce(self, dim_list(dim), keepdim, false);
            }
            KTT_REQUIRE(
                !dtype || *dtype == self.scalar_type(),
                "sum.dim_IntList",
                "dtype cast not supported yet"
            );
            return reduce("sum.dim_IntList", Reduce::Sum, self, dim_list(dim), keepdim);
        }

        at::Tensor mean(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
            if (holds_complex(self)) {
                return complex_reduce(self, {}, false, true);
            }
            KTT_REQUIRE(
                !dtype || *dtype == self.scalar_type(),
                "mean",
                "dtype cast not supported yet"
            );
            return reduce("mean", Reduce::Mean, self, {}, false);
        }

        at::Tensor mean_dim(
            const at::Tensor& self,
            at::OptionalIntArrayRef dim,
            bool keepdim,
            std::optional<at::ScalarType> dtype
        ) {
            if (holds_complex(self)) {
                return complex_reduce(self, dim_list(dim), keepdim, true);
            }
            KTT_REQUIRE(
                !dtype || *dtype == self.scalar_type(),
                "mean.dim",
                "dtype cast not supported yet"
            );
            return reduce("mean.dim", Reduce::Mean, self, dim_list(dim), keepdim);
        }

        at::Tensor amax(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
            return reduce("amax", Reduce::Max, self, to_vector(dim), keepdim);
        }

        at::Tensor& amax_out(
            const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::Tensor& out
        ) {
            auto result = (
                reduce("amax.out", Reduce::Max, self, to_vector(dim), keepdim)
            );
            if (out.sizes() != result.sizes()) {
                out.resize_(result.sizes());
            }
            return write_into("amax.out", out, result.to(out.scalar_type()));
        }

        // Product of floating `self` over logical `dim` (all dims when none).
        // True unless KTT_TREE_PROD=0 (TT-NN's prod, which rounds to bf16).
        bool tree_prod() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_TREE_PROD");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // Product over the last dim of fp32 `x` by halving: x[:h] * x[h:2h], with an
        // odd last element multiplied into the first.
        at::Tensor tree_product(at::Tensor x) {
            while (x.size(-1) > 1) {
                const int64_t n = x.size(-1), h = n / 2;
                auto halves = x.narrow(-1, 0, h) * x.narrow(-1, h, h);
                if (n % 2 == 1) {
                    auto first = halves.narrow(-1, 0, 1) * x.narrow(-1, n - 1, 1);
                    halves = h == 1 ? first
                                    : at::cat({first, halves.narrow(-1, 1, h - 1)}, -1);
                }
                x = halves;
            }
            return x.squeeze(-1);
        }

        at::Tensor prod_impl(
            const char* op,
            const at::Tensor& self,
            std::optional<int64_t> dim,
            bool keep
        ) {
            KTT_REQUIRE(
                at::isFloatingType(self.scalar_type()), op, "floating dtypes only"
            );
            if (self.dim() == 0) {
                return self.clone();
            }
            std::vector<int64_t> out;
            std::optional<int64_t> device_dim;
            if (dim) {
                const int64_t d = c10::maybe_wrap_dim(*dim, self.dim());
                out = self.sizes().vec();
                if (keep) {
                    out[d] = 1;
                } else {
                    out.erase(out.begin() + d);
                }
                device_dim = d + std::max<int64_t>(0, 2 - self.dim());
            }
            if (self.numel() == 0) {
                return at::ones(out, self.options());
            }
            if (tree_prod()) {
                auto f = self.to(at::kFloat);
                auto rows = dim ? f.movedim(*dim, -1) : f.reshape({-1});
                return tree_product(rows).reshape(out).to(self.scalar_type());
            }
            auto h = bridge::prod(dense_as(op, self, at::kFloat), device_dim, out);
            return wrap_output(h, out, at::kFloat).to(self.scalar_type());
        }

        at::Tensor prod(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
            KTT_REQUIRE(!dtype || *dtype == self.scalar_type(), "prod", "dtype cast");
            return prod_impl("prod", self, std::nullopt, false);
        }

        at::Tensor prod_dim(
            const at::Tensor& self,
            int64_t dim,
            bool keepdim,
            std::optional<at::ScalarType> dtype
        ) {
            KTT_REQUIRE(
                !dtype || *dtype == self.scalar_type(), "prod.dim_int", "dtype cast"
            );
            return prod_impl("prod.dim_int", self, dim, keepdim);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("sum", &sum);
        m.impl("sum.dim_IntList", &sum_dim);
        m.impl("mean", &mean);
        m.impl("mean.dim", &mean_dim);
        m.impl("amax", &amax);
        m.impl("amax.out", &amax_out);
        m.impl("prod", &prod);
        m.impl("prod.dim_int", &prod_dim);
    }

}  // namespace ktt
