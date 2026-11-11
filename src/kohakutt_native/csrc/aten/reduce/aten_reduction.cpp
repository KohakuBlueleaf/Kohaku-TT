// aten reduction kernels for tt: sum, mean, amax, prod (all dims or a dim list).
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <limits>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"
#include "core/torch/mesh_rules.hpp"

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

            // Callers pass sums of bool / integers as int64 (accumulated()).
            auto dtype = self.scalar_type();

            // Mesh-placed input: per chip on its shard. Reduced dims are reduction
            // factors (a sharded one leaves P(sum) / P(max)), kept dims pass through.
            if (auto base = mesh::placed_base(self)) {
                const int64_t lift_in = std::max<int64_t>(0, 2 - rank);
                const int64_t out_rank = static_cast<int64_t>(out.size());
                const int64_t lift_out = std::max<int64_t>(0, 2 - out_rank);
                mesh::Factors f;
                std::vector<int> in_f(lift_in, -1), out_f(lift_out, -1);
                f.kinds.assign(rank, mesh::FactorKind::Pass);
                for (int64_t d = 0; d < rank; ++d) {
                    in_f.push_back(static_cast<int>(d));
                    if (reduced[d]) {
                        f.kinds[d] = mesh::FactorKind::Reduce;
                        if (keep) {
                            out_f.push_back(-1);
                        }
                    } else {
                        out_f.push_back(static_cast<int>(d));
                    }
                }
                f.inputs = {in_f};
                f.output = out_f;
                f.reduce =
                    (kind == Reduce::Max ? bridge::ReduceKind::Max
                                         : bridge::ReduceKind::Sum);
                // Sum and mean are linear: a P(sum) input stays P(sum).
                if (kind != Reduce::Max) {
                    f.partial = mesh::Factors::Partial::All;
                }
                auto placed = mesh::run_placed(
                    op, f, {base}, bridge::device_sizes_of(out), [&](const auto& in) {
                        // The local shard's output sizes.
                        const auto local = bridge::device_sizes(in[0]);
                        std::vector<int64_t> local_out;
                        for (int64_t d = 0; d < rank; ++d) {
                            if (!reduced[d]) {
                                local_out.push_back(local[d + lift_in]);
                            } else if (keep) {
                                local_out.push_back(1);
                            }
                        }
                        if (kind == Reduce::Max) {
                            return bridge::amax(in[0], device_dims, local_out);
                        }
                        auto s = bridge::sum(in[0], device_dims, local_out);
                        return (
                            kind == Reduce::Mean ? bridge::binary_scalar(
                                                       bridge::BinaryOp::Div,
                                                       s,
                                                       static_cast<double>(count)
                                                   )
                                                 : s
                        );
                    }
                );
                if (placed) {
                    return wrap_output(*placed, out, dtype);
                }
            }
            // All dims of a whole stored-format tensor (e.g. Flat): reduced as stored.
            auto x = dense_stored(op, self);
            // Kept dims of a head-major tensor: reduced as stored, the result
            // head-major.
            const bool major =
                (bridge::format(x) == bridge::Format::HeadMajor && keep &&
                 static_cast<int64_t>(device_dims.size()) != rank);
            auto stored_out = out;
            if (bridge::format(x) != bridge::Format::Nchw) {
                if (static_cast<int64_t>(device_dims.size()) == rank) {
                    x = bridge::as_stored(x);
                    device_dims = {0, 1, 2, 3};
                    device_dims.resize(bridge::device_sizes(x).size());
                } else if (major) {
                    x = bridge::as_stored(x);
                    for (auto& d : device_dims) {
                        d = d == rank - 3 ? rank - 2 : d == rank - 2 ? rank - 3 : d;
                    }
                    std::swap(stored_out[rank - 3], stored_out[rank - 2]);
                } else {
                    x = bridge::to_nchw(x);
                }
            }
            auto wrap = [&](const bridge::HandlePtr& h) {
                return wrap_output(
                    major ? bridge::view_as(h, out, bridge::Format::HeadMajor) : h,
                    out,
                    dtype
                );
            };
            if (kind == Reduce::Max) {
                return wrap(bridge::amax(x, device_dims, stored_out));
            }
            auto h = bridge::sum(x, device_dims, stored_out);
            if (kind == Reduce::Mean) {
                h = (bridge::binary_scalar(
                    bridge::BinaryOp::Div, h, static_cast<double>(count)
                ));
            }
            return wrap(h);
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

        // `self` in the dtype PyTorch accumulates a sum or product in: `dtype` when
        // given, else int64 for bool and integer inputs, else its own.
        at::Tensor accumulated(
            const at::Tensor& self, std::optional<at::ScalarType> dtype
        ) {
            const bool include_bool = true;
            auto target =
                (dtype ? *dtype
                     : at::isIntegralType(self.scalar_type(), include_bool)
                     ? at::kLong
                     : self.scalar_type());
            return target == self.scalar_type() ? self : self.to(target);
        }

        // `self` in `dtype` for mean; integer and bool inputs without one are an
        // error, as in PyTorch.
        at::Tensor mean_input(
            const char* op, const at::Tensor& self, std::optional<at::ScalarType> dtype
        ) {
            auto x = dtype && *dtype != self.scalar_type() ? self.to(*dtype) : self;
            TORCH_CHECK(
                at::isFloatingType(x.scalar_type()),
                op,
                "(): could not infer output dtype. Input dtype must be either a "
                "floating point or complex dtype. Got: ",
                x.scalar_type()
            );
            return x;
        }

        at::Tensor sum(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
            if (holds_complex(self)) {
                return complex_reduce(self, {}, false, false);
            }
            return reduce("sum", Reduce::Sum, accumulated(self, dtype), {}, false);
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
            return reduce(
                "sum.dim_IntList",
                Reduce::Sum,
                accumulated(self, dtype),
                dim_list(dim),
                keepdim
            );
        }

        at::Tensor mean(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
            if (holds_complex(self)) {
                return complex_reduce(self, {}, false, true);
            }
            auto x = mean_input("mean", self, dtype);
            return reduce("mean", Reduce::Mean, x, {}, false);
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
            auto x = mean_input("mean", self, dtype);
            return reduce("mean.dim", Reduce::Mean, x, dim_list(dim), keepdim);
        }

        at::Tensor amax(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
            return reduce("amax", Reduce::Max, self, to_vector(dim), keepdim);
        }

        at::Tensor& amax_out(
            const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::Tensor& out
        ) {
            auto result = (reduce(
                "amax.out", Reduce::Max, self, to_vector(dim), keepdim
            ));
            if (out.sizes() != result.sizes()) {
                out.resize_(result.sizes());
            }
            return write_into("amax.out", out, result.to(out.scalar_type()));
        }

        // Product of `self` over logical `dim` (all dims when none).
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
            // Integers (stored int32): the product by halving in integer multiplies.
            if (!at::isFloatingType(self.scalar_type())) {
                auto rows = dim ? self.movedim(*dim, -1) : self.reshape({-1});
                return tree_product(rows).reshape(out);
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
            return prod_impl("prod", accumulated(self, dtype), std::nullopt, false);
        }

        at::Tensor prod_dim(
            const at::Tensor& self,
            int64_t dim,
            bool keepdim,
            std::optional<at::ScalarType> dtype
        ) {
            return prod_impl("prod.dim_int", accumulated(self, dtype), dim, keepdim);
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
