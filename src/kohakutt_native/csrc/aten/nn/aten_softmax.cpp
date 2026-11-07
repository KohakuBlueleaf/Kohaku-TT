// aten softmax kernels for tt: softmax, safe softmax, log_softmax (+ backward).

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        using B = bridge::BinaryOp;
        using U = bridge::UnaryOp;

        // The device dim of logical dim `dim` of a rank-`rank` tensor (rank < 2 is
        // padded); 0-d tensors are rejected by the callers.
        int64_t device_dim(int64_t dim, int64_t rank) {
            TORCH_CHECK_NOT_IMPLEMENTED(
                rank >= 1, "KohakuTTUnsupportedOperator: softmax family: 0-d input"
            );
            return c10::maybe_wrap_dim(dim, rank) + std::max<int64_t>(0, 2 - rank);
        }

        // Logical sizes of `t` with dim `dim` set to 1 (keepdim reductions).
        std::vector<int64_t> kept(const at::Tensor& t, int64_t dim) {
            auto sizes = to_vector(t.sizes());
            if (!sizes.empty()) {
                sizes[c10::maybe_wrap_dim(dim, t.dim())] = 1;
            }
            return sizes;
        }

        at::Tensor tt_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
            KTT_REQUIRE(!half_to_float, "_softmax", "half_to_float is not supported");
            if (self.dim() == 0) {
                return at::ones_like(self);  // softmax of one element
            }
            auto h = (
                bridge::softmax(dense("_softmax", self), device_dim(dim, self.dim()))
            );
            return wrap_output(h, self.sizes(), self.scalar_type());
        }

        // softmax that yields 0 (not NaN) for rows that are entirely -inf (SDPA's
        // masking).
        at::Tensor tt_safe_softmax(
            const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype
        ) {
            KTT_REQUIRE(
                !dtype || *dtype == self.scalar_type(), "_safe_softmax", "dtype cast"
            );
            const auto d = device_dim(dim, self.dim());
            auto x = dense("_safe_softmax", self);
            auto probs = bridge::softmax(x, d);
            auto row_max = bridge::amax(x, {d}, kept(self, dim));
            auto dead = (
                bridge::binary_scalar(
                    B::Eq, row_max, -std::numeric_limits<double>::infinity()
                )
            );
            auto zeros = bridge::full_like(probs, 0.0);
            auto h = (
                bridge::where(
                    bridge::typecast(dead, bridge::dtype(probs)), zeros, probs
                )
            );
            return wrap_output(h, self.sizes(), self.scalar_type());
        }

        // x - max - log(sum(exp(x - max))) along `dim`.
        at::Tensor tt_log_softmax(
            const at::Tensor& self, int64_t dim, bool half_to_float
        ) {
            KTT_REQUIRE(
                !half_to_float, "_log_softmax", "half_to_float is not supported"
            );
            if (self.dim() == 0) {
                return at::zeros_like(self);  // log_softmax of one element
            }
            const auto d = device_dim(dim, self.dim());
            const auto keep = kept(self, dim);
            auto x = dense("_log_softmax", self);
            auto shifted = bridge::binary(B::Sub, x, bridge::amax(x, {d}, keep));
            auto lse = bridge::unary(
                U::Log, bridge::sum(bridge::unary(U::Exp, shifted), {d}, keep)
            );
            return (
                wrap_output(
                    bridge::binary(B::Sub, shifted, lse), self.sizes(), self.scalar_type()
                )
            );
        }

        // output * (grad - sum(grad * output, dim)).
        at::Tensor tt_softmax_backward(
            const at::Tensor& grad,
            const at::Tensor& output,
            int64_t dim,
            at::ScalarType
        ) {
            const auto d = device_dim(dim, grad.dim());
            auto g = dense("_softmax_backward_data", grad);
            auto y = dense("_softmax_backward_data", output);
            auto dot = bridge::sum(bridge::binary(B::Mul, g, y), {d}, kept(grad, dim));
            auto h = bridge::binary(B::Mul, y, bridge::binary(B::Sub, g, dot));
            return wrap_output(h, grad.sizes(), grad.scalar_type());
        }

        // grad - exp(output) * sum(grad, dim).
        at::Tensor tt_log_softmax_backward(
            const at::Tensor& grad,
            const at::Tensor& output,
            int64_t dim,
            at::ScalarType
        ) {
            const auto d = device_dim(dim, grad.dim());
            auto g = dense("_log_softmax_backward_data", grad);
            auto total = bridge::sum(g, {d}, kept(grad, dim));
            auto p = bridge::unary(U::Exp, dense("_log_softmax_backward_data", output));
            auto h = bridge::binary(B::Sub, g, bridge::binary(B::Mul, p, total));
            return wrap_output(h, grad.sizes(), grad.scalar_type());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("_softmax", &tt_softmax);
        m.impl("_safe_softmax", &tt_safe_softmax);
        m.impl("_log_softmax", &tt_log_softmax);
        m.impl("_softmax_backward_data", &tt_softmax_backward);
        m.impl("_log_softmax_backward_data", &tt_log_softmax_backward);
    }

}  // namespace ktt
