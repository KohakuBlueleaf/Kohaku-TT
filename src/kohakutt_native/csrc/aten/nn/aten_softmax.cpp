// aten softmax kernels for tt: softmax, safe softmax, log_softmax (+ backward).

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>

#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"

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

        // True when mesh-placed `base` is sharded on device dim `d`.
        bool sharded_on(const bridge::HandlePtr& base, int64_t d) {
            for (const auto& ax : bridge::placement(base).axes) {
                if (ax.kind == bridge::AxisKind::S && ax.dim == d) {
                    return true;
                }
            }
            return false;
        }

        // A mesh-placed input whose softmax dim is not sharded: the softmax per chip
        // on its shard (that dim whole on every chip). nullopt otherwise.
        std::optional<at::Tensor> local_softmax(
            const char* op, const at::Tensor& self, int64_t d, bool log
        ) {
            auto base = mesh::placed_base(self);
            if (!base || sharded_on(base, d) || self.dim() < 2) {
                return std::nullopt;
            }
            const auto dims = bridge::device_sizes(base);
            auto f = mesh::elementwise(1, dims.size());
            f.kinds[d] = mesh::FactorKind::Replicate;
            auto out = mesh::run_placed(op, f, {base}, dims, [&](const auto& in) {
                if (!log) {
                    return bridge::softmax(in[0], d);
                }
                std::vector<int64_t> keep = bridge::device_sizes(in[0]);
                keep[d] = 1;
                auto shifted = bridge::binary(
                    B::Sub, in[0], bridge::amax(in[0], {d}, keep)
                );
                auto lse = bridge::unary(
                    U::Log, bridge::sum(bridge::unary(U::Exp, shifted), {d}, keep)
                );
                return bridge::binary(B::Sub, shifted, lse);
            });
            if (!out) {
                return std::nullopt;
            }
            return wrap_output(*out, self.sizes(), self.scalar_type());
        }

        at::Tensor tt_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
            KTT_REQUIRE(!half_to_float, "_softmax", "half_to_float is not supported");
            if (self.dim() == 0) {
                return at::ones_like(self);  // softmax of one element
            }
            if (auto base = mesh::placed_base(self)) {
                const int64_t d = device_dim(dim, self.dim());
                if (auto local = local_softmax("_softmax", self, d, false)) {
                    return *local;
                }
                // Sharded softmax dim: max and sum over it as statistics (P -> R of
                // [rows, 1]), the rest per chip.
                if (sharded_on(base, d)) {
                    auto e = (self - self.amax(dim, true)).exp();
                    return e / e.sum(dim, true);
                }
            }
            auto h = (bridge::softmax(
                dense("_softmax", self), device_dim(dim, self.dim())
            ));
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
            auto dead = (bridge::binary_scalar(
                B::Eq, row_max, -std::numeric_limits<double>::infinity()
            ));
            auto zeros = bridge::full_like(probs, 0.0);
            auto h = (bridge::where(
                bridge::typecast(dead, bridge::dtype(probs)), zeros, probs
            ));
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
            if (auto base = mesh::placed_base(self)) {
                if (auto local = local_softmax("_log_softmax", self, d, true)) {
                    return *local;
                }
                if (sharded_on(base, d)) {
                    auto shifted = self - self.amax(dim, true);
                    return shifted - shifted.exp().sum(dim, true).log();
                }
            }
            const auto keep = kept(self, dim);
            auto x = dense("_log_softmax", self);
            auto shifted = bridge::binary(B::Sub, x, bridge::amax(x, {d}, keep));
            auto lse = bridge::unary(
                U::Log, bridge::sum(bridge::unary(U::Exp, shifted), {d}, keep)
            );
            return (wrap_output(
                bridge::binary(B::Sub, shifted, lse), self.sizes(), self.scalar_type()
            ));
        }

        // output * (grad - sum(grad * output, dim)).
        at::Tensor tt_softmax_backward(
            const at::Tensor& grad,
            const at::Tensor& output,
            int64_t dim,
            at::ScalarType
        ) {
            // Mesh-placed operands: the same formula through the placement rules.
            if (mesh::placed_base(grad) || mesh::placed_base(output)) {
                return output * (grad - (grad * output).sum(dim, true));
            }
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
            // Mesh-placed operands: the same formula through the placement rules.
            if (mesh::placed_base(grad) || mesh::placed_base(output)) {
                return grad - output.exp() * grad.sum(dim, true);
            }
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
