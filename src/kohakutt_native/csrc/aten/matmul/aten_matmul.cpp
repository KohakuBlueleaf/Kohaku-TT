// aten matmul kernels for tt: mm, bmm, linear (+ backward); fidelity per
// KTT_MATMUL_FIDELITY (default HiFi4), fp32 accumulation.
#include <torch/library.h>

#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"

namespace ktt {
    namespace {

        // The device operand of a 2-D matmul input: block-float storage as is, a
        // transpose of a contiguous tensor as that tensor with `transposed` set,
        // anything else resolved densely.
        bridge::HandlePtr operand(
            const char* op, const at::Tensor& t, bool& transposed
        ) {
            if (auto block = block_base(t, transposed)) {
                return block;
            }
            transposed =
                (t.size(0) > 1 && t.size(1) > 1 && t.stride(0) == 1 &&
                 t.stride(1) == t.size(0));
            return dense(op, transposed ? t.t() : t);
        }

        // The local bias operand of a mesh linear whose output (device rank
        // `out_rank`) is placed `out`, when `bias` has `input`'s dtype: with no partial
        // axis, `bias`'s shard on each chip (per mesh axis sharded on its last dim
        // where `out` is sharded on its last dim, replicated elsewhere); with P(sum)
        // axes over 2^k chips and R elsewhere, replicated `bias` / 2^k; null otherwise.
        bridge::HandlePtr local_bias(
            const at::Tensor& input,
            const at::Tensor& bias,
            const bridge::Placement& out,
            size_t out_rank
        ) {
            if (bias.scalar_type() != input.scalar_type()) {
                return nullptr;
            }
            auto b = mesh::placed_base(bias);
            if (!b) {
                b = dense("linear", bias);
            }
            const auto placed = bridge::placement(b);
            // A partial-sum output over 2^k chips: each chip adds bias / 2^k (exact).
            if (out.has_partial()) {
                const auto mesh = bridge::mesh_shape();
                int64_t chips = 1;
                for (size_t a = 0; a < out.axes.size(); ++a) {
                    const auto& o = out.axes[a];
                    if (o.kind == bridge::AxisKind::P &&
                        o.op != bridge::ReduceKind::Sum) {
                        return nullptr;
                    }
                    if (o.kind == bridge::AxisKind::S) {
                        return nullptr;
                    }
                    if (o.kind == bridge::AxisKind::P && a < mesh.size()) {
                        chips *= mesh[a];
                    }
                }
                if (!placed.is_replicated() || (chips & (chips - 1)) != 0) {
                    return nullptr;
                }
                return bridge::scaled_copy(b, 1.0 / static_cast<double>(chips));
            }
            const int out_last = static_cast<int>(out_rank) - 1;
            const int bias_last = static_cast<int>(bridge::device_sizes(b).size()) - 1;
            const size_t axes = std::max(out.axes.size(), placed.axes.size());
            for (size_t a = 0; a < axes; ++a) {
                const auto o = a < out.axes.size() ? out.axes[a]
                                                   : bridge::AxisPlacement{};
                const auto p =
                    (a < placed.axes.size() ? placed.axes[a] : bridge::AxisPlacement{});
                if (o.kind == bridge::AxisKind::P) {
                    return nullptr;
                }
                const auto want =
                    (o.kind == bridge::AxisKind::S && o.dim == out_last
                         ? bridge::AxisPlacement::shard(bias_last)
                         : bridge::AxisPlacement{});
                if (!(p == want)) {
                    return nullptr;
                }
            }
            return bridge::local_view(b);
        }

        at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
            KTT_REQUIRE(self.dim() == 2 && mat2.dim() == 2, "mm", "needs 2-D tensors");
            KTT_REQUIRE(self.size(1) == mat2.size(0), "mm", "inner sizes differ");
            KTT_REQUIRE(self.scalar_type() == mat2.scalar_type(), "mm", "mixed dtypes");
            // Mesh-placed operand (as stored, or the transpose of a contiguous one):
            // the matmul per chip on the shards.
            {
                auto placed = [](const at::Tensor& t, bool& transposed) {
                    transposed = !t.is_contiguous() && t.t().is_contiguous();
                    return mesh::placed_base(transposed ? t.t() : t);
                };
                bool ta = false, tb = false;
                auto as = placed(self, ta);
                auto bs = placed(mat2, tb);
                if (as || bs) {
                    auto a = as ? as : dense("mm", ta ? self.t() : self);
                    auto b = bs ? bs : dense("mm", tb ? mat2.t() : mat2);
                    auto out = mesh::run_placed(
                        "mm",
                        mesh::matmul(2, ta, tb),
                        {a, b},
                        {self.size(0), mat2.size(1)},
                        [&](const auto& in) {
                            return bridge::mm(in[0], in[1], ta, tb);
                        }
                    );
                    if (out) {
                        return wrap_output(
                            *out, {self.size(0), mat2.size(1)}, self.scalar_type()
                        );
                    }
                }
            }
            bool ta = false, tb = false;
            auto a = operand("mm", self, ta);
            auto b = operand("mm", mat2, tb);
            return wrap_output(
                bridge::mm(a, b, ta, tb),
                {self.size(0), mat2.size(1)},
                self.scalar_type()
            );
        }

        // Batched a @ b over the leading dim; transposes of the last two dims are
        // absorbed.
        at::Tensor bmm(const at::Tensor& self, const at::Tensor& mat2) {
            KTT_REQUIRE(self.dim() == 3 && mat2.dim() == 3, "bmm", "needs 3-D tensors");
            KTT_REQUIRE(self.size(0) == mat2.size(0), "bmm", "batch sizes differ");
            KTT_REQUIRE(self.size(2) == mat2.size(1), "bmm", "inner sizes differ");
            KTT_REQUIRE(
                self.scalar_type() == mat2.scalar_type(), "bmm", "mixed dtypes"
            );
            auto operand3 = [](const at::Tensor& t, bool& transposed) {
                auto swapped = t.transpose(1, 2);
                transposed = !t.is_contiguous() && swapped.is_contiguous();
                return dense("bmm", transposed ? swapped : t);
            };
            // Mesh-placed operand: the batched matmul per chip on the shards.
            {
                auto placed = [](const at::Tensor& t, bool& transposed) {
                    auto swapped = t.transpose(1, 2);
                    transposed = !t.is_contiguous() && swapped.is_contiguous();
                    return mesh::placed_base(transposed ? swapped : t);
                };
                bool ta = false, tb = false;
                auto as = placed(self, ta);
                auto bs = placed(mat2, tb);
                if (as || bs) {
                    auto a = as ? as : dense("bmm", ta ? self.transpose(1, 2) : self);
                    auto b = bs ? bs : dense("bmm", tb ? mat2.transpose(1, 2) : mat2);
                    const std::vector<int64_t> sizes{
                        self.size(0), self.size(1), mat2.size(2)
                    };
                    auto out = mesh::run_placed(
                        "bmm", mesh::bmm(ta, tb), {a, b}, sizes, [&](const auto& in) {
                            return bridge::mm(in[0], in[1], ta, tb);
                        }
                    );
                    if (out) {
                        return wrap_output(*out, sizes, self.scalar_type());
                    }
                }
            }
            bool ta = false, tb = false;
            auto a = operand3(self, ta);
            auto b = operand3(mat2, tb);
            return wrap_output(
                bridge::mm(a, b, ta, tb),
                {self.size(0), self.size(1), mat2.size(2)},
                self.scalar_type()
            );
        }

        // Sizes of `t` with the last dim replaced by `last`.
        std::vector<int64_t> with_last(const at::Tensor& t, int64_t last) {
            auto sizes = to_vector(t.sizes());
            sizes.back() = last;
            return sizes;
        }

        // F.linear(input, weight, bias): input (..., in), weight (out, in), bias
        // (out,).
        at::Tensor linear(
            const at::Tensor& input,
            const at::Tensor& weight,
            const std::optional<at::Tensor>& bias
        ) {
            // A 1-D weight (in,): input @ weight, the last dim dropped.
            if (weight.dim() == 1 && input.dim() >= 1) {
                auto out = (input.to(at::kFloat) * weight.to(at::kFloat)).sum(-1);
                if (bias && bias->defined()) {
                    out = out + bias->to(at::kFloat);
                }
                return out.to(at::result_type(input, weight));
            }
            KTT_REQUIRE(
                input.dim() >= 1 && weight.dim() == 2, "linear", "needs 2-D weight"
            );
            KTT_REQUIRE(
                input.size(-1) == weight.size(1), "linear", "inner sizes differ"
            );
            if (bias && bias->defined()) {
                KTT_REQUIRE(
                    bias->dim() == 1 && bias->size(0) == weight.size(0),
                    "linear",
                    "bias must be (out,)"
                );
            }
            // Degenerate sizes: no rows, or no inner dim (the result is the bias, or
            // zeros).
            auto out_sizes = with_last(input, weight.size(0));
            if (input.numel() == 0 || weight.numel() == 0) {
                auto zeros = at::zeros(out_sizes, input.options());
                return (bias && bias->defined()) ? zeros + *bias : zeros;
            }
            // Mesh-placed input or weight (rank 2-4 input, plain weight): the matmul
            // per chip on the shards (column-, row- or data-parallel by the cheapest
            // conversion). A bias placed as the output's last dim goes to each chip's
            // matmul (local_bias); any other is added through the elementwise rules.
            if (input.dim() >= 2 && input.dim() <= 4) {
                const bool wt = !weight.is_contiguous() && weight.t().is_contiguous();
                auto xs = mesh::placed_base(input);
                auto ws = mesh::placed_base(wt ? weight.t() : weight);
                if (xs || ws) {
                    auto x = xs ? xs : dense("linear", input);
                    auto w = ws ? ws : dense("linear", wt ? weight.t() : weight);
                    auto f = mesh::matmul(bridge::device_sizes(x).size(), false, !wt);
                    const auto out_device = bridge::device_sizes_of(out_sizes);
                    const bool biased = bias && bias->defined();
                    mesh::Choice chosen;
                    bridge::HandlePtr local;
                    auto out = mesh::run_placed(
                        "linear",
                        f,
                        {x, w},
                        out_device,
                        [&](const auto& in) {
                            if (biased) {
                                local = local_bias(
                                    input, *bias, chosen.output, out_device.size()
                                );
                            }
                            if (auto fused = bridge::reduce_linear(
                                    in[0], in[1], local, wt
                                )) {
                                return fused;
                            }
                            return bridge::linear(in[0], in[1], local, wt);
                        },
                        &chosen
                    );
                    if (out) {
                        auto result = wrap_output(*out, out_sizes, input.scalar_type());
                        return (biased && !local) ? result + *bias : result;
                    }
                }
            }

            bridge::HandlePtr b;
            if (bias && bias->defined()) {
                b = dense("linear", *bias);
            }

            // The weight goes to the matmul as stored: block floats (mixed with the
            // input's dtype), and a transpose of a contiguous (in, out) tensor as that
            // tensor (`kn`).
            bool kn = false;
            auto w = block_base(weight, kn);
            if (!w) {
                kn =
                    (!weight.is_contiguous() && weight.t().is_contiguous() &&
                     weight.size(0) > 1 && weight.size(1) > 1);
                w = dense("linear", kn ? weight.t() : weight);
            }

            // N-d inputs (rank 2-4) go to TT-NN as they are; it batches over leading
            // dims.
            if (input.dim() >= 2 && input.dim() <= 4) {
                auto h = bridge::linear(dense("linear", input), w, b, kn);
                if (bridge::device_sizes(h) == bridge::device_sizes_of(out_sizes)) {
                    return wrap_output(h, out_sizes, input.scalar_type());
                }
            }
            auto rows = input.reshape({input.numel() / input.size(-1), input.size(-1)});
            auto h = bridge::linear(dense("linear", rows), w, b, kn);
            auto out = wrap_output(
                h, {rows.size(0), weight.size(0)}, input.scalar_type()
            );
            return out.view(out_sizes);
        }

        // Gradients of linear: grad @ weight, grad^T @ input, sum of grad over rows.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
            const at::Tensor& self,
            const at::Tensor& grad_output,
            const at::Tensor& weight,
            std::array<bool, 3> mask
        ) {
            auto x = self.reshape({-1, self.size(-1)});
            auto g = grad_output.reshape({-1, grad_output.size(-1)});
            at::Tensor grad_input, grad_weight, grad_bias;
            if (mask[0]) {
                grad_input = at::mm(g, weight).view(self.sizes());
            }
            if (mask[1]) {
                grad_weight = at::mm(g.t(), x);
            }
            if (mask[2]) {
                grad_bias = g.sum(0);
            }
            return {grad_input, grad_weight, grad_bias};
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("mm", &mm);
        m.impl("bmm", &bmm);
        m.impl("linear", &linear);
        m.impl("linear_backward", &linear_backward);
    }

}  // namespace ktt
