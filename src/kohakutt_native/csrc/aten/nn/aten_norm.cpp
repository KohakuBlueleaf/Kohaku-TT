// aten normalization kernels for tt: layer norm and batch norm (+ backward), group
// norm backward, RMS norm; composed of tt kernels (at:: calls run below autograd).
#include <ATen/core/grad_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <initializer_list>
#include <limits>

#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"

namespace ktt {
    namespace {

        bool float_type(at::ScalarType t) {
            return t == at::kBFloat16 || t == at::kFloat;
        }

        // True when Kohaku-TT's norm kernels take x over its last dim with parameters
        // `params` (undefined, or 1-D bf16 / fp32 tt tensors).
        bool own_norm(
            const at::Tensor& x,
            size_t normalized,
            std::initializer_list<const std::optional<at::Tensor>*> params
        ) {
            if (!bridge::own_norm_kernels() || normalized != 1 || x.dim() < 1 ||
                !float_type(x.scalar_type()) || mesh::placed_base(x) != nullptr) {
                return false;
            }
            for (const auto* p : params) {
                if (*p && (*p)->defined() &&
                    (!(*p)->is_privateuseone() || (*p)->dim() != 1 ||
                     !float_type((*p)->scalar_type()) ||
                     mesh::placed_base(**p) != nullptr)) {
                    return false;
                }
            }
            return true;
        }

        // The torch dtype of a bridge statistics handle (fp32 or bf16).
        at::ScalarType stat_dtype(const bridge::HandlePtr& h) {
            return bridge::dtype(h) == bridge::DType::Float32 ? at::kFloat
                                                              : at::kBFloat16;
        }

        // The trailing `normalized` dims of `x`, as reduction dims.
        std::vector<int64_t> norm_dims(
            const at::Tensor& x, c10::IntArrayRef normalized
        ) {
            std::vector<int64_t> dims;
            for (
                int64_t i = x.dim() - static_cast<int64_t>(normalized.size());
                i < x.dim();
                ++i
            ) {
                dims.push_back(i);
            }
            return dims;
        }

        // (y, mean, rstd): y = (x - mean) * rstd * weight + bias over the trailing
        // dims.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm(
            const at::Tensor& input,
            c10::IntArrayRef normalized_shape,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            double eps
        ) {
            auto dims = norm_dims(input, normalized_shape);
            if (input.numel() == 0) {
                auto stat_sizes = to_vector(input.sizes());
                for (auto d : dims) {
                    stat_sizes[d] = 1;
                }
                auto stats = at::empty(stat_sizes, input.options().dtype(at::kFloat));
                return {at::empty_like(input), stats, stats.clone()};
            }
            // Last dim, bf16 / fp32: Kohaku-TT's norm kernels (fp32 statistics); bf16
            // over trailing dims: TT-NN's moreh layer norm (bf16 statistics).
            auto bf16 = [](const std::optional<at::Tensor>& t) {
                return !t || !t->defined() || t->scalar_type() == at::kBFloat16;
            };
            // A mesh-placed input takes the composite below, which runs through the
            // placement rules (statistics-only communication over sharded dims).
            const bool placed = mesh::placed_base(input) != nullptr;
            const bool own = own_norm(input, normalized_shape.size(), {&weight, &bias});
            if (own ||
                (!placed && input.scalar_type() == at::kBFloat16 && bf16(weight) &&
                 bf16(bias))) {
                const char* op = "native_layer_norm";
                auto param = [&](const std::optional<at::Tensor>& t) {
                    return t && t->defined() ? dense(op, *t) : bridge::HandlePtr();
                };
                auto r = (bridge::layer_norm(
                    dense(op, input),
                    static_cast<int64_t>(normalized_shape.size()),
                    eps,
                    param(weight),
                    param(bias)
                ));
                auto stat_sizes = to_vector(input.sizes());
                for (auto d : dims) {
                    stat_sizes[d] = 1;
                }
                return {
                    wrap_output(r.y, input.sizes(), input.scalar_type()),
                    wrap_output(r.mean, stat_sizes, stat_dtype(r.mean)),
                    wrap_output(r.rstd, stat_sizes, stat_dtype(r.rstd))
                };
            }
            auto x = input.to(at::kFloat);
            const bool keepdim = true;
            auto mean = x.mean(dims, keepdim);
            auto centered = x - mean;
            auto rstd = ((centered * centered).mean(dims, keepdim) + eps).rsqrt();
            auto y = centered * rstd;
            if (weight && weight->defined()) {
                y = y * weight->to(at::kFloat);
            }
            if (bias && bias->defined()) {
                y = y + bias->to(at::kFloat);
            }
            return {y.to(input.scalar_type()), mean, rstd};
        }

        // Gradients of layer_norm w.r.t. input, weight and bias (per output_mask).
        std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward(
            const at::Tensor& grad_out,
            const at::Tensor& input,
            c10::IntArrayRef normalized_shape,
            const at::Tensor& mean,
            const at::Tensor& rstd,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            std::array<bool, 3> mask
        ) {
            auto dims = norm_dims(input, normalized_shape);
            std::vector<int64_t> batch_dims;
            for (
                int64_t i = 0;
                i < input.dim() - static_cast<int64_t>(normalized_shape.size());
                ++i
            ) {
                batch_dims.push_back(i);
            }
            // Mesh-placed operands take the composite.
            const bool placed =
                (mesh::placed_base(input) || mesh::placed_base(grad_out) ||
                 mesh::placed_base(mean));
            // Kohaku-TT's norm kernels with fp32 statistics, or TT-NN's moreh backward
            // with the moreh forward's bf16 statistics.
            const bool own =
                (!placed && own_norm(input, normalized_shape.size(), {&weight}) &&
                 grad_out.scalar_type() == input.scalar_type() &&
                 mean.scalar_type() == at::kFloat && rstd.scalar_type() == at::kFloat);
            if (own ||
                (!placed && input.scalar_type() == at::kBFloat16 &&
                 mean.scalar_type() == at::kBFloat16 &&
                 grad_out.scalar_type() == at::kBFloat16)) {
                const char* op = "native_layer_norm_backward";
                const bool has_weight = weight && weight->defined();
                const bool has_bias = bias && bias->defined();
                // A forward's statistics are whole storages (their base as is); others
                // resolve.
                auto stats = [&](const at::Tensor& t) {
                    auto& st = storage_of(t);
                    const bool whole =
                        (st.base && t.storage_offset() == 0 &&
                         t.numel() * t.element_size() ==
                             static_cast<int64_t>(t.storage().nbytes()));
                    return (
                        whole ? bridge::to_nchw(bridge::replicated(st.base))
                              : dense(op, t)
                    );
                };
                auto r = (bridge::layer_norm_backward(
                    dense(op, grad_out),
                    dense(op, input),
                    stats(mean),
                    stats(rstd),
                    static_cast<int64_t>(normalized_shape.size()),
                    has_weight ? dense(op, *weight) : bridge::HandlePtr(),
                    mask[0],
                    mask[1] && has_weight,
                    mask[2] && has_bias
                ));
                at::Tensor gi, gw, gb;
                if (r.input) {
                    gi = wrap_output(r.input, input.sizes(), input.scalar_type());
                }
                if (r.gamma) {
                    gw = wrap_output(r.gamma, weight->sizes(), stat_dtype(r.gamma))
                             .to(weight->scalar_type());
                }
                if (r.beta) {
                    gb = wrap_output(r.beta, bias->sizes(), stat_dtype(r.beta))
                             .to(bias->scalar_type());
                }
                return {gi, gw, gb};
            }
            auto g = grad_out.to(at::kFloat);
            auto xhat = (input.to(at::kFloat) - mean) * rstd;
            at::Tensor grad_input, grad_weight, grad_bias;
            if (mask[0]) {
                auto gx =
                    ((weight && weight->defined()) ? g * weight->to(at::kFloat) : g);
                const bool keepdim = true;
                auto a = gx.mean(dims, keepdim);
                auto b = (gx * xhat).mean(dims, keepdim);
                grad_input = ((gx - a - xhat * b) * rstd).to(input.scalar_type());
            }
            if (mask[1] && weight && weight->defined()) {
                auto gw = batch_dims.empty() ? g * xhat : (g * xhat).sum(batch_dims);
                grad_weight = gw.to(weight->scalar_type());
            }
            if (mask[2] && bias && bias->defined()) {
                auto gb = batch_dims.empty() ? g : g.sum(batch_dims);
                grad_bias = gb.to(bias->scalar_type());
            }
            return {grad_input, grad_weight, grad_bias};
        }

        // Shape [1, C, 1, ...] for broadcasting per-channel values against `input`.
        std::vector<int64_t> channel_shape(const at::Tensor& input) {
            std::vector<int64_t> shape(input.dim(), 1);
            shape[1] = input.size(1);
            return shape;
        }

        // Batch norm over all dims but 1 in fp32 -> (out, save_mean, save_invstd);
        // training uses batch stats and updates given running buffers, eval uses the
        // running buffers.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm(
            const at::Tensor& input,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            const std::optional<at::Tensor>& running_mean,
            const std::optional<at::Tensor>& running_var,
            bool training,
            double momentum,
            double eps
        ) {
            const char* op = "_native_batch_norm_legit";
            KTT_REQUIRE(input.dim() >= 2, op, "input needs a channel dim");
            auto shape = channel_shape(input);
            std::vector<int64_t> dims;
            int64_t count = 1;
            for (int64_t d = 0; d < input.dim(); ++d) {
                if (d != 1) {
                    dims.push_back(d);
                    count *= input.size(d);
                }
            }
            auto x = input.to(at::kFloat);
            at::Tensor mean, var;
            if (training) {
                mean = x.mean(dims);
                auto centered = x - mean.reshape(shape);
                var = (centered * centered).mean(dims);
                if (running_mean && running_mean->defined()) {
                    running_mean->copy_(
                        *running_mean * (1.0 - momentum) + mean * momentum
                    );
                }
                if (running_var && running_var->defined()) {
                    const double bessel =
                        (static_cast<double>(count) / std::max<int64_t>(count - 1, 1));
                    auto unbiased = var * bessel;
                    running_var->copy_(
                        *running_var * (1.0 - momentum) + unbiased * momentum
                    );
                }
            } else {
                KTT_REQUIRE(
                    running_mean && running_var, op, "eval needs running statistics"
                );
                mean = running_mean->to(at::kFloat);
                var = running_var->to(at::kFloat);
            }
            auto invstd = at::rsqrt(var + eps);
            auto out = (x - mean.reshape(shape)) * invstd.reshape(shape);
            if (weight && weight->defined()) {
                out = out * weight->to(at::kFloat).reshape(shape);
            }
            if (bias && bias->defined()) {
                out = out + bias->to(at::kFloat).reshape(shape);
            }
            out = out.to(input.scalar_type());
            if (!training) {
                auto none = at::empty({0}, input.options().dtype(at::kFloat));
                return {out, none, none.clone()};
            }
            return {out, mean, invstd};
        }

        std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_legit(
            const at::Tensor& input,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            at::Tensor& running_mean,
            at::Tensor& running_var,
            bool training,
            double momentum,
            double eps
        ) {
            return (batch_norm(
                input, weight, bias, running_mean, running_var, training, momentum, eps
            ));
        }

        // Training batch norm that updates the running buffers: batch_norm's outputs
        // plus an empty uint8 reserve (as on CPU).
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
        batch_norm_with_update(
            const at::Tensor& input,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            at::Tensor& running_mean,
            at::Tensor& running_var,
            double momentum,
            double eps
        ) {
            auto [out, mean, invstd] = (batch_norm(
                input, weight, bias, running_mean, running_var, true, momentum, eps
            ));
            auto reserve = at::empty({0}, input.options().dtype(at::kByte));
            return {out, mean, invstd, reserve};
        }

        std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_no_stats(
            const at::Tensor& input,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            bool training,
            double momentum,
            double eps
        ) {
            return (batch_norm(
                input, weight, bias, std::nullopt, std::nullopt, training, momentum, eps
            ));
        }

        // Batch norm gradients in fp32 -> (grad_input, grad_weight, grad_bias) per
        // mask; training uses the saved batch stats, eval the running buffers.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward(
            const at::Tensor& grad_out,
            const at::Tensor& input,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& running_mean,
            const std::optional<at::Tensor>& running_var,
            const std::optional<at::Tensor>& save_mean,
            const std::optional<at::Tensor>& save_invstd,
            bool train,
            double eps,
            std::array<bool, 3> mask
        ) {
            const char* op = "native_batch_norm_backward";
            auto shape = channel_shape(input);
            std::vector<int64_t> dims;
            int64_t count = 1;
            for (int64_t d = 0; d < input.dim(); ++d) {
                if (d != 1) {
                    dims.push_back(d);
                    count *= input.size(d);
                }
            }
            at::Tensor mean, invstd;
            if (train) {
                KTT_REQUIRE(
                    save_mean && save_invstd, op, "training needs saved statistics"
                );
                mean = save_mean->to(at::kFloat);
                invstd = save_invstd->to(at::kFloat);
            } else {
                KTT_REQUIRE(
                    running_mean && running_var, op, "eval needs running statistics"
                );
                mean = running_mean->to(at::kFloat);
                invstd = at::rsqrt(running_var->to(at::kFloat) + eps);
            }
            auto g = grad_out.to(at::kFloat);
            auto x_hat =
                ((input.to(at::kFloat) - mean.reshape(shape)) * invstd.reshape(shape));
            auto grad_bias = g.sum(dims);
            auto grad_weight = (g * x_hat).sum(dims);
            auto w =
                (weight && weight->defined() ? weight->to(at::kFloat)
                                             : at::ones_like(invstd));

            at::Tensor grad_input;
            if (mask[0]) {
                auto scale = (w * invstd).reshape(shape);
                if (train) {
                    const double n = static_cast<double>(count);
                    auto centered =
                        (g * n - grad_bias.reshape(shape) -
                         x_hat * grad_weight.reshape(shape));
                    grad_input = scale * centered / n;
                } else {
                    grad_input = g * scale;
                }
                grad_input = grad_input.to(input.scalar_type());
            }
            auto param_dtype =
                (weight && weight->defined() ? weight->scalar_type()
                                             : input.scalar_type());
            return {
                grad_input,
                mask[1] ? grad_weight.to(param_dtype) : at::Tensor(),
                mask[2] ? grad_bias.to(param_dtype) : at::Tensor()
            };
        }

        bool own_group_norm();

        // Group norm gradients in fp32 over [N, G, D] (D = C / G * HxW), with the
        // forward's saved mean / rstd [N, G] -> (grad_input, grad_weight, grad_bias)
        // per mask.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> group_norm_backward(
            const at::Tensor& grad_out,
            const at::Tensor& input,
            const at::Tensor& mean,
            const at::Tensor& rstd,
            const std::optional<at::Tensor>& weight,
            int64_t n,
            int64_t c,
            int64_t hxw,
            int64_t groups,
            std::array<bool, 3> mask
        ) {
            // Channels-last storage (as the forward's row form takes it): Kohaku-TT's
            // backward programs, the result channels-last.
            const auto dtype = input.scalar_type();
            const auto plain = [](const at::Tensor& t) {
                return t.scalar_type() == at::kFloat ||
                    t.scalar_type() == at::kBFloat16;
            };
            const bool weighted = weight && weight->defined();
            if (own_group_norm() && input.dim() == 4 && input.numel() > 0 &&
                plain(input) && grad_out.scalar_type() == dtype &&
                grad_out.sizes() == input.sizes() && (!weighted || plain(*weight)) &&
                !mesh::placed_base(input) && !mesh::placed_base(grad_out)) {
                const char* op = "native_group_norm_backward";
                const auto x_h = dense_propagate(op, input);
                if (bridge::format(x_h) == bridge::Format::Nhwc &&
                    bridge::group_norm_rows_supported(x_h, groups)) {
                    auto g_h = dense_propagate(op, grad_out);
                    if (bridge::format(g_h) != bridge::Format::Nhwc) {
                        g_h = bridge::to_nhwc(g_h);
                    }
                    // mean / rstd [N, G] as fp32 [N, C]: each group's value at its
                    // channels. The row-form forward's are views of such tensors
                    // already.
                    const auto per_channel = [&](const at::Tensor& t) {
                        if (t.is_privateuseone() && t.scalar_type() == at::kFloat &&
                            t.storage_offset() == 0 && t.dim() == 2 &&
                            t.stride(0) == c && t.stride(1) == c / groups) {
                            const auto& st = storage_of(t);
                            if (st.base && st.dtype == at::kFloat &&
                                st.sizes == std::vector<int64_t>{n, c} &&
                                bridge::format(st.base) == bridge::Format::Nchw) {
                                return st.base;
                            }
                        }
                        return dense(
                            op,
                            t.to(at::kFloat)
                                .reshape({n, groups, 1})
                                .expand({n, groups, c / groups})
                                .reshape({n, c})
                        );
                    };
                    auto r = bridge::group_norm_rows_backward(
                        x_h,
                        g_h,
                        per_channel(mean),
                        per_channel(rstd),
                        weighted ? dense(op, *weight) : bridge::HandlePtr{},
                        groups
                    );
                    const auto params = weighted ? weight->scalar_type() : dtype;
                    const auto summed = [&](const bridge::HandlePtr& h) {
                        auto rows = wrap_output(h, {n, c}, at::kFloat);
                        return (n == 1 ? rows.reshape({c}) : rows.sum(0)).to(params);
                    };
                    return {
                        mask[0] ? wrap_output(r[0], input.sizes().vec(), dtype)
                                : at::Tensor(),
                        mask[1] ? summed(r[1]) : at::Tensor(),
                        mask[2] ? summed(r[2]) : at::Tensor()
                    };
                }
            }
            const int64_t d = c / groups * hxw;
            auto g = grad_out.to(at::kFloat).reshape({n, c, hxw});
            auto x = input.to(at::kFloat).reshape({n, groups, d});
            auto mu = mean.to(at::kFloat).reshape({n, groups, 1});
            auto r = rstd.to(at::kFloat).reshape({n, groups, 1});
            auto x_hat = ((x - mu) * r).reshape({n, c, hxw});
            const bool has_weight = weight && weight->defined();
            auto param_dtype = has_weight ? weight->scalar_type() : input.scalar_type();

            at::Tensor grad_input, grad_weight, grad_bias;
            if (mask[0]) {
                auto dx_hat = has_weight ? g * weight->to(at::kFloat).reshape({1, c, 1})
                                         : g;
                auto dxh = dx_hat.reshape({n, groups, d});
                auto xh = x_hat.reshape({n, groups, d});
                auto sum_dxh = dxh.sum({2}, true);
                auto sum_dxh_xh = (dxh * xh).sum({2}, true);
                const double dd = static_cast<double>(d);
                grad_input = (r / dd) * (dxh * dd - sum_dxh - xh * sum_dxh_xh);
                grad_input = grad_input.reshape(input.sizes()).to(input.scalar_type());
            }
            if (mask[1]) {
                grad_weight = (g * x_hat).sum({0, 2}).to(param_dtype);
            }
            if (mask[2]) {
                grad_bias = g.sum({0, 2}).to(param_dtype);
            }
            return {grad_input, grad_weight, grad_bias};
        }

        // RMS norm over the last dim on Kohaku-TT's kernels with saved fp32 rstd, and
        // its backward.
        struct RmsNormFunction : public torch::autograd::Function<RmsNormFunction> {
            static at::Tensor forward(
                torch::autograd::AutogradContext* ctx,
                const at::Tensor& x,
                const std::optional<at::Tensor>& gamma,
                double eps
            ) {
                at::AutoDispatchBelowADInplaceOrView guard;
                const char* op = "rms_norm";
                const at::Tensor weight = gamma ? *gamma : at::Tensor();
                auto r = bridge::rms_norm_train(
                    dense(op, x),
                    eps,
                    weight.defined() ? dense(op, weight) : bridge::HandlePtr{}
                );
                auto stat_sizes = to_vector(x.sizes());
                stat_sizes.back() = 1;
                auto rstd = wrap_output(r.rstd, stat_sizes, at::kFloat);
                ctx->save_for_backward({x, weight, rstd});
                return wrap_output(r.y, x.sizes(), x.scalar_type());
            }

            static torch::autograd::variable_list backward(
                torch::autograd::AutogradContext* ctx,
                torch::autograd::variable_list grads
            ) {
                const auto saved = ctx->get_saved_variables();
                const auto &x = saved[0], &weight = saved[1], &rstd = saved[2];
                if (!grads[0].defined()) {
                    return {at::Tensor(), at::Tensor(), at::Tensor()};
                }
                const char* op = "rms_norm_backward";
                const bool want_w = weight.defined() && ctx->needs_input_grad(1);
                auto r = bridge::rms_norm_backward(
                    dense(op, grads[0].to(x.scalar_type())),
                    dense(op, x),
                    dense(op, rstd),
                    weight.defined() ? dense(op, weight) : bridge::HandlePtr{},
                    ctx->needs_input_grad(0),
                    want_w
                );
                at::Tensor dx, dw;
                if (r.input) {
                    dx = wrap_output(r.input, x.sizes(), x.scalar_type());
                }
                if (r.gamma) {
                    dw = wrap_output(r.gamma, weight.sizes(), stat_dtype(r.gamma))
                             .to(weight.scalar_type());
                }
                return {dx, dw, at::Tensor()};
            }
        };

        // F.rms_norm at the autograd key: over the last dim, Kohaku-TT's kernels (with
        // its backward when gradients are needed); without gradients, TT-NN's program
        // for bf16 with W % 32 == 0 when they are off; else the composite.
        at::Tensor rms_norm(
            const at::Tensor& input,
            c10::SymIntArrayRef normalized_shape,
            const std::optional<at::Tensor>& weight,
            std::optional<double> eps
        ) {
            const double e = eps.value_or(std::numeric_limits<float>::epsilon());
            const int64_t n = static_cast<int64_t>(normalized_shape.size());
            const bool has_weight = weight && weight->defined();
            const bool needs_grad =
                (at::GradMode::is_enabled() &&
                 (input.requires_grad() || (has_weight && weight->requires_grad())));
            const auto dtype = input.scalar_type();
            if (needs_grad && input.is_privateuseone() && n == 1 && input.dim() >= 1 &&
                normalized_shape[0] == input.sym_size(-1) && input.numel() > 0 &&
                own_norm(input, 1, {&weight})) {
                return RmsNormFunction::apply(
                    input, has_weight ? weight : std::optional<at::Tensor>(), e
                );
            }
            const bool fused =
                (!needs_grad && input.is_privateuseone() && !mesh::placed_base(input) &&
                 input.numel() > 0 && n == 1 && input.dim() >= 1 &&
                 normalized_shape[0] == input.sym_size(-1) &&
                 (own_norm(input, 1, {&weight}) ||
                  (input.size(-1) % 32 == 0 && dtype == at::kBFloat16 &&
                   (!has_weight ||
                    (weight->dim() == 1 && weight->scalar_type() == dtype)))));
            if (fused) {
                auto g = has_weight ? dense("rms_norm", *weight) : bridge::HandlePtr{};
                return (wrap_output(
                    bridge::rms_norm(dense("rms_norm", input), e, g),
                    input.sizes(),
                    dtype
                ));
            }

            // The composite: x * rsqrt(mean(x^2) + eps) in the op-math dtype, then *
            // weight.
            const auto opmath =
                (dtype == at::kBFloat16 || dtype == at::kHalf ? at::kFloat : dtype);
            std::vector<int64_t> dims;
            for (int64_t k = 0; k < n; ++k) {
                dims.push_back(input.dim() - 1 - k);
            }
            auto x = input.to(opmath);
            auto result = (x * at::rsqrt(x.pow(2).mean(dims, true) + e)).to(dtype);
            return has_weight ? result * *weight : result;
        }

        // True unless KTT_OWN_GROUP_NORM=0 (the composite of TT-NN reductions and
        // binaries instead of Kohaku-TT's moments / affine kernels).
        bool own_group_norm() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_OWN_GROUP_NORM");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        // (y, mean [N, G], rstd [N, G]): group norm of input [N, C, *] in the op-math
        // dtype from per-channel sums over the spatial dims (no view of the input
        // regrouped), then one per-channel scale and shift [N, C, 1, ...]; y in
        // input's dtype.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> group_norm(
            const at::Tensor& input,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            int64_t n,
            int64_t c,
            int64_t hw,
            int64_t groups,
            double eps
        ) {
            const auto dtype = input.scalar_type();
            const auto opmath =
                (dtype == at::kBFloat16 || dtype == at::kHalf ? at::kFloat : dtype);
            std::vector<int64_t> spatial;
            for (int64_t d = 2; d < input.dim(); ++d) {
                spatial.push_back(d);
            }
            const double count = static_cast<double>((c / groups) * hw);
            // [N, C, H, W] bf16 / fp32: Kohaku-TT's kernels (moments, group statistics,
            // affine).
            const auto vector_ok = [](const std::optional<at::Tensor>& t) {
                return (
                    !t || !t->defined() || t->scalar_type() == at::kFloat ||
                    t->scalar_type() == at::kBFloat16
                );
            };
            const char* op = "native_group_norm";
            const auto opt = [&](const std::optional<at::Tensor>& t) {
                return t && t->defined() ? dense(op, *t) : bridge::HandlePtr{};
            };
            const bool own =
                (own_group_norm() && input.numel() > 0 && !mesh::placed_base(input) &&
                 (dtype == at::kBFloat16 || dtype == at::kFloat) && vector_ok(weight) &&
                 vector_ok(bias));
            // Channels-last storage, and without gradients [N, C, L] through its [N, L,
            // C] transpose, on the row-form programs. mean / rstd (read by the backward
            // only): from the per-channel sums when a gradient is needed, else zeros.
            const auto wants_grad = [](const std::optional<at::Tensor>& t) {
                return t && t->defined() && t->requires_grad();
            };
            const bool grad =
                (at::GradMode::is_enabled() &&
                 (input.requires_grad() || wants_grad(weight) || wants_grad(bias)));
            const bool rows =
                (input.dim() == 3 && hw % 32 == 0 && c % 32 == 0 && c % groups == 0);
            if (own && (input.dim() == 4 || (rows && !grad))) {
                const auto x_h = rows ? dense(op, input.transpose(1, 2))
                                      : dense_propagate(op, input);
                if (grad && bridge::group_norm_rows_supported(x_h, groups)) {
                    auto st = bridge::group_norm_rows_sums(
                        x_h, opt(weight), opt(bias), groups, eps
                    );
                    // mean / rstd [N, G] (fp32): every (C / G)-th column of the
                    // per-channel tensors, as views.
                    const auto per_group = [&](const bridge::HandlePtr& h) {
                        return wrap_output(h, {n, c}, at::kFloat)
                            .slice(1, 0, c, c / groups);
                    };
                    return {
                        wrap_output(st[0], input.sizes().vec(), dtype),
                        per_group(st[1]),
                        per_group(st[2])
                    };
                }
                if (bridge::group_norm_rows_supported(x_h, groups)) {
                    auto h = bridge::group_norm_rows(
                        x_h, opt(weight), opt(bias), groups, eps
                    );
                    auto y = rows ? wrap_output(h, {n, hw, c}, dtype).transpose(1, 2)
                                  : wrap_output(h, input.sizes().vec(), dtype);
                    return {
                        y,
                        wrap_output(
                            bridge::shared_full({n, groups}, 0.0, to_bridge(dtype)),
                            {n, groups},
                            dtype
                        ),
                        wrap_output(
                            bridge::shared_full({n, groups}, 0.0, to_bridge(dtype)),
                            {n, groups},
                            dtype
                        )
                    };
                }
            }
            if (own && input.dim() == 4) {
                const auto x_h = dense(op, input);
                auto [h1, h2] = bridge::group_norm_moments(x_h);
                auto st = (bridge::group_norm_stats(
                    h1, h2, opt(weight), opt(bias), groups, count, eps, to_bridge(dtype)
                ));
                auto h = bridge::group_norm_affine(x_h, st[0], st[1]);
                return {
                    wrap_output(h, input.sizes().vec(), dtype),
                    wrap_output(st[2], {n, groups}, dtype),
                    wrap_output(st[3], {n, groups}, dtype)
                };
            }
            auto x = input.to(opmath);
            auto s1 = spatial.empty() ? x : x.sum(spatial);
            auto s2 = spatial.empty() ? x * x : (x * x).sum(spatial);
            auto mean = s1.reshape({n, groups, c / groups}).sum(-1) / count;
            auto var = s2.reshape({n, groups, c / groups}).sum(-1) / count -
                mean * mean;
            auto rstd = at::rsqrt(var.clamp_min(0) + eps);
            // Per-channel scale a = rstd * w and shift b = bias - mean * a.
            auto a = rstd.unsqueeze(-1).expand({n, groups, c / groups}).reshape({n, c});
            auto m = mean.unsqueeze(-1).expand({n, groups, c / groups}).reshape({n, c});
            if (weight && weight->defined()) {
                a = a * weight->to(opmath);
            }
            auto b = -m * a;
            if (bias && bias->defined()) {
                b = b + bias->to(opmath);
            }
            std::vector<int64_t> per_channel{n, c};
            for (size_t k = 0; k < spatial.size(); ++k) {
                per_channel.push_back(1);
            }
            auto y = x * a.reshape(per_channel) + b.reshape(per_channel);
            return {y.to(dtype), mean.to(dtype), rstd.to(dtype)};
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
        m.impl("rms_norm", &rms_norm);
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("native_group_norm", &group_norm);
        m.impl("native_group_norm_backward", &group_norm_backward);
        m.impl("native_batch_norm_backward", &batch_norm_backward);
        m.impl("native_batch_norm", &batch_norm);
        m.impl("_native_batch_norm_legit", &batch_norm_legit);
        m.impl("_batch_norm_with_update", &batch_norm_with_update);
        m.impl("_native_batch_norm_legit.no_stats", &batch_norm_no_stats);
        m.impl("native_layer_norm", &layer_norm);
        m.impl("native_layer_norm_backward", &layer_norm_backward);
    }

}  // namespace ktt
