// Bridge normalization and loss ops (TT-NN moreh kernels, bf16): layer norm with its
// statistics and backward, nll loss and its backward, GELU backward.
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"

namespace ktt::bridge {
    namespace {

        // The device shape of x's statistics over its trailing `normalized` dims.
        std::vector<int64_t> stats_shape(const HandlePtr& x, int64_t normalized) {
            std::vector<int64_t> lead(x->shape.begin(), x->shape.end() - normalized);
            return detail::device_dims(lead);
        }

        ttnn::Tensor empty_like_shape(const std::vector<int64_t>& shape, DType dtype) {
            return ttnn::empty(
                detail::device_shape(shape),
                detail::to_ttnn(dtype),
                tt::tt_metal::Layout::TILE,
                detail::device(),
                tt::tt_metal::MemoryConfig{}
            );
        }

    }  // namespace

    HandlePtr rms_norm(
        const HandlePtr& x, double eps, const HandlePtr& gamma, const HandlePtr& beta
    ) {
        // Memory-bound: fidelity floor HiFi4.
        const auto config = numerics::compute_config(
            numerics::effective({.floor = tt::tt_metal::MathFidelity::HiFi4})
        );
        const int64_t bytes = x->bytes;
        return detail::timed_op(x->dtype, x->shape, [=] {
            std::optional<const ttnn::Tensor> g, b;
            if (gamma) {
                g.emplace(gamma->tensor);
            }
            if (beta) {
                b.emplace(beta->tensor);
            }
            return (
                ttnn::rms_norm(
                    x->tensor,
                    static_cast<float>(eps),
                    g,
                    b,
                    std::nullopt,
                    detail::activation_config(bytes),
                    std::nullopt,
                    config
                )
            );
        });
    }

    LayerNorm layer_norm(
        const HandlePtr& x,
        int64_t normalized,
        double eps,
        const HandlePtr& gamma,
        const HandlePtr& beta
    ) {
        LayerNorm out{
            detail::make_handle(x->shape, x->dtype),
            detail::make_handle(stats_shape(x, normalized), x->dtype),
            detail::make_handle(stats_shape(x, normalized), x->dtype)
        };
        detail::counters().ops++;
        detail::submit([out, x, normalized, eps, gamma, beta] {
            out.mean->tensor = empty_like_shape(out.mean->shape, x->dtype);
            out.rstd->tensor = empty_like_shape(out.rstd->shape, x->dtype);
            std::optional<const ttnn::Tensor> g, b;
            if (gamma) {
                g.emplace(gamma->tensor);
            }
            if (beta) {
                b.emplace(beta->tensor);
            }
            auto results = (
                ttnn::moreh_layer_norm(
                    x->tensor,
                    static_cast<uint32_t>(normalized),
                    static_cast<float>(eps),
                    g,
                    b,
                    std::nullopt,
                    out.mean->tensor,
                    out.rstd->tensor
                )
            );
            out.y->tensor = *results[0];
            detail::check_shape(*out.y);
        });
        return out;
    }

    LayerNormGrads layer_norm_backward(
        const HandlePtr& grad,
        const HandlePtr& x,
        const HandlePtr& mean,
        const HandlePtr& rstd,
        int64_t normalized,
        const HandlePtr& gamma,
        bool want_input,
        bool want_gamma,
        bool want_beta
    ) {
        std::vector<int64_t> param(x->shape.end() - normalized, x->shape.end());
        LayerNormGrads out;
        if (want_input) {
            out.input = detail::make_handle(x->shape, x->dtype);
        }
        if (want_gamma) {
            out.gamma = detail::make_handle(detail::device_dims(param), x->dtype);
        }
        if (want_beta) {
            out.beta = detail::make_handle(detail::device_dims(param), x->dtype);
        }
        detail::counters().ops++;
        detail::submit([out, grad, x, mean, rstd, normalized, gamma] {
            std::optional<const ttnn::Tensor> gi, gg, gb, g;
            if (out.input) {
                out.input->tensor = empty_like_shape(out.input->shape, x->dtype);
                gi.emplace(out.input->tensor);
            }
            if (out.gamma) {
                out.gamma->tensor = empty_like_shape(out.gamma->shape, x->dtype);
                gg.emplace(out.gamma->tensor);
            }
            if (out.beta) {
                out.beta->tensor = empty_like_shape(out.beta->shape, x->dtype);
                gb.emplace(out.beta->tensor);
            }
            if (gamma) {
                g.emplace(gamma->tensor);
            }
            // Statistics stored keepdim-shaped (trace outputs) go back to moreh's
            // shape.
            auto stats = detail::device_shape(stats_shape(x, normalized));
            auto as_stats = [&](const HandlePtr& h) {
                return (
                    h->shape == stats_shape(x, normalized)
                        ? h->tensor
                        : ttnn::reshape(h->tensor, stats)
                );
            };
            ttnn::moreh_layer_norm_backward(
                grad->tensor,
                x->tensor,
                as_stats(mean),
                as_stats(rstd),
                static_cast<uint32_t>(normalized),
                g,
                gi,
                gg,
                gb
            );
        });
        return out;
    }

    NllLoss nll_loss(
        const HandlePtr& logp, const HandlePtr& target, bool mean, int64_t ignore_index
    ) {
        NllLoss out{
            detail::make_handle(detail::device_dims({}), logp->dtype),
            mean ? detail::make_handle(detail::device_dims({}), logp->dtype) : nullptr
        };
        detail::counters().ops++;
        detail::submit([out, logp, target, mean, ignore_index] {
            std::optional<ttnn::Tensor> divisor;
            if (mean) {
                out.total->tensor = empty_like_shape({}, logp->dtype);
                divisor = out.total->tensor;
            }
            out.loss->tensor = (
                ttnn::moreh_nll_loss(
                    logp->tensor,
                    target->tensor,
                    mean ? "mean" : "sum",
                    std::nullopt,
                    divisor,
                    std::nullopt,
                    static_cast<int32_t>(ignore_index)
                )
            );
            detail::check_shape(*out.loss);
        });
        return out;
    }

    HandlePtr nll_loss_backward(
        const HandlePtr& logp,
        const HandlePtr& target,
        const HandlePtr& grad,
        const HandlePtr& divisor,
        int64_t ignore_index
    ) {
        return detail::timed_op(logp->dtype, logp->shape, [=] {
            std::optional<ttnn::Tensor> d;
            if (divisor) {
                d.emplace(divisor->tensor);
            }
            // TT-NN needs the gradient buffer: the class count is not in its inputs.
            auto input_grad = empty_like_shape(logp->shape, logp->dtype);
            return (
                ttnn::moreh_nll_loss_backward(
                    target->tensor,
                    grad->tensor,
                    divisor != nullptr,
                    std::nullopt,
                    input_grad,
                    d,
                    static_cast<int32_t>(ignore_index)
                )
            );
        });
    }

    HandlePtr gelu_backward(const HandlePtr& grad, const HandlePtr& x) {
        return detail::timed_op(x->dtype, x->shape, [=] {
            return *ttnn::gelu_bw(grad->tensor, x->tensor)[0];
        });
    }

}  // namespace ktt::bridge
