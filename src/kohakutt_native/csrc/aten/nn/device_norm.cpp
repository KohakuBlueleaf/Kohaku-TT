// Bridge normalization and loss ops: layer norm / RMS norm over the last dim on
// Kohaku-TT's kernels (ktt_norm), TT-NN's moreh layer norm (bf16) and rms_norm
// otherwise; nll loss and its backward, GELU backward.
#include <chrono>

#include "aten/nn/ktt_norm/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

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

        bool float_dtype(DType d) {
            return d == DType::BFloat16 || d == DType::Float32;
        }

        // True when Kohaku-TT's norm kernels take x over its last `normalized` dims
        // with gamma / beta (null or [1, W] rows).
        bool own_norm(
            const HandlePtr& x,
            int64_t normalized,
            const HandlePtr& gamma,
            const HandlePtr& beta
        ) {
            if (!program::own_kernels_enabled() || normalized != 1 ||
                !float_dtype(x->dtype) || !x->placement.is_replicated() ||
                x->shape.size() < 2) {
                return false;
            }
            for (const auto* p : {&gamma, &beta}) {
                if (*p &&
                    (!float_dtype((*p)->dtype) || (*p)->shape.size() != 2 ||
                     (*p)->shape[0] != 1 || (*p)->shape[1] != x->shape.back() ||
                     !(*p)->placement.is_replicated())) {
                    return false;
                }
            }
            return true;
        }

        // Kohaku-TT's norm forward on the stream: y, and fp32 mean (layer norm) / rstd
        // of x's device shape with the last dim 1 when `stats`.
        LayerNorm own_forward(
            const HandlePtr& x,
            const HandlePtr& gamma,
            const HandlePtr& beta,
            double eps,
            bool rms,
            bool stats
        ) {
            auto stat_shape = x->shape;
            stat_shape.back() = 1;
            LayerNorm out{detail::make_handle(x->shape, x->dtype), nullptr, nullptr};
            if (stats) {
                if (!rms) {
                    out.mean = detail::make_handle(stat_shape, DType::Float32);
                }
                out.rstd = detail::make_handle(stat_shape, DType::Float32);
            }
            detail::counters().ops++;
            const auto n = numerics::effective({.exact = true});
            detail::submit([out, x, gamma, beta, eps, rms, stats, n] {
                const auto start = std::chrono::steady_clock::now();
                detail::phase_clock() = start;
                const auto r = ktt_norm::forward(
                    x->tensor,
                    gamma ? &gamma->tensor : nullptr,
                    beta ? &beta->tensor : nullptr,
                    static_cast<float>(eps),
                    rms,
                    stats,
                    detail::activation_config(out.y->bytes),
                    n.fidelity,
                    n.approx
                );
                out.y->tensor = r.y;
                if (out.mean) {
                    out.mean->tensor = *r.mean;
                }
                if (out.rstd) {
                    out.rstd->tensor = *r.rstd;
                }
                const auto elapsed = std::chrono::steady_clock::now() - start;
                using std::chrono::duration_cast, std::chrono::nanoseconds;
                detail::counters().op_ns += duration_cast<nanoseconds>(elapsed).count();
            });
            return out;
        }

        // Kohaku-TT's norm backward on the stream: dx in x's dtype, dgamma / dbeta [1,
        // W] in gamma's dtype (x's without gamma).
        LayerNormGrads own_backward(
            const HandlePtr& grad,
            const HandlePtr& x,
            const HandlePtr& mean,
            const HandlePtr& rstd,
            const HandlePtr& gamma,
            bool rms,
            bool want_input,
            bool want_gamma,
            bool want_beta
        ) {
            const DType param = gamma ? gamma->dtype : x->dtype;
            const std::vector<int64_t> row{1, x->shape.back()};
            LayerNormGrads out;
            if (want_input) {
                out.input = detail::make_handle(x->shape, x->dtype);
            }
            if (want_gamma) {
                out.gamma = detail::make_handle(row, param);
            }
            if (want_beta) {
                out.beta = detail::make_handle(row, param);
            }
            detail::counters().ops++;
            const auto n = numerics::effective({.exact = true});
            detail::submit([out, grad, x, mean, rstd, gamma, rms, param, n] {
                const auto start = std::chrono::steady_clock::now();
                detail::phase_clock() = start;
                const auto r = ktt_norm::backward(
                    grad->tensor,
                    x->tensor,
                    mean ? &mean->tensor : nullptr,
                    rstd->tensor,
                    gamma ? &gamma->tensor : nullptr,
                    rms,
                    out.input != nullptr,
                    out.gamma != nullptr,
                    out.beta != nullptr,
                    detail::to_ttnn(param),
                    out.input ? detail::activation_config(out.input->bytes)
                              : tt::tt_metal::MemoryConfig{},
                    n.fidelity,
                    n.approx
                );
                if (out.input) {
                    out.input->tensor = *r.dx;
                }
                if (out.gamma) {
                    out.gamma->tensor = *r.dgamma;
                }
                if (out.beta) {
                    out.beta->tensor = *r.dbeta;
                }
                const auto elapsed = std::chrono::steady_clock::now() - start;
                using std::chrono::duration_cast, std::chrono::nanoseconds;
                detail::counters().op_ns += duration_cast<nanoseconds>(elapsed).count();
            });
            return out;
        }

    }  // namespace

    bool own_norm_kernels() {
        return program::own_kernels_enabled();
    }

    LayerNorm rms_norm_train(const HandlePtr& x, double eps, const HandlePtr& gamma) {
        return own_forward(x, gamma, nullptr, eps, true, true);
    }

    LayerNormGrads rms_norm_backward(
        const HandlePtr& grad,
        const HandlePtr& x,
        const HandlePtr& rstd,
        const HandlePtr& gamma,
        bool want_input,
        bool want_gamma
    ) {
        return own_backward(
            grad, x, nullptr, rstd, gamma, true, want_input, want_gamma, false
        );
    }

    HandlePtr rms_norm(
        const HandlePtr& x, double eps, const HandlePtr& gamma, const HandlePtr& beta
    ) {
        if (own_norm(x, 1, gamma, beta)) {
            return own_forward(x, gamma, beta, eps, true, false).y;
        }
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
            return (ttnn::rms_norm(
                x->tensor,
                static_cast<float>(eps),
                g,
                b,
                std::nullopt,
                detail::activation_config(bytes),
                std::nullopt,
                config
            ));
        });
    }

    LayerNorm layer_norm(
        const HandlePtr& x,
        int64_t normalized,
        double eps,
        const HandlePtr& gamma,
        const HandlePtr& beta
    ) {
        if (own_norm(x, normalized, gamma, beta)) {
            return own_forward(x, gamma, beta, eps, false, true);
        }
        LayerNorm out{
            detail::make_handle(x->shape, x->dtype),
            detail::make_handle(stats_shape(x, normalized), x->dtype),
            detail::make_handle(stats_shape(x, normalized), x->dtype)
        };
        // Memory-bound statistics: fidelity floor HiFi4, as the bridge reductions.
        const auto config = numerics::compute_config(
            numerics::effective({.floor = tt::tt_metal::MathFidelity::HiFi4})
        );
        detail::counters().ops++;
        detail::submit([out, x, normalized, eps, gamma, beta, config] {
            // Over one dim, moreh writes only every 32nd row's statistics: they come
            // from a mean and a Welford variance, in moreh's backward stats layout.
            const bool lastdim = normalized == 1;
            std::optional<const ttnn::Tensor> m, r;
            if (!lastdim) {
                out.mean->tensor = empty_like_shape(out.mean->shape, x->dtype);
                out.rstd->tensor = empty_like_shape(out.rstd->shape, x->dtype);
                m.emplace(out.mean->tensor);
                r.emplace(out.rstd->tensor);
            }
            std::optional<const ttnn::Tensor> g, b;
            if (gamma) {
                g.emplace(gamma->tensor);
            }
            if (beta) {
                b.emplace(beta->tensor);
            }
            auto results = (ttnn::moreh_layer_norm(
                x->tensor,
                static_cast<uint32_t>(normalized),
                static_cast<float>(eps),
                g,
                b,
                std::nullopt,
                m,
                r
            ));
            out.y->tensor = *results[0];
            detail::check_shape(*out.y);
            if (lastdim) {
                const auto stats = detail::device_shape(out.mean->shape);
                const bool keepdim = true;
                const bool unbiased = false;
                auto mean = ttnn::mean(x->tensor, -1, keepdim, std::nullopt, config);
                auto var = (ttnn::var(
                    x->tensor, -1, keepdim, std::nullopt, config, 1.0f, unbiased
                ));
                out.mean->tensor = ttnn::reshape(mean, stats);
                out.rstd->tensor = (ttnn::reshape(
                    ttnn::rsqrt(ttnn::add(var, static_cast<float>(eps))), stats
                ));
            }
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
        if (own_norm(x, normalized, gamma, nullptr) && grad->dtype == x->dtype &&
            mean->dtype == DType::Float32 && rstd->dtype == DType::Float32 &&
            grad->shape == x->shape && mean->shape.size() == x->shape.size() &&
            mean->shape.back() == 1 && rstd->shape == mean->shape &&
            grad->placement.is_replicated() && mean->placement.is_replicated() &&
            rstd->placement.is_replicated()) {
            return own_backward(
                grad, x, mean, rstd, gamma, false, want_input, want_gamma, want_beta
            );
        }
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
            // Over one dim, x and grad go to moreh as rows [M, W] and the statistics as
            // [1, M]: its reader misreads statistics rows past the first.
            const bool rows = normalized == 1 && x->shape.size() > 2;
            const int64_t w = x->shape.back();
            int64_t m = 1;
            for (size_t i = 0; i + 1 < x->shape.size(); ++i) {
                m *= x->shape[i];
            }
            auto x_t = x->tensor;
            auto grad_t = grad->tensor;
            if (rows) {
                x_t = ttnn::reshape(x_t, detail::device_shape({m, w}));
                grad_t = ttnn::reshape(grad_t, detail::device_shape({m, w}));
            }
            std::optional<const ttnn::Tensor> gi, gg, gb, g;
            if (out.input) {
                gi.emplace(empty_like_shape(
                    rows ? std::vector<int64_t>{m, w} : out.input->shape, x->dtype
                ));
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
            const auto stats_dims = rows ? detail::device_dims({m})
                                         : stats_shape(x, normalized);
            auto stats = detail::device_shape(stats_dims);
            auto as_stats = [&](const HandlePtr& h) {
                return (
                    h->shape == stats_dims ? h->tensor : ttnn::reshape(h->tensor, stats)
                );
            };
            ttnn::moreh_layer_norm_backward(
                grad_t,
                x_t,
                as_stats(mean),
                as_stats(rstd),
                static_cast<uint32_t>(normalized),
                g,
                gi,
                gg,
                gb
            );
            if (out.input) {
                out.input->tensor =
                    (rows ? ttnn::reshape(*gi, detail::device_shape(out.input->shape))
                          : *gi);
            }
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
            out.loss->tensor = (ttnn::moreh_nll_loss(
                logp->tensor,
                target->tensor,
                mean ? "mean" : "sum",
                std::nullopt,
                divisor,
                std::nullopt,
                static_cast<int32_t>(ignore_index)
            ));
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
            return (ttnn::moreh_nll_loss_backward(
                target->tensor,
                grad->tensor,
                divisor != nullptr,
                std::nullopt,
                input_grad,
                d,
                static_cast<int32_t>(ignore_index)
            ));
        });
    }

    HandlePtr gelu_backward(const HandlePtr& grad, const HandlePtr& x) {
        return detail::timed_op(x->dtype, x->shape, [=] {
            return *ttnn::gelu_bw(grad->tensor, x->tensor)[0];
        });
    }

}  // namespace ktt::bridge
