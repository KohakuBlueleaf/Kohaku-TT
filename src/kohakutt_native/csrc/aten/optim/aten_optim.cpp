// aten fused optimizer kernels for tt (torch.optim.Adam / AdamW with fused=True):
// _fused_adam_ and _fused_adamw_, one program per parameter (aten/optim/ktt_adam).
#include <cmath>

#include <torch/library.h>

#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"

namespace ktt {
    namespace {

        // The step count a 0-d step tensor holds: host-known after the first read.
        double step_of(const at::Tensor& step) {
            if (auto known = known_scalar(step)) {
                return *known;
            }
            const double value = step.item<double>();
            set_known_scalar(step, value);
            return value;
        }

        // The whole-tensor operand of a step: the mesh-placed or stored base when `t`
        // covers its storage, else dense().
        bridge::HandlePtr operand(const char* op, const at::Tensor& t) {
            auto h = mesh::placed_base(t);
            return h ? h : dense_any_layout(op, t);
        }

        // PyTorch's single-tensor Adam / AdamW step on tt ops.
        void composite(
            at::Tensor p,
            at::Tensor g,
            at::Tensor m,
            at::Tensor v,
            const at::Tensor& vmax,
            double t,
            double lr,
            double beta1,
            double beta2,
            double weight_decay,
            double eps,
            bool amsgrad,
            bool maximize,
            bool decoupled
        ) {
            if (maximize) {
                g = g.neg();
            }
            if (weight_decay != 0) {
                if (decoupled) {
                    p.mul_(1 - lr * weight_decay);
                } else {
                    g = g.add(p, weight_decay);
                }
            }
            m.lerp_(g, 1 - beta1);
            v.mul_(beta2).addcmul_(g, g, 1 - beta2);
            const double c1 = 1 - std::pow(beta1, t);
            const double c2 = std::sqrt(1 - std::pow(beta2, t));
            at::Tensor denom;
            if (amsgrad) {
                at::Tensor peak = vmax;
                peak.copy_(at::maximum(peak, v));
                denom = peak.sqrt().div_(c2).add_(eps);
            } else {
                denom = v.sqrt().div_(c2).add_(eps);
            }
            p.addcdiv_(m, denom, -lr / c1);
        }

        void fused(
            const char* op,
            bool decoupled,
            at::TensorList self,
            at::TensorList grads,
            at::TensorList exp_avgs,
            at::TensorList exp_avg_sqs,
            at::TensorList max_exp_avg_sqs,
            at::TensorList state_steps,
            double lr,
            double beta1,
            double beta2,
            double weight_decay,
            double eps,
            bool amsgrad,
            bool maximize,
            const std::optional<at::Tensor>& grad_scale,
            const std::optional<at::Tensor>& found_inf
        ) {
            const bool scaled = grad_scale && grad_scale->defined();
            KTT_REQUIRE(
                !(found_inf && found_inf->defined()),
                op,
                "found_inf (a GradScaler) is not supported"
            );
            for (size_t i = 0; i < self.size(); ++i) {
                const at::Tensor& p = self[i];
                at::Tensor g = scaled ? grads[i] / *grad_scale : grads[i];
                const double t = step_of(state_steps[i]);
                const double c1 = 1 - std::pow(beta1, t);
                const double c2 = std::sqrt(1 - std::pow(beta2, t));
                if (!amsgrad && g.sizes() == p.sizes() &&
                    g.scalar_type() == p.scalar_type()) {
                    auto ph = operand(op, p);
                    const auto format = bridge::format(ph);
                    auto in_format = [&](const at::Tensor& x) {
                        auto h = operand(op, x);
                        return bridge::format(h) == format
                            ? h
                            : bridge::to_format(h, format);
                    };
                    bridge::AdamStep s;
                    s.beta1 = beta1;
                    s.beta2 = beta2;
                    s.scale = 1 / c2;
                    s.eps = eps;
                    s.step = lr / c1;
                    s.decay = decoupled ? 1 - lr * weight_decay : 1.0;
                    s.l2 = decoupled ? 0.0 : weight_decay;
                    s.negate = maximize;
                    const auto out = bridge::adam_step(
                        ph,
                        in_format(g),
                        in_format(exp_avgs[i]),
                        in_format(exp_avg_sqs[i]),
                        s
                    );
                    if (!out.empty()) {
                        commit(op, p, out[0]);
                        commit(op, exp_avgs[i], out[1]);
                        commit(op, exp_avg_sqs[i], out[2]);
                        continue;
                    }
                }
                composite(
                    p,
                    g,
                    exp_avgs[i],
                    exp_avg_sqs[i],
                    amsgrad ? max_exp_avg_sqs[i] : at::Tensor(),
                    t,
                    lr,
                    beta1,
                    beta2,
                    weight_decay,
                    eps,
                    amsgrad,
                    maximize,
                    decoupled
                );
            }
        }

// _fused_<name>_ and its tensor_lr overload.
#define KTT_FUSED(name, decoupled)                   \
    void tt_fused_##name(                            \
        at::TensorList self,                         \
        at::TensorList grads,                        \
        at::TensorList exp_avgs,                     \
        at::TensorList exp_avg_sqs,                  \
        at::TensorList max_exp_avg_sqs,              \
        at::TensorList state_steps,                  \
        double lr,                                   \
        double beta1,                                \
        double beta2,                                \
        double weight_decay,                         \
        double eps,                                  \
        bool amsgrad,                                \
        bool maximize,                               \
        const std::optional<at::Tensor>& grad_scale, \
        const std::optional<at::Tensor>& found_inf   \
    ) {                                              \
        fused(                                       \
            "_fused_" #name "_",                     \
            decoupled,                               \
            self,                                    \
            grads,                                   \
            exp_avgs,                                \
            exp_avg_sqs,                             \
            max_exp_avg_sqs,                         \
            state_steps,                             \
            lr,                                      \
            beta1,                                   \
            beta2,                                   \
            weight_decay,                            \
            eps,                                     \
            amsgrad,                                 \
            maximize,                                \
            grad_scale,                              \
            found_inf                                \
        );                                           \
    }                                                \
    void tt_fused_##name##_tensor_lr(                \
        at::TensorList self,                         \
        at::TensorList grads,                        \
        at::TensorList exp_avgs,                     \
        at::TensorList exp_avg_sqs,                  \
        at::TensorList max_exp_avg_sqs,              \
        at::TensorList state_steps,                  \
        const at::Tensor& lr,                        \
        double beta1,                                \
        double beta2,                                \
        double weight_decay,                         \
        double eps,                                  \
        bool amsgrad,                                \
        bool maximize,                               \
        const std::optional<at::Tensor>& grad_scale, \
        const std::optional<at::Tensor>& found_inf   \
    ) {                                              \
        fused(                                       \
            "_fused_" #name "_",                     \
            decoupled,                               \
            self,                                    \
            grads,                                   \
            exp_avgs,                                \
            exp_avg_sqs,                             \
            max_exp_avg_sqs,                         \
            state_steps,                             \
            lr.item<double>(),                       \
            beta1,                                   \
            beta2,                                   \
            weight_decay,                            \
            eps,                                     \
            amsgrad,                                 \
            maximize,                                \
            grad_scale,                              \
            found_inf                                \
        );                                           \
    }

        KTT_FUSED(adam, false)
        KTT_FUSED(adamw, true)

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("_fused_adam_", &tt_fused_adam);
        m.impl("_fused_adam_.tensor_lr", &tt_fused_adam_tensor_lr);
        m.impl("_fused_adamw_", &tt_fused_adamw);
        m.impl("_fused_adamw_.tensor_lr", &tt_fused_adamw_tensor_lr);
    }

}  // namespace ktt
