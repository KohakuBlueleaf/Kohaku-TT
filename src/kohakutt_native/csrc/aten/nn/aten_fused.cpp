// Fused transformer ops (torch.ops.ktt): affine RMSNorm and rotate-half RoPE. tt
// tensors run one TT-NN program each; other devices (and meta) run the composite.
#include <ATen/ATen.h>
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // rms_norm(x) * weight + bias over the last dim in fp32, in x's dtype.
        at::Tensor rms_norm_composite(
            const at::Tensor& x,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            double eps
        ) {
            std::optional<at::Tensor> w;
            if (weight && weight->defined()) {
                w = weight->to(at::kFloat);
            }
            auto y = at::rms_norm(x.to(at::kFloat), {x.size(-1)}, w, eps);
            if (bias && bias->defined()) {
                y = y + bias->to(at::kFloat);
            }
            return y.to(x.scalar_type());
        }

        // x * cos + cat(-x2, x1) * sin over the last dim in fp32 (cos / sin rows 0 ..
        // S - 1), in x's dtype.
        at::Tensor rotary_composite(
            const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin
        ) {
            const auto s = x.size(-2);
            auto xf = x.to(at::kFloat);
            auto halves = xf.chunk(2, -1);
            auto rotated = at::cat({-halves[1], halves[0]}, -1);
            auto c = cos.narrow(-2, 0, s).to(at::kFloat);
            auto n = sin.narrow(-2, 0, s).to(at::kFloat);
            return (xf * c + rotated * n).to(x.scalar_type());
        }

        bool is_row(const std::optional<at::Tensor>& t, const at::Tensor& x) {
            return (
                !t || !t->defined() ||
                (t->is_privateuseone() && t->dim() == 1 && t->size(0) == x.size(-1) &&
                 t->scalar_type() == x.scalar_type())
            );
        }

        at::Tensor rms_norm_tt(
            const at::Tensor& x,
            const std::optional<at::Tensor>& weight,
            const std::optional<at::Tensor>& bias,
            double eps
        ) {
            const bool fused =
                (x.dim() >= 1 && x.size(-1) % 32 == 0 &&
                 (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat) &&
                 is_row(weight, x) && is_row(bias, x));
            if (!fused || x.numel() == 0) {
                return rms_norm_composite(x, weight, bias, eps);
            }
            auto row = [](const std::optional<at::Tensor>& t) {
                return t && t->defined() ? dense("ktt.rms_norm", *t)
                                         : bridge::HandlePtr{};
            };
            return (
                wrap_output(
                    bridge::rms_norm(dense("ktt.rms_norm", x), eps, row(weight), row(bias)),
                    x.sizes(),
                    x.scalar_type()
                )
            );
        }

        at::Tensor rotary_tt(
            const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin
        ) {
            const auto d = x.dim() == 4 ? x.size(-1) : 0;
            const bool fused =
                ((d == 32 || (d > 0 && d % 64 == 0)) &&
                 (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat) &&
                 cos.dim() == 4 && cos.sizes() == sin.sizes() && cos.size(0) == 1 &&
                 cos.size(1) == 1 && cos.size(-1) == d && cos.size(-2) >= x.size(-2) &&
                 cos.scalar_type() == x.scalar_type() &&
                 sin.scalar_type() == x.scalar_type() && x.numel() > 0);
            if (!fused) {
                return rotary_composite(x, cos, sin);
            }
            return (
                wrap_output(
                    bridge::rotary(
                        dense("ktt.rotary", x),
                        dense("ktt.rotary", cos),
                        dense("ktt.rotary", sin)
                    ),
                    x.sizes(),
                    x.scalar_type()
                )
            );
        }

    }  // namespace

    TORCH_LIBRARY_FRAGMENT(ktt, m) {
        m.def("rms_norm(Tensor x, Tensor? weight, Tensor? bias, float eps) -> Tensor");
        m.def("rotary(Tensor x, Tensor cos, Tensor sin) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(ktt, CompositeExplicitAutograd, m) {
        m.impl("rms_norm", &rms_norm_composite);
        m.impl("rotary", &rotary_composite);
    }

    TORCH_LIBRARY_IMPL(ktt, PrivateUse1, m) {
        m.impl("rms_norm", &rms_norm_tt);
        m.impl("rotary", &rotary_tt);
    }

}  // namespace ktt
