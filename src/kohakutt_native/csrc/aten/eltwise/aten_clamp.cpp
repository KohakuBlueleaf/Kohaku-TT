// aten clamp kernels for tt: clamp / clamp_min / clamp_max with scalar or tensor
// bounds, in-place and out= forms.

#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        at::Tensor tt_clamp(
            const at::Tensor& self,
            const std::optional<at::Scalar>& min,
            const std::optional<at::Scalar>& max
        ) {
            auto lo = min ? std::optional<double>(min->toDouble()) : std::nullopt;
            auto hi = max ? std::optional<double>(max->toDouble()) : std::nullopt;
            return (
                wrap_output(
                    bridge::clamp(dense("clamp", self), lo, hi),
                    self.sizes(),
                    self.scalar_type()
                )
            );
        }

        at::Tensor& tt_clamp_(
            at::Tensor& self,
            const std::optional<at::Scalar>& min,
            const std::optional<at::Scalar>& max
        ) {
            return write_into("clamp_", self, tt_clamp(self, min, max));
        }

        // clamp_min / clamp_max with a scalar bound (clamp) or a tensor bound (maximum
        // / minimum), plus their in-place and out= forms.
        at::Tensor tt_clamp_min(const at::Tensor& self, const at::Scalar& lo) {
            return tt_clamp(self, lo, std::nullopt);
        }

        at::Tensor tt_clamp_max(const at::Tensor& self, const at::Scalar& hi) {
            return tt_clamp(self, std::nullopt, hi);
        }

        at::Tensor tt_clamp_min_tensor(const at::Tensor& self, const at::Tensor& lo) {
            return at::maximum(self, lo);
        }

        at::Tensor tt_clamp_max_tensor(const at::Tensor& self, const at::Tensor& hi) {
            return at::minimum(self, hi);
        }

        at::Tensor& tt_clamp_min_(at::Tensor& self, const at::Scalar& lo) {
            return write_into("clamp_min_", self, tt_clamp_min(self, lo));
        }

        at::Tensor& tt_clamp_max_(at::Tensor& self, const at::Scalar& hi) {
            return write_into("clamp_max_", self, tt_clamp_max(self, hi));
        }

        at::Tensor& tt_clamp_min_out(
            const at::Tensor& self, const at::Scalar& lo, at::Tensor& out
        ) {
            return write_out("clamp_min.out", out, tt_clamp_min(self, lo));
        }

        at::Tensor& tt_clamp_max_out(
            const at::Tensor& self, const at::Scalar& hi, at::Tensor& out
        ) {
            return write_out("clamp_max.out", out, tt_clamp_max(self, hi));
        }

        at::Tensor& tt_clamp_min_tensor_out(
            const at::Tensor& self, const at::Tensor& lo, at::Tensor& out
        ) {
            return into("clamp_min.Tensor_out", out, tt_clamp_min_tensor(self, lo));
        }

        at::Tensor& tt_clamp_max_tensor_out(
            const at::Tensor& self, const at::Tensor& hi, at::Tensor& out
        ) {
            return into("clamp_max.Tensor_out", out, tt_clamp_max_tensor(self, hi));
        }

        at::Tensor tt_clamp_tensor(
            const at::Tensor& self,
            const std::optional<at::Tensor>& min,
            const std::optional<at::Tensor>& max
        ) {
            auto out = self;
            if (min && min->defined()) {
                out = at::maximum(out, *min);
            }
            if (max && max->defined()) {
                out = at::minimum(out, *max);
            }
            return out;
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("clamp", &tt_clamp);
        m.impl("clamp_", &tt_clamp_);
        m.impl("clamp_min", &tt_clamp_min);
        m.impl("clamp_max", &tt_clamp_max);
        m.impl("clamp_min.Tensor", &tt_clamp_min_tensor);
        m.impl("clamp_max.Tensor", &tt_clamp_max_tensor);
        m.impl("clamp_min_", &tt_clamp_min_);
        m.impl("clamp_max_", &tt_clamp_max_);
        m.impl("clamp_min.out", &tt_clamp_min_out);
        m.impl("clamp_max.out", &tt_clamp_max_out);
        m.impl("clamp_min.Tensor_out", &tt_clamp_min_tensor_out);
        m.impl("clamp_max.Tensor_out", &tt_clamp_max_tensor_out);
        m.impl("clamp.Tensor", &tt_clamp_tensor);
    }

}  // namespace ktt
