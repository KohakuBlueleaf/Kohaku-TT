// aten ternary kernels for tt: addcmul, addcdiv, lerp (optimizer steps), where,
// masked_fill.
#include <c10/util/accumulate.h>
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // The broadcast shape of three tensors.
        std::vector<int64_t> shape3(
            const at::Tensor& a, const at::Tensor& b, const at::Tensor& c
        ) {
            auto ab = at::infer_size(a.sizes(), b.sizes());
            auto abc = at::infer_size(ab, c.sizes());
            return std::vector<int64_t>(abc.begin(), abc.end());
        }

        // self + value * t1 * t2 (or t1 / t2 for addcdiv).
        at::Tensor addc(
            const char* op,
            bool divide,
            const at::Tensor& self,
            const at::Tensor& t1,
            const at::Tensor& t2,
            const at::Scalar& value
        ) {
            auto shape = shape3(self, t1, t2);
            auto dtype = self.scalar_type();
            const double v = value.toDouble();
            auto placed = placed_pointwise(
                op, {self, t1, t2}, shape, dtype, [&](const auto& in) {
                    return (
                        divide ? bridge::addcdiv(in[0], in[1], in[2], v)
                               : bridge::addcmul(in[0], in[1], in[2], v)
                    );
                }
            );
            if (placed) {
                return *placed;
            }
            auto a = dense_propagate_as(op, self, shape, dtype);
            auto b = dense_propagate_as(op, t1, shape, dtype);
            auto c = dense_propagate_as(op, t2, shape, dtype);
            auto h =
                (divide ? bridge::addcdiv(a, b, c, value.toDouble())
                        : bridge::addcmul(a, b, c, value.toDouble()));
            return wrap_output(h, shape, dtype);
        }

        at::Tensor tt_addcmul(
            const at::Tensor& s,
            const at::Tensor& t1,
            const at::Tensor& t2,
            const at::Scalar& v
        ) {
            return addc("addcmul", false, s, t1, t2, v);
        }

        at::Tensor& tt_addcmul_(
            at::Tensor& s,
            const at::Tensor& t1,
            const at::Tensor& t2,
            const at::Scalar& v
        ) {
            return write_into("addcmul_", s, tt_addcmul(s, t1, t2, v));
        }

        at::Tensor& tt_addcmul_out(
            const at::Tensor& s,
            const at::Tensor& t1,
            const at::Tensor& t2,
            const at::Scalar& v,
            at::Tensor& out
        ) {
            return write_out("addcmul.out", out, tt_addcmul(s, t1, t2, v));
        }

        at::Tensor tt_addcdiv(
            const at::Tensor& s,
            const at::Tensor& t1,
            const at::Tensor& t2,
            const at::Scalar& v
        ) {
            return addc("addcdiv", true, s, t1, t2, v);
        }

        at::Tensor& tt_addcdiv_(
            at::Tensor& s,
            const at::Tensor& t1,
            const at::Tensor& t2,
            const at::Scalar& v
        ) {
            return write_into("addcdiv_", s, tt_addcdiv(s, t1, t2, v));
        }

        at::Tensor& tt_addcdiv_out(
            const at::Tensor& s,
            const at::Tensor& t1,
            const at::Tensor& t2,
            const at::Scalar& v,
            at::Tensor& out
        ) {
            return write_out("addcdiv.out", out, tt_addcdiv(s, t1, t2, v));
        }

        // self + weight * (end - self).
        at::Tensor tt_lerp(
            const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight
        ) {
            auto shape = broadcast_shape(self, end);
            auto dtype = self.scalar_type();
            const double w = weight.toDouble();
            auto placed = placed_pointwise(
                "lerp.Scalar", {self, end}, shape, dtype, [&](const auto& in) {
                    return bridge::lerp(in[0], in[1], w);
                }
            );
            if (placed) {
                return *placed;
            }
            auto h = (bridge::lerp(
                dense_propagate_as("lerp.Scalar", self, shape, dtype),
                dense_propagate_as("lerp.Scalar", end, shape, dtype),
                weight.toDouble()
            ));
            return wrap_output(h, shape, dtype);
        }

        at::Tensor& tt_lerp_(
            at::Tensor& self, const at::Tensor& end, const at::Scalar& weight
        ) {
            return write_into("lerp_.Scalar", self, tt_lerp(self, end, weight));
        }

        at::Tensor& tt_lerp_out(
            const at::Tensor& self,
            const at::Tensor& end,
            const at::Scalar& weight,
            at::Tensor& out
        ) {
            return write_out("lerp.Scalar_out", out, tt_lerp(self, end, weight));
        }

        // condition ? self : other, broadcast; operands promoted to a common dtype.
        at::Tensor tt_where(
            const at::Tensor& cond, const at::Tensor& self, const at::Tensor& other
        ) {
            const char* op = "where.self";
            auto shape = shape3(cond, self, other);
            auto dtype = at::result_type(self, other);
            if (c10::multiply_integers(shape) == 0) {
                return at::empty(shape, self.options().dtype(dtype));
            }
            // Operands of the output's shape or of one element (read unexpanded:
            // TT-NN's where broadcasts them) in one program, else every operand
            // expanded.
            const std::vector<const at::Tensor*> operands = {&cond, &self, &other};
            std::vector<bridge::HandlePtr> in(3);
            bool direct = true;
            for (size_t i = 0; i < 3; ++i) {
                const auto& t = *operands[i];
                if (t.sizes() == c10::IntArrayRef(shape)) {
                    in[i] = dense_propagate_as(op, t, shape, dtype);
                    direct = direct && bridge::format(in[i]) == bridge::Format::Nchw;
                } else {
                    direct = direct && t.numel() == 1;
                }
            }
            const std::vector<int64_t> ones(shape.size(), 1);
            for (size_t i = 0; i < 3; ++i) {
                if (!in[i]) {
                    const auto& t = *operands[i];
                    in[i] = direct ? dense_as(op, t.reshape(ones), dtype)
                                   : dense_propagate_as(op, t, shape, dtype);
                }
            }
            auto h = bridge::where(in[0], in[1], in[2]);
            return wrap_output(h, shape, dtype);
        }

        // self = mask ? value : self.
        at::Tensor& tt_masked_fill_(
            at::Tensor& self, const at::Tensor& mask, const at::Scalar& value
        ) {
            auto fill = bridge::full_like(
                dense_propagate("masked_fill_.Scalar", self), value.toDouble()
            );
            auto filled = wrap_output(fill, self.sizes(), self.scalar_type());
            return (
                write_into("masked_fill_.Scalar", self, tt_where(mask, filled, self))
            );
        }

        at::Tensor& tt_masked_fill_tensor_(
            at::Tensor& self, const at::Tensor& mask, const at::Tensor& value
        ) {
            KTT_REQUIRE(value.dim() == 0, "masked_fill_.Tensor", "value must be 0-d");
            auto v = value.is_cpu() ? value.item() : value.cpu().item();
            return tt_masked_fill_(self, mask, v);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("addcmul", &tt_addcmul);
        m.impl("addcmul_", &tt_addcmul_);
        m.impl("addcmul.out", &tt_addcmul_out);
        m.impl("addcdiv", &tt_addcdiv);
        m.impl("addcdiv_", &tt_addcdiv_);
        m.impl("addcdiv.out", &tt_addcdiv_out);
        m.impl("lerp.Scalar", &tt_lerp);
        m.impl("lerp_.Scalar", &tt_lerp_);
        m.impl("lerp.Scalar_out", &tt_lerp_out);
        m.impl("where.self", &tt_where);
        m.impl("masked_fill_.Scalar", &tt_masked_fill_);
        m.impl("masked_fill_.Tensor", &tt_masked_fill_tensor_);
    }

}  // namespace ktt
