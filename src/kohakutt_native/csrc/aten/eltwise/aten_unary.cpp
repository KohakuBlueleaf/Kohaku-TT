// aten elementwise unary kernels for tt: one table row per op (functional, in-place
// and out= overloads).
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        using U = bridge::UnaryOp;

// Row: tt_<op> (functional), tt_<op>_ (in-place) and tt_<op>_out (out=).
#define KTT_UNARY(op, id)                                                \
    at::Tensor tt_##op(const at::Tensor& self) {                         \
        return unary_kernel(#op, id, self);                              \
    }                                                                    \
    at::Tensor& tt_##op##_(at::Tensor& self) {                           \
        return write_into(#op "_", self, tt_##op(self));                 \
    }                                                                    \
    at::Tensor& tt_##op##_out(const at::Tensor& self, at::Tensor& out) { \
        return write_out(#op ".out", out, tt_##op(self));                \
    }

        KTT_UNARY(neg, U::Neg)
        KTT_UNARY(abs, U::Abs)
        KTT_UNARY(exp, U::Exp)
        KTT_UNARY(log, U::Log)
        KTT_UNARY(sqrt, U::Sqrt)
        KTT_UNARY(rsqrt, U::Rsqrt)
        KTT_UNARY(reciprocal, U::Reciprocal)
        KTT_UNARY(tanh, U::Tanh)
        KTT_UNARY(sigmoid, U::Sigmoid)
        KTT_UNARY(relu, U::Relu)
        KTT_UNARY(silu, U::Silu)
        KTT_UNARY(sin, U::Sin)
        KTT_UNARY(cos, U::Cos)
        KTT_UNARY(erf, U::Erf)
        KTT_UNARY(acos, U::Acos)
        KTT_UNARY(asin, U::Asin)
        KTT_UNARY(asinh, U::Asinh)
        KTT_UNARY(atan, U::Atan)
        KTT_UNARY(atanh, U::Atanh)
        KTT_UNARY(acosh, U::Acosh)
        KTT_UNARY(cosh, U::Cosh)
        KTT_UNARY(sinh, U::Sinh)
        KTT_UNARY(tan, U::Tan)
        KTT_UNARY(exp2, U::Exp2)
        KTT_UNARY(expm1, U::Expm1)
        KTT_UNARY(log2, U::Log2)
        KTT_UNARY(log10, U::Log10)
        KTT_UNARY(log1p, U::Log1p)
        KTT_UNARY(erfc, U::Erfc)
        KTT_UNARY(erfinv, U::Erfinv)
        KTT_UNARY(i0, U::I0)
        KTT_UNARY(lgamma, U::Lgamma)
        KTT_UNARY(sign, U::Sign)
        KTT_UNARY(floor, U::Floor)
        KTT_UNARY(ceil, U::Ceil)
        KTT_UNARY(trunc, U::Trunc)
        KTT_UNARY(frac, U::Frac)
        KTT_UNARY(hardsigmoid, U::Hardsigmoid)
        KTT_UNARY(hardswish, U::Hardswish)
        KTT_UNARY(mish, U::Mish)

        // gelu(self, approximate): the exact ("none") form, or the tanh form
        // 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))); one program each.
        at::Tensor tt_gelu(const at::Tensor& self, c10::string_view approximate) {
            if (approximate == "tanh") {
                return unary_kernel("gelu", U::GeluTanh, self);
            }
            KTT_REQUIRE(
                approximate == "none", "gelu", "approximate '", approximate, "'"
            );
            return unary_kernel("gelu", U::Gelu, self);
        }

        // !self: true where self == 0.
        at::Tensor tt_logical_not(const at::Tensor& self) {
            return at::eq(self, 0);
        }

        // ~x: logical not for bool (stored 0/1), -x - 1 for integers.
        at::Tensor tt_bitwise_not(const at::Tensor& self) {
            const char* op = "bitwise_not";
            auto x = dense(op, self);
            if (self.scalar_type() == at::kBool) {
                auto h = bridge::binary_scalar(bridge::BinaryOp::Eq, x, 0.0);
                return wrap_output(h, self.sizes(), at::kBool);
            }
            KTT_REQUIRE(
                at::isIntegralType(self.scalar_type(), false),
                op,
                "needs a bool or integer tensor"
            );
            auto h = (bridge::binary_scalar(
                bridge::BinaryOp::Sub, bridge::unary(U::Neg, x), 1.0
            ));
            return wrap_output(h, self.sizes(), self.scalar_type());
        }

        at::Tensor& tt_bitwise_not_(at::Tensor& self) {
            return write_into("bitwise_not_", self, tt_bitwise_not(self));
        }

        at::Tensor& tt_bitwise_not_out(const at::Tensor& self, at::Tensor& out) {
            return write_out("bitwise_not.out", out, tt_bitwise_not(self));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("bitwise_not", &tt_bitwise_not);
        m.impl("bitwise_not_", &tt_bitwise_not_);
        m.impl("bitwise_not.out", &tt_bitwise_not_out);
        m.impl("logical_not", &tt_logical_not);
#define KTT_REGISTER_UNARY(op)    \
    m.impl(#op, &tt_##op);        \
    m.impl(#op "_", &tt_##op##_); \
    m.impl(#op ".out", &tt_##op##_out);
        KTT_REGISTER_UNARY(neg)
        KTT_REGISTER_UNARY(abs)
        KTT_REGISTER_UNARY(exp)
        KTT_REGISTER_UNARY(log)
        KTT_REGISTER_UNARY(sqrt)
        KTT_REGISTER_UNARY(rsqrt)
        KTT_REGISTER_UNARY(reciprocal)
        KTT_REGISTER_UNARY(tanh)
        KTT_REGISTER_UNARY(sigmoid)
        KTT_REGISTER_UNARY(silu)
        KTT_REGISTER_UNARY(sin)
        KTT_REGISTER_UNARY(cos)
        KTT_REGISTER_UNARY(erf)
        KTT_REGISTER_UNARY(acos)
        KTT_REGISTER_UNARY(asin)
        KTT_REGISTER_UNARY(asinh)
        KTT_REGISTER_UNARY(atan)
        KTT_REGISTER_UNARY(atanh)
        KTT_REGISTER_UNARY(acosh)
        KTT_REGISTER_UNARY(cosh)
        KTT_REGISTER_UNARY(sinh)
        KTT_REGISTER_UNARY(tan)
        KTT_REGISTER_UNARY(exp2)
        KTT_REGISTER_UNARY(expm1)
        KTT_REGISTER_UNARY(log2)
        KTT_REGISTER_UNARY(log10)
        KTT_REGISTER_UNARY(log1p)
        KTT_REGISTER_UNARY(erfc)
        KTT_REGISTER_UNARY(erfinv)
        KTT_REGISTER_UNARY(i0)
        KTT_REGISTER_UNARY(lgamma)
        KTT_REGISTER_UNARY(sign)
        KTT_REGISTER_UNARY(floor)
        KTT_REGISTER_UNARY(ceil)
        KTT_REGISTER_UNARY(trunc)
        KTT_REGISTER_UNARY(frac)
        KTT_REGISTER_UNARY(hardsigmoid)
        KTT_REGISTER_UNARY(hardswish)
        KTT_REGISTER_UNARY(mish)
#undef KTT_REGISTER_UNARY
        m.impl("relu", &tt_relu);
        m.impl("relu_", &tt_relu_);
        m.impl("gelu", &tt_gelu);
    }

}  // namespace ktt
