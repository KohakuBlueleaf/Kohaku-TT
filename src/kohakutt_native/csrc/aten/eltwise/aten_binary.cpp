// aten elementwise binary kernels for tt: one table row per op (functional, in-place
// and out= overloads; Tensor and Scalar forms).
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        using B = bridge::BinaryOp;

// Tensor-Tensor op with alpha (add/sub): tt_<op>, tt_<op>_, tt_<op>_out.
#define KTT_BINARY_ALPHA(op, name, id)                                                 \
    at::Tensor tt_##op(                                                                \
        const at::Tensor& s, const at::Tensor& o, const at::Scalar& a                  \
    ) {                                                                                \
        return binary_kernel(name ".Tensor", id, s, o, a);                             \
    }                                                                                  \
    at::Tensor& tt_##op##_(at::Tensor& s, const at::Tensor& o, const at::Scalar& a) {  \
        return write_into(name "_.Tensor", s, tt_##op(s, o, a));                       \
    }                                                                                  \
    at::Tensor& tt_##op##_out(                                                         \
        const at::Tensor& s, const at::Tensor& o, const at::Scalar& a, at::Tensor& out \
    ) {                                                                                \
        return write_into(name ".out", out, tt_##op(s, o, a));                         \
    }                                                                                  \
    at::Tensor tt_##op##_scalar(                                                       \
        const at::Tensor& s, const at::Scalar& o, const at::Scalar& a                  \
    ) {                                                                                \
        return binary_scalar_kernel(                                                   \
            name ".Scalar",                                                            \
            id,                                                                        \
            s,                                                                         \
            a.toDouble() == 1.0 ? o : at::Scalar(o.toDouble() * a.toDouble())          \
        );                                                                             \
    }                                                                                  \
    at::Tensor& tt_##op##_scalar_(                                                     \
        at::Tensor& s, const at::Scalar& o, const at::Scalar& a                        \
    ) {                                                                                \
        return write_into(name "_.Scalar", s, tt_##op##_scalar(s, o, a));              \
    }

// Tensor-Tensor op without alpha, plus its Scalar form.
#define KTT_BINARY(op, name, id, tensor_overload, scalar_overload)                   \
    at::Tensor tt_##op(const at::Tensor& s, const at::Tensor& o) {                   \
        return binary_kernel(name tensor_overload, id, s, o, 1);                     \
    }                                                                                \
    at::Tensor& tt_##op##_(at::Tensor& s, const at::Tensor& o) {                     \
        return write_into(name "_" tensor_overload, s, tt_##op(s, o));               \
    }                                                                                \
    at::Tensor& tt_##op##_out(                                                       \
        const at::Tensor& s, const at::Tensor& o, at::Tensor& out                    \
    ) {                                                                              \
        return write_into(name ".out", out, tt_##op(s, o));                          \
    }                                                                                \
    at::Tensor tt_##op##_scalar(const at::Tensor& s, const at::Scalar& o) {          \
        return binary_scalar_kernel(name scalar_overload, id, s, o);                 \
    }                                                                                \
    at::Tensor& tt_##op##_scalar_(at::Tensor& s, const at::Scalar& o) {              \
        return write_into(name "_" scalar_overload, s, tt_##op##_scalar(s, o));      \
    }                                                                                \
    at::Tensor& tt_##op##_scalar_out(                                                \
        const at::Tensor& s, const at::Scalar& o, at::Tensor& out                    \
    ) {                                                                              \
        return write_into(name scalar_overload "_out", out, tt_##op##_scalar(s, o)); \
    }

        KTT_BINARY_ALPHA(add, "add", B::Add)
        KTT_BINARY_ALPHA(sub, "sub", B::Sub)
        KTT_BINARY(mul, "mul", B::Mul, ".Tensor", ".Scalar")
        KTT_BINARY(div, "div", B::Div, ".Tensor", ".Scalar")
        KTT_BINARY(pow, "pow", B::Pow, ".Tensor_Tensor", ".Tensor_Scalar")
        KTT_BINARY(eq, "eq", B::Eq, ".Tensor", ".Scalar")
        KTT_BINARY(ne, "ne", B::Ne, ".Tensor", ".Scalar")
        KTT_BINARY(lt, "lt", B::Lt, ".Tensor", ".Scalar")
        KTT_BINARY(le, "le", B::Le, ".Tensor", ".Scalar")
        KTT_BINARY(gt, "gt", B::Gt, ".Tensor", ".Scalar")
        KTT_BINARY(ge, "ge", B::Ge, ".Tensor", ".Scalar")

        at::Tensor tt_maximum(const at::Tensor& s, const at::Tensor& o) {
            return binary_kernel("maximum", B::Maximum, s, o, 1);
        }

        at::Tensor tt_minimum(const at::Tensor& s, const at::Tensor& o) {
            return binary_kernel("minimum", B::Minimum, s, o, 1);
        }

        // atan2(s, o): integer operands promote to the default float dtype.
        at::Tensor tt_atan2(const at::Tensor& s, const at::Tensor& o) {
            const bool include_bool = true;
            if (at::isIntegralType(at::result_type(s, o), include_bool)) {
                auto f = at::get_default_dtype_as_scalartype();
                return binary_kernel("atan2", B::Atan2, s.to(f), o.to(f), 1);
            }
            return binary_kernel("atan2", B::Atan2, s, o, 1);
        }

        at::Tensor& tt_atan2_(at::Tensor& s, const at::Tensor& o) {
            return write_into("atan2_", s, tt_atan2(s, o));
        }

        at::Tensor& tt_atan2_out(
            const at::Tensor& s, const at::Tensor& o, at::Tensor& out
        ) {
            return write_into("atan2.out", out, tt_atan2(s, o));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        // add / sub: Tensor, in-place, out, Scalar, in-place Scalar.
        m.impl("add.Tensor", &tt_add);
        m.impl("add_.Tensor", &tt_add_);
        m.impl("add.out", &tt_add_out);
        m.impl("add.Scalar", &tt_add_scalar);
        m.impl("add_.Scalar", &tt_add_scalar_);
        m.impl("sub.Tensor", &tt_sub);
        m.impl("sub_.Tensor", &tt_sub_);
        m.impl("sub.out", &tt_sub_out);
        m.impl("sub.Scalar", &tt_sub_scalar);
        m.impl("sub_.Scalar", &tt_sub_scalar_);

        // mul / div.
        m.impl("mul.Tensor", &tt_mul);
        m.impl("mul_.Tensor", &tt_mul_);
        m.impl("mul.out", &tt_mul_out);
        m.impl("mul.Scalar", &tt_mul_scalar);
        m.impl("mul_.Scalar", &tt_mul_scalar_);
        m.impl("div.Tensor", &tt_div);
        m.impl("div_.Tensor", &tt_div_);
        m.impl("div.out", &tt_div_out);
        m.impl("div.Scalar", &tt_div_scalar);
        m.impl("div_.Scalar", &tt_div_scalar_);

        // pow.
        m.impl("pow.Tensor_Tensor", &tt_pow);
        m.impl("pow_.Tensor", &tt_pow_);
        m.impl("pow.Tensor_Tensor_out", &tt_pow_out);
        m.impl("pow.Tensor_Scalar", &tt_pow_scalar);
        m.impl("pow_.Scalar", &tt_pow_scalar_);
        m.impl("pow.Tensor_Scalar_out", &tt_pow_scalar_out);

        // maximum / minimum.
        m.impl("maximum", &tt_maximum);
        m.impl("minimum", &tt_minimum);

        // atan2.
        m.impl("atan2", &tt_atan2);
        m.impl("atan2_", &tt_atan2_);
        m.impl("atan2.out", &tt_atan2_out);

        // Comparisons: Tensor, Scalar and their out= forms (results are bool).
#define KTT_REGISTER_COMPARE(op)               \
    m.impl(#op ".Tensor", &tt_##op);           \
    m.impl(#op ".Tensor_out", &tt_##op##_out); \
    m.impl(#op ".Scalar", &tt_##op##_scalar);  \
    m.impl(#op ".Scalar_out", &tt_##op##_scalar_out);
        KTT_REGISTER_COMPARE(eq)
        KTT_REGISTER_COMPARE(ne)
        KTT_REGISTER_COMPARE(lt)
        KTT_REGISTER_COMPARE(le)
        KTT_REGISTER_COMPARE(gt)
        KTT_REGISTER_COMPARE(ge)
#undef KTT_REGISTER_COMPARE
    }

}  // namespace ktt
