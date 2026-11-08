// Bridge elementwise ops: one TT-NN call per UnaryOp / BinaryOp id, plus typecast,
// where and the fused ternaries (addcmul, addcdiv, lerp).
#include <stdexcept>

#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "core/device/program.hpp"
#include "aten/eltwise/ktt_binary/program.hpp"
#include "aten/eltwise/ktt_unary/program.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

namespace ktt::bridge {
    namespace {

        bool is_compare(BinaryOp op) {
            return (
                op == BinaryOp::Eq || op == BinaryOp::Ne || op == BinaryOp::Lt ||
                op == BinaryOp::Le || op == BinaryOp::Gt || op == BinaryOp::Ge
            );
        }

        // Comparisons produce bf16 0/1 (the physical form of bool); others keep the
        // inputs'.
        DType result_dtype(BinaryOp op, DType input) {
            return is_compare(op) ? DType::BFloat16 : input;
        }

    }  // namespace

    HandlePtr unary(UnaryOp op, const HandlePtr& a) {
        if (detail::is_empty(a)) {
            return detail::empty_of(a, a->dtype);
        }
        const bool own = ktt_unary::supports(op, a->dtype, a->shape);
        const bool approx = numerics::effective().approx;
        return detail::timed_op(a->dtype, a->shape, [=]() -> ttnn::Tensor {
            const auto& t = a->tensor;
            const auto mc = detail::activation_config(a->bytes);
            if (own) {
                return ktt_unary::run(op, t, approx, mc);
            }
            const auto g = detail::worker_cores(a->shape, detail::kUnaryTilesPerCore);
            switch (op) {
                case UnaryOp::Neg:
                    return ttnn::neg(t, mc, std::nullopt, g);
                case UnaryOp::Abs:
                    return ttnn::abs(t, mc, std::nullopt, g);
                case UnaryOp::Exp:
                    return ttnn::exp(t, approx, mc, std::nullopt, g);
                case UnaryOp::Log:
                    return ttnn::log(t, approx, mc, std::nullopt, g);
                case UnaryOp::Sqrt:
                    return ttnn::sqrt(t, approx, mc, std::nullopt, g);
                case UnaryOp::Rsqrt:
                    return ttnn::rsqrt(t, approx, mc, std::nullopt, g);
                case UnaryOp::Reciprocal:
                    return ttnn::reciprocal(t, mc, std::nullopt, g);
                case UnaryOp::Tanh:
                    return ttnn::tanh(t, mc, std::nullopt, approx, g);
                case UnaryOp::Sigmoid:
                    return (
                        ttnn::sigmoid(
                            t,
                            static_cast<int>(ttnn::operations::unary::VecMode::RC),
                            (approx ? ttnn::operations::unary::SigmoidMode::
                                          ACCURATE_FAST_EXP
                                    : ttnn::operations::unary::SigmoidMode::ACCURATE),
                            mc,
                            std::nullopt,
                            g
                        )
                    );
                case UnaryOp::Relu:
                    return ttnn::relu(t, mc, std::nullopt, g);
                case UnaryOp::Gelu:
                    return (
                        ttnn::gelu(
                            t,
                            ttnn::operations::unary::GeluVariant::ACCURATE,
                            mc,
                            std::nullopt,
                            g
                        )
                    );
                case UnaryOp::Silu:
                    return ttnn::silu(t, mc, std::nullopt, g);
                case UnaryOp::Sin:
                    return ttnn::sin(t, mc, std::nullopt, g);
                case UnaryOp::Cos:
                    return ttnn::cos(t, mc, std::nullopt, g);
                case UnaryOp::Erf:
                    return ttnn::erf(t, approx, mc, std::nullopt, g);
                case UnaryOp::Acos:
                    return ttnn::acos(t, mc, std::nullopt, g);
                case UnaryOp::Asin:
                    return ttnn::asin(t, mc, std::nullopt, g);
                case UnaryOp::Asinh:
                    return ttnn::asinh(t, mc, std::nullopt, g);
                case UnaryOp::Atan:
                    return ttnn::atan(t, mc, std::nullopt, g);
                case UnaryOp::Atanh:
                    return ttnn::atanh(t, mc, std::nullopt, g);
                case UnaryOp::Acosh:
                    return ttnn::acosh(t, mc, std::nullopt, g);
                case UnaryOp::Cosh:
                    return ttnn::cosh(t, mc, std::nullopt, g);
                case UnaryOp::Sinh:
                    return ttnn::sinh(t, mc, std::nullopt, g);
                case UnaryOp::Tan:
                    return ttnn::tan(t, mc, std::nullopt, g);
                case UnaryOp::Exp2:
                    return ttnn::exp2(t, mc, std::nullopt, g);
                case UnaryOp::Expm1:
                    return ttnn::expm1(t, mc, std::nullopt, g);
                case UnaryOp::Log2:
                    return ttnn::log2(t, approx, mc, std::nullopt, g);
                case UnaryOp::Log10:
                    return ttnn::log10(t, approx, mc, std::nullopt, g);
                case UnaryOp::Log1p:
                    return ttnn::log1p(t, approx, mc, std::nullopt, g);
                case UnaryOp::Erfc:
                    return ttnn::erfc(t, mc, std::nullopt, g);
                case UnaryOp::Erfinv:
                    return ttnn::erfinv(t, mc, std::nullopt, g);
                case UnaryOp::Lgamma:
                    return ttnn::lgamma(t, mc, std::nullopt, g);
                case UnaryOp::Digamma:
                    return ttnn::digamma(t, mc, std::nullopt, g);
                case UnaryOp::Sign:
                    return ttnn::sign(t, mc, std::nullopt, g);
                case UnaryOp::Floor:
                    return ttnn::floor(t, mc, std::nullopt, g);
                case UnaryOp::Ceil:
                    return ttnn::ceil(t, mc, std::nullopt, g);
                case UnaryOp::Trunc:
                    return ttnn::trunc(t, mc, std::nullopt, g);
                case UnaryOp::Frac:
                    return ttnn::frac(t, mc, std::nullopt, g);
                case UnaryOp::Hardsigmoid:
                    return ttnn::hardsigmoid(t, mc, std::nullopt, g);
                case UnaryOp::Hardswish:
                    return ttnn::hardswish(t, mc, std::nullopt, g);
                case UnaryOp::Mish:
                    return ttnn::mish(t, approx, mc, std::nullopt, g);
                case UnaryOp::IsNan:
                    return ttnn::isnan(t, mc, std::nullopt, g);
                case UnaryOp::IsInf:
                    return ttnn::isinf(t, mc, std::nullopt, g);
                case UnaryOp::IsFinite:
                    return ttnn::isfinite(t, mc, std::nullopt, g);
                case UnaryOp::I0:
                    return ttnn::i0(t, mc, std::nullopt, g);
            }
            throw std::invalid_argument("kohakutt_native: unmapped UnaryOp");
        });
    }

    HandlePtr binary(BinaryOp op, const HandlePtr& a, const HandlePtr& b) {
        auto out = result_dtype(op, a->dtype);
        if (detail::is_empty(a)) {
            return detail::empty_of(a, out);
        }
        auto bf16 = std::optional<const tt::tt_metal::DataType>(detail::to_ttnn(out));
        auto shape = detail::broadcast_dims(a->shape, b->shape);
        const bool same_dtype = a->dtype == b->dtype;
        const auto plan =
            (same_dtype ? ktt_binary::plan(op, a->dtype, a->shape, b->shape)
                        : std::nullopt);
        return detail::timed_op(out, shape, [=]() -> ttnn::Tensor {
            const auto &x = a->tensor, &y = b->tensor;
            const auto mc = detail::activation_config(detail::padded_bytes(shape, out));
            if (plan) {
                return ktt_binary::run(op, x, y, *plan, mc);
            }
            const auto g = detail::worker_cores(shape, detail::kBinaryTilesPerCore);
            switch (op) {
                case BinaryOp::Add:
                    return ttnn::add(
                        x,
                        y,
                        std::nullopt,
                        mc,
                        std::nullopt,
                        {},
                        {},
                        {},
                        std::nullopt,
                        g
                    );
                case BinaryOp::Sub:
                    return ttnn::subtract(
                        x,
                        y,
                        std::nullopt,
                        mc,
                        std::nullopt,
                        {},
                        {},
                        {},
                        std::nullopt,
                        g
                    );
                case BinaryOp::Mul:
                    return ttnn::multiply(
                        x,
                        y,
                        std::nullopt,
                        mc,
                        std::nullopt,
                        {},
                        {},
                        {},
                        std::nullopt,
                        g
                    );
                case BinaryOp::Div:
                    return ttnn::divide(
                        x,
                        y,
                        std::nullopt,
                        mc,
                        std::nullopt,
                        {},
                        {},
                        {},
                        std::nullopt,
                        g
                    );
                case BinaryOp::Maximum:
                    return ttnn::maximum(x, y);
                case BinaryOp::Minimum:
                    return ttnn::minimum(x, y);
                case BinaryOp::Pow:
                    return ttnn::pow(x, y);
                case BinaryOp::Eq:
                    return ttnn::eq(x, y, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Ne:
                    return ttnn::ne(x, y, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Lt:
                    return ttnn::lt(x, y, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Le:
                    return ttnn::le(x, y, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Gt:
                    return ttnn::gt(x, y, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Ge:
                    return ttnn::ge(x, y, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::BitAnd:
                    return ttnn::bitwise_and(x, y);
                case BinaryOp::BitOr:
                    return ttnn::bitwise_or(x, y);
                case BinaryOp::BitXor:
                    return ttnn::bitwise_xor(x, y);
                case BinaryOp::Shl:
                    return ttnn::bitwise_left_shift(x, y);
                case BinaryOp::Shr:
                    return ttnn::bitwise_right_shift(x, y);
                case BinaryOp::Gcd:
                    return ttnn::gcd(x, y);
                case BinaryOp::Lcm:
                    return ttnn::lcm(x, y);
                case BinaryOp::Atan2:
                    return ttnn::atan2(x, y);
            }
            throw std::invalid_argument("kohakutt_native: unmapped BinaryOp");
        });
    }

    HandlePtr binary_scalar(BinaryOp op, const HandlePtr& a, double scalar) {
        auto out = result_dtype(op, a->dtype);
        if (detail::is_empty(a)) {
            return detail::empty_of(a, out);
        }
        auto bf16 = std::optional<const tt::tt_metal::DataType>(detail::to_ttnn(out));

        // The scalar takes the tensor's kind: int32 tensors need an int32 scalar.
        auto s =
            (a->dtype == DType::Int32
                 ? ttnn::operations::unary::ScalarVariant(
                       static_cast<int32_t>(std::llround(scalar))
                   )
                 : ttnn::operations::unary::ScalarVariant(static_cast<float>(scalar)));
        auto f = static_cast<float>(scalar);
        return detail::timed_op(out, a->shape, [=]() -> ttnn::Tensor {
            const auto& x = a->tensor;
            const auto mc = (
                detail::activation_config(detail::padded_bytes(a->shape, out))
            );
            const auto g = detail::worker_cores(a->shape, detail::kBinaryTilesPerCore);
            switch (op) {
                case BinaryOp::Add:
                    return ttnn::add(
                        x,
                        s,
                        std::nullopt,
                        mc,
                        std::nullopt,
                        {},
                        {},
                        {},
                        std::nullopt,
                        g
                    );
                case BinaryOp::Sub:
                    return ttnn::subtract(
                        x,
                        s,
                        std::nullopt,
                        mc,
                        std::nullopt,
                        {},
                        {},
                        {},
                        std::nullopt,
                        g
                    );
                case BinaryOp::Mul:
                    return ttnn::multiply(
                        x,
                        s,
                        std::nullopt,
                        mc,
                        std::nullopt,
                        {},
                        {},
                        {},
                        std::nullopt,
                        g
                    );
                case BinaryOp::Div:
                    return ttnn::divide(
                        x,
                        s,
                        std::nullopt,
                        mc,
                        std::nullopt,
                        {},
                        {},
                        {},
                        std::nullopt,
                        g
                    );
                case BinaryOp::Pow:
                    // Exponents with an exact cheaper form; others through pow.
                    if (a->dtype != DType::Int32) {
                        if (f == 2.0f) {
                            return ttnn::multiply(x, x);
                        }
                        if (f == 1.0f) {
                            return (
                                ttnn::clone(x, std::nullopt, std::nullopt, std::nullopt)
                            );
                        }
                        if (f == 0.5f) {
                            return ttnn::sqrt(x);
                        }
                        if (f == -0.5f) {
                            return ttnn::rsqrt(x);
                        }
                        if (f == -1.0f) {
                            return ttnn::reciprocal(x);
                        }
                    }
                    return ttnn::pow(x, f);
                case BinaryOp::Eq:
                    return ttnn::eq(x, s, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Ne:
                    return ttnn::ne(x, s, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Lt:
                    return ttnn::lt(x, s, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Le:
                    return ttnn::le(x, s, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Gt:
                    return ttnn::gt(x, s, bf16, mc, std::nullopt, {}, {}, {}, g);
                case BinaryOp::Ge:
                    return ttnn::ge(x, s, bf16, mc, std::nullopt, {}, {}, {}, g);
                default:
                    break;
            }
            throw std::invalid_argument("kohakutt_native: BinaryOp has no scalar form");
        });
    }

    HandlePtr clamp(
        const HandlePtr& a, std::optional<double> lo, std::optional<double> hi
    ) {
        // Bounds take the tensor's kind: int32 tensors need int32 bounds (a float bound
        // crashed the device op).
        const bool ints = a->dtype == DType::Int32;
        auto bound = [ints](
                         std::optional<double> v
                     ) -> std::optional<std::variant<float, int32_t>> {
            if (!v) {
                return std::nullopt;
            }
            if (ints) {
                return static_cast<int32_t>(*v);
            }
            return static_cast<float>(*v);
        };
        if (detail::is_empty(a)) {
            return detail::empty_of(a, a->dtype);
        }
        return detail::timed_op(a->dtype, a->shape, [=] {
            return ttnn::clamp(a->tensor, bound(lo), bound(hi));
        });
    }

    HandlePtr typecast(const HandlePtr& a, DType dtype) {
        if (a->dtype == dtype) {
            return a;
        }
        if (detail::is_empty(a)) {
            return detail::empty_of(a, dtype);
        }
        // bf16 <-> fp32 on tiled tensors: Kohaku-TT's copy program (sizes as runtime
        // arguments).
        const auto is_float = [](DType t) {
            return t == DType::BFloat16 || t == DType::Float32;
        };
        const bool own =
            (program::own_kernels_enabled() && is_float(a->dtype) && is_float(dtype));
        return detail::timed_op(dtype, a->shape, [=] {
            const auto mc = (
                detail::activation_config(detail::padded_bytes(a->shape, dtype))
            );
            if (own && a->tensor.layout() == tt::tt_metal::Layout::TILE) {
                return ktt_unary::cast(a->tensor, detail::to_ttnn(dtype), mc);
            }
            return ttnn::typecast(a->tensor, detail::to_ttnn(dtype), mc);
        });
    }

    HandlePtr bitcast(const HandlePtr& a, DType dtype) {
        if (a->dtype == dtype) {
            return a;
        }
        if (detail::is_empty(a)) {
            return detail::empty_of(a, dtype);
        }
        return detail::timed_op(dtype, a->shape, [=] {
            const auto mc = (
                detail::activation_config(detail::padded_bytes(a->shape, dtype))
            );
            return ttnn::bitcast(a->tensor, detail::to_ttnn(dtype), mc);
        });
    }

    HandlePtr where(const HandlePtr& cond, const HandlePtr& a, const HandlePtr& b) {
        if (detail::is_empty(a)) {
            return detail::empty_of(a, a->dtype);
        }
        return detail::timed_op(
            a->dtype,
            detail::broadcast_dims(
                cond->shape, detail::broadcast_dims(a->shape, b->shape)
            ),
            [=] {
                return ttnn::where(cond->tensor, a->tensor, b->tensor);
            }
        );
    }

    HandlePtr addcmul(
        const HandlePtr& a, const HandlePtr& b, const HandlePtr& c, double v
    ) {
        return detail::timed_op(a->dtype, a->shape, [=] {
            return (
                ttnn::addcmul(a->tensor, b->tensor, c->tensor, static_cast<float>(v))
            );
        });
    }

    HandlePtr addcdiv(
        const HandlePtr& a, const HandlePtr& b, const HandlePtr& c, double v
    ) {
        return detail::timed_op(a->dtype, a->shape, [=] {
            return (
                ttnn::addcdiv(a->tensor, b->tensor, c->tensor, static_cast<float>(v))
            );
        });
    }

    HandlePtr lerp(const HandlePtr& a, const HandlePtr& end, double weight) {
        return detail::timed_op(a->dtype, a->shape, [=] {
            return ttnn::lerp(a->tensor, end->tensor, static_cast<float>(weight));
        });
    }

}  // namespace ktt::bridge
