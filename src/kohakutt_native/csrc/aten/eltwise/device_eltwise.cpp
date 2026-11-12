// Bridge elementwise ops: one TT-NN call per UnaryOp / BinaryOp id, plus typecast,
// where and the fused ternaries (addcmul, addcdiv, lerp). Unary, binary, scalar and
// typecast ops keep Nhwc operands Nhwc.
#include <algorithm>
#include <cmath>
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

        // Ops mapping 0 to 0: an Nhwc result keeps the operands' zero tile padding.
        bool keeps_zero(UnaryOp op) {
            using U = UnaryOp;
            return (
                op == U::Neg || op == U::Abs || op == U::Relu || op == U::Gelu ||
                op == U::GeluTanh || op == U::Silu
            );
        }

        bool keeps_zero(BinaryOp op) {
            using B = BinaryOp;
            return (
                op == B::Add || op == B::Sub || op == B::Mul || op == B::Maximum ||
                op == B::Minimum
            );
        }

        bool keeps_zero(BinaryOp op, double s) {
            using B = BinaryOp;
            return (
                std::isfinite(s) &&
                (op == B::Mul || (op == B::Div && s != 0) ||
                 ((op == B::Add || op == B::Sub) && s == 0))
            );
        }

        // An Nhwc [1, C or 1, 1, 1] vector `v` broadcast against Nhwc `full`.
        bool channel_vector(const Handle& v, const Handle& full) {
            const auto &s = v.shape, &f = full.shape;
            return (
                s.size() == 4 && f.size() == 4 && s[0] == 1 && s[2] == 1 && s[3] == 1 &&
                (s[1] == f[1] || s[1] == 1)
            );
        }

        // A one-element `v` of `full`'s rank (broadcast over it in any format).
        bool one_element(const Handle& v, const Handle& full) {
            return (
                v.shape.size() == full.shape.size() &&
                std::all_of(v.shape.begin(), v.shape.end(), [](int64_t d) {
                    return d == 1;
                })
            );
        }

        // a and b operate as stored: one format, and one shape or a broadcast vector.
        bool stored_pair(const Handle& a, const Handle& b) {
            return (
                a.format == b.format &&
                (a.shape == b.shape || one_element(a, b) || one_element(b, a) ||
                 (a.format == Format::HeadMajor && a.shape.size() == b.shape.size()) ||
                 (a.format == Format::Nhwc &&
                  (channel_vector(a, b) || channel_vector(b, a))))
            );
        }

        // `ops` in one format: a non-Nchw format one of them has when all share a
        // shape (the others converted, cached), else Nchw.
        std::vector<HandlePtr> one_format(std::vector<HandlePtr> ops) {
            Format f = Format::Nchw;
            bool same = true;
            for (const auto& h : ops) {
                f = f == Format::Nchw ? h->format : f;
                same = same && h->shape == ops.front()->shape;
            }
            // Stored formats hold bf16 / fp32 only: other dtypes keep all in Nchw.
            for (const auto& h : ops) {
                same = same &&
                    (h->dtype == DType::BFloat16 || h->dtype == DType::Float32);
            }
            if (f != Format::Nchw && !same) {
                f = Format::Nchw;
            }
            for (auto& h : ops) {
                h = to_format(h, f);
            }
            return ops;
        }

        // `compute`'s result, its tile padding zeroed for an Nhwc result of padded
        // stored dims `dims` unless `kept`.
        template <typename F>
        HandlePtr stored_op(
            DType dtype,
            const std::vector<int64_t>& shape,
            Format format,
            bool kept,
            F compute,
            const char* label = __builtin_FUNCTION()
        ) {
            const bool fill =
                (format == Format::Nhwc && !kept &&
                 detail::has_padding(detail::stored_dims(shape, format)));
            return detail::timed_op(
                dtype,
                shape,
                [=] {
                    auto t = compute();
                    return fill ? detail::zero_padding(t) : t;
                },
                format,
                label
            );
        }

    }  // namespace

    HandlePtr unary(UnaryOp op, const HandlePtr& a) {
        if (detail::is_empty(a)) {
            return detail::empty_of(a, a->dtype);
        }
        const auto dims = detail::stored_dims(a->shape, a->format);
        const bool fill =
            (a->format == Format::Nhwc && !keeps_zero(op) && detail::has_padding(dims));
        const bool own = ktt_unary::supports(op, a->dtype, dims);
        const bool approx = numerics::effective().approx;
        auto compute = [=]() -> ttnn::Tensor {
            const auto& t = a->tensor;
            const auto mc = detail::activation_config(a->bytes);
            if (own) {
                return ktt_unary::run(op, t, approx, mc);
            }
            const auto g = detail::worker_cores(dims, detail::kUnaryTilesPerCore);
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
                    return (ttnn::sigmoid(
                        t,
                        static_cast<int>(ttnn::operations::unary::VecMode::RC),
                        (approx
                             ? ttnn::operations::unary::SigmoidMode::ACCURATE_FAST_EXP
                             : ttnn::operations::unary::SigmoidMode::ACCURATE),
                        mc,
                        std::nullopt,
                        g
                    ));
                case UnaryOp::Relu:
                    return ttnn::relu(t, mc, std::nullopt, g);
                case UnaryOp::Gelu:
                    return (ttnn::gelu(
                        t,
                        ttnn::operations::unary::GeluVariant::ACCURATE,
                        mc,
                        std::nullopt,
                        g
                    ));
                case UnaryOp::GeluTanh:
                    return (ttnn::gelu(
                        t,
                        ttnn::operations::unary::GeluVariant::TANH,
                        mc,
                        std::nullopt,
                        g
                    ));
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
        };
        return detail::timed_op(
            a->dtype,
            a->shape,
            [=] {
                auto t = compute();
                return fill ? detail::zero_padding(t) : t;
            },
            a->format
        );
    }

    HandlePtr binary(BinaryOp op, const HandlePtr& a, const HandlePtr& b) {
        for (size_t k = 0; k < std::min(a->shape.size(), b->shape.size()); ++k) {
            const int64_t x = a->shape[a->shape.size() - 1 - k];
            const int64_t y = b->shape[b->shape.size() - 1 - k];
            if (x != y && x != 1 && y != 1) {
                throw std::invalid_argument(
                    "kohakutt_native: binary: shapes " + detail::dims_str(a->shape) +
                    " and " + detail::dims_str(b->shape) + " do not broadcast"
                );
            }
        }
        const bool stored = a->format != Format::Nchw || b->format != Format::Nchw;
        if (stored && !stored_pair(*a, *b)) {
            return binary(op, to_nchw(a), to_nchw(b));
        }
        const bool nhwc = a->format == Format::Nhwc;
        auto out = result_dtype(op, a->dtype);
        if (detail::is_empty(a)) {
            return detail::empty_of(a, out);
        }
        auto bf16 = std::optional<const tt::tt_metal::DataType>(detail::to_ttnn(out));
        auto shape = detail::broadcast_dims(a->shape, b->shape);
        const auto format = a->format;
        const auto dims = detail::stored_dims(shape, format);
        const bool fill =
            (nhwc && detail::has_padding(dims) &&
             !(a->shape == b->shape && keeps_zero(op)));
        const bool same_dtype = a->dtype == b->dtype;
        const auto plan =
            (same_dtype ? ktt_binary::plan(
                              op,
                              a->dtype,
                              detail::stored_dims(a->shape, format),
                              detail::stored_dims(b->shape, format)
                          )
                        : std::nullopt);
        auto compute = [=]() -> ttnn::Tensor {
            const auto &x = a->tensor, &y = b->tensor;
            const auto mc = detail::activation_config(detail::padded_bytes(dims, out));
            if (plan) {
                return ktt_binary::run(op, x, y, *plan, mc);
            }
            const auto g = detail::worker_cores(dims, detail::kBinaryTilesPerCore);
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
        };
        return detail::timed_op(
            out,
            shape,
            [=] {
                auto t = compute();
                return fill ? detail::zero_padding(t) : t;
            },
            format
        );
    }

    HandlePtr complex_mul(
        const HandlePtr& a, const HandlePtr& b, bool conj_a, bool conj_b
    ) {
        const bool plain =
            (a->dtype == DType::Float32 && b->dtype == DType::Float32 &&
             a->format == Format::Nchw && b->format == Format::Nchw &&
             a->placement.is_replicated() && b->placement.is_replicated() &&
             !detail::is_empty(a));
        const auto plan =
            (plain ? ktt_binary::complex_plan(
                         detail::stored_dims(a->shape, a->format),
                         detail::stored_dims(b->shape, b->format)
                     )
                   : std::nullopt);
        if (!plan) {
            return nullptr;
        }
        const auto dims = detail::stored_dims(a->shape, a->format);
        return detail::timed_op(a->dtype, a->shape, [=] {
            const auto mc = (detail::activation_config(
                detail::padded_bytes(dims, a->dtype)
            ));
            return ktt_binary::run_complex_mul(
                a->tensor, b->tensor, *plan, conj_a, conj_b, mc
            );
        });
    }

    HandlePtr binary_scalar(BinaryOp op, const HandlePtr& a, double scalar) {
        // max/min against a scalar are one-sided clamps.
        if (op == BinaryOp::Maximum) {
            return clamp(a, scalar, std::nullopt);
        }
        if (op == BinaryOp::Minimum) {
            return clamp(a, std::nullopt, scalar);
        }
        auto out = result_dtype(op, a->dtype);
        if (detail::is_empty(a)) {
            return detail::empty_of(a, out);
        }
        const auto dims = detail::stored_dims(a->shape, a->format);
        const bool fill =
            (a->format == Format::Nhwc && !keeps_zero(op, scalar) &&
             detail::has_padding(dims));
        auto bf16 = std::optional<const tt::tt_metal::DataType>(detail::to_ttnn(out));

        // The scalar takes the tensor's kind: int32 tensors need an int32 scalar.
        auto s =
            (a->dtype == DType::Int32
                 ? ttnn::operations::unary::ScalarVariant(
                       static_cast<int32_t>(std::llround(scalar))
                   )
                 : ttnn::operations::unary::ScalarVariant(static_cast<float>(scalar)));
        auto f = static_cast<float>(scalar);
        const bool approx = numerics::effective().approx;
        auto compute = [=]() -> ttnn::Tensor {
            const auto& x = a->tensor;
            const auto mc = detail::activation_config(detail::padded_bytes(dims, out));
            const auto g = detail::worker_cores(dims, detail::kBinaryTilesPerCore);
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
                            return ttnn::multiply(
                                x,
                                x,
                                std::nullopt,
                                mc,
                                std::nullopt,
                                {},
                                {},
                                {},
                                std::nullopt,
                                g
                            );
                        }
                        if (f == 1.0f) {
                            return (
                                ttnn::clone(x, std::nullopt, std::nullopt, std::nullopt)
                            );
                        }
                        if (f == 0.5f) {
                            return ttnn::sqrt(x, approx, mc, std::nullopt, g);
                        }
                        if (f == -0.5f) {
                            return ttnn::rsqrt(x, approx, mc, std::nullopt, g);
                        }
                        if (f == -1.0f) {
                            return ttnn::reciprocal(x, mc, std::nullopt, g);
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
                case BinaryOp::BitAnd:
                    if (a->dtype == DType::Int32) {
                        const auto bits = static_cast<int32_t>(std::llround(scalar));
                        return ttnn::bitwise_and(x, bits, mc);
                    }
                    break;
                default:
                    break;
            }
            throw std::invalid_argument("kohakutt_native: BinaryOp has no scalar form");
        };
        return detail::timed_op(
            out,
            a->shape,
            [=] {
                auto t = compute();
                return fill ? detail::zero_padding(t) : t;
            },
            a->format
        );
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
        const bool kept = lo.value_or(0) <= 0 && hi.value_or(0) >= 0;
        return stored_op(a->dtype, a->shape, a->format, kept, [=] {
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
        const int64_t bytes = (detail::padded_bytes(
            detail::stored_dims(a->shape, a->format), dtype
        ));
        return detail::timed_op(
            dtype,
            a->shape,
            [=] {
                const auto mc = detail::activation_config(bytes);
                if (own && a->tensor.layout() == tt::tt_metal::Layout::TILE) {
                    return ktt_unary::cast(a->tensor, detail::to_ttnn(dtype), mc);
                }
                return ttnn::typecast(a->tensor, detail::to_ttnn(dtype), mc);
            },
            a->format
        );
    }

    HandlePtr bitcast(const HandlePtr& a, DType dtype) {
        if (a->dtype == dtype) {
            return a;
        }
        if (detail::is_empty(a)) {
            return detail::empty_of(a, dtype);
        }
        const int64_t bytes = (detail::padded_bytes(
            detail::stored_dims(a->shape, a->format), dtype
        ));
        return stored_op(dtype, a->shape, a->format, true, [=] {
            const auto mc = detail::activation_config(bytes);
            return ttnn::bitcast(a->tensor, detail::to_ttnn(dtype), mc);
        });
    }

    // The ternaries run on their operands as stored when they share a non-Nchw format
    // and shape (the others converted to it), else on Nchw.
    HandlePtr where(const HandlePtr& cond, const HandlePtr& a, const HandlePtr& b) {
        if (detail::is_empty(a)) {
            return detail::empty_of(a, a->dtype);
        }
        const auto in = one_format({cond, a, b});
        return stored_op(
            a->dtype,
            detail::broadcast_dims(
                cond->shape, detail::broadcast_dims(a->shape, b->shape)
            ),
            in[0]->format,
            true,
            [=] {
                return ttnn::where(in[0]->tensor, in[1]->tensor, in[2]->tensor);
            }
        );
    }

    HandlePtr addcmul(
        const HandlePtr& a, const HandlePtr& b, const HandlePtr& c, double v
    ) {
        const auto in = one_format({a, b, c});
        return stored_op(a->dtype, a->shape, in[0]->format, true, [=] {
            return (ttnn::addcmul(
                in[0]->tensor, in[1]->tensor, in[2]->tensor, static_cast<float>(v)
            ));
        });
    }

    HandlePtr addcdiv(
        const HandlePtr& a, const HandlePtr& b, const HandlePtr& c, double v
    ) {
        const auto in = one_format({a, b, c});
        return stored_op(a->dtype, a->shape, in[0]->format, false, [=] {
            return (ttnn::addcdiv(
                in[0]->tensor, in[1]->tensor, in[2]->tensor, static_cast<float>(v)
            ));
        });
    }

    HandlePtr lerp(const HandlePtr& a, const HandlePtr& end, double weight) {
        const auto in = one_format({a, end});
        return stored_op(a->dtype, a->shape, in[0]->format, true, [=] {
            return (
                ttnn::lerp(in[0]->tensor, in[1]->tensor, static_cast<float>(weight))
            );
        });
    }

}  // namespace ktt::bridge
