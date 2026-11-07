// aten predicate and logical kernels for tt: isnan / isinf / isneginf, logical and
// bitwise and / or / xor, where.out.

#include <torch/library.h>

#include <limits>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // NaN / inf tests by TT-NN's own ops (x != x is false for NaN on device);
        // integer tensors are never NaN or inf.
        at::Tensor float_test(
            const char* op, bridge::UnaryOp id, const at::Tensor& self
        ) {
            if (!at::isFloatingType(self.scalar_type())) {
                return at::zeros(self.sizes(), self.options().dtype(at::kBool));
            }
            auto h = bridge::unary(id, dense(op, self));
            return wrap_output(h, self.sizes(), self.scalar_type()).to(at::kBool);
        }

        at::Tensor tt_isnan(const at::Tensor& self) {
            return float_test("isnan", bridge::UnaryOp::IsNan, self);
        }

        at::Tensor tt_isinf(const at::Tensor& self) {
            return float_test("isinf", bridge::UnaryOp::IsInf, self);
        }

        // Logical ops on any dtype: nonzero -> true.
        at::Tensor tt_logical_and(const at::Tensor& a, const at::Tensor& b) {
            return at::bitwise_and(at::ne(a, 0), at::ne(b, 0));
        }

        at::Tensor tt_logical_or(const at::Tensor& a, const at::Tensor& b) {
            return at::bitwise_or(at::ne(a, 0), at::ne(b, 0));
        }

        at::Tensor tt_logical_xor(const at::Tensor& a, const at::Tensor& b) {
            return at::ne(at::ne(a, 0), at::ne(b, 0));
        }

        // and/or: bool (stored 0/1) as a * b / max(a, b); integers by int32 bitwise
        // ops.
        at::Tensor bool_logic(
            const char* op, const at::Tensor& a, const at::Tensor& b, bool is_and
        ) {
            auto dtype = at::result_type(a, b);
            if (dtype == at::kBool) {
                auto x = a.to(at::kFloat), y = b.to(at::kFloat);
                return at::ne(is_and ? x * y : at::maximum(x, y), 0);
            }
            KTT_REQUIRE(
                at::isIntegralType(dtype, false), op, "needs bool or integer operands"
            );
            auto shape = broadcast_shape(a, b);
            auto h = (
                bridge::binary(
                    is_and ? bridge::BinaryOp::BitAnd : bridge::BinaryOp::BitOr,
                    dense_as(op, a.expand(shape), at::kInt),
                    dense_as(op, b.expand(shape), at::kInt)
                )
            );
            return wrap_output(h, shape, dtype);
        }

        at::Tensor& tt_bitwise_and_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            return (
                into(
                    "bitwise_and.Tensor_out", out, bool_logic("bitwise_and", a, b, true)
                )
            );
        }

        at::Tensor& tt_bitwise_or_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            return (
                into(
                    "bitwise_or.Tensor_out", out, bool_logic("bitwise_or", a, b, false)
                )
            );
        }

        // xor: bool as a != b; integers by the int32 bitwise op.
        at::Tensor& tt_bitwise_xor_out(
            const at::Tensor& a, const at::Tensor& b, at::Tensor& out
        ) {
            const char* op = "bitwise_xor.Tensor_out";
            auto dtype = at::result_type(a, b);
            if (dtype == at::kBool) {
                return into(op, out, at::ne(a.to(at::kFloat), b.to(at::kFloat)));
            }
            KTT_REQUIRE(
                at::isIntegralType(dtype, false), op, "needs bool or integer operands"
            );
            auto shape = broadcast_shape(a, b);
            auto h = (
                bridge::binary(
                    bridge::BinaryOp::BitXor,
                    dense_as(op, a.expand(shape), at::kInt),
                    dense_as(op, b.expand(shape), at::kInt)
                )
            );
            return into(op, out, wrap_output(h, shape, at::kInt).to(dtype));
        }

        at::Tensor& tt_isneginf_out(const at::Tensor& self, at::Tensor& out) {
            return (
                into(
                    "isneginf.out",
                    out,
                    at::eq(self, -std::numeric_limits<double>::infinity())
                )
            );
        }

        at::Tensor& tt_where_out(
            const at::Tensor& cond,
            const at::Tensor& self,
            const at::Tensor& other,
            at::Tensor& out
        ) {
            return into("where.self_out", out, at::where(cond, self, other));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("isnan", &tt_isnan);
        m.impl("isinf", &tt_isinf);
        m.impl("logical_and", &tt_logical_and);
        m.impl("logical_or", &tt_logical_or);
        m.impl("logical_xor", &tt_logical_xor);
        m.impl("bitwise_and.Tensor_out", &tt_bitwise_and_out);
        m.impl("bitwise_or.Tensor_out", &tt_bitwise_or_out);
        m.impl("bitwise_xor.Tensor_out", &tt_bitwise_xor_out);
        m.impl("isneginf.out", &tt_isneginf_out);
        m.impl("where.self_out", &tt_where_out);
    }

}  // namespace ktt
