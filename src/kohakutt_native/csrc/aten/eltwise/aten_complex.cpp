// Complex paths of the elementwise table kernels for tt (fp32 parts): add / sub /
// mul / div / eq / ne, neg / abs / sgn / exp / reciprocal.
#include <c10/util/complex.h>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"

namespace ktt {
    namespace {

        using B = bridge::BinaryOp;
        using U = bridge::UnaryOp;

        // (re, im) of a tensor operand; a wrapped Python number (0-d CPU) as a 0-d tt
        // pair.
        std::pair<at::Tensor, at::Tensor> operand(const at::Tensor& t) {
            if (t.is_cpu()) {
                auto z = t.item().toComplexDouble();
                auto options = (
                    at::TensorOptions().dtype(at::kFloat).device(at::kPrivateUse1)
                );
                return {
                    at::full({}, z.real(), options), at::full({}, z.imag(), options)
                };
            }
            return parts(t);
        }

        // The binary op `id` of (ar, ai) and (br, bi) as (re, im), or a bool tensor.
        at::Tensor combine(
            const char* op,
            bridge::BinaryOp id,
            const at::Tensor& ar,
            const at::Tensor& ai,
            const at::Tensor& br,
            const at::Tensor& bi
        ) {
            switch (id) {
                case B::Add:
                    return from_parts(ar + br, ai + bi);
                case B::Sub:
                    return from_parts(ar - br, ai - bi);
                case B::Mul:
                    return from_parts(ar * br - ai * bi, ar * bi + ai * br);
                case B::Div: {
                    auto den = br * br + bi * bi;
                    return (
                        from_parts((ar * br + ai * bi) / den, (ai * br - ar * bi) / den)
                    );
                }
                case B::Eq:
                    return at::logical_and(at::eq(ar, br), at::eq(ai, bi));
                case B::Ne:
                    return at::logical_or(at::ne(ar, br), at::ne(ai, bi));
                default:
                    unsupported(op, "this op on complex tensors");
            }
        }

    }  // namespace

    at::Tensor complex_binary(
        const char* op,
        bridge::BinaryOp id,
        const at::Tensor& self,
        const at::Tensor& other,
        const at::Scalar& alpha
    ) {
        auto [ar, ai] = operand(self);
        auto [br, bi] = operand(other);
        if ((id == B::Add || id == B::Sub) &&
            !(alpha.isIntegral(false) && alpha.toLong() == 1)) {
            auto z = alpha.toComplexDouble();
            auto scaled_re = br * z.real() - bi * z.imag();
            bi = br * z.imag() + bi * z.real();
            br = scaled_re;
        }
        return combine(op, id, ar, ai, br, bi);
    }

    at::Tensor complex_binary_scalar(
        const char* op, bridge::BinaryOp id, const at::Tensor& self, const at::Scalar& s
    ) {
        auto [ar, ai] = parts(self);
        auto z = s.toComplexDouble();
        auto br = at::full_like(ar, z.real()), bi = at::full_like(ar, z.imag());
        return combine(op, id, ar, ai, br, bi);
    }

    at::Tensor complex_unary(
        const char* op, bridge::UnaryOp id, const at::Tensor& self
    ) {
        auto [re, im] = parts(self);
        switch (id) {
            case U::Neg:
                return from_parts(-re, -im);
            case U::Abs:
                return at::hypot(re, im);
            case U::Exp: {
                auto scale = at::exp(re);
                return from_parts(scale * at::cos(im), scale * at::sin(im));
            }
            case U::Sign: {
                // sgn(z) = z / |z|, 0 at z = 0.
                auto mag = at::hypot(re, im);
                auto safe = at::where(at::eq(mag, 0.0), at::ones_like(mag), mag);
                return from_parts(re / safe, im / safe);
            }
            case U::Reciprocal: {
                auto den = re * re + im * im;
                return from_parts(re / den, -im / den);
            }
            default:
                unsupported(op, "this op on complex tensors");
        }
    }

}  // namespace ktt
