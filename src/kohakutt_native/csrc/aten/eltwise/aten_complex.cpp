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
                auto options = (at::TensorOptions()
                                    .dtype(at::kFloat)
                                    .device(at::kPrivateUse1));
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

        // True when `small` broadcasts into `full` as bridge::complex_mul() takes it:
        // the same sizes, or full's last two dims under leading dims of 1.
        bool broadcasts(c10::IntArrayRef full, c10::IntArrayRef small) {
            if (small.size() > full.size() || small.size() < 2) {
                return small == full;
            }
            const size_t lead = full.size() - small.size();
            bool same = lead == 0, head = true;
            for (size_t i = 0; i < small.size(); ++i) {
                same = same && small[i] == full[lead + i];
                const bool tail = i + 2 >= small.size();
                head = head && small[i] == (tail ? full[lead + i] : 1);
            }
            return same || head;
        }

        // self * other in one program on their interleaved pairs; nullopt where the
        // operands are not two complex tt tensors bridge::complex_mul() takes.
        std::optional<at::Tensor> interleaved_mul(
            const char* op, const at::Tensor& self, const at::Tensor& other
        ) {
            if (!holds_complex(self) || !holds_complex(other)) {
                return std::nullopt;
            }
            auto a = interleaved(self), b = interleaved(other);
            if (!a || !b) {
                return std::nullopt;
            }
            if (!broadcasts(a->x.sizes(), b->x.sizes())) {
                std::swap(a, b);
                if (!broadcasts(a->x.sizes(), b->x.sizes())) {
                    return std::nullopt;
                }
            }
            auto h = (bridge::complex_mul(
                dense(op, a->x), dense(op, b->x), a->conj, b->conj
            ));
            if (!h) {
                return std::nullopt;
            }
            return from_interleaved(wrap_output(h, a->x.sizes(), at::kFloat));
        }

        // The stored interleaved pairs of complex tt tensor `t`; nullopt for a
        // conjugated view and where interleaved() has none.
        std::optional<at::Tensor> plain_pairs(const at::Tensor& t) {
            if (!holds_complex(t)) {
                return std::nullopt;
            }
            auto p = interleaved(t);
            return p && !p->conj ? std::optional<at::Tensor>(p->x) : std::nullopt;
        }

    }  // namespace

    at::Tensor complex_binary(
        const char* op,
        bridge::BinaryOp id,
        const at::Tensor& self,
        const at::Tensor& other,
        const at::Scalar& alpha
    ) {
        if (id == B::Mul) {
            if (auto out = interleaved_mul(op, self, other)) {
                return *out;
            }
        }
        // add / sub of two tensors of one last dim: the fp32 op on their pairs.
        const bool unit_alpha = alpha.isIntegral(false) && alpha.toLong() == 1;
        if ((id == B::Add || id == B::Sub) && unit_alpha && self.dim() > 0 &&
            other.dim() > 0 && self.size(-1) == other.size(-1)) {
            auto a = plain_pairs(self), b = plain_pairs(other);
            if (a && b) {
                return from_interleaved(id == B::Add ? *a + *b : *a - *b);
            }
        }
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
        auto z = s.toComplexDouble();
        // A real factor or divisor: the fp32 op on the pairs.
        if ((id == B::Mul || id == B::Div) && z.imag() == 0.0) {
            if (auto x = plain_pairs(self)) {
                return from_interleaved(id == B::Mul ? *x * z.real() : *x / z.real());
            }
        }
        auto [ar, ai] = parts(self);
        auto br = at::full_like(ar, z.real()), bi = at::full_like(ar, z.imag());
        return combine(op, id, ar, ai, br, bi);
    }

    at::Tensor complex_unary(
        const char* op, bridge::UnaryOp id, const at::Tensor& self
    ) {
        if (id == U::Neg) {
            if (auto x = plain_pairs(self)) {
                return from_interleaved(-*x);
            }
        }
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
