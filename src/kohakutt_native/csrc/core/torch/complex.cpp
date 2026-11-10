// complex64 on tt: views between complex tensors and their fp32 pair storage,
// conjugation, complex / polar construction.
#include "core/torch/complex.hpp"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {

    bool holds_complex(const at::Tensor& t) {
        return t.defined() && at::isComplexType(t.scalar_type());
    }

    at::Tensor as_real(const at::Tensor& t) {
        KTT_REQUIRE(
            t.scalar_type() == at::kComplexFloat,
            "complex",
            "only complex64 is supported"
        );
        if (!t.is_conj()) {
            return at::view_as_real(t);
        }
        // A conjugated view: its stored pairs with the imaginary half negated (never
        // resolve_conj, which copies through this function).
        auto stored = t.alias();
        stored._set_conj(false);
        auto r = at::view_as_real(stored);
        return at::stack({r.select(-1, 0), -r.select(-1, 1)}, -1);
    }

    at::Tensor as_complex(const at::Tensor& pairs) {
        auto p = pairs.scalar_type() == at::kFloat ? pairs : pairs.to(at::kFloat);
        bool viewable =
            (p.size(-1) == 2 && p.stride(-1) == 1 && p.storage_offset() % 2 == 0);
        for (int64_t d = 0; viewable && d + 1 < p.dim(); ++d) {
            viewable = p.stride(d) % 2 == 0;
        }
        return at::view_as_complex(viewable ? p : p.contiguous());
    }

    std::pair<at::Tensor, at::Tensor> parts(const at::Tensor& t) {
        if (holds_complex(t)) {
            auto r = as_real(t);
            return {r.select(-1, 0), r.select(-1, 1)};
        }
        auto re = t.to(at::kFloat);
        return {re, at::zeros_like(re)};
    }

    at::Tensor from_parts(const at::Tensor& re, const at::Tensor& im) {
        auto shape = at::infer_size(re.sizes(), im.sizes());
        auto r = re.to(at::kFloat).expand(shape), i = im.to(at::kFloat).expand(shape);
        return as_complex(at::stack({r, i}, -1));
    }

    namespace {

        // Pairs `r` [..., N, 2] viewed as [..., 2 N]; nullopt where that is no view.
        std::optional<at::Tensor> merged(const at::Tensor& r) {
            if (r.dim() < 2 || r.stride(-1) != 1 || r.stride(-2) != 2) {
                return std::nullopt;
            }
            auto sizes = r.sizes().vec();
            sizes.pop_back();
            sizes.back() *= 2;
            return r.view(sizes);
        }

    }  // namespace

    std::optional<Interleaved> interleaved(const at::Tensor& t) {
        if (!t.is_privateuseone() || t.scalar_type() != at::kComplexFloat ||
            t.is_neg()) {
            return std::nullopt;
        }
        auto stored = t.alias();
        stored._set_conj(false);
        auto x = merged(at::view_as_real(stored));
        if (!x) {
            return std::nullopt;
        }
        return Interleaved{*x, t.is_conj()};
    }

    at::Tensor from_interleaved(const at::Tensor& x) {
        auto sizes = x.sizes().vec();
        sizes.back() /= 2;
        sizes.push_back(2);
        return at::view_as_complex(x.contiguous().view(sizes));
    }

    void copy_pairs(const at::Tensor& dst, const at::Tensor& src) {
        auto d = as_real(dst), s = as_real(src.to(at::kComplexFloat));
        auto md = merged(d), ms = merged(s);
        if (md && ms && d.sizes() == s.sizes()) {
            md->copy_(*ms);
        } else {
            d.copy_(s);
        }
    }

    at::Tensor complex_empty(c10::IntArrayRef sizes) {
        auto flat = sizes.vec();
        if (flat.empty()) {
            flat.push_back(2);
        } else {
            flat.back() *= 2;
        }
        auto pair_sizes = sizes.vec();
        pair_sizes.push_back(2);
        auto options = at::TensorOptions().dtype(at::kFloat).device(at::kPrivateUse1);
        return at::view_as_complex(at::empty(flat, options).view(pair_sizes));
    }

    namespace {

        // conj of self's values: as_real of self with its conjugate bit flipped.
        at::Tensor tt_conj_physical(const at::Tensor& self) {
            if (!holds_complex(self)) {
                return self.clone();
            }
            auto stored = self.alias();
            stored._set_conj(!self.is_conj());
            return as_complex(as_real(stored));
        }

        at::Tensor& tt_conj_physical_out(const at::Tensor& self, at::Tensor& out) {
            return into("conj_physical.out", out, tt_conj_physical(self));
        }

        at::Tensor& tt_conj_physical_(at::Tensor& self) {
            return write_into("conj_physical_", self, tt_conj_physical(self));
        }

        at::Tensor& tt_complex_out(
            const at::Tensor& re, const at::Tensor& im, at::Tensor& out
        ) {
            return into("complex.out", out, from_parts(re, im));
        }

        // abs * (cos angle, sin angle).
        at::Tensor& tt_polar_out(
            const at::Tensor& abs, const at::Tensor& angle, at::Tensor& out
        ) {
            auto r = abs.to(at::kFloat), a = angle.to(at::kFloat);
            return into("polar.out", out, from_parts(r * at::cos(a), r * at::sin(a)));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("view_as_real", &at::native::view_as_real);
        m.impl("view_as_complex", &at::native::view_as_complex);
        m.impl("_conj_physical", &tt_conj_physical);
        m.impl("conj_physical.out", &tt_conj_physical_out);
        m.impl("conj_physical_", &tt_conj_physical_);
        m.impl("complex.out", &tt_complex_out);
        m.impl("polar.out", &tt_polar_out);
    }

}  // namespace ktt
