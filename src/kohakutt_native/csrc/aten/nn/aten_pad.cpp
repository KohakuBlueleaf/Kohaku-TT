// aten padding for tt: constant_pad_nd (cat / narrow), reflection / replication 1-D to
// 3-D (+ backward) from host-built source indices (index_select / scatter_add).
#include <torch/library.h>

#include <algorithm>
#include <array>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // Source position in [0, n) of each of the n + before + after padded positions.
        std::vector<int64_t> source_index(
            int64_t n, int64_t before, int64_t after, bool reflect
        ) {
            std::vector<int64_t> index;
            for (int64_t j = -before; j < n + after; ++j) {
                int64_t s = j;
                if (reflect) {
                    s = s < 0 ? -s : s;
                    s = s >= n ? 2 * (n - 1) - s : s;
                } else {
                    s = std::clamp<int64_t>(s, 0, n - 1);
                }
                index.push_back(s);
            }
            return index;
        }

        // source_index as a cached device constant.
        at::Tensor pad_index(int64_t n, int64_t before, int64_t after, bool reflect) {
            auto key = c10::str("pad|", n, "|", before, "|", after, "|", reflect);
            return device_constant(key, [&] {
                return at::tensor(source_index(n, before, after, reflect), at::kLong);
            });
        }

        // The padded dims, last first: (dim, before, after) from PyTorch's pad list.
        std::vector<std::array<int64_t, 3>> padded_dims(
            const char* op, const at::Tensor& self, at::IntArrayRef pad, bool reflect
        ) {
            std::vector<std::array<int64_t, 3>> dims;
            for (size_t k = 0; k < pad.size() / 2; ++k) {
                const int64_t d = self.dim() - 1 - static_cast<int64_t>(k);
                const int64_t before = pad[2 * k], after = pad[2 * k + 1];
                KTT_REQUIRE(
                    self.size(d) + before + after >= 0, op, "padding crops below size 0"
                );
                if (reflect) {
                    KTT_REQUIRE(
                        before < self.size(d) && after < self.size(d),
                        op,
                        "padding must be smaller than the input size"
                    );
                }
                dims.push_back({d, before, after});
            }
            return dims;
        }

        at::Tensor pad_forward(
            const char* op, const at::Tensor& self, at::IntArrayRef pad, bool reflect
        ) {
            const auto dims = padded_dims(op, self, pad, reflect);
            if (self.numel() == 0) {
                auto sizes = self.sizes().vec();
                for (const auto& [d, before, after] : dims) {
                    sizes[d] += before + after;
                }
                return at::empty(sizes, self.options());
            }
            auto out = self;
            for (const auto& [d, before, after] : dims) {
                if (before == 0 && after == 0) {
                    continue;
                }
                out = (at::index_select(
                    out, d, pad_index(self.size(d), before, after, reflect)
                ));
            }
            return out.is_same(self) ? self.clone() : out;
        }

        // grad of the padded output summed back onto each source position, dim by dim.
        at::Tensor pad_backward(
            const char* op,
            const at::Tensor& grad,
            const at::Tensor& self,
            at::IntArrayRef pad,
            bool reflect
        ) {
            if (self.numel() == 0) {
                return at::zeros(self.sizes(), grad.options());
            }
            auto g = grad;
            for (const auto& [d, before, after] : padded_dims(op, self, pad, reflect)) {
                if (before == 0 && after == 0) {
                    continue;
                }
                const int64_t n = self.size(d);
                auto index = pad_index(n, before, after, reflect);
                std::vector<int64_t> view(g.dim(), 1);
                view[d] = index.numel();
                auto expanded = index.reshape(view).expand(g.sizes());
                auto sizes = g.sizes().vec();
                sizes[d] = n;
                g = at::scatter_add(at::zeros(sizes, g.options()), d, expanded, g);
            }
            return g.is_same(grad) ? grad.clone() : g;
        }

// Forward, out=, backward and grad_input= kernels of one pad op.
#define KTT_PAD(name, reflect)                                              \
    at::Tensor tt_##name(const at::Tensor& self, at::IntArrayRef pad) {     \
        return pad_forward(#name, self, pad, reflect);                      \
    }                                                                       \
    at::Tensor& tt_##name##_out(                                            \
        const at::Tensor& self, at::IntArrayRef pad, at::Tensor& out        \
    ) {                                                                     \
        return into(#name ".out", out, tt_##name(self, pad));               \
    }                                                                       \
    at::Tensor tt_##name##_backward(                                        \
        const at::Tensor& grad, const at::Tensor& self, at::IntArrayRef pad \
    ) {                                                                     \
        return pad_backward(#name "_backward", grad, self, pad, reflect);   \
    }                                                                       \
    at::Tensor& tt_##name##_backward_out(                                   \
        const at::Tensor& grad,                                             \
        const at::Tensor& self,                                             \
        at::IntArrayRef pad,                                                \
        at::Tensor& grad_input                                              \
    ) {                                                                     \
        auto result = tt_##name##_backward(grad, self, pad);                \
        return into(#name "_backward.grad_input", grad_input, result);      \
    }

        KTT_PAD(reflection_pad1d, true)
        KTT_PAD(reflection_pad2d, true)
        KTT_PAD(reflection_pad3d, true)
        KTT_PAD(replication_pad1d, false)
        KTT_PAD(replication_pad2d, false)
        KTT_PAD(replication_pad3d, false)

        // Pads (or crops, for negative pads) the last dims with `value`, by cat /
        // narrow.
        at::Tensor tt_constant_pad_nd(
            const at::Tensor& self, c10::IntArrayRef pad, const at::Scalar& value
        ) {
            KTT_REQUIRE(
                pad.size() % 2 == 0, "constant_pad_nd", "pad must have even length"
            );

            // Zero-size path: empty results (zero-size input or a crop to width 0)
            // directly.
            auto final_sizes = self.sizes().vec();
            for (size_t i = 0; i < pad.size() / 2; ++i) {
                final_sizes[self.dim() - 1 - i] += pad[2 * i] + pad[2 * i + 1];
            }
            if (c10::multiply_integers(final_sizes) == 0) {
                return at::empty(final_sizes, self.options());
            }
            if (self.numel() == 0) {
                return at::full(final_sizes, value, self.options());
            }
            auto out = self;
            for (size_t i = 0; i < pad.size() / 2; ++i) {
                const int64_t d = self.dim() - 1 - static_cast<int64_t>(i);
                int64_t before = pad[2 * i], after = pad[2 * i + 1];
                if (before < 0) {
                    out = out.narrow(d, -before, out.size(d) + before);
                    before = 0;
                }
                if (after < 0) {
                    out = out.narrow(d, 0, out.size(d) + after);
                    after = 0;
                }
                std::vector<at::Tensor> parts;
                auto block = [&](int64_t n) {
                    auto sizes = out.sizes().vec();
                    sizes[d] = n;
                    return at::full(sizes, value, out.options());
                };
                if (before > 0) {
                    parts.push_back(block(before));
                }
                if (out.numel() > 0) {
                    parts.push_back(out);
                }
                if (after > 0) {
                    parts.push_back(block(after));
                }
                out = parts.size() == 1 ? parts[0] : at::cat(parts, d);
            }
            return out;
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("constant_pad_nd", &tt_constant_pad_nd);
#define KTT_REGISTER_PAD(name)                        \
    m.impl(#name, &tt_##name);                        \
    m.impl(#name ".out", &tt_##name##_out);           \
    m.impl(#name "_backward", &tt_##name##_backward); \
    m.impl(#name "_backward.grad_input", &tt_##name##_backward_out);
        KTT_REGISTER_PAD(reflection_pad1d)
        KTT_REGISTER_PAD(reflection_pad2d)
        KTT_REGISTER_PAD(reflection_pad3d)
        KTT_REGISTER_PAD(replication_pad1d)
        KTT_REGISTER_PAD(replication_pad2d)
        KTT_REGISTER_PAD(replication_pad3d)
#undef KTT_REGISTER_PAD
    }

}  // namespace ktt
