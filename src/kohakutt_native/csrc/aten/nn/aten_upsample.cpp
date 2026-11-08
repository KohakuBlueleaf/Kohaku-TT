// aten upsampling for tt (nearest, nearest-exact, linear / bilinear; + backward): the
// interpolation is linear, so out = A_h @ x @ A_w^T with host-built [out, in] matrices.

#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        enum class Mode { Nearest, NearestExact, Linear, Cubic, LinearAA };

        // PyTorch's bicubic weights (a = -0.75) of the taps i - 1 .. i + 2 at offset t.
        std::array<double, 4> cubic_weights(double t) {
            constexpr double a = -0.75;
            auto near = [](double x) {
                return ((a + 2) * x - (a + 3)) * x * x + 1;
            };
            auto far = [](double x) {
                return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
            };
            return {far(t + 1), near(t), near(1 - t), far(2 - t)};
        }

        // PyTorch's source-to-destination scale: 1 / scale when given, else in / out
        // (linear with align_corners: (in - 1) / (out - 1)).
        double area_scale(
            int64_t in, int64_t out, std::optional<double> scale, bool align, Mode mode
        ) {
            if ((mode == Mode::Linear || mode == Mode::Cubic ||
                 mode == Mode::LinearAA) &&
                align) {
                return (
                    out > 1 ? static_cast<double>(in - 1) / static_cast<double>(out - 1)
                            : 0.0
                );
            }
            if (scale && *scale > 0) {
                return 1.0 / *scale;
            }
            return static_cast<double>(in) / static_cast<double>(out);
        }

        // The [out, in] interpolation matrix along one dim (fp32 device constant).
        at::Tensor interpolation(
            int64_t out,
            int64_t in,
            std::optional<double> scale,
            bool align,
            Mode mode,
            const at::Device&
        ) {
            const double s = area_scale(in, out, scale, align, mode);
            auto key = (
                c10::str(
                    "upsample|",
                    out,
                    "|",
                    in,
                    "|",
                    s,
                    "|",
                    align,
                    "|",
                    static_cast<int>(mode)
                )
            );
            return device_constant(key, [&] {
                auto a = at::zeros({out, in}, at::kFloat);
                auto acc = a.accessor<float, 2>();
                for (int64_t o = 0; o < out; ++o) {
                    if (mode == Mode::Nearest || mode == Mode::NearestExact) {
                        const double offset = mode == Mode::NearestExact ? 0.5 : 0.0;
                        const auto src = (
                            static_cast<int64_t>(std::floor((o + offset) * s))
                        );
                        acc[o][std::min(src, in - 1)] = 1.0f;
                        continue;
                    }
                    if (mode == Mode::LinearAA) {
                        // Triangle filter widened by the downscale factor, normalized.
                        const double support = s >= 1.0 ? s : 1.0;
                        const double inv = s >= 1.0 ? 1.0 / s : 1.0;
                        const double center = s * (o + 0.5);
                        const auto lo = (
                            std::max<int64_t>(
                                static_cast<int64_t>(center - support + 0.5), 0
                            )
                        );
                        const auto hi = (
                            std::min<int64_t>(
                                static_cast<int64_t>(center + support + 0.5), in
                            )
                        );
                        double total = 0.0;
                        std::vector<double> w;
                        for (int64_t j = lo; j < hi; ++j) {
                            const double x = std::abs((j - center + 0.5) * inv);
                            w.push_back(x < 1.0 ? 1.0 - x : 0.0);
                            total += w.back();
                        }
                        for (int64_t j = lo; j < hi; ++j) {
                            acc[o][j] += (
                                static_cast<float>(
                                    total != 0.0 ? w[j - lo] / total : 0.0
                                )
                            );
                        }
                        continue;
                    }
                    if (mode == Mode::Cubic) {
                        // Taps clamped to the edge; the source index unclamped.
                        const double src = align ? o * s : (o + 0.5) * s - 0.5;
                        const auto i = static_cast<int64_t>(std::floor(src));
                        const auto w = cubic_weights(src - static_cast<double>(i));
                        for (int64_t k = 0; k < 4; ++k) {
                            const int64_t tap = (
                                std::clamp<int64_t>(i - 1 + k, 0, in - 1)
                            );
                            acc[o][tap] += static_cast<float>(w[k]);
                        }
                        continue;
                    }
                    double src = align ? o * s : (o + 0.5) * s - 0.5;
                    src = std::max(src, 0.0);
                    const auto i0 = (
                        std::min(static_cast<int64_t>(std::floor(src)), in - 1)
                    );
                    const int64_t i1 = std::min(i0 + 1, in - 1);
                    const auto lambda = (
                        static_cast<float>(src - static_cast<double>(i0))
                    );
                    acc[o][i0] += 1.0f - lambda;
                    acc[o][i1] += lambda;
                }
                return a;
            });
        }

        // x [..., H, W] (or [..., W] for 1-D) resized to `size`, or its gradient
        // (backward: transposed matrices), in fp32.
        at::Tensor resize(
            const at::Tensor& x,
            at::IntArrayRef in_size,
            at::IntArrayRef out_size,
            std::vector<std::optional<double>> scales,
            bool align,
            Mode mode,
            bool backward
        ) {
            // Interpolation matrices copy and blend values: fidelity floor HiFi4 (a
            // bf16 value's 8 mantissa bits need the second slice of the operand).
            const FidelityFloor exact(3);
            const auto dims = static_cast<int64_t>(out_size.size());
            // Nearest forward selects values: exact copies (exact_select), in x's
            // dtype between dims.
            const bool select = !backward && mode != Mode::Linear;
            auto y = select ? x : x.to(at::kFloat);
            auto right = [&](const at::Tensor& t, const at::Tensor& m) {
                return (
                    select ? exact_select(t, m, true).to(x.scalar_type())
                           : at::matmul(t, m)
                );
            };
            auto left = [&](const at::Tensor& m, const at::Tensor& t) {
                return (
                    select ? exact_select(m, t, false).to(x.scalar_type())
                           : at::matmul(m, t)
                );
            };
            for (int64_t k = 0; k < dims; ++k) {
                auto a = (
                    interpolation(
                        out_size[k], in_size[k], scales[k], align, mode, x.device()
                    )
                );
                // The last dim: y @ A^T; the second last: A @ y (no transposes of y);
                // an earlier dim: moved last, multiplied, moved back.
                const int64_t d = y.dim() - dims + k;
                if (d == y.dim() - 1) {
                    y = right(y, backward ? a : a.t());
                } else if (d == y.dim() - 2) {
                    y = left(backward ? a.t() : a, y);
                } else {
                    y = (
                        right(y.transpose(d, -1), backward ? a : a.t()).transpose(d, -1)
                    );
                }
            }
            return y.contiguous().to(x.scalar_type());
        }

        // The [r n, n] matrix with a 1 at (i r + phase, i): spreads n rows to every
        // r-th row of r n, starting at `phase` (fp32 device constant).
        at::Tensor spread(int64_t n, int64_t r, int64_t phase) {
            auto key = c10::str("spread|", n, "|", r, "|", phase);
            return device_constant(key, [&] {
                auto m = at::zeros({r * n, n}, at::kFloat);
                auto acc = m.accessor<float, 2>();
                for (int64_t i = 0; i < n; ++i) {
                    acc[i * r + phase][i] = 1.0f;
                }
                return m;
            });
        }

        // pixel_shuffle: [*, C r^2, H, W] -> [*, C, r H, r W], out[c, h r + i, w r + j]
        // = x[c r^2 + i r + j, h, w]; the sum over (i, j) of spread rows and columns
        // of each phase's channels (exact copies).
        // True unless KTT_OWN_UPSAMPLE=0 (matrix path for nearest resizes and pixel
        // shuffle).
        bool own_upsample() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_OWN_UPSAMPLE");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

        at::Tensor tt_pixel_shuffle(const at::Tensor& self, int64_t r) {
            const char* op = "pixel_shuffle";
            KTT_REQUIRE(self.dim() >= 3 && r > 0, op, "needs [*, C r^2, H, W], r > 0");
            const int64_t c = self.size(-3), h = self.size(-2), w = self.size(-1);
            KTT_REQUIRE(c % (r * r) == 0, op, "channels not divisible by r^2");
            const auto type = self.scalar_type();
            if (own_upsample() && 32 % r == 0 && self.numel() > 0 &&
                (type == at::kBFloat16 || type == at::kFloat)) {
                auto out_sizes = self.sizes().vec();
                out_sizes[out_sizes.size() - 3] = c / (r * r);
                out_sizes[out_sizes.size() - 2] = h * r;
                out_sizes[out_sizes.size() - 1] = w * r;
                return (
                    wrap_output(
                        bridge::pixel_shuffle(dense(op, self), r), out_sizes, type
                    )
                );
            }
            auto sizes = self.sizes().vec();
            sizes.erase(sizes.end() - 3, sizes.end());
            auto phases = sizes;
            phases.insert(phases.end(), {c / (r * r), r, r, h, w});
            auto x = self.reshape(phases);
            at::Tensor out;
            for (int64_t i = 0; i < r; ++i) {
                for (int64_t j = 0; j < r; ++j) {
                    auto part = x.select(-4, i).select(-3, j);  // [*, C, H, W]
                    auto cols = exact_select(part, spread(w, r, j).t(), true);
                    auto term = exact_select(spread(h, r, i), cols, false);
                    out = out.defined() ? out + term : term;
                }
            }
            return out.to(self.scalar_type());
        }

        // pixel_unshuffle: the inverse, [*, C, r H, r W] -> [*, C r^2, H, W].
        at::Tensor tt_pixel_unshuffle(const at::Tensor& self, int64_t r) {
            const char* op = "pixel_unshuffle";
            KTT_REQUIRE(self.dim() >= 3 && r > 0, op, "needs [*, C, r H, r W], r > 0");
            const int64_t c = self.size(-3), h = self.size(-2), w = self.size(-1);
            KTT_REQUIRE(
                h % r == 0 && w % r == 0, op, "height / width not divisible by r"
            );
            std::vector<at::Tensor> parts;
            for (int64_t i = 0; i < r; ++i) {
                for (int64_t j = 0; j < r; ++j) {
                    auto rows = exact_select(spread(h / r, r, i).t(), self, false);
                    parts.push_back(exact_select(rows, spread(w / r, r, j), true)
                                        .to(self.scalar_type()));
                }
            }
            // [*, C, r r, H, W] -> [*, C r^2, H, W].
            auto stacked = at::stack(parts, -3);
            auto sizes = self.sizes().vec();
            sizes[sizes.size() - 3] = c * r * r;
            sizes[sizes.size() - 2] = h / r;
            sizes[sizes.size() - 1] = w / r;
            return stacked.reshape(sizes);
        }

        std::vector<int64_t> spatial(const at::Tensor& t, int64_t dims) {
            return t.sizes().slice(t.dim() - dims).vec();
        }

        at::Tensor tt_upsample_nearest1d(
            const at::Tensor& self, at::IntArrayRef size, std::optional<double> scale
        ) {
            return resize(
                self, spatial(self, 1), size, {scale}, false, Mode::Nearest, false
            );
        }

        // The integer factor out / in when nearest resizing is an exact repeat by a
        // divisor of 32 (scale, when given, equal to it); else 0.
        int64_t repeat_factor(int64_t in, int64_t out, std::optional<double> scale) {
            if (in <= 0 || out % in != 0) {
                return 0;
            }
            const int64_t k = out / in;
            if (32 % k != 0 ||
                (scale && *scale > 0 && *scale != static_cast<double>(k))) {
                return 0;
            }
            return k;
        }

        at::Tensor tt_upsample_nearest2d(
            const at::Tensor& self,
            at::IntArrayRef size,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            const auto type = self.scalar_type();
            if (own_upsample() && self.dim() >= 2 && size.size() == 2 &&
                (type == at::kBFloat16 || type == at::kFloat) && self.numel() > 0) {
                const int64_t kh = repeat_factor(self.size(-2), size[0], scale_h);
                const int64_t kw = repeat_factor(self.size(-1), size[1], scale_w);
                if (kh > 0 && kw > 0) {
                    const char* op = "upsample_nearest2d";
                    auto sizes = self.sizes().vec();
                    sizes[sizes.size() - 2] = size[0];
                    sizes[sizes.size() - 1] = size[1];
                    auto h = bridge::upsample_nearest(dense(op, self), kh, kw);
                    return wrap_output(h, sizes, type);
                }
            }
            return resize(
                self,
                spatial(self, 2),
                size,
                {scale_h, scale_w},
                false,
                Mode::Nearest,
                false
            );
        }

        at::Tensor tt_upsample_nearest_exact2d(
            const at::Tensor& self,
            at::IntArrayRef size,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            auto s = std::vector<std::optional<double>>{scale_h, scale_w};
            return (
                resize(
                    self, spatial(self, 2), size, s, false, Mode::NearestExact, false
                )
            );
        }

        at::Tensor tt_upsample_bilinear2d(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            auto s = std::vector<std::optional<double>>{scale_h, scale_w};
            return resize(self, spatial(self, 2), size, s, align, Mode::Linear, false);
        }

        at::Tensor tt_upsample_bicubic2d(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            auto s = std::vector<std::optional<double>>{scale_h, scale_w};
            return resize(self, spatial(self, 2), size, s, align, Mode::Cubic, false);
        }

        at::Tensor& tt_upsample_bicubic2d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& out
        ) {
            auto result = tt_upsample_bicubic2d(self, size, align, scale_h, scale_w);
            return into("upsample_bicubic2d.out", out, result);
        }

        at::Tensor tt_upsample_bicubic2d_backward(
            const at::Tensor& grad,
            at::IntArrayRef out_size,
            at::IntArrayRef in_size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            auto in = in_size.slice(in_size.size() - 2);
            auto s = std::vector<std::optional<double>>{scale_h, scale_w};
            return (
                resize(grad, in, out_size, s, align, Mode::Cubic, true).reshape(in_size)
            );
        }

        at::Tensor tt_upsample_bilinear2d_aa(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            auto s = std::vector<std::optional<double>>{scale_h, scale_w};
            return (
                resize(self, spatial(self, 2), size, s, align, Mode::LinearAA, false)
            );
        }

        at::Tensor& tt_upsample_bilinear2d_aa_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& out
        ) {
            auto result = (
                tt_upsample_bilinear2d_aa(self, size, align, scale_h, scale_w)
            );
            return into("_upsample_bilinear2d_aa.out", out, result);
        }

        at::Tensor tt_upsample_bilinear2d_aa_backward(
            const at::Tensor& grad,
            at::IntArrayRef out_size,
            at::IntArrayRef in_size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            auto in = in_size.slice(in_size.size() - 2);
            auto s = std::vector<std::optional<double>>{scale_h, scale_w};
            return (
                resize(grad, in, out_size, s, align, Mode::LinearAA, true)
                .reshape(in_size)
            );
        }

        at::Tensor tt_upsample_linear1d(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale
        ) {
            return resize(
                self, spatial(self, 1), size, {scale}, align, Mode::Linear, false
            );
        }

        at::Tensor tt_upsample_nearest1d_backward(
            const at::Tensor& grad,
            at::IntArrayRef out_size,
            at::IntArrayRef in_size,
            std::optional<double> scale
        ) {
            auto in = in_size.slice(in_size.size() - 1);
            return resize(grad, in, out_size, {scale}, false, Mode::Nearest, true)
                .reshape(in_size);
        }

        at::Tensor tt_upsample_nearest2d_backward(
            const at::Tensor& grad,
            at::IntArrayRef out_size,
            at::IntArrayRef in_size,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            auto in = in_size.slice(in_size.size() - 2);
            auto s = std::vector<std::optional<double>>{scale_h, scale_w};
            return (
                resize(grad, in, out_size, s, false, Mode::Nearest, true)
                .reshape(in_size)
            );
        }

        at::Tensor tt_upsample_bilinear2d_backward(
            const at::Tensor& grad,
            at::IntArrayRef out_size,
            at::IntArrayRef in_size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w
        ) {
            auto in = in_size.slice(in_size.size() - 2);
            auto s = std::vector<std::optional<double>>{scale_h, scale_w};
            return (
                resize(grad, in, out_size, s, align, Mode::Linear, true)
                .reshape(in_size)
            );
        }

        at::Tensor& tt_upsample_nearest1d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            std::optional<double> scale,
            at::Tensor& out
        ) {
            auto result = tt_upsample_nearest1d(self, size, scale);
            return into("upsample_nearest1d.out", out, result);
        }

        at::Tensor& tt_upsample_nearest2d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& out
        ) {
            auto result = tt_upsample_nearest2d(self, size, scale_h, scale_w);
            return into("upsample_nearest2d.out", out, result);
        }

        at::Tensor& tt_upsample_nearest_exact2d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& out
        ) {
            auto result = tt_upsample_nearest_exact2d(self, size, scale_h, scale_w);
            return into("_upsample_nearest_exact2d.out", out, result);
        }

        at::Tensor& tt_upsample_bilinear2d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& out
        ) {
            auto result = tt_upsample_bilinear2d(self, size, align, scale_h, scale_w);
            return into("upsample_bilinear2d.out", out, result);
        }

        at::Tensor& tt_upsample_linear1d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale,
            at::Tensor& out
        ) {
            auto result = tt_upsample_linear1d(self, size, align, scale);
            return into("upsample_linear1d.out", out, result);
        }

        at::Tensor& tt_upsample_nearest2d_backward_out(
            const at::Tensor& grad,
            at::IntArrayRef out_size,
            at::IntArrayRef in_size,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& grad_input
        ) {
            auto result = (
                tt_upsample_nearest2d_backward(
                    grad, out_size, in_size, scale_h, scale_w
                )
            );
            return into("upsample_nearest2d_backward.grad_input", grad_input, result);
        }

        at::Tensor& tt_upsample_bilinear2d_backward_out(
            const at::Tensor& grad,
            at::IntArrayRef out_size,
            at::IntArrayRef in_size,
            bool align,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& grad_input
        ) {
            auto result = (
                tt_upsample_bilinear2d_backward(
                    grad, out_size, in_size, align, scale_h, scale_w
                )
            );
            return into("upsample_bilinear2d_backward.grad_input", grad_input, result);
        }

        // 1-D / 3-D variants (resize is rank-generic).
        at::Tensor& tt_upsample_nearest3d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            std::optional<double> scale_d,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& out
        ) {
            auto s = std::vector<std::optional<double>>{scale_d, scale_h, scale_w};
            auto result = (
                resize(self, spatial(self, 3), size, s, false, Mode::Nearest, false)
            );
            return into("upsample_nearest3d.out", out, result);
        }

        at::Tensor& tt_upsample_nearest_exact1d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            std::optional<double> scale,
            at::Tensor& out
        ) {
            auto result = resize(
                self, spatial(self, 1), size, {scale}, false, Mode::NearestExact, false
            );
            return into("_upsample_nearest_exact1d.out", out, result);
        }

        at::Tensor& tt_upsample_nearest_exact3d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            std::optional<double> scale_d,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& out
        ) {
            auto s = std::vector<std::optional<double>>{scale_d, scale_h, scale_w};
            auto result = (
                resize(
                    self, spatial(self, 3), size, s, false, Mode::NearestExact, false
                )
            );
            return into("_upsample_nearest_exact3d.out", out, result);
        }

        at::Tensor& tt_upsample_trilinear3d_out(
            const at::Tensor& self,
            at::IntArrayRef size,
            bool align,
            std::optional<double> scale_d,
            std::optional<double> scale_h,
            std::optional<double> scale_w,
            at::Tensor& out
        ) {
            auto s = std::vector<std::optional<double>>{scale_d, scale_h, scale_w};
            auto result = (
                resize(self, spatial(self, 3), size, s, align, Mode::Linear, false)
            );
            return into("upsample_trilinear3d.out", out, result);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("upsample_nearest3d.out", &tt_upsample_nearest3d_out);
        m.impl("_upsample_nearest_exact1d.out", &tt_upsample_nearest_exact1d_out);
        m.impl("_upsample_nearest_exact3d.out", &tt_upsample_nearest_exact3d_out);
        m.impl("upsample_trilinear3d.out", &tt_upsample_trilinear3d_out);
        m.impl("upsample_nearest1d.out", &tt_upsample_nearest1d_out);
        m.impl("upsample_nearest2d.out", &tt_upsample_nearest2d_out);
        m.impl("_upsample_nearest_exact2d.out", &tt_upsample_nearest_exact2d_out);
        m.impl("upsample_bilinear2d.out", &tt_upsample_bilinear2d_out);
        m.impl("upsample_linear1d.out", &tt_upsample_linear1d_out);
        m.impl(
            "upsample_nearest2d_backward.grad_input",
            &tt_upsample_nearest2d_backward_out
        );
        m.impl(
            "upsample_bilinear2d_backward.grad_input",
            &tt_upsample_bilinear2d_backward_out
        );
        m.impl("pixel_shuffle", &tt_pixel_shuffle);
        m.impl("pixel_unshuffle", &tt_pixel_unshuffle);
        m.impl("upsample_nearest1d", &tt_upsample_nearest1d);
        m.impl("upsample_nearest2d", &tt_upsample_nearest2d);
        m.impl("_upsample_nearest_exact2d", &tt_upsample_nearest_exact2d);
        m.impl("upsample_bilinear2d", &tt_upsample_bilinear2d);
        m.impl("upsample_bicubic2d", &tt_upsample_bicubic2d);
        m.impl("_upsample_bilinear2d_aa", &tt_upsample_bilinear2d_aa);
        m.impl("_upsample_bilinear2d_aa.out", &tt_upsample_bilinear2d_aa_out);
        m.impl("_upsample_bilinear2d_aa_backward", &tt_upsample_bilinear2d_aa_backward);
        m.impl("upsample_bicubic2d.out", &tt_upsample_bicubic2d_out);
        m.impl("upsample_bicubic2d_backward", &tt_upsample_bicubic2d_backward);
        m.impl("upsample_linear1d", &tt_upsample_linear1d);
        m.impl("upsample_nearest1d_backward", &tt_upsample_nearest1d_backward);
        m.impl("upsample_nearest2d_backward", &tt_upsample_nearest2d_backward);
        m.impl("upsample_bilinear2d_backward", &tt_upsample_bilinear2d_backward);
    }

}  // namespace ktt
