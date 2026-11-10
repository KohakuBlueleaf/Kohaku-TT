// aten FFT kernels for tt (_fft_r2c / _fft_c2r / _fft_c2c), fp32: direct DFT matmuls
// up to kDirectMax points, four-step for longer composite n, Bluestein for long primes.
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <cmath>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"
#include "core/torch/constants.hpp"

namespace ktt {
    namespace {

        // Longest axis transformed by one [n, n] DFT matmul (two 4 MB matrices at
        // 1024).
        constexpr int64_t kDirectMax = 1024;

        // fp32 parts of a complex signal.
        struct Signal {
            at::Tensor re, im;
        };

        // [rows, cols] w_r cos / w_r sin of 2 pi ((r * c) mod n) / n as fp32 device
        // constants; w_r = 1, or c2r's Hermitian weights (2, with 1 for the DC and
        // Nyquist rows).
        std::pair<at::Tensor, at::Tensor> phase_table(
            const std::string& tag,
            int64_t rows,
            int64_t cols,
            int64_t n,
            bool hermitian = false
        ) {
            auto weight = [=](int64_t r) {
                if (!hermitian || r == 0 || (n % 2 == 0 && r == n / 2)) {
                    return 1.0;
                }
                return 2.0;
            };
            auto build = [=](bool sine) {
                return [=] {
                    auto t = at::empty({rows, cols}, at::kDouble);
                    auto acc = t.accessor<double, 2>();
                    for (int64_t r = 0; r < rows; ++r) {
                        for (int64_t c = 0; c < cols; ++c) {
                            const double
                                a = (2.0 * M_PI * static_cast<double>((r * c) % n) / n);
                            acc[r][c] = weight(r) * (sine ? std::sin(a) : std::cos(a));
                        }
                    }
                    return t.to(at::kFloat);
                };
            };
            auto key = c10::str("fft.", tag, "|", rows, "|", cols, "|", n);
            return {
                device_constant(key + "|cos", build(false)),
                device_constant(key + "|sin", build(true))
            };
        }

        // x [..., k] @ m [k, j] as one 2-D matmul.
        at::Tensor apply(const at::Tensor& x, const at::Tensor& m) {
            const FidelityFloor exact(3);  // fp32 DFT matrices
            auto sizes = x.sizes().vec();
            sizes.back() = m.size(1);
            return at::mm(x.reshape({-1, x.size(-1)}), m).reshape(sizes);
        }

        // (a + i b) * (c + i s * d) for a sign s, elementwise.
        Signal twiddle(
            const Signal& x, const at::Tensor& c, const at::Tensor& d, double s
        ) {
            return {x.re * c - x.im * d * s, x.im * c + x.re * d * s};
        }

        Signal dft(const Signal& x, double sign);

        // Direct DFT along the last dim: X_k = sum_j x_j e^{sign i 2 pi j k / n}.
        Signal direct(const Signal& x, double sign) {
            const int64_t n = x.re.size(-1);
            auto [c, s] = phase_table("dft", n, n, n);
            return {
                apply(x.re, c) - apply(x.im, s) * sign,
                apply(x.im, c) + apply(x.re, s) * sign
            };
        }

        // Four-step DFT for n = n1 * n2: DFTs over j1, twiddles e^{sign i 2 pi j2 k1 /
        // n}, DFTs over j2; the output index is k1 + n1 * k2.
        Signal four_step(const Signal& x, double sign, int64_t n1, int64_t n2) {
            auto lead = x.re.sizes().vec();
            lead.pop_back();
            auto grid = lead;
            grid.insert(grid.end(), {n1, n2});
            auto split = [&](const at::Tensor& t) {
                return t.reshape(grid).transpose(-1, -2);
            };
            auto b = dft({split(x.re), split(x.im)}, sign);  // [..., j2, k1]
            auto [c, s] = phase_table("twiddle", n2, n1, n1 * n2);
            b = twiddle(b, c, s, sign);
            auto stage = dft(
                {b.re.transpose(-1, -2), b.im.transpose(-1, -2)}, sign
            );  // [..., k1, k2]
            auto out = lead;
            out.push_back(n1 * n2);
            auto join = [&](const at::Tensor& t) {
                return t.transpose(-1, -2).reshape(out);
            };
            return {join(stage.re), join(stage.im)};
        }

        // The largest divisor of n at most sqrt(n) (1 for primes).
        int64_t inner_factor(int64_t n) {
            for (
                auto f = static_cast<int64_t>(std::sqrt(static_cast<double>(n))); f > 1;
                --f
            ) {
                if (n % f == 0) {
                    return f;
                }
            }
            return 1;
        }

        // Bluestein: X_k = w_k sum_j (x_j w_j) conj(w_{k-j}), w_j = e^{sign i pi j^2 /
        // n}, the convolution as a power-of-two circular one through dft().
        Signal bluestein(const Signal& x, double sign) {
            const int64_t n = x.re.size(-1);
            int64_t m = 1;
            while (m < 2 * n - 1) {
                m <<= 1;
            }
            auto chirp = [=](bool sine) {
                return [=] {
                    auto t = at::empty({n}, at::kDouble);
                    auto acc = t.accessor<double, 1>();
                    for (int64_t j = 0; j < n; ++j) {
                        const double
                            a = (M_PI * static_cast<double>((j * j) % (2 * n)) / n);
                        acc[j] = sine ? std::sin(a) : std::cos(a);
                    }
                    return t.to(at::kFloat);
                };
            };
            auto key = c10::str("fft.chirp|", n);
            auto wc = device_constant(key + "|cos", chirp(false));
            auto ws = device_constant(key + "|sin", chirp(true));

            // a = x * w, zero-padded to m; b = conj(w) at lags 0..n-1 and m-n+1..m-1.
            auto a = twiddle(x, wc, ws, sign);
            auto pad = [&](const at::Tensor& t) {
                return at::constant_pad_nd(t, {0, m - n}, 0);
            };
            a = {pad(a.re), pad(a.im)};
            auto tail = [&](const at::Tensor& t) {
                return t.narrow(0, 1, n - 1).flip({0});
            };
            auto gap = at::zeros({m - 2 * n + 1}, wc.options());
            Signal b{
                at::cat({wc, gap, tail(wc)}),
                at::cat({-ws * sign, gap, -tail(ws) * sign})
            };

            auto fa = dft(a, -1.0), fb = dft({b.re, b.im}, -1.0);
            Signal prod{fa.re * fb.re - fa.im * fb.im, fa.re * fb.im + fa.im * fb.re};
            auto y = dft(prod, 1.0);
            Signal conv{y.re.narrow(-1, 0, n) / m, y.im.narrow(-1, 0, n) / m};
            return twiddle(conv, wc, ws, sign);
        }

        // Unnormalized DFT along the last dim.
        Signal dft(const Signal& x, double sign) {
            const int64_t n = x.re.size(-1);
            if (n <= kDirectMax) {
                return direct(x, sign);
            }
            const int64_t f = inner_factor(n);
            return f == 1 ? bluestein(x, sign) : four_step(x, sign, f, n / f);
        }

        // dft() along `dim` of a signal.
        Signal dft_dim(const Signal& x, int64_t dim, double sign) {
            auto moved = dft({x.re.movedim(dim, -1), x.im.movedim(dim, -1)}, sign);
            return {moved.re.movedim(-1, dim), moved.im.movedim(-1, dim)};
        }

        // PyTorch's normalization codes: 0 none, 1 by 1 / sqrt(n), 2 by 1 / n.
        double norm_scale(int64_t normalization, int64_t n) {
            if (normalization == 1) {
                return 1.0 / std::sqrt(static_cast<double>(n));
            }
            return normalization == 2 ? 1.0 / static_cast<double>(n) : 1.0;
        }

        std::vector<int64_t> wrapped(c10::IntArrayRef dims, int64_t rank) {
            std::vector<int64_t> out;
            for (auto d : dims) {
                out.push_back(c10::maybe_wrap_dim(d, rank));
            }
            return out;
        }

        Signal signal_of(const at::Tensor& t) {
            auto [re, im] = parts(t);
            return {re.contiguous(), im.contiguous()};
        }

        at::Tensor tt_fft_c2c(
            const at::Tensor& self,
            c10::SymIntArrayRef dim,
            int64_t normalization,
            bool forward
        ) {
            auto x = signal_of(self);
            auto dims = wrapped(C10_AS_INTARRAYREF_SLOW(dim), self.dim());
            int64_t n = 1;
            for (auto d : dims) {
                x = dft_dim(x, d, forward ? -1.0 : 1.0);
                n *= self.size(d);
            }
            const double scale = norm_scale(normalization, n);
            return from_parts(x.re * scale, x.im * scale);
        }

        // Real-to-complex: the last dim's half spectrum (onesided) or full, then c2c
        // over the other dims.
        at::Tensor tt_fft_r2c(
            const at::Tensor& self,
            at::IntArrayRef dim,
            int64_t normalization,
            bool onesided
        ) {
            auto dims = wrapped(dim, self.dim());
            const int64_t last = dims.back(), n = self.size(last);
            auto x = self.to(at::kFloat).movedim(last, -1).contiguous();
            const int64_t keep = onesided ? n / 2 + 1 : n;
            Signal y;
            if (n <= kDirectMax) {
                auto [c, s] = phase_table("r2c", n, keep, n);
                y = {apply(x, c), -apply(x, s)};
            } else {
                auto full = dft({x, at::zeros_like(x)}, -1.0);
                y = {full.re.narrow(-1, 0, keep), full.im.narrow(-1, 0, keep)};
            }
            y = {y.re.movedim(-1, last), y.im.movedim(-1, last)};
            int64_t total = n;
            for (size_t k = 0; k + 1 < dims.size(); ++k) {
                y = dft_dim(y, dims[k], -1.0);
                total *= self.size(dims[k]);
            }
            const double scale = norm_scale(normalization, total);
            return from_parts(y.re * scale, y.im * scale);
        }

        // Complex-to-real: inverse c2c over all but the last dim, then the last dim's
        // Hermitian inverse to `n` real points: sum_k w_k (Re X_k cos - Im X_k sin),
        // w_k = 2 except the DC and Nyquist terms.
        at::Tensor tt_fft_c2r(
            const at::Tensor& self,
            at::IntArrayRef dim,
            int64_t normalization,
            c10::SymInt last_dim_size
        ) {
            auto dims = wrapped(dim, self.dim());
            const int64_t last = dims.back(), n = last_dim_size.expect_int();
            auto x = signal_of(self);
            int64_t total = n;
            for (size_t k = 0; k + 1 < dims.size(); ++k) {
                x = dft_dim(x, dims[k], 1.0);
                total *= self.size(dims[k]);
            }
            const int64_t h = std::min<int64_t>(self.size(last), n / 2 + 1);
            auto re = x.re.movedim(last, -1).narrow(-1, 0, h).contiguous();
            auto im = x.im.movedim(last, -1).narrow(-1, 0, h).contiguous();
            at::Tensor out;
            if (n <= kDirectMax) {
                const bool hermitian = true;
                auto [c, s] = phase_table("c2r", h, n, n, hermitian);
                out = apply(re, c) - apply(im, s);
            } else {
                // Full Hermitian spectrum: X_{n-k} = conj(X_k), then an inverse c2c.
                const int64_t rest = n - h;
                auto mirror = [&](const at::Tensor& t) {
                    return t.narrow(-1, 1, rest).flip({-1});
                };
                Signal full{
                    at::cat({re, mirror(re)}, -1), at::cat({im, -mirror(im)}, -1)
                };
                out = dft(full, 1.0).re;
            }
            return (
                (out * norm_scale(normalization, total)).movedim(-1, last).contiguous()
            );
        }

        at::Tensor& tt_fft_c2c_out(
            const at::Tensor& self,
            c10::SymIntArrayRef dim,
            int64_t normalization,
            bool forward,
            at::Tensor& out
        ) {
            return (
                into("_fft_c2c.out", out, tt_fft_c2c(self, dim, normalization, forward))
            );
        }

        at::Tensor& tt_fft_r2c_out(
            const at::Tensor& self,
            at::IntArrayRef dim,
            int64_t normalization,
            bool onesided,
            at::Tensor& out
        ) {
            return (into(
                "_fft_r2c.out", out, tt_fft_r2c(self, dim, normalization, onesided)
            ));
        }

        at::Tensor& tt_fft_c2r_out(
            const at::Tensor& self,
            at::IntArrayRef dim,
            int64_t normalization,
            c10::SymInt last_dim_size,
            at::Tensor& out
        ) {
            auto result = (tt_fft_c2r(
                self, dim, normalization, std::move(last_dim_size)
            ));
            return into("_fft_c2r.out", out, result);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("_fft_c2c", &tt_fft_c2c);
        m.impl("_fft_c2c.out", &tt_fft_c2c_out);
        m.impl("_fft_r2c", &tt_fft_r2c);
        m.impl("_fft_r2c.out", &tt_fft_r2c_out);
        m.impl("_fft_c2r", &tt_fft_c2r);
        m.impl("_fft_c2r.out", &tt_fft_c2r_out);
    }

}  // namespace ktt
