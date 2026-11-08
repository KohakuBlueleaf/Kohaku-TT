// aten pooling kernels for tt: max / avg pool 2-D (per-tap strided slices), max pool
// backward (scatter_add), adaptive avg pool and backward (averaging-matrix matmuls).

#include <c10/util/accumulate.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <limits>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // The [out, in] averaging matrix of adaptive pooling along one dim: row o is
        // 1 / len over PyTorch's window [floor(o * in / out), ceil((o + 1) * in /
        // out)).
        at::Tensor averaging(int64_t out, int64_t in, const at::TensorOptions&) {
            return device_constant(c10::str("pool.avg|", out, "|", in), [&] {
                auto a = at::zeros({out, in}, at::kFloat);
                auto acc = a.accessor<float, 2>();
                for (int64_t o = 0; o < out; ++o) {
                    const int64_t begin = (o * in) / out;
                    const int64_t end = ((o + 1) * in + out - 1) / out;
                    for (int64_t i = begin; i < end; ++i) {
                        acc[o][i] = 1.0f / static_cast<float>(end - begin);
                    }
                }
                return a;
            });
        }

        void check_adaptive(
            const char* op, const at::Tensor& input, int64_t oh, int64_t ow
        ) {
            KTT_REQUIRE(
                input.dim() == 3 || input.dim() == 4, op, "input must be 3-D or 4-D"
            );
            KTT_REQUIRE(oh > 0 && ow > 0, op, "output size must be positive");
        }

        // A_h @ x @ A_w^T over the last two dims, in fp32.
        at::Tensor tt_adaptive_avg_pool2d(
            const at::Tensor& input, at::IntArrayRef output_size
        ) {
            const char* op = "_adaptive_avg_pool2d";
            KTT_REQUIRE(output_size.size() == 2, op, "output_size must have 2 values");
            const int64_t oh = output_size[0], ow = output_size[1];
            check_adaptive(op, input, oh, ow);
            auto ah = averaging(oh, input.size(-2), input.options());
            auto aw = averaging(ow, input.size(-1), input.options());
            auto x = input.to(at::kFloat);
            const FidelityFloor exact(3);  // averaging matrices on values
            auto out = at::matmul(at::matmul(ah, x), aw.t());
            return out.to(input.scalar_type());
        }

        // 3-D: A_d, A_h, A_w applied to the last three dims of [..., D, H, W] in fp32.
        at::Tensor tt_adaptive_avg_pool3d(
            const at::Tensor& input, at::IntArrayRef output_size
        ) {
            const char* op = "_adaptive_avg_pool3d";
            KTT_REQUIRE(output_size.size() == 3, op, "output_size must have 3 values");
            KTT_REQUIRE(
                input.dim() == 4 || input.dim() == 5, op, "input must be 4-D or 5-D"
            );
            const int64_t od = output_size[0], oh = output_size[1], ow = output_size[2];
            KTT_REQUIRE(od > 0 && oh > 0 && ow > 0, op, "output size must be positive");
            const int64_t d = input.size(-3);
            auto ad = averaging(od, d, input.options());
            auto ah = averaging(oh, input.size(-2), input.options());
            auto aw = averaging(ow, input.size(-1), input.options());
            const FidelityFloor exact(3);
            auto x = at::matmul(at::matmul(ah, input.to(at::kFloat)), aw.t());
            auto lead = input.sizes().slice(0, input.dim() - 3).vec();
            auto flat = lead;
            flat.insert(flat.end(), {d, oh * ow});
            auto out = at::matmul(ad, x.reshape(flat));
            auto sizes = lead;
            sizes.insert(sizes.end(), {od, oh, ow});
            return out.reshape(sizes).to(input.scalar_type());
        }

        at::Tensor tt_adaptive_avg_pool3d_backward(
            const at::Tensor& grad, const at::Tensor& input
        ) {
            const int64_t od = grad.size(-3), oh = grad.size(-2), ow = grad.size(-1);
            const int64_t d = input.size(-3), h = input.size(-2), w = input.size(-1);
            auto ad = averaging(od, d, input.options());
            auto ah = averaging(oh, h, input.options());
            auto aw = averaging(ow, w, input.options());
            const FidelityFloor exact(3);
            auto g = at::matmul(at::matmul(ah.t(), grad.to(at::kFloat)), aw);
            auto lead = input.sizes().slice(0, input.dim() - 3).vec();
            auto flat = lead;
            flat.insert(flat.end(), {od, h * w});
            auto out = at::matmul(ad.t(), g.reshape(flat));
            return out.reshape(input.sizes()).to(input.scalar_type());
        }

        // A_h^T @ grad @ A_w: each output gradient spread evenly over its window.
        at::Tensor tt_adaptive_avg_pool2d_backward(
            const at::Tensor& grad, const at::Tensor& input
        ) {
            const char* op = "_adaptive_avg_pool2d_backward";
            const int64_t oh = grad.size(-2), ow = grad.size(-1);
            check_adaptive(op, input, oh, ow);
            auto ah = averaging(oh, input.size(-2), input.options());
            auto aw = averaging(ow, input.size(-1), input.options());
            auto g = grad.to(at::kFloat);
            const FidelityFloor exact(3);
            auto out = at::matmul(at::matmul(ah.t(), g), aw);
            return out.to(input.scalar_type());
        }

        // One spatial dim of a 2-D pool: output size and the extra right padding
        // ceil_mode needs so every window lies inside the padded input.
        struct Axis {
            int64_t size, kernel, stride, pad, dilation, out, extra;
        };

        Axis axis(int64_t size, int64_t k, int64_t s, int64_t p, int64_t d, bool ceil) {
            const int64_t span = size + 2 * p - d * (k - 1) - 1;
            int64_t out = (ceil ? (span + s - 1) / s : span / s) + 1;
            if (ceil && (out - 1) * s >= size + p) {
                --out;
            }
            const int64_t reach = (out - 1) * s + d * (k - 1) + 1;
            const int64_t extra = std::max<int64_t>(0, reach - (size + 2 * p));
            return {size, k, s, p, d, out, extra};
        }

        // x [..., H, W] padded by (pad, pad + extra) on both spatial dims with `value`.
        at::Tensor pad_spatial(
            const at::Tensor& x, const Axis& h, const Axis& w, double value
        ) {
            if (h.pad == 0 && w.pad == 0 && h.extra == 0 && w.extra == 0) {
                return x;
            }
            return at::constant_pad_nd(
                x, {w.pad, w.pad + w.extra, h.pad, h.pad + h.extra}, value
            );
        }

        // The [..., OH, OW] window elements of padded `x` under kernel tap (i, j);
        // `gather`: strided taps (fp32 / bf16) by index_select (the row gather).
        at::Tensor tap(
            const at::Tensor& x,
            const Axis& h,
            const Axis& w,
            int64_t i,
            int64_t j,
            bool gather = false
        ) {
            const int64_t top = i * h.dilation, left = j * w.dilation;
            if (gather && (h.stride > 1 || w.stride > 1) &&
                (x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16)) {
                const auto idx_opts = x.options().dtype(at::kLong);
                auto rows = (
                    at::arange(
                        top, top + (h.out - 1) * h.stride + 1, h.stride, idx_opts
                    )
                );
                auto cols = (
                    at::arange(
                        left, left + (w.out - 1) * w.stride + 1, w.stride, idx_opts
                    )
                );
                return x.index_select(-2, rows).index_select(-1, cols);
            }
            return (
                x.slice(-2, top, top + (h.out - 1) * h.stride + 1, h.stride)
                    .slice(-1, left, left + (w.out - 1) * w.stride + 1, w.stride)
            );
        }

        // 2-D pool arguments (kernel, stride, padding, dilation as 1 or 2 values).
        std::pair<Axis, Axis> pool_axes(
            const char* op,
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool ceil
        ) {
            KTT_REQUIRE(
                input.dim() == 3 || input.dim() == 4, op, "input must be 3-D or 4-D"
            );
            // Value i of a 1- or 2-value argument (empty -> fallback).
            auto pick = [](at::IntArrayRef v, size_t i, int64_t fallback) {
                return v.empty() ? fallback : v[std::min(i, v.size() - 1)];
            };
            const int64_t kh = pick(kernel, 0, 1), kw = pick(kernel, 1, 1);
            auto h = (
                axis(
                    input.size(-2),
                    kh,
                    pick(stride, 0, kh),
                    pick(padding, 0, 0),
                    pick(dilation, 0, 1),
                    ceil
                )
            );
            auto w = (
                axis(
                    input.size(-1),
                    kw,
                    pick(stride, 1, kw),
                    pick(padding, 1, 0),
                    pick(dilation, 1, 1),
                    ceil
                )
            );
            KTT_REQUIRE(h.out > 0 && w.out > 0, op, "output size would be empty");
            return {h, w};
        }

        // (max over each window, flat H*W input index of its first maximum).
        std::tuple<at::Tensor, at::Tensor> tt_max_pool2d_with_indices(
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool ceil
        ) {
            const char* op = "max_pool2d_with_indices";
            auto [h, w] = pool_axes(op, input, kernel, stride, padding, dilation, ceil);
            auto out_sizes = input.sizes().vec();
            out_sizes[out_sizes.size() - 2] = h.out;
            out_sizes.back() = w.out;
            auto long_opts = input.options().dtype(at::kLong);
            if (input.numel() == 0) {
                return {
                    at::empty(out_sizes, input.options()),
                    at::empty(out_sizes, long_opts)
                };
            }
            const double lowest = -std::numeric_limits<double>::infinity();
            auto padded = pad_spatial(input, h, w, lowest);

            // Input row / column of each output position's tap (0, 0).
            auto int_opts = input.options().dtype(at::kInt);
            auto rows = at::arange(h.out, int_opts) * h.stride - h.pad;
            auto cols = at::arange(w.out, int_opts) * w.stride - w.pad;
            at::Tensor best, best_index;
            for (int64_t i = 0; i < h.kernel; ++i) {
                for (int64_t j = 0; j < w.kernel; ++j) {
                    auto value = tap(padded, h, w, i, j, true);
                    auto r = rows + i * h.dilation, c = cols + j * w.dilation;
                    auto index = (r.reshape({h.out, 1}) * w.size +
                                  c.reshape({1, w.out}))
                                     .expand(out_sizes);
                    if (!best.defined()) {
                        best = value;
                        best_index = index;
                        continue;
                    }
                    auto better = at::gt(value, best);
                    best = at::where(better, value, best);
                    best_index = at::where(better, index, best_index);
                }
            }
            return {best.contiguous(), best_index.to(at::kLong)};
        }

        // 3-D: (max over each window, flat D*H*W input index of its first maximum).
        std::tuple<at::Tensor, at::Tensor> tt_max_pool3d_with_indices(
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool ceil
        ) {
            const char* op = "max_pool3d_with_indices";
            KTT_REQUIRE(
                input.dim() == 4 || input.dim() == 5, op, "input must be 4-D or 5-D"
            );
            auto pick = [](at::IntArrayRef v, size_t i, int64_t fallback) {
                return v.empty() ? fallback : v[std::min(i, v.size() - 1)];
            };
            std::array<Axis, 3> ax;
            for (size_t k = 0; k < 3; ++k) {
                const int64_t size = (
                    input.size(input.dim() - 3 + static_cast<int64_t>(k))
                );
                const int64_t kern = pick(kernel, k, 1);
                ax[k] = (
                    axis(
                        size,
                        kern,
                        pick(stride, k, kern),
                        pick(padding, k, 0),
                        pick(dilation, k, 1),
                        ceil
                    )
                );
                KTT_REQUIRE(ax[k].out > 0, op, "output size would be empty");
            }
            auto out_sizes = input.sizes().vec();
            for (size_t k = 0; k < 3; ++k) {
                out_sizes[out_sizes.size() - 3 + k] = ax[k].out;
            }
            auto long_opts = input.options().dtype(at::kLong);
            if (input.numel() == 0) {
                return {
                    at::empty(out_sizes, input.options()),
                    at::empty(out_sizes, long_opts)
                };
            }
            const double lowest = -std::numeric_limits<double>::infinity();
            std::vector<int64_t> pads;
            for (int64_t k = 2; k >= 0; --k) {
                pads.push_back(ax[k].pad);
                pads.push_back(ax[k].pad + ax[k].extra);
            }
            auto padded = at::constant_pad_nd(input, pads, lowest);

            // Input coordinate of each output position's tap 0, per spatial dim.
            auto int_opts = input.options().dtype(at::kInt);
            std::array<at::Tensor, 3> starts;
            for (size_t k = 0; k < 3; ++k) {
                starts[k] = at::arange(ax[k].out, int_opts) * ax[k].stride - ax[k].pad;
            }
            const auto& [d, h, w] = ax;
            at::Tensor best, best_index;
            for (int64_t a = 0; a < d.kernel; ++a) {
                for (int64_t b = 0; b < h.kernel; ++b) {
                    for (int64_t c = 0; c < w.kernel; ++c) {
                        const int64_t z0 = a * d.dilation, y0 = b * h.dilation;
                        const int64_t x0 = c * w.dilation;
                        const int64_t z1 = z0 + (d.out - 1) * d.stride + 1;
                        const int64_t y1 = y0 + (h.out - 1) * h.stride + 1;
                        const int64_t x1 = x0 + (w.out - 1) * w.stride + 1;
                        auto value = (
                            padded.slice(-3, z0, z1, d.stride)
                                           .slice(-2, y0, y1, h.stride)
                            .slice(-1, x0, x1, w.stride)
                        );
                        auto zi = (starts[0] + z0).reshape({d.out, 1, 1});
                        auto yi = (starts[1] + y0).reshape({1, h.out, 1});
                        auto xi = (starts[2] + x0).reshape({1, 1, w.out});
                        auto index = (
                            ((zi * h.size + yi) * w.size + xi).expand(out_sizes)
                        );
                        if (!best.defined()) {
                            best = value;
                            best_index = index;
                            continue;
                        }
                        auto better = at::gt(value, best);
                        best = at::where(better, value, best);
                        best_index = at::where(better, index, best_index);
                    }
                }
            }
            return {best.contiguous(), best_index.to(at::kLong)};
        }

        // 3-D backward: each output gradient added at its saved maximum's D*H*W index.
        at::Tensor tt_max_pool3d_with_indices_backward(
            const at::Tensor& grad,
            const at::Tensor& input,
            at::IntArrayRef,
            at::IntArrayRef,
            at::IntArrayRef,
            at::IntArrayRef,
            bool,
            const at::Tensor& indices
        ) {
            const int64_t dhw = input.size(-3) * input.size(-2) * input.size(-1);
            const int64_t planes = input.numel() / dhw;
            auto zeros = at::zeros({planes, dhw}, grad.options());
            auto out = at::scatter_add(
                zeros, 1, indices.reshape({planes, -1}), grad.reshape({planes, -1})
            );
            return out.reshape(input.sizes());
        }

        // Adaptive window positions along one axis: [out, kmax] source indices (windows
        // shorter than kmax repeat their last element, which leaves the max unchanged).
        std::vector<std::vector<int64_t>> adaptive_windows(int64_t in, int64_t out) {
            std::vector<std::vector<int64_t>> rows(out);
            size_t kmax = 0;
            for (int64_t o = 0; o < out; ++o) {
                const int64_t begin = (o * in) / out;
                const int64_t end = ((o + 1) * in + out - 1) / out;
                for (int64_t i = begin; i < end; ++i) {
                    rows[o].push_back(i);
                }
                kmax = std::max(kmax, rows[o].size());
            }
            for (auto& row : rows) {
                while (row.size() < kmax) {
                    row.push_back(row.back());
                }
            }
            return rows;
        }

        // (max over each adaptive window, flat H*W input index of its first maximum).
        std::tuple<at::Tensor, at::Tensor> tt_adaptive_max_pool2d(
            const at::Tensor& input, at::IntArrayRef output_size
        ) {
            const char* op = "adaptive_max_pool2d";
            KTT_REQUIRE(
                input.dim() == 3 || input.dim() == 4, op, "input must be 3-D or 4-D"
            );
            const int64_t h = input.size(-2), w = input.size(-1);
            const int64_t oh = output_size[0], ow = output_size[1];
            auto out_sizes = input.sizes().vec();
            out_sizes[out_sizes.size() - 2] = oh;
            out_sizes.back() = ow;
            auto long_opts = input.options().dtype(at::kLong);
            if (input.numel() == 0) {
                return {
                    at::empty(out_sizes, input.options()),
                    at::empty(out_sizes, long_opts)
                };
            }
            auto rows = adaptive_windows(h, oh), cols = adaptive_windows(w, ow);
            const int64_t kh = static_cast<int64_t>(rows[0].size());
            const int64_t kw = static_cast<int64_t>(cols[0].size());

            // Flat source index of every (output position, window element), in that
            // order.
            std::vector<int64_t> table;
            table.reserve(oh * ow * kh * kw);
            for (int64_t a = 0; a < oh; ++a) {
                for (int64_t b = 0; b < ow; ++b) {
                    for (int64_t i = 0; i < kh; ++i) {
                        for (int64_t j = 0; j < kw; ++j) {
                            table.push_back(rows[a][i] * w + cols[b][j]);
                        }
                    }
                }
            }
            const int64_t planes = input.numel() / (h * w);
            const int64_t cells = oh * ow;
            const int64_t k = kh * kw;
            auto key = c10::str("pool.amax|", h, "|", w, "|", oh, "|", ow);
            auto index = device_constant(key, [&] {
                return at::tensor(table);
            });
            auto flat = input.reshape({planes, h * w});
            auto spread = index.reshape({1, cells * k}).expand({planes, -1});
            auto values = at::gather(flat, 1, spread).reshape({planes, cells, k});
            auto best = at::amax(values, {-1});
            const bool keepdim = true;
            auto pick = at::argmax(values, -1, keepdim);
            auto where = index.reshape({1, cells, k}).expand({planes, cells, k});
            auto best_index = at::gather(where, 2, pick).squeeze(-1);
            return {
                best.reshape(out_sizes), best_index.reshape(out_sizes).to(at::kLong)
            };
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_adaptive_max_pool2d_out(
            const at::Tensor& input,
            at::IntArrayRef output_size,
            at::Tensor& out,
            at::Tensor& indices
        ) {
            auto [v, i] = tt_adaptive_max_pool2d(input, output_size);
            into("adaptive_max_pool2d.out", out, v);
            into("adaptive_max_pool2d.out", indices, i);
            return {out, indices};
        }

        at::Tensor& tt_adaptive_max_pool2d_backward_out(
            const at::Tensor& grad,
            const at::Tensor& input,
            const at::Tensor& indices,
            at::Tensor& grad_input
        ) {
            const int64_t hw = input.size(-2) * input.size(-1);
            const int64_t planes = input.numel() / std::max<int64_t>(hw, 1);
            auto zeros = at::zeros({planes, hw}, grad.options());
            auto result = at::scatter_add(
                zeros, 1, indices.reshape({planes, -1}), grad.reshape({planes, -1})
            );
            auto name = "adaptive_max_pool2d_backward.grad_input";
            return into(name, grad_input, result.reshape(input.sizes()));
        }

        // max_unpool: zeros of the spatial `size` with each value written at its saved
        // flat index (per plane); `spatial` = 2 or 3 trailing dims.
        at::Tensor max_unpool(
            const at::Tensor& self,
            const at::Tensor& indices,
            at::IntArrayRef size,
            int64_t spatial
        ) {
            const char* op = "max_unpool";
            KTT_REQUIRE(
                self.sizes() == indices.sizes(), op, "indices must match the input"
            );
            auto lead = self.sizes().slice(0, self.dim() - spatial).vec();
            int64_t area = 1;
            for (auto s : size) {
                area *= s;
            }
            auto out_sizes = lead;
            out_sizes.insert(out_sizes.end(), size.begin(), size.end());
            const int64_t planes = c10::multiply_integers(lead);
            if (self.numel() == 0 || planes == 0) {
                return at::zeros(out_sizes, self.options());
            }
            auto zeros = at::zeros({planes, area}, self.options());
            auto out = at::scatter(
                zeros, 1, indices.reshape({planes, -1}), self.reshape({planes, -1})
            );
            return out.reshape(out_sizes);
        }

        at::Tensor tt_max_unpool2d(
            const at::Tensor& self, const at::Tensor& indices, at::IntArrayRef size
        ) {
            return max_unpool(self, indices, size, 2);
        }

        at::Tensor& tt_max_unpool2d_out(
            const at::Tensor& self,
            const at::Tensor& indices,
            at::IntArrayRef size,
            at::Tensor& out
        ) {
            return into("max_unpool2d.out", out, max_unpool(self, indices, size, 2));
        }

        at::Tensor tt_max_unpool3d(
            const at::Tensor& self,
            const at::Tensor& indices,
            at::IntArrayRef size,
            at::IntArrayRef,
            at::IntArrayRef
        ) {
            return max_unpool(self, indices, size, 3);
        }

        at::Tensor& tt_max_unpool3d_out(
            const at::Tensor& self,
            const at::Tensor& indices,
            at::IntArrayRef size,
            at::IntArrayRef,
            at::IntArrayRef,
            at::Tensor& out
        ) {
            return into("max_unpool3d.out", out, max_unpool(self, indices, size, 3));
        }

        // grad_input [..., H, W]: each output gradient added at its saved maximum's
        // index.
        at::Tensor tt_max_pool2d_with_indices_backward(
            const at::Tensor& grad,
            const at::Tensor& input,
            at::IntArrayRef,
            at::IntArrayRef,
            at::IntArrayRef,
            at::IntArrayRef,
            bool,
            const at::Tensor& indices
        ) {
            const int64_t planes = input.numel() / (input.size(-2) * input.size(-1));
            const int64_t hw = input.size(-2) * input.size(-1);
            auto zeros = at::zeros({planes, hw}, grad.options());
            auto out = at::scatter_add(
                zeros, 1, indices.reshape({planes, -1}), grad.reshape({planes, -1})
            );
            return out.reshape(input.sizes());
        }

        // Mean over each window; the divisor counts padding only with count_include_pad
        // (never ceil_mode's extra cells), or is divisor_override.
        at::Tensor tt_avg_pool2d(
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            bool ceil,
            bool count_include_pad,
            std::optional<int64_t> divisor_override
        ) {
            const char* op = "avg_pool2d";
            auto [h, w] = pool_axes(op, input, kernel, stride, padding, {}, ceil);
            auto out_sizes = input.sizes().vec();
            out_sizes[out_sizes.size() - 2] = h.out;
            out_sizes.back() = w.out;
            if (input.numel() == 0) {
                return at::empty(out_sizes, input.options());
            }
            auto x = pad_spatial(input.to(at::kFloat), h, w, 0.0);

            // Cells that count: the input, plus the explicit padding when requested.
            auto valid = at::ones(
                {1, h.size, w.size}, input.options().dtype(at::kFloat)
            );
            if (count_include_pad) {
                valid = at::constant_pad_nd(valid, {w.pad, w.pad, h.pad, h.pad}, 1.0);
                valid = at::constant_pad_nd(valid, {0, w.extra, 0, h.extra}, 0.0);
            } else {
                valid = pad_spatial(valid, h, w, 0.0);
            }
            at::Tensor total, count;
            for (int64_t i = 0; i < h.kernel; ++i) {
                for (int64_t j = 0; j < w.kernel; ++j) {
                    auto t = tap(x, h, w, i, j);
                    auto c = tap(valid, h, w, i, j);
                    total = total.defined() ? total + t : t;
                    count = count.defined() ? count + c : c;
                }
            }
            auto out =
                (divisor_override ? total / static_cast<double>(*divisor_override)
                                  : total / count);
            return out.to(input.scalar_type()).contiguous();
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_max_pool2d_with_indices_out(
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool ceil,
            at::Tensor& out,
            at::Tensor& indices
        ) {
            auto [v, i] = (
                tt_max_pool2d_with_indices(
                    input, kernel, stride, padding, dilation, ceil
                )
            );
            into("max_pool2d_with_indices.out", out, v);
            into("max_pool2d_with_indices.out", indices, i);
            return {out, indices};
        }

        at::Tensor& tt_max_pool2d_with_indices_backward_out(
            const at::Tensor& grad,
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool ceil,
            const at::Tensor& indices,
            at::Tensor& grad_input
        ) {
            auto result = (
                tt_max_pool2d_with_indices_backward(
                    grad, input, kernel, stride, padding, dilation, ceil, indices
                )
            );
            return (
                into("max_pool2d_with_indices_backward.grad_input", grad_input, result)
            );
        }

        at::Tensor& tt_avg_pool2d_out(
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            bool ceil,
            bool count_include_pad,
            std::optional<int64_t> divisor_override,
            at::Tensor& out
        ) {
            auto result = (
                tt_avg_pool2d(
                    input,
                    kernel,
                    stride,
                    padding,
                    ceil,
                    count_include_pad,
                    divisor_override
                )
            );
            return into("avg_pool2d.out", out, result);
        }

        // PyTorch's fractional pooling window starts along one dim [N, C, out] (int64,
        // host): ((i + u) * alpha) - (u * alpha) truncated, alpha = (in - k) / (out -
        // 1), in the input's dtype (as its CPU kernel rounds); the last start in - k.
        at::Tensor fractional_starts(
            const at::Tensor& u,
            int64_t in,
            int64_t out,
            int64_t k,
            at::ScalarType dtype
        ) {
            auto options = at::TensorOptions().dtype(dtype);
            auto starts = at::empty({u.size(0), u.size(1), out}, at::kLong);
            if (out > 1) {
                auto alpha =
                    (at::scalar_tensor(static_cast<double>(in - k), options) /
                     at::scalar_tensor(static_cast<double>(out - 1), options));
                auto i = at::arange(out - 1, options).view({1, 1, -1});
                auto uu = u.to(dtype).unsqueeze(-1);
                auto first = ((i + uu) * alpha).to(at::kLong);
                auto offset = (uu * alpha).to(at::kLong);
                starts.narrow(2, 0, out - 1).copy_(first - offset);
            }
            if (out > 0) {
                starts.narrow(2, out - 1, 1).fill_(in - k);
            }
            return starts;
        }

        // (values, flat spatial indices) of fractional max pooling over the last
        // `kernel.size()` dims of `self` [N, C, *S] (or [C, *S]); `samples` [N, C,
        // dims] with sample `order[d]` driving spatial dim d.
        std::tuple<at::Tensor, at::Tensor> fractional_max_pool(
            const char* op,
            const at::Tensor& self,
            at::IntArrayRef kernel,
            at::IntArrayRef output_size,
            const at::Tensor& samples,
            std::vector<int64_t> order
        ) {
            const auto dims = static_cast<int64_t>(kernel.size());
            KTT_REQUIRE(
                self.dim() == dims + 1 || self.dim() == dims + 2,
                op,
                "needs [C, *S] or [N, C, *S]"
            );
            const bool batched = self.dim() == dims + 2;
            auto x = batched ? self : self.unsqueeze(0);
            const int64_t n = x.size(0), c = x.size(1);
            auto in = x.sizes().slice(2).vec();
            auto u = samples.cpu().reshape({n, c, dims});
            int64_t positions = 1, window = 1, flat_size = 1;
            for (int64_t d = 0; d < dims; ++d) {
                positions *= output_size[d];
                window *= kernel[d];
                flat_size *= in[d];
            }
            auto out_sizes = std::vector<int64_t>{n, c};
            out_sizes.insert(out_sizes.end(), output_size.begin(), output_size.end());
            if (n * c * positions == 0) {
                auto empty = at::empty(out_sizes, x.options());
                auto empty_idx = at::empty(out_sizes, x.options().dtype(at::kLong));
                return {
                    batched ? empty : empty.squeeze(0),
                    batched ? empty_idx : empty_idx.squeeze(0)
                };
            }
            // Flat input index [N, C, positions, window]: per dim, start + offset times
            // that dim's stride, broadcast over the other dims.
            auto index = at::zeros({n, c, positions, window}, at::kLong);
            int64_t stride = flat_size, out_stride = positions, k_stride = window;
            for (int64_t d = 0; d < dims; ++d) {
                stride /= in[d];
                out_stride /= output_size[d];
                k_stride /= kernel[d];
                auto starts = (
                    fractional_starts(
                        u.select(2, order[d]),
                        in[d],
                        output_size[d],
                        kernel[d],
                        self.scalar_type()
                    )
                );
                auto pos =
                    (at::arange(positions, at::kLong).div(out_stride, "floor") %
                     output_size[d]);
                auto off =
                    (at::arange(window, at::kLong).div(k_stride, "floor") % kernel[d]);
                auto start = starts.index_select(2, pos);  // [N, C, positions]
                index = index +
                    (start.unsqueeze(-1) + off.view({1, 1, 1, -1})) * stride;
            }
            auto device_index = index.to(x.device());
            auto flat = x.reshape({n, c, flat_size}).to(at::kFloat);
            auto values = at::gather(flat, 2, device_index.reshape({n, c, -1}))
                              .reshape({n, c, positions, window});
            // First maximum on ties (PyTorch's strict > scan).
            auto arg = at::argmax(values, -1);
            auto best = values.gather(-1, arg.unsqueeze(-1)).squeeze(-1);
            auto picked = device_index.gather(-1, arg.unsqueeze(-1)).squeeze(-1);
            auto out = best.reshape(out_sizes).to(self.scalar_type());
            auto idx = picked.reshape(out_sizes);
            return {batched ? out : out.squeeze(0), batched ? idx : idx.squeeze(0)};
        }

        std::tuple<at::Tensor, at::Tensor> tt_fractional_max_pool2d(
            const at::Tensor& self,
            at::IntArrayRef kernel,
            at::IntArrayRef output_size,
            const at::Tensor& samples
        ) {
            // Sample 0 drives W, sample 1 H.
            return fractional_max_pool(
                "fractional_max_pool2d", self, kernel, output_size, samples, {1, 0}
            );
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_fractional_max_pool2d_out(
            const at::Tensor& self,
            at::IntArrayRef kernel,
            at::IntArrayRef output_size,
            const at::Tensor& samples,
            at::Tensor& out,
            at::Tensor& indices
        ) {
            auto [v, i] = tt_fractional_max_pool2d(self, kernel, output_size, samples);
            into("fractional_max_pool2d.output", out, v);
            into("fractional_max_pool2d.output", indices, i);
            return {out, indices};
        }

        std::tuple<at::Tensor, at::Tensor> tt_fractional_max_pool3d(
            const at::Tensor& self,
            at::IntArrayRef kernel,
            at::IntArrayRef output_size,
            const at::Tensor& samples
        ) {
            // Samples 0, 1, 2 drive T, H, W.
            return fractional_max_pool(
                "fractional_max_pool3d", self, kernel, output_size, samples, {0, 1, 2}
            );
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_fractional_max_pool3d_out(
            const at::Tensor& self,
            at::IntArrayRef kernel,
            at::IntArrayRef output_size,
            const at::Tensor& samples,
            at::Tensor& out,
            at::Tensor& indices
        ) {
            auto [v, i] = tt_fractional_max_pool3d(self, kernel, output_size, samples);
            into("fractional_max_pool3d.output", out, v);
            into("fractional_max_pool3d.output", indices, i);
            return {out, indices};
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("fractional_max_pool2d", &tt_fractional_max_pool2d);
        m.impl("fractional_max_pool2d.output", &tt_fractional_max_pool2d_out);
        m.impl("fractional_max_pool3d", &tt_fractional_max_pool3d);
        m.impl("fractional_max_pool3d.output", &tt_fractional_max_pool3d_out);
        m.impl("max_pool2d_with_indices", &tt_max_pool2d_with_indices);
        m.impl("max_pool2d_with_indices.out", &tt_max_pool2d_with_indices_out);
        m.impl(
            "max_pool2d_with_indices_backward", &tt_max_pool2d_with_indices_backward
        );
        m.impl(
            "max_pool2d_with_indices_backward.grad_input",
            &tt_max_pool2d_with_indices_backward_out
        );
        m.impl("avg_pool2d", &tt_avg_pool2d);
        m.impl("avg_pool2d.out", &tt_avg_pool2d_out);
        m.impl("max_unpool2d", &tt_max_unpool2d);
        m.impl("max_unpool2d.out", &tt_max_unpool2d_out);
        m.impl("max_unpool3d", &tt_max_unpool3d);
        m.impl("max_unpool3d.out", &tt_max_unpool3d_out);
        m.impl("adaptive_max_pool2d.out", &tt_adaptive_max_pool2d_out);
        m.impl(
            "adaptive_max_pool2d_backward.grad_input",
            &tt_adaptive_max_pool2d_backward_out
        );
        m.impl("max_pool3d_with_indices", &tt_max_pool3d_with_indices);
        m.impl(
            "max_pool3d_with_indices_backward", &tt_max_pool3d_with_indices_backward
        );
        m.impl("_adaptive_avg_pool3d", &tt_adaptive_avg_pool3d);
        m.impl("_adaptive_avg_pool3d_backward", &tt_adaptive_avg_pool3d_backward);
        m.impl("_adaptive_avg_pool2d", &tt_adaptive_avg_pool2d);
        m.impl("_adaptive_avg_pool2d_backward", &tt_adaptive_avg_pool2d_backward);
    }

}  // namespace ktt
