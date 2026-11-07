// aten N-d pooling kernels for tt from host geometry tables (cached device constants):
// avg pool 3-D, avg pool 2-D / 3-D backward, adaptive max pool 3-D (+ backward).

#include <c10/util/accumulate.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <string>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // One spatial dim of an avg pool: input size, kernel, stride, pad, output size,
        // and the extra right padding ceil_mode needs (padded size = size + 2 pad +
        // extra).
        struct Axis {
            int64_t size, kernel, stride, pad, out, extra;

            int64_t padded() const {
                return size + 2 * pad + extra;
            }
        };

        // Pool geometry over the trailing `nd` dims (kernel / stride / padding as 1 or
        // nd).
        std::vector<Axis> geometry(
            const char* op,
            const at::Tensor& input,
            int64_t nd,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            bool ceil
        ) {
            KTT_REQUIRE(
                input.dim() == nd + 1 || input.dim() == nd + 2,
                op,
                "unexpected input rank"
            );
            auto pick = [](at::IntArrayRef v, size_t i, int64_t fallback) {
                return v.empty() ? fallback : v[std::min(i, v.size() - 1)];
            };
            std::vector<Axis> ax;
            for (int64_t k = 0; k < nd; ++k) {
                const int64_t size = input.size(input.dim() - nd + k);
                const int64_t kern = pick(kernel, k, 1), s = pick(stride, k, kern);
                const int64_t p = pick(padding, k, 0);
                const int64_t span = size + 2 * p - kern;
                int64_t out = (ceil ? (span + s - 1) / s : span / s) + 1;
                if (ceil && (out - 1) * s >= size + p) {
                    --out;
                }
                KTT_REQUIRE(out > 0, op, "output size would be empty");
                const int64_t reach = (out - 1) * s + kern;
                ax.push_back({
                    size,
                    kern,
                    s,
                    p,
                    out,
                    std::max<int64_t>(0, reach - (size + 2 * p))
                });
            }
            return ax;
        }

        std::string geometry_key(
            const char* tag, const std::vector<Axis>& ax, int64_t flag
        ) {
            std::string key = tag;
            for (const auto& a : ax) {
                key += (
                    c10::str(
                        "|",
                        a.size,
                        ",",
                        a.kernel,
                        ",",
                        a.stride,
                        ",",
                        a.pad,
                        ",",
                        a.out
                    )
                );
            }
            return key + c10::str("|", flag);
        }

        // Calls fn(flat output cell, flat padded index) for every (tap, output cell),
        // taps outermost, both row-major.
        template <typename Fn>
        void for_each_tap(const std::vector<Axis>& ax, Fn fn) {
            const size_t nd = ax.size();
            int64_t taps = 1, cells = 1;
            for (const auto& a : ax) {
                taps *= a.kernel;
                cells *= a.out;
            }
            std::vector<int64_t> t(nd), o(nd);
            for (int64_t ti = 0; ti < taps; ++ti) {
                int64_t rest = ti;
                for (size_t k = nd; k-- > 0;) {
                    t[k] = rest % ax[k].kernel;
                    rest /= ax[k].kernel;
                }
                for (int64_t oi = 0; oi < cells; ++oi) {
                    int64_t r = oi;
                    for (size_t k = nd; k-- > 0;) {
                        o[k] = r % ax[k].out;
                        r /= ax[k].out;
                    }
                    int64_t flat = 0;
                    for (size_t k = 0; k < nd; ++k) {
                        flat = flat * ax[k].padded() + o[k] * ax[k].stride + t[k];
                    }
                    fn(oi, flat);
                }
            }
        }

        // Per-output divisor: divisor_override, else the cells that count in each
        // window (the input, plus explicit padding with count_include_pad; never ceil's
        // extra).
        at::Tensor divisor(
            const std::vector<Axis>& ax,
            bool count_include_pad,
            std::optional<int64_t> over
        ) {
            auto key = (
                geometry_key("avgpool.div", ax, over ? -*over : count_include_pad)
            );
            return device_constant(key, [&] {
                int64_t cells = 1;
                for (const auto& a : ax) {
                    cells *= a.out;
                }
                std::vector<float> count(cells, 0.0f);
                const size_t nd = ax.size();
                for_each_tap(ax, [&](int64_t oi, int64_t flat) {
                    // Padded coordinate per axis from the flat index.
                    bool inside = true;
                    for (size_t k = nd; k-- > 0;) {
                        const int64_t c = flat % ax[k].padded();
                        flat /= ax[k].padded();
                        const int64_t lo = count_include_pad ? 0 : ax[k].pad;
                        const int64_t hi =
                            (count_include_pad ? ax[k].size + 2 * ax[k].pad
                                               : ax[k].size + ax[k].pad);
                        inside = inside && c >= lo && c < hi;
                    }
                    count[oi] += inside ? 1.0f : 0.0f;
                });
                if (over) {
                    std::fill(count.begin(), count.end(), static_cast<float>(*over));
                }
                std::vector<int64_t> shape;
                for (const auto& a : ax) {
                    shape.push_back(a.out);
                }
                return at::tensor(count).reshape(shape);
            });
        }

        std::vector<int64_t> pads_of(const std::vector<Axis>& ax) {
            std::vector<int64_t> pads;
            for (size_t k = ax.size(); k-- > 0;) {
                pads.push_back(ax[k].pad);
                pads.push_back(ax[k].pad + ax[k].extra);
            }
            return pads;
        }

        // Sum of every tap's strided window of zero-padded x over the trailing dims, /
        // divisor.
        at::Tensor avg_pool(
            const char* op,
            const at::Tensor& input,
            int64_t nd,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            bool ceil,
            bool count_include_pad,
            std::optional<int64_t> divisor_override
        ) {
            auto ax = geometry(op, input, nd, kernel, stride, padding, ceil);
            auto out_sizes = input.sizes().vec();
            for (int64_t k = 0; k < nd; ++k) {
                out_sizes[input.dim() - nd + k] = ax[k].out;
            }
            if (input.numel() == 0) {
                return at::empty(out_sizes, input.options());
            }
            auto x = at::constant_pad_nd(input.to(at::kFloat), pads_of(ax), 0.0);
            int64_t taps = 1;
            for (const auto& a : ax) {
                taps *= a.kernel;
            }
            at::Tensor total;
            for (int64_t ti = 0; ti < taps; ++ti) {
                auto window = x;
                int64_t rest = ti;
                for (int64_t k = nd; k-- > 0;) {
                    const int64_t t = rest % ax[k].kernel;
                    rest /= ax[k].kernel;
                    const int64_t dim = x.dim() - nd + k;
                    const int64_t end = t + (ax[k].out - 1) * ax[k].stride + 1;
                    window = window.slice(dim, t, end, ax[k].stride);
                }
                total = total.defined() ? total + window : window;
            }
            auto result = total / divisor(ax, count_include_pad, divisor_override);
            return result.to(input.scalar_type()).contiguous();
        }

        // Each output gradient / divisor added to every padded cell of its window (one
        // scatter_add over all taps), then the padding cropped.
        at::Tensor avg_pool_backward(
            const char* op,
            const at::Tensor& grad,
            const at::Tensor& input,
            int64_t nd,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            bool ceil,
            bool count_include_pad,
            std::optional<int64_t> divisor_override
        ) {
            if (input.numel() == 0) {
                return at::zeros_like(input);
            }
            auto ax = geometry(op, input, nd, kernel, stride, padding, ceil);
            int64_t taps = 1, cells = 1, volume = 1;
            std::vector<int64_t> padded;
            for (const auto& a : ax) {
                taps *= a.kernel;
                cells *= a.out;
                volume *= a.padded();
                padded.push_back(a.padded());
            }
            auto index = device_constant(geometry_key("avgpool.bwd", ax, 0), [&] {
                std::vector<int64_t> table;
                table.reserve(taps * cells);
                for_each_tap(ax, [&](int64_t, int64_t flat) {
                    table.push_back(flat);
                });
                return at::tensor(table);
            });
            const int64_t planes =
                (input.numel() /
                 c10::multiply_integers(input.sizes().slice(input.dim() - nd)));
            auto share =
                (grad.to(at::kFloat) /
                 divisor(ax, count_include_pad, divisor_override));
            auto values = share.reshape({planes, 1, cells})
                              .expand({planes, taps, cells});
            auto zeros = at::zeros({planes, volume}, grad.options().dtype(at::kFloat));
            auto spread = index.reshape({1, taps * cells}).expand({planes, -1});
            auto sums = at::scatter_add(zeros, 1, spread, values.reshape({planes, -1}));
            std::vector<int64_t> shape{planes};
            shape.insert(shape.end(), padded.begin(), padded.end());
            auto full = sums.reshape(shape);
            for (int64_t k = 0; k < nd; ++k) {
                full = full.narrow(1 + k, ax[k].pad, ax[k].size);
            }
            return full.reshape(input.sizes()).to(input.scalar_type()).contiguous();
        }

        at::Tensor tt_avg_pool3d(
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            bool ceil,
            bool count_include_pad,
            std::optional<int64_t> divisor_override
        ) {
            return (
                avg_pool(
                    "avg_pool3d",
                    input,
                    3,
                    kernel,
                    stride,
                    padding,
                    ceil,
                    count_include_pad,
                    divisor_override
                )
            );
        }

        at::Tensor& tt_avg_pool3d_out(
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
                tt_avg_pool3d(
                    input,
                    kernel,
                    stride,
                    padding,
                    ceil,
                    count_include_pad,
                    divisor_override
                )
            );
            return into("avg_pool3d.out", out, result);
        }

        template <int64_t ND>
        at::Tensor tt_avg_pool_backward(
            const at::Tensor& grad,
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            bool ceil,
            bool count_include_pad,
            std::optional<int64_t> divisor_override
        ) {
            const char* op = ND == 2 ? "avg_pool2d_backward" : "avg_pool3d_backward";
            return (
                avg_pool_backward(
                    op,
                    grad,
                    input,
                    ND,
                    kernel,
                    stride,
                    padding,
                    ceil,
                    count_include_pad,
                    divisor_override
                )
            );
        }

        template <int64_t ND>
        at::Tensor& tt_avg_pool_backward_out(
            const at::Tensor& grad,
            const at::Tensor& input,
            at::IntArrayRef kernel,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            bool ceil,
            bool count_include_pad,
            std::optional<int64_t> divisor_override,
            at::Tensor& grad_input
        ) {
            auto result = (
                tt_avg_pool_backward<ND>(
                    grad,
                    input,
                    kernel,
                    stride,
                    padding,
                    ceil,
                    count_include_pad,
                    divisor_override
                )
            );
            const char* op =
                (ND == 2 ? "avg_pool2d_backward.grad_input"
                         : "avg_pool3d_backward.grad_input");
            return into(op, grad_input, result);
        }

        // Adaptive window positions along one axis: [out][kmax] source indices (shorter
        // windows repeat their last element, which leaves the max unchanged).
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

        // (max over each adaptive 3-D window, flat D*H*W input index of its first
        // maximum).
        std::tuple<at::Tensor, at::Tensor> tt_adaptive_max_pool3d(
            const at::Tensor& input, at::IntArrayRef output_size
        ) {
            const char* op = "adaptive_max_pool3d";
            KTT_REQUIRE(
                input.dim() == 4 || input.dim() == 5, op, "input must be 4-D or 5-D"
            );
            KTT_REQUIRE(output_size.size() == 3, op, "output_size must have 3 values");
            std::array<int64_t, 3> in{input.size(-3), input.size(-2), input.size(-1)};
            auto out_sizes = input.sizes().vec();
            for (size_t k = 0; k < 3; ++k) {
                out_sizes[out_sizes.size() - 3 + k] = output_size[k];
            }
            auto long_opts = input.options().dtype(at::kLong);
            if (input.numel() == 0) {
                return {
                    at::empty(out_sizes, input.options()),
                    at::empty(out_sizes, long_opts)
                };
            }
            std::array<std::vector<std::vector<int64_t>>, 3> win;
            int64_t cells = 1, k = 1;
            for (size_t a = 0; a < 3; ++a) {
                win[a] = adaptive_windows(in[a], output_size[a]);
                cells *= output_size[a];
                k *= static_cast<int64_t>(win[a][0].size());
            }
            auto key = (
                c10::str(
                    "pool.amax3|",
                    in[0],
                    ",",
                    in[1],
                    ",",
                    in[2],
                    "|",
                    output_size[0],
                    ",",
                    output_size[1],
                    ",",
                    output_size[2]
                )
            );
            // Flat source index of every (output cell, window element), in that order.
            auto index = device_constant(key, [&] {
                std::vector<int64_t> table;
                table.reserve(cells * k);
                for (const auto& zs : win[0]) {
                    for (const auto& ys : win[1]) {
                        for (const auto& xs : win[2]) {
                            for (auto z : zs) {
                                for (auto y : ys) {
                                    for (auto x : xs) {
                                        table.push_back((z * in[1] + y) * in[2] + x);
                                    }
                                }
                            }
                        }
                    }
                }
                return at::tensor(table);
            });
            const int64_t volume = in[0] * in[1] * in[2];
            const int64_t planes = input.numel() / volume;
            auto flat = input.reshape({planes, volume});
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

        std::tuple<at::Tensor&, at::Tensor&> tt_adaptive_max_pool3d_out(
            const at::Tensor& input,
            at::IntArrayRef output_size,
            at::Tensor& out,
            at::Tensor& indices
        ) {
            auto [v, i] = tt_adaptive_max_pool3d(input, output_size);
            into("adaptive_max_pool3d.out", out, v);
            into("adaptive_max_pool3d.out", indices, i);
            return {out, indices};
        }

        at::Tensor& tt_adaptive_max_pool3d_backward_out(
            const at::Tensor& grad,
            const at::Tensor& input,
            const at::Tensor& indices,
            at::Tensor& grad_input
        ) {
            const int64_t volume = input.size(-3) * input.size(-2) * input.size(-1);
            const int64_t planes = input.numel() / std::max<int64_t>(volume, 1);
            auto zeros = at::zeros({planes, volume}, grad.options());
            auto result = at::scatter_add(
                zeros, 1, indices.reshape({planes, -1}), grad.reshape({planes, -1})
            );
            auto name = "adaptive_max_pool3d_backward.grad_input";
            return into(name, grad_input, result.reshape(input.sizes()));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("avg_pool3d", &tt_avg_pool3d);
        m.impl("avg_pool3d.out", &tt_avg_pool3d_out);
        m.impl("avg_pool2d_backward", &tt_avg_pool_backward<2>);
        m.impl("avg_pool2d_backward.grad_input", &tt_avg_pool_backward_out<2>);
        m.impl("avg_pool3d_backward", &tt_avg_pool_backward<3>);
        m.impl("avg_pool3d_backward.grad_input", &tt_avg_pool_backward_out<3>);
        m.impl("adaptive_max_pool3d.out", &tt_adaptive_max_pool3d_out);
        m.impl(
            "adaptive_max_pool3d_backward.grad_input",
            &tt_adaptive_max_pool3d_backward_out
        );
    }

}  // namespace ktt
