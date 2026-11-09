// aten convolution for tt (forward, transposed and backward; 1-D / 2-D, any stride,
// padding, dilation, groups) and im2col / col2im: grouped, depthwise and strided
// one-row convs and their gradients on Kohaku-TT's ktt_conv2, dense convs on ktt_conv_nhwc
// (Nhwc storage) / ktt_conv / TT-NN, the rest as per-tap matmuls summed in fp32.
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include "core/torch/constants.hpp"
#include "core/torch/contract.hpp"
#include "core/torch/tt_tensor.hpp"

namespace ktt {
    namespace {

        // Flat padded-image position (row-major, width wp) of each of the oh x ow
        // output positions for the kernel tap at (top, left): a cached device constant.
        at::Tensor tap_positions(
            int64_t top,
            int64_t left,
            int64_t oh,
            int64_t ow,
            std::array<int64_t, 2> stride,
            int64_t wp
        ) {
            auto key = (
                c10::str(
                    "conv.tap|",
                    top,
                    "|",
                    left,
                    "|",
                    oh,
                    "|",
                    ow,
                    "|",
                    stride[0],
                    "|",
                    stride[1],
                    "|",
                    wp
                )
            );
            return device_constant(key, [&] {
                std::vector<int64_t> where(oh * ow);
                for (int64_t a = 0; a < oh; ++a) {
                    for (int64_t b = 0; b < ow; ++b) {
                        where
                            [a * ow +
                             b] = ((top + a * stride[0]) * wp + left + b * stride[1]);
                    }
                }
                return at::tensor(where, at::kLong);
            });
        }

        // 2-D convolution of x [N, C, H, W] with w [O, C/groups, KH, KW] (no bias).
        at::Tensor conv2d(
            const at::Tensor& x,
            const at::Tensor& w,
            std::array<int64_t, 2> stride,
            std::array<int64_t, 2> padding,
            std::array<int64_t, 2> dilation,
            int64_t groups
        ) {
            const char* op = "convolution";
            const int64_t n = x.size(0), c = x.size(1), o = w.size(0);
            const int64_t kh = w.size(2), kw = w.size(3);
            auto padded = x;
            if (padding[0] > 0 || padding[1] > 0) {
                padded = at::constant_pad_nd(
                    x, {padding[1], padding[1], padding[0], padding[0]}
                );
            }
            const int64_t h = padded.size(2), wd = padded.size(3);
            const int64_t oh = (h - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
            const int64_t ow = (wd - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
            KTT_REQUIRE(oh > 0 && ow > 0, op, "kernel larger than the padded input");
            if (n == 0) {
                return at::empty({0, o, oh, ow}, x.options());
            }

            const int64_t cg = c / groups, og = o / groups, m = n * oh * ow;
            const bool depthwise = cg == 1 && og == 1;
            if (groups == 1 && stride[0] == 1 && stride[1] == 1) {
                // Shifted matmuls over the flattened channels-last input [N, H W', C]
                // (W' the padded width, zero rows appended): tap (i, j) reads the rows
                // from i d0 W' + j d1, its output rows run over [oh, W'] and the extra
                // W' - ow columns are cropped once at the end.
                const int64_t rows_out = oh * wd;
                const int64_t tail = (kw - 1) * dilation[1];
                auto flat = padded.permute({0, 2, 3, 1})
                                .contiguous()
                                .reshape({n, h * wd, c});
                if (tail > 0) {
                    flat = at::constant_pad_nd(flat, {0, 0, 0, tail});
                }
                at::Tensor sum;
                for (int64_t i = 0; i < kh; ++i) {
                    for (int64_t j = 0; j < kw; ++j) {
                        const int64_t start = i * dilation[0] * wd + j * dilation[1];
                        auto rows = flat.narrow(1, start, rows_out);
                        auto term = at::matmul(rows, w.select(3, j).select(2, i).t());
                        sum = sum.defined() ? sum + term : term;
                    }
                }
                return sum.reshape({n, oh, wd, o})
                    .narrow(2, 0, ow)
                    .permute({0, 3, 1, 2})
                    .contiguous();
            }
            if (groups == 1) {
                // Channels last once; per tap rows [M, C] @ tap^T [C, O] -> [M, O].
                auto nhwc = padded.permute({0, 2, 3, 1}).contiguous();
                at::Tensor sum;
                for (int64_t i = 0; i < kh; ++i) {
                    for (int64_t j = 0; j < kw; ++j) {
                        const int64_t top = i * dilation[0], left = j * dilation[1];
                        auto rows = nhwc.slice(
                                            1,
                                            top,
                                            top + (oh - 1) * stride[0] + 1,
                                            stride[0]
                        )
                                        .slice(
                                            2,
                                            left,
                                            left + (ow - 1) * stride[1] + 1,
                                            stride[1]
                                        )
                                        .reshape({m, c});
                        auto term = at::mm(rows, w.select(3, j).select(2, i).t());
                        sum = sum.defined() ? sum + term : term;
                    }
                }
                return sum.reshape({n, oh, ow, o}).permute({0, 3, 1, 2}).contiguous();
            }
            at::Tensor acc;
            for (int64_t i = 0; i < kh; ++i) {
                for (int64_t j = 0; j < kw; ++j) {
                    // Input pixels under kernel tap (i, j) for every output position.
                    const int64_t top = i * dilation[0], left = j * dilation[1];
                    auto patch = (
                        padded.slice(2, top, top + (oh - 1) * stride[0] + 1, stride[0])
                            .slice(3, left, left + (ow - 1) * stride[1] + 1, stride[1])
                    );
                    auto tap = w.select(3, j).select(2, i);  // [O, cg]
                    at::Tensor term;
                    if (depthwise) {
                        // One weight per channel: an elementwise multiply ([N, C, oh,
                        // ow]).
                        term = patch * tap.reshape({1, c, 1, 1});
                    } else {
                        // All groups at once: [G, M, cg] @ [G, cg, og] -> [G, M, og].
                        auto rows = patch.reshape({n, groups, cg, oh, ow})
                                        .permute({1, 0, 3, 4, 2})
                                        .reshape({groups, m, cg});
                        auto weights = tap.reshape({groups, og, cg}).transpose(1, 2);
                        term = at::bmm(rows, weights);
                    }
                    acc = acc.defined() ? acc + term : term;
                }
            }
            if (depthwise) {
                return acc.contiguous();
            }
            // [G, N, oh, ow, og] -> [N, G * og, oh, ow].
            return acc.reshape({groups, n, oh, ow, og})
                .permute({1, 0, 4, 2, 3})
                .reshape({n, o, oh, ow})
                .contiguous();
        }

        // conv2d gradients (fp32) for input sizes `in` [N, C, H, W]: grad_weight per
        // tap from conv input `x`; grad_input per tap scatter-added at the tap's padded
        // positions.
        std::pair<at::Tensor, at::Tensor> conv2d_backward(
            const at::Tensor& go,
            const at::Tensor& x,
            const at::Tensor& w,
            std::array<int64_t, 4> in,
            std::array<int64_t, 2> stride,
            std::array<int64_t, 2> padding,
            std::array<int64_t, 2> dilation,
            int64_t groups,
            bool need_input,
            bool need_weight
        ) {
            const int64_t n = in[0], c = in[1], o = w.size(0);
            const int64_t kh = w.size(2), kw = w.size(3);
            const int64_t oh = go.size(2), ow = go.size(3);
            const int64_t hp = in[2] + 2 * padding[0], wp = in[3] + 2 * padding[1];
            auto padded = x;
            if (need_weight && (padding[0] > 0 || padding[1] > 0)) {
                padded = at::constant_pad_nd(
                    x, {padding[1], padding[1], padding[0], padding[0]}
                );
            }
            const int64_t cg = c / groups, og = o / groups, m = n * oh * ow;
            const bool depthwise = cg == 1 && og == 1;
            // grad rows per group: [G, M, og] (depthwise: go itself, [N, C, oh, ow]).
            auto go_rows = depthwise ? go
                                     : go.reshape({n, groups, og, oh, ow})
                                           .permute({1, 0, 3, 4, 2})
                                           .reshape({groups, m, og});
            at::Tensor gi;
            std::vector<at::Tensor> taps;
            for (int64_t i = 0; i < kh; ++i) {
                for (int64_t j = 0; j < kw; ++j) {
                    const int64_t top = i * dilation[0], left = j * dilation[1];
                    if (need_weight) {
                        const int64_t bottom = top + (oh - 1) * stride[0] + 1;
                        const int64_t right = left + (ow - 1) * stride[1] + 1;
                        auto patch = (
                            padded.slice(2, top, bottom, stride[0])
                            .slice(3, left, right, stride[1])
                        );
                        if (depthwise) {
                            taps.push_back((go * patch).sum({0, 2, 3}).reshape({c, 1}));
                        } else {
                            auto rows = patch.reshape({n, groups, cg, oh, ow})
                                            .permute({1, 0, 3, 4, 2})
                                            .reshape({groups, m, cg});
                            // [G, og, M] @ [G, M, cg] -> [G, og, cg] -> [O, cg].
                            auto tap = at::bmm(go_rows.transpose(1, 2), rows);
                            taps.push_back(tap.reshape({o, cg}));
                        }
                    }
                    if (need_input) {
                        auto tap_w = w.select(3, j).select(2, i);  // [O, cg]
                        at::Tensor part;                           // [N * C, oh * ow]
                        if (depthwise) {
                            part = (go * tap_w.reshape({1, c, 1, 1}))
                                       .reshape({n * c, oh * ow});
                        } else {
                            // [G, M, og] @ [G, og, cg] -> [G, M, cg] -> [N, G, cg, oh *
                            // ow].
                            auto rows = at::bmm(
                                go_rows, tap_w.reshape({groups, og, cg})
                            );
                            part = rows.reshape({groups, n, oh * ow, cg})
                                       .permute({1, 0, 3, 2})
                                       .reshape({n * c, oh * ow});
                        }
                        auto index = tap_positions(top, left, oh, ow, stride, wp)
                                         .reshape({1, oh * ow})
                                         .expand({n * c, oh * ow});
                        if (!gi.defined()) {
                            gi = at::zeros({n * c, hp * wp}, go.options());
                        }
                        gi = at::scatter_add(gi, 1, index, part);
                    }
                }
            }
            at::Tensor grad_input, grad_weight;
            if (need_input) {
                grad_input = gi.reshape({n, c, hp, wp})
                                 .slice(2, padding[0], padding[0] + in[2])
                                 .slice(3, padding[1], padding[1] + in[3])
                                 .contiguous();
            }
            if (need_weight) {
                grad_weight = at::stack(taps, 2).reshape({o, cg, kh, kw});
            }
            return {grad_input, grad_weight};
        }

        // 1-D convolutions run as 2-D with a unit height: per-arg 2-D values and fp32
        // lifts.
        struct Geometry {
            int64_t spatial;
            std::array<int64_t, 2> stride, padding, dilation, output_padding;

            Geometry(
                const char* op,
                const at::Tensor& weight,
                at::IntArrayRef stride_,
                at::IntArrayRef padding_,
                at::IntArrayRef dilation_,
                at::IntArrayRef output_padding_
            )
                : spatial(weight.dim() - 2) {
                KTT_REQUIRE(
                    spatial == 1 || spatial == 2, op, "only 1-D and 2-D convolution"
                );
                stride = two(stride_, 1);
                padding = two(padding_, 0);
                dilation = two(dilation_, 1);
                output_padding = two(output_padding_, 0);
            }

            std::array<int64_t, 2> two(at::IntArrayRef v, int64_t height) const {
                if (v.empty()) {
                    return {height, 0};
                }
                return spatial == 1 ? std::array<int64_t, 2>{height, v[0]}
                                    : std::array<int64_t, 2>{v[0], v[1]};
            }

            at::Tensor lift(const at::Tensor& t) const {
                return (spatial == 1 ? t.unsqueeze(2) : t).to(at::kFloat);
            }

            at::Tensor drop(const at::Tensor& t, at::ScalarType dtype) const {
                auto r = t.to(dtype);
                return spatial == 1 ? r.squeeze(2) : r;
            }
        };

        // A forward 2-D conv TT-NN's conv2d runs: bf16 input and weight on tt, one
        // group (KTT_NATIVE_CONV=0: never).
        bool native_conv(
            const at::Tensor& input, const at::Tensor& weight, int64_t groups
        ) {
            static const bool on = [] {
                const char* v = std::getenv("KTT_NATIVE_CONV");
                return !(v && std::string(v) == "0");
            }();
            // TT-NN's conv2d is wrong for one input and one output channel over a
            // single input row with a kernel wider than 1.
            const bool single_row_channel =
                (input.dim() == 4 && weight.dim() == 4 && input.size(1) == 1 &&
                 weight.size(0) == 1 && input.size(2) == 1 && weight.size(3) > 1);
            return (
                on && groups == 1 && input.dim() == 4 && weight.dim() == 4 &&
                input.is_privateuseone() && input.scalar_type() == at::kBFloat16 &&
                weight.scalar_type() == at::kBFloat16 && input.numel() > 0 &&
                !single_row_channel
            );
        }

        // Output sizes [N, O, OH, OW] of a 2-D conv of `input` with `weight`.
        std::array<int64_t, 4> sizes4_of(
            const at::Tensor& input, const at::Tensor& weight, const Geometry& g
        ) {
            std::array<int64_t, 4> out{input.size(0), weight.size(0), 0, 0};
            for (int k = 0; k < 2; ++k) {
                out[2 + k] =
                    ((input.size(2 + k) + 2 * g.padding[k] -
                      g.dilation[k] * (weight.size(2 + k) - 1) - 1) /
                         g.stride[k] +
                     1);
            }
            return out;
        }

        std::array<int64_t, 4> sizes4(const at::Tensor& t) {
            return {t.size(0), t.size(1), t.size(2), t.size(3)};
        }

        // Transposed conv output sizes [N, C_out, H, W] for 2-D input x and weight w.
        std::array<int64_t, 4> transposed_sizes(
            const at::Tensor& x, const at::Tensor& w, const Geometry& g, int64_t groups
        ) {
            std::array<int64_t, 4> out{x.size(0), w.size(1) * groups, 0, 0};
            for (int k = 0; k < 2; ++k) {
                out[2 + k] =
                    ((x.size(2 + k) - 1) * g.stride[k] - 2 * g.padding[k] +
                     g.dilation[k] * (w.size(2 + k) - 1) + g.output_padding[k] + 1);
            }
            return out;
        }

        // bf16 tt tensors, all non-empty.
        bool bf16_tt(std::initializer_list<const at::Tensor*> ts) {
            for (const auto* t : ts) {
                if (!t->is_privateuseone() || t->scalar_type() != at::kBFloat16 ||
                    t->numel() == 0) {
                    return false;
                }
            }
            return true;
        }

        // The block-float format w is stored as (BFloat16 otherwise).
        bridge::DType weight_format(const at::Tensor& w) {
            if (const auto& st = storage_of(w); st.base &&
                bridge::is_block(bridge::dtype(st.base))) {
                return bridge::dtype(st.base);
            }
            return bridge::DType::BFloat16;
        }

        std::vector<int64_t> sizes_of(const at::Tensor& t) {
            return std::vector<int64_t>(t.sizes().begin(), t.sizes().end());
        }

        // The [N, C, L] base of t when t is a [N, C, 1, L] view of a whole contiguous
        // [N, C, L] storage of t's dtype (PyTorch lifts 1-D convs so); else nullptr.
        bridge::HandlePtr row_base(const at::Tensor& t) {
            if (t.dim() != 4 || t.size(2) != 1 || !t.is_privateuseone() ||
                !t.is_contiguous() || t.storage_offset() != 0) {
                return nullptr;
            }
            const auto& st = storage_of(t);
            const std::vector<int64_t> want{t.size(0), t.size(1), t.size(3)};
            if (!st.base || st.sizes != want || st.dtype != t.scalar_type()) {
                return nullptr;
            }
            return st.base;
        }

        // A conv over one row: 1-D operands, or [N, C, 1, L] views that all have a
        // row_base().
        bool one_row(std::initializer_list<const at::Tensor*> ts) {
            bool all_3d = true, all_rows = true;
            for (const auto* t : ts) {
                all_3d = all_3d && t->dim() == 3;
                all_rows = all_rows && row_base(*t) != nullptr;
            }
            return all_3d || all_rows;
        }

        // A conv operand for the bridge: its handle and sizes (rank 3 for a one-row conv).
        struct Operand {
            bridge::HandlePtr handle;
            std::vector<int64_t> sizes;
        };

        Operand operand(const char* op, const at::Tensor& t, bool rows) {
            if (rows && t.dim() == 4) {
                return {row_base(t), {t.size(0), t.size(1), t.size(3)}};
            }
            return {dense(op, t), sizes_of(t)};
        }

        // A bridge result of sizes `sizes` as a tensor of `like`'s rank.
        at::Tensor result(
            const bridge::HandlePtr& h,
            const std::vector<int64_t>& sizes,
            at::ScalarType dtype,
            const at::Tensor& like
        ) {
            auto out = wrap_output(h, sizes, dtype);
            return like.dim() == 4 && out.dim() == 3 ? out.unsqueeze(2) : out;
        }

        // A forward grouped / depthwise conv, or a one-group strided conv over one row,
        // of bf16 tt input and weight on Kohaku-TT's ktt_conv2 (or TT-NN's conv2d under
        // KTT_GROUPED_CONV=ttnn), bias added; nullopt for other convs.
        std::optional<at::Tensor> grouped_forward(
            const at::Tensor& input,
            const at::Tensor& weight,
            const std::optional<at::Tensor>& bias,
            const Geometry& g,
            int64_t groups
        ) {
            const char* op = "convolution_overrideable";
            if (!bf16_tt({&input, &weight}) || input.dim() != weight.dim()) {
                return std::nullopt;
            }
            const auto format = weight_format(weight);
            const bool rows = one_row({&input, &weight});
            // The route from the sizes; the operands are read only for these convs.
            auto bridge_sizes = [&](const at::Tensor& t) {
                return rows && t.dim() == 4
                    ? std::vector<int64_t>{t.size(0), t.size(1), t.size(3)}
                    : sizes_of(t);
            };
            const auto xs = bridge_sizes(input), ws = bridge_sizes(weight);
            const bool own = bridge::own_grouped_conv(
                xs, ws, g.stride, g.padding, g.dilation, groups, format
            );
            if (!own && !bridge::ttnn_grouped_conv(xs, ws, g.stride, groups)) {
                return std::nullopt;
            }
            auto x = operand(op, input, rows), w = operand(op, weight, rows);
            // TT-NN's conv2d takes [N, C, 1, L].
            const bool lift = !own && x.sizes.size() == 3;
            if (lift) {
                x = operand(op, input.dim() == 3 ? input.unsqueeze(2) : input, false);
                w = operand(op, weight.dim() == 3 ? weight.unsqueeze(2) : weight, false);
            }
            std::vector<int64_t> sizes{x.sizes[0], w.sizes[0]};
            const size_t spatial = x.sizes.size() - 2;
            for (size_t d = 0; d < spatial; ++d) {
                const size_t k = 2 - spatial + d;
                sizes.push_back(
                    (x.sizes[2 + d] + 2 * g.padding[k] -
                     g.dilation[k] * (w.sizes[2 + d] - 1) - 1) /
                        g.stride[k] +
                    1
                );
            }
            auto out = result(
                bridge::conv2d(
                    x.handle, w.handle, g.stride, g.padding, g.dilation, groups, format
                ),
                sizes,
                input.scalar_type(),
                input
            );
            if (out.dim() != input.dim()) {
                out = out.squeeze(2);
            }
            if (bias && bias->defined()) {
                std::vector<int64_t> view(out.dim(), 1);
                view[1] = -1;
                out = out + bias->to(input.scalar_type()).reshape(view);
            }
            return out;
        }

        // A native_conv() forward on Kohaku-TT's channels-last conv, bias added on its
        // Nhwc output: an input stored as Nhwc is read as stored, an Nchw input is
        // converted under KTT_NHWC=1; nullopt when bridge::conv2d_nhwc() does not take it.
        std::optional<at::Tensor> nhwc_forward(
            const at::Tensor& input,
            const at::Tensor& weight,
            const std::optional<at::Tensor>& bias,
            const Geometry& g,
            int64_t groups
        ) {
            const char* op = "convolution_overrideable";
            const auto xs = sizes_of(input), ws = sizes_of(weight);
            if (!bridge::nhwc_conv(xs, ws, g.stride, g.padding, g.dilation, groups)) {
                return std::nullopt;
            }
            bridge::HandlePtr b;
            if (bias && bias->defined()) {
                b = dense_as(op, *bias, input.scalar_type());
            }
            auto out = bridge::conv2d_nhwc(
                dense_any_layout(op, input),
                dense(op, weight),
                b,
                g.stride,
                g.padding,
                g.dilation,
                groups
            );
            if (!out) {
                return std::nullopt;
            }
            const auto s = sizes4_of(input, weight, g);
            return wrap_output(
                out, std::vector<int64_t>(s.begin(), s.end()), input.scalar_type()
            );
        }

        // The input gradient of a conv nhwc_forward() takes, on Kohaku-TT's channels-last
        // conv; nullopt when bridge::conv2d_nhwc_input_grad() does not take it.
        std::optional<at::Tensor> nhwc_input_grad(
            const at::Tensor& grad_output,
            const at::Tensor& input,
            const at::Tensor& weight,
            const Geometry& g,
            int64_t groups
        ) {
            const char* op = "convolution_backward_overrideable";
            const auto xs = sizes_of(input), ws = sizes_of(weight);
            if (!bridge::nhwc_conv(xs, ws, g.stride, g.padding, g.dilation, groups) ||
                !bf16_tt({&grad_output, &weight, &input})) {
                return std::nullopt;
            }
            auto out = bridge::conv2d_nhwc_input_grad(
                dense_any_layout(op, grad_output), dense(op, weight)
            );
            if (!out) {
                return std::nullopt;
            }
            return wrap_output(out, xs, input.scalar_type());
        }

        // The input gradient of a conv grouped_forward() runs on ktt_conv2; nullopt
        // otherwise.
        std::optional<at::Tensor> grouped_input_grad(
            const at::Tensor& grad_output,
            const at::Tensor& input,
            const at::Tensor& weight,
            const Geometry& g,
            int64_t groups
        ) {
            const char* op = "convolution_backward_overrideable";
            if (!bf16_tt({&grad_output, &weight, &input})) {
                return std::nullopt;
            }
            const auto format = weight_format(weight);
            const bool rows = one_row({&grad_output, &input, &weight});
            const auto xs = rows && input.dim() == 4
                ? std::vector<int64_t>{input.size(0), input.size(1), input.size(3)}
                : sizes_of(input);
            const auto w = operand(op, weight, rows);
            if (!bridge::own_grouped_input_grad(
                    xs, w.sizes, g.stride, g.padding, g.dilation, groups, format
                )) {
                return std::nullopt;
            }
            return result(
                bridge::conv2d_input_grad(
                    operand(op, grad_output, rows).handle, w.handle, xs, g.stride,
                    g.padding, g.dilation, groups, format
                ),
                xs,
                input.scalar_type(),
                input
            );
        }

        // The weight gradient of a conv grouped_forward() runs on ktt_conv2; nullopt
        // otherwise.
        std::optional<at::Tensor> grouped_weight_grad(
            const at::Tensor& grad_output,
            const at::Tensor& input,
            const at::Tensor& weight,
            const Geometry& g,
            int64_t groups
        ) {
            const char* op = "convolution_backward_overrideable";
            if (!bf16_tt({&grad_output, &input}) || !weight.is_privateuseone()) {
                return std::nullopt;
            }
            const bool rows = one_row({&grad_output, &input}) &&
                (weight.dim() == 3 || weight.size(2) == 1);
            const auto x = operand(op, input, rows);
            auto ws = sizes_of(weight);
            if (rows && ws.size() == 4) {
                ws.erase(ws.begin() + 2);
            }
            if (!bridge::own_grouped_weight_grad(
                    x.sizes, ws, g.stride, g.padding, g.dilation, groups
                )) {
                return std::nullopt;
            }
            auto gw = result(
                bridge::conv2d_weight_grad(
                    operand(op, grad_output, rows).handle, x.handle, ws, g.stride,
                    g.padding, g.dilation, groups
                ),
                ws,
                at::kBFloat16,
                weight
            );
            return gw.to(weight.scalar_type());
        }

        // A 2-D conv over non-overlapping patches of at least 4 x 4 (stride == kernel,
        // no padding / dilation / groups): patch extraction and one matmul beat per-tap
        // matmuls.
        bool is_patchify(
            const at::Tensor& input,
            const at::Tensor& weight,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            int64_t groups
        ) {
            if (input.dim() != 4 || weight.dim() != 4 || groups != 1 ||
                stride.size() != 2) {
                return false;
            }
            const int64_t kh = weight.size(2), kw = weight.size(3);
            return stride[0] == kh && stride[1] == kw && kh * kw >= 16 &&
                std::all_of(
                       padding.begin(),
                       padding.end(),
                       [](int64_t p) {
                           return p == 0;
                       }
                ) &&
                std::all_of(dilation.begin(), dilation.end(), [](int64_t d) {
                       return d == 1;
                   });
        }

        // x [N, C, H, W] -> rows [N HP WP, C KH KW] -> rows @ w^T + bias -> [N, O, HP,
        // WP].
        at::Tensor patchify(
            const at::Tensor& input,
            const at::Tensor& weight,
            const std::optional<at::Tensor>& bias
        ) {
            const int64_t n = input.size(0), c = input.size(1), o = weight.size(0);
            const int64_t kh = weight.size(2), kw = weight.size(3);
            const int64_t hp = input.size(2) / kh, wp = input.size(3) / kw;
            auto x = input.narrow(2, 0, hp * kh).narrow(3, 0, wp * kw);
            auto rows = x.reshape({n, c, hp, kh, wp, kw})
                            .permute({0, 2, 4, 1, 3, 5})
                            .reshape({n * hp * wp, c * kh * kw});
            std::optional<at::Tensor> b;
            if (bias && bias->defined()) {
                b = bias->to(input.scalar_type());
            }
            auto out = at::linear(
                rows, weight.reshape({o, c * kh * kw}).to(input.scalar_type()), b
            );
            return out.reshape({n, hp, wp, o}).permute({0, 3, 1, 2}).contiguous();
        }

        // Gradients of patchify(): grad rows [N HP WP, O] give grad_w = rows^T-product,
        // grad_x = grad rows @ w scattered back into patches (zero outside the cropped
        // region).
        std::tuple<at::Tensor, at::Tensor, at::Tensor> patchify_backward(
            const at::Tensor& grad_output,
            const at::Tensor& input,
            const at::Tensor& weight,
            std::array<bool, 3> mask
        ) {
            const int64_t n = input.size(0), c = input.size(1), o = weight.size(0);
            const int64_t kh = weight.size(2), kw = weight.size(3);
            const int64_t hp = input.size(2) / kh, wp = input.size(3) / kw;
            const auto dtype = input.scalar_type();
            auto grad_rows = grad_output.to(dtype)
                                 .permute({0, 2, 3, 1})
                                 .reshape({n * hp * wp, o});
            at::Tensor grad_input, grad_weight, grad_bias;
            if (mask[0]) {
                auto w = weight.reshape({o, c * kh * kw}).to(dtype);
                auto patches = at::mm(grad_rows, w)
                                   .reshape({n, hp, wp, c, kh, kw})
                                   .permute({0, 3, 1, 4, 2, 5})
                                   .reshape({n, c, hp * kh, wp * kw});
                const bool cropped = (
                    hp * kh != input.size(2) || wp * kw != input.size(3)
                );
                grad_input = cropped
                    ? at::constant_pad_nd(
                          patches,
                          {0, input.size(3) - wp * kw, 0, input.size(2) - hp * kh}
                      )
                    : patches;
            }
            if (mask[1]) {
                auto x = input.narrow(2, 0, hp * kh).narrow(3, 0, wp * kw);
                auto rows = x.reshape({n, c, hp, kh, wp, kw})
                                .permute({0, 2, 4, 1, 3, 5})
                                .reshape({n * hp * wp, c * kh * kw});
                grad_weight = (
                    at::mm(grad_rows.t(), rows)
                        .reshape(weight.sizes())
                        .to(weight.scalar_type())
                );
            }
            if (mask[2]) {
                grad_bias = (
                    grad_rows.to(at::kFloat).sum(0).to(grad_output.scalar_type())
                );
            }
            return {grad_input, grad_weight, grad_bias};
        }

        at::Tensor convolution(
            const at::Tensor& input,
            const at::Tensor& weight,
            const std::optional<at::Tensor>& bias,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool transposed,
            at::IntArrayRef output_padding,
            int64_t groups
        );

        // 3-D convolution (or its transpose) as a sum over depth taps of 2-D ones:
        // the depths each tap reads folded into the batch, results summed in fp32.
        at::Tensor convolution3d(
            const at::Tensor& input,
            const at::Tensor& weight,
            const std::optional<at::Tensor>& bias,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool transposed,
            at::IntArrayRef output_padding,
            int64_t groups
        ) {
            const char* op = "convolution_overrideable";
            KTT_REQUIRE(input.dim() == 5, op, "3-D convolution needs [N, C, T, H, W]");
            auto at3 = [](at::IntArrayRef v, int64_t fill) {
                return v.empty() ? std::array<int64_t, 3>{fill, fill, fill}
                                 : std::array<int64_t, 3>{v[0], v[1], v[2]};
            };
            auto s = at3(stride, 1), p = at3(padding, 0), d = at3(dilation, 1);
            auto op_pad = at3(output_padding, 0);
            const int64_t n = input.size(0), c = input.size(1), t_in = input.size(2);
            const int64_t h = input.size(3), w = input.size(4), taps = weight.size(2);
            std::vector<int64_t> s2{s[1], s[2]}, p2{p[1], p[2]}, d2{d[1], d[2]};
            std::vector<int64_t> op2{op_pad[1], op_pad[2]};
            const int64_t out_channels = (
                transposed ? weight.size(1) * groups : weight.size(0)
            );
            at::Tensor acc;
            int64_t t_out = 0;
            if (!transposed) {
                t_out = (t_in + 2 * p[0] - d[0] * (taps - 1) - 1) / s[0] + 1;
                auto padded = at::constant_pad_nd(input, {0, 0, 0, 0, p[0], p[0]}, 0);
                for (int64_t k = 0; k < taps; ++k) {
                    auto depth = at::arange(t_out, at::kLong) * s[0] + k * d[0];
                    auto x = padded.index_select(2, depth.to(input.device()))
                                 .transpose(1, 2)
                                 .reshape({n * t_out, c, h, w});
                    auto y = ktt::convolution(
                        x,
                        weight.select(2, k),
                        std::nullopt,
                        s2,
                        p2,
                        d2,
                        false,
                        {0, 0},
                        groups
                    );
                    auto f = y.to(at::kFloat);
                    acc = acc.defined() ? acc + f : f;
                }
                acc = acc.reshape({n, t_out, out_channels, acc.size(2), acc.size(3)});
            } else {
                t_out =
                    ((t_in - 1) * s[0] - 2 * p[0] + d[0] * (taps - 1) + op_pad[0] + 1);
                auto x = input.transpose(1, 2).reshape({n * t_in, c, h, w});
                for (int64_t k = 0; k < taps; ++k) {
                    auto y = (
                        ktt::convolution(
                            x,
                            weight.select(2, k),
                            std::nullopt,
                            s2,
                            p2,
                            d2,
                            true,
                            op2,
                            groups
                        )
                    );
                    y = y.to(at::kFloat)
                            .reshape({n, t_in, out_channels, y.size(2), y.size(3)});
                    if (!acc.defined()) {
                        acc = at::zeros(
                            {n, t_out, out_channels, y.size(3), y.size(4)}, y.options()
                        );
                    }
                    std::vector<int64_t> src, dst;
                    for (int64_t t = 0; t < t_in; ++t) {
                        const int64_t pos = t * s[0] + k * d[0] - p[0];
                        if (pos >= 0 && pos < t_out) {
                            src.push_back(t);
                            dst.push_back(pos);
                        }
                    }
                    if (src.empty()) {
                        continue;
                    }
                    auto device = input.device();
                    auto picked = (
                        y.index_select(1, at::tensor(src, at::kLong).to(device))
                    );
                    acc = (
                        acc.index_add(1, at::tensor(dst, at::kLong).to(device), picked)
                    );
                }
            }
            auto out = acc.transpose(1, 2);
            if (bias && bias->defined()) {
                out = out + bias->to(at::kFloat).view({1, out_channels, 1, 1, 1});
            }
            return out.contiguous().to(input.scalar_type());
        }

        // Convolution forward; transposed = the gradient of a conv wrt its input.
        at::Tensor convolution(
            const at::Tensor& input,
            const at::Tensor& weight,
            const std::optional<at::Tensor>& bias,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool transposed,
            at::IntArrayRef output_padding,
            int64_t groups
        ) {
            const char* op = "convolution_overrideable";
            if (weight.dim() == 5) {
                return (
                    convolution3d(
                        input,
                        weight,
                        bias,
                        stride,
                        padding,
                        dilation,
                        transposed,
                        output_padding,
                        groups
                    )
                );
            }
            Geometry g(op, weight, stride, padding, dilation, output_padding);
            KTT_REQUIRE(input.dim() == g.spatial + 2, op, "input needs a batch dim");
            if (!transposed &&
                is_patchify(input, weight, stride, padding, dilation, groups)) {
                return patchify(input, weight, bias);
            }
            if (!transposed) {
                if (auto out = grouped_forward(input, weight, bias, g, groups)) {
                    return *out;
                }
            }
            if (!transposed && native_conv(input, weight, groups)) {
                if (auto out = nhwc_forward(input, weight, bias, g, groups)) {
                    return *out;
                }
                const auto s = sizes4_of(input, weight, g);
                auto out = (
                    wrap_output(
                        bridge::conv2d(
                            dense_propagate(op, input),
                            dense(op, weight),
                            g.stride,
                            g.padding,
                            g.dilation,
                            groups
                        ),
                        std::vector<int64_t>(s.begin(), s.end()),
                        input.scalar_type()
                    )
                );
                if (bias && bias->defined()) {
                    out = out + bias->to(input.scalar_type()).reshape({1, -1, 1, 1});
                }
                return out;
            }
            auto x = g.lift(input), w = g.lift(weight);
            at::Tensor out;
            if (transposed) {
                auto grads = conv2d_backward(
                    x,
                    x,
                    w,
                    transposed_sizes(x, w, g, groups),
                    g.stride,
                    g.padding,
                    g.dilation,
                    groups,
                    true,  // need_input
                    false  // need_weight
                );
                out = grads.first;
            } else {
                out = conv2d(x, w, g.stride, g.padding, g.dilation, groups);
            }
            if (bias && bias->defined()) {
                out = out + bias->to(at::kFloat).reshape({1, -1, 1, 1});
            }
            return g.drop(out, input.scalar_type());
        }

        // Convolution gradients; for transposed convs the roles of input and output
        // swap.
        std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
            const at::Tensor& grad_output,
            const at::Tensor& input,
            const at::Tensor& weight,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef dilation,
            bool transposed,
            at::IntArrayRef output_padding,
            int64_t groups,
            std::array<bool, 3> mask
        ) {
            const char* op = "convolution_backward_overrideable";
            if (!transposed &&
                is_patchify(input, weight, stride, padding, dilation, groups)) {
                return patchify_backward(grad_output, input, weight, mask);
            }
            Geometry g(op, weight, stride, padding, dilation, output_padding);
            std::optional<at::Tensor> own_gi, own_gw;
            if (!transposed && mask[0]) {
                own_gi = nhwc_input_grad(grad_output, input, weight, g, groups);
                if (!own_gi) {
                    own_gi = grouped_input_grad(grad_output, input, weight, g, groups);
                }
            }
            if (!transposed && mask[1]) {
                own_gw = grouped_weight_grad(grad_output, input, weight, g, groups);
            }
            const bool need_gi = mask[0] && !own_gi, need_gw = mask[1] && !own_gw;
            at::Tensor gi, gw;
            if (!transposed && (need_gi || need_gw)) {
                auto go = g.lift(grad_output), x = g.lift(input), w = g.lift(weight);
                std::tie(gi, gw) = (conv2d_backward(
                    go,
                    x,
                    w,
                    sizes4(x),
                    g.stride,
                    g.padding,
                    g.dilation,
                    groups,
                    need_gi,
                    need_gw
                ));
            } else if (transposed) {
                auto go = g.lift(grad_output), x = g.lift(input), w = g.lift(weight);
                if (mask[0]) {
                    gi = conv2d(go, w, g.stride, g.padding, g.dilation, groups);
                    gi = gi.narrow(2, 0, x.size(2)).narrow(3, 0, x.size(3));
                }
                if (mask[1]) {
                    // A conv from grad_output to input has this weight: its weight
                    // gradient.
                    auto grads = conv2d_backward(
                        x,
                        go,
                        w,
                        sizes4(go),
                        g.stride,
                        g.padding,
                        g.dilation,
                        groups,
                        false,  // need_input
                        true    // need_weight
                    );
                    gw = grads.second;
                }
            }
            at::Tensor grad_input, grad_weight, grad_bias;
            if (mask[0]) {
                grad_input = own_gi ? *own_gi : g.drop(gi, input.scalar_type());
            }
            if (mask[1]) {
                grad_weight = own_gw ? *own_gw : g.drop(gw, weight.scalar_type());
            }
            if (mask[2]) {
                std::vector<int64_t> dims{0};
                for (int64_t d = 2; d < grad_output.dim(); ++d) {
                    dims.push_back(d);
                }
                auto sum = grad_output.to(at::kFloat).sum(dims);
                grad_bias = sum.to(grad_output.scalar_type());
            }
            return {grad_input, grad_weight, grad_bias};
        }

        // fold: columns [N, C * KH * KW, L] summed into an image [N, C, H, W]; each
        // tap's L values are scatter-added at that tap's padded-image positions, then
        // cropped.
        at::Tensor tt_col2im(
            const at::Tensor& self,
            at::IntArrayRef output_size,
            at::IntArrayRef kernel_size,
            at::IntArrayRef dilation,
            at::IntArrayRef padding,
            at::IntArrayRef stride
        ) {
            const char* op = "col2im";
            KTT_REQUIRE(
                self.dim() == 2 || self.dim() == 3, op, "input must be 2-D or 3-D"
            );
            auto cols = self.dim() == 2 ? self.unsqueeze(0) : self;
            const int64_t n = cols.size(0), kh = kernel_size[0], kw = kernel_size[1];
            const int64_t c = cols.size(1) / (kh * kw);
            const int64_t h = output_size[0], w = output_size[1];
            const int64_t hp = h + 2 * padding[0], wp = w + 2 * padding[1];
            const int64_t oh = (hp - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
            const int64_t ow = (wp - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
            KTT_REQUIRE(
                oh * ow == cols.size(2), op, "column count does not match the sizes"
            );

            auto taps = cols.to(at::kFloat).reshape({n, c, kh * kw, oh * ow});
            auto image = at::zeros({n * c, hp * wp}, cols.options().dtype(at::kFloat));
            for (int64_t i = 0; i < kh; ++i) {
                for (int64_t j = 0; j < kw; ++j) {
                    const int64_t top = i * dilation[0], left = j * dilation[1];
                    std::array<int64_t, 2> steps{stride[0], stride[1]};
                    auto index = tap_positions(top, left, oh, ow, steps, wp)
                                     .reshape({1, oh * ow})
                                     .expand({n * c, oh * ow});
                    auto part = taps.select(2, i * kw + j).reshape({n * c, oh * ow});
                    image = at::scatter_add(image, 1, index, part);
                }
            }
            auto out = image.reshape({n, c, hp, wp})
                           .slice(2, padding[0], padding[0] + h)
                           .slice(3, padding[1], padding[1] + w)
                           .contiguous()
                           .to(self.scalar_type());
            return self.dim() == 2 ? out.squeeze(0) : out;
        }

        // unfold: image [N, C, H, W] -> columns [N, C * KH * KW, L], each tap's strided
        // patch of the padded image stacked along the kernel axis.
        at::Tensor tt_im2col(
            const at::Tensor& self,
            at::IntArrayRef kernel_size,
            at::IntArrayRef dilation,
            at::IntArrayRef padding,
            at::IntArrayRef stride
        ) {
            const char* op = "im2col";
            KTT_REQUIRE(
                self.dim() == 3 || self.dim() == 4, op, "input must be 3-D or 4-D"
            );
            auto x = self.dim() == 3 ? self.unsqueeze(0) : self;
            const int64_t n = x.size(0), c = x.size(1), kh = kernel_size[0];
            const int64_t kw = kernel_size[1];
            const int64_t hp = x.size(2) + 2 * padding[0];
            const int64_t wp = x.size(3) + 2 * padding[1];
            const int64_t oh = (hp - dilation[0] * (kh - 1) - 1) / stride[0] + 1;
            const int64_t ow = (wp - dilation[1] * (kw - 1) - 1) / stride[1] + 1;
            KTT_REQUIRE(oh > 0 && ow > 0, op, "kernel larger than the padded input");
            if (n == 0 || c == 0) {
                auto out = at::empty({n, c * kh * kw, oh * ow}, self.options());
                return self.dim() == 3 ? out.squeeze(0) : out;
            }
            auto padded = x;
            if (padding[0] > 0 || padding[1] > 0) {
                padded = at::constant_pad_nd(
                    x, {padding[1], padding[1], padding[0], padding[0]}
                );
            }
            std::vector<at::Tensor> taps;
            for (int64_t i = 0; i < kh; ++i) {
                for (int64_t j = 0; j < kw; ++j) {
                    const int64_t top = i * dilation[0], left = j * dilation[1];
                    const int64_t bottom = top + (oh - 1) * stride[0] + 1;
                    const int64_t right = left + (ow - 1) * stride[1] + 1;
                    auto patch = (
                        padded.slice(2, top, bottom, stride[0])
                        .slice(3, left, right, stride[1])
                    );
                    taps.push_back(patch.reshape({n, c, oh * ow}));
                }
            }
            auto out = at::stack(taps, 2).reshape({n, c * kh * kw, oh * ow});
            return self.dim() == 3 ? out.squeeze(0) : out;
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("im2col", &tt_im2col);
        m.impl("col2im", &tt_col2im);
        m.impl("convolution_overrideable", &convolution);
        m.impl("convolution_backward_overrideable", &convolution_backward);
    }

}  // namespace ktt
