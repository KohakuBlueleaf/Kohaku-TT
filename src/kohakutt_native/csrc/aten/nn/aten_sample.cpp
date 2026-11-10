// aten grid_sampler_2d for tt (bilinear / nearest; zeros / border / reflection) and its
// backward: fp32 corner indices and weights on device, gathers / scatter_adds.
#include <torch/library.h>

#include <array>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        constexpr int64_t kBilinear = 0, kNearest = 1;
        constexpr int64_t kZeros = 0, kBorder = 1, kReflection = 2;

        // A pixel coordinate along one dim and its derivative wrt the normalized
        // coordinate.
        struct Coord {
            at::Tensor value, grad;
        };

        // PyTorch's reflect_coordinates into [twice_low / 2, twice_high / 2]; the
        // derivative factor is +-1 (direction after the reflections).
        Coord reflect(const at::Tensor& x, int64_t twice_low, int64_t twice_high) {
            if (twice_low == twice_high) {
                return {at::zeros_like(x), at::zeros_like(x)};
            }
            const double low = twice_low / 2.0, span = (twice_high - twice_low) / 2.0;
            auto offset = x - low;
            auto d = at::abs(offset);
            auto flips = at::floor(d / span);
            auto extra = d - flips * span;
            auto even = at::eq(flips - at::floor(flips * 0.5) * 2.0, 0);
            auto direction = at::where(at::lt(offset, 0), -1.0, 1.0);
            return {
                at::where(even, extra + low, (span - extra) + low),
                at::where(even, direction, -direction)
            };
        }

        // Normalized coordinate -> pixel coordinate after the padding mode's border /
        // reflection handling, with d(pixel) / d(normalized).
        Coord source_coord(
            const at::Tensor& coord, int64_t size, int64_t padding, bool align_corners
        ) {
            const double scale =
                (align_corners ? 0.5 * static_cast<double>(size - 1)
                               : 0.5 * static_cast<double>(size));
            const double shift = align_corners ? 0.0 : 0.5;
            Coord c{(coord + 1.0) * scale - shift, at::full_like(coord, scale)};
            if (padding == kReflection) {
                auto r =
                    (align_corners ? reflect(c.value, 0, 2 * (size - 1))
                                   : reflect(c.value, -1, 2 * size - 1));
                c = {r.value, c.grad * r.grad};
            }
            if (padding == kBorder || padding == kReflection) {
                // Gradient flows only strictly inside (PyTorch's
                // clip_coordinates_set_grad).
                auto inside = (at::logical_and(
                    at::gt(c.value, 0), at::lt(c.value, size - 1)
                ));
                c = {at::clamp(c.value, 0, size - 1), c.grad * inside.to(at::kFloat)};
            }
            return c;
        }

        // One sampling corner: flat input index [N, 1, L], weight and its x / y
        // derivatives [N, 1, L] (0 when the corner is outside the input).
        struct Corner {
            at::Tensor index, weight, dweight_dx, dweight_dy;
        };

        // The corners (4 bilinear, 1 nearest) of pixel coordinates ix, iy [N, Ho, Wo].
        std::vector<Corner> corners(
            const at::Tensor& ix,
            const at::Tensor& iy,
            int64_t h,
            int64_t w,
            bool nearest
        ) {
            const int64_t n = ix.size(0), count = ix.numel() / n;
            auto make = [&](const at::Tensor& x,
                            const at::Tensor& y,
                            const at::Tensor& wt,
                            const at::Tensor& dx,
                            const at::Tensor& dy) {
                auto in_x = at::logical_and(at::ge(x, 0), at::le(x, w - 1));
                auto in_y = at::logical_and(at::ge(y, 0), at::le(y, h - 1));
                auto inside = at::logical_and(in_x, in_y).to(at::kFloat);
                auto cx = at::clamp(x, 0, w - 1), cy = at::clamp(y, 0, h - 1);
                auto index = (cy * static_cast<double>(w) + cx).to(at::kLong);
                auto flat = [&](const at::Tensor& t) {
                    return (t * inside).reshape({n, 1, count});
                };
                return Corner{
                    index.reshape({n, 1, count}), flat(wt), flat(dx), flat(dy)
                };
            };
            if (nearest) {
                auto zero = at::zeros_like(ix);
                return {
                    make(at::round(ix), at::round(iy), at::ones_like(ix), zero, zero)
                };
            }
            auto x0 = at::floor(ix), y0 = at::floor(iy);
            auto x1 = x0 + 1.0, y1 = y0 + 1.0;
            auto wx1 = ix - x0, wy1 = iy - y0;
            auto wx0 = 1.0 - wx1, wy0 = 1.0 - wy1;
            return {
                make(x0, y0, wx0 * wy0, -wy0, -wx0),
                make(x1, y0, wx1 * wy0, wy0, -wx1),
                make(x0, y1, wx0 * wy1, -wy1, wx0),
                make(x1, y1, wx1 * wy1, wy1, wx1)
            };
        }

        constexpr int64_t kBicubic = 2;

        void check_mode(const char* op, int64_t interpolation) {
            KTT_REQUIRE(
                interpolation == kBilinear || interpolation == kNearest,
                op,
                "bicubic interpolation is not supported yet"
            );
        }

        // PyTorch's bicubic weights (a = -0.75) of the taps floor(x) - 1 .. + 2 at
        // offset t = x - floor(x).
        std::array<at::Tensor, 4> cubic_weights(const at::Tensor& t) {
            constexpr double a = -0.75;
            auto near = [](const at::Tensor& x) {
                return ((x * (a + 2) - (a + 3)) * x) * x + 1.0;
            };
            auto far = [](const at::Tensor& x) {
                return ((x * a - 5 * a) * x + 8 * a) * x - 4 * a;
            };
            return {far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)};
        }

        // A bicubic tap coordinate after the padding mode (border: clipped;
        // reflection: reflected, then clipped).
        at::Tensor tap_coord(
            const at::Tensor& x, int64_t size, int64_t padding, bool align_corners
        ) {
            if (padding == kBorder) {
                return at::clamp(x, 0, size - 1);
            }
            if (padding == kReflection) {
                auto r =
                    (align_corners ? reflect(x, 0, 2 * (size - 1))
                                   : reflect(x, -1, 2 * size - 1));
                return at::clamp(r.value, 0, size - 1);
            }
            return x;
        }

        // Bicubic grid sampling: 4 x 4 taps around the unnormalized coordinate, each
        // tap's coordinate through the padding mode, zero outside the input.
        at::Tensor bicubic_sample(
            const at::Tensor& input,
            const at::Tensor& grid,
            int64_t padding,
            bool align_corners
        ) {
            const int64_t n = input.size(0), c = input.size(1);
            const int64_t h = input.size(2), w = input.size(3);
            const int64_t ho = grid.size(1), wo = grid.size(2), count = ho * wo;
            auto g = grid.to(at::kFloat);
            auto unnormalize = [&](const at::Tensor& coord, int64_t size) {
                return (
                    align_corners ? (coord + 1.0) * (0.5 * (size - 1))
                                  : ((coord + 1.0) * size - 1.0) * 0.5
                );
            };
            auto ix = unnormalize(g.select(3, 0), w),
                 iy = unnormalize(g.select(3, 1), h);
            auto x0 = at::floor(ix), y0 = at::floor(iy);
            auto wx = cubic_weights(ix - x0), wy = cubic_weights(iy - y0);
            auto flat = input.to(at::kFloat).reshape({n, c, h * w});
            at::Tensor out;
            for (int64_t j = 0; j < 4; ++j) {
                auto ty = (tap_coord(
                    y0 + static_cast<double>(j - 1), h, padding, align_corners
                ));
                for (int64_t i = 0; i < 4; ++i) {
                    auto tx = (tap_coord(
                        x0 + static_cast<double>(i - 1), w, padding, align_corners
                    ));
                    auto inside = (at::logical_and(
                        at::logical_and(at::ge(tx, 0), at::le(tx, w - 1)),
                        at::logical_and(at::ge(ty, 0), at::le(ty, h - 1))
                    ));
                    auto index = (at::clamp(ty, 0, h - 1) * static_cast<double>(w) +
                                  at::clamp(tx, 0, w - 1))
                                     .to(at::kLong)
                                     .reshape({n, 1, count});
                    auto weight = (wx[i] * wy[j] * inside.to(at::kFloat))
                                      .reshape({n, 1, count});
                    auto term = at::gather(flat, 2, index.expand({n, c, count})) *
                        weight;
                    out = out.defined() ? out + term : term;
                }
            }
            return out.reshape({n, c, ho, wo}).to(input.scalar_type());
        }

        // input [N, C, H, W] sampled at grid [N, Ho, Wo, 2] (x, y) -> [N, C, Ho, Wo].
        at::Tensor tt_grid_sampler_2d(
            const at::Tensor& input,
            const at::Tensor& grid,
            int64_t interpolation,
            int64_t padding,
            bool align_corners
        ) {
            const int64_t n = input.size(0), c = input.size(1);
            const int64_t h = input.size(2), w = input.size(3);
            const int64_t ho = grid.size(1), wo = grid.size(2), count = ho * wo;
            if (n == 0 || c == 0 || count == 0) {
                return at::empty({n, c, ho, wo}, input.options());
            }
            if (interpolation == kBicubic) {
                return bicubic_sample(input, grid, padding, align_corners);
            }
            check_mode("grid_sampler_2d", interpolation);
            auto g = grid.to(at::kFloat);
            auto ix = source_coord(g.select(3, 0), w, padding, align_corners).value;
            auto iy = source_coord(g.select(3, 1), h, padding, align_corners).value;
            auto flat = input.to(at::kFloat).reshape({n, c, h * w});
            at::Tensor out;
            for (const auto& k : corners(ix, iy, h, w, interpolation == kNearest)) {
                auto values = at::gather(flat, 2, k.index.expand({n, c, count}));
                auto term = values * k.weight;
                out = out.defined() ? out + term : term;
            }
            return out.reshape({n, c, ho, wo}).to(input.scalar_type());
        }

        // (grad_input: grad * weight scatter-added at each corner; grad_grid: sum over
        // C of grad * value * d weight / d pixel, times d pixel / d grid) per
        // output_mask.
        std::tuple<at::Tensor, at::Tensor> tt_grid_sampler_2d_backward(
            const at::Tensor& grad_output,
            const at::Tensor& input,
            const at::Tensor& grid,
            int64_t interpolation,
            int64_t padding,
            bool align_corners,
            std::array<bool, 2> mask
        ) {
            check_mode("grid_sampler_2d_backward", interpolation);
            const int64_t n = input.size(0), c = input.size(1);
            const int64_t h = input.size(2), w = input.size(3);
            const int64_t ho = grid.size(1), wo = grid.size(2), count = ho * wo;
            auto g = grid.to(at::kFloat);
            auto cx = source_coord(g.select(3, 0), w, padding, align_corners);
            auto cy = source_coord(g.select(3, 1), h, padding, align_corners);
            auto go = grad_output.to(at::kFloat).reshape({n, c, count});
            auto flat = input.to(at::kFloat).reshape({n, c, h * w});

            at::Tensor grad_input = at::zeros({n, c, h * w}, flat.options());
            at::Tensor dx = at::zeros({n, 1, count}, flat.options());
            at::Tensor dy = at::zeros({n, 1, count}, flat.options());
            for (
                const auto& k :
                corners(cx.value, cy.value, h, w, interpolation == kNearest)
            ) {
                auto index = k.index.expand({n, c, count});
                if (mask[0]) {
                    grad_input = at::scatter_add(grad_input, 2, index, go * k.weight);
                }
                if (mask[1] && interpolation == kBilinear) {
                    auto gv = (go * at::gather(flat, 2, index))
                                  .sum(1, true);  // [N, 1, L]
                    dx = dx + gv * k.dweight_dx;
                    dy = dy + gv * k.dweight_dy;
                }
            }
            at::Tensor grad_grid;
            if (mask[1]) {
                auto gx = dx.reshape({n, ho, wo}) * cx.grad;
                auto gy = dy.reshape({n, ho, wo}) * cy.grad;
                grad_grid = at::stack({gx, gy}, 3).to(grid.scalar_type());
            }
            if (mask[0]) {
                grad_input = grad_input.reshape(input.sizes()).to(input.scalar_type());
            } else {
                grad_input = at::Tensor();
            }
            return {grad_input, grad_grid};
        }

        // input [N, C, D, H, W] sampled at grid [N, Do, Ho, Wo, 3] (x, y, z) ->
        // [N, C, Do, Ho, Wo]: trilinear (8 corners) or nearest, zero outside.
        at::Tensor tt_grid_sampler_3d(
            const at::Tensor& input,
            const at::Tensor& grid,
            int64_t interpolation,
            int64_t padding,
            bool align_corners
        ) {
            check_mode("grid_sampler_3d", interpolation);
            const int64_t n = input.size(0), c = input.size(1);
            const int64_t d = input.size(2), h = input.size(3), w = input.size(4);
            const int64_t od = grid.size(1), oh = grid.size(2), ow = grid.size(3);
            const int64_t count = od * oh * ow;
            if (n == 0 || c == 0 || count == 0) {
                return at::empty({n, c, od, oh, ow}, input.options());
            }
            auto g = grid.to(at::kFloat);
            auto ix = source_coord(g.select(4, 0), w, padding, align_corners).value;
            auto iy = source_coord(g.select(4, 1), h, padding, align_corners).value;
            auto iz = source_coord(g.select(4, 2), d, padding, align_corners).value;
            auto flat = input.to(at::kFloat).reshape({n, c, d * h * w});
            auto sample = [&](const at::Tensor& x,
                              const at::Tensor& y,
                              const at::Tensor& z,
                              const at::Tensor& weight) {
                auto inside = (at::logical_and(
                    at::logical_and(
                        at::logical_and(at::ge(x, 0), at::le(x, w - 1)),
                        at::logical_and(at::ge(y, 0), at::le(y, h - 1))
                    ),
                    at::logical_and(at::ge(z, 0), at::le(z, d - 1))
                ));
                auto index = ((at::clamp(z, 0, d - 1) * static_cast<double>(h) +
                               at::clamp(y, 0, h - 1)) *
                                  static_cast<double>(w) +
                              at::clamp(x, 0, w - 1))
                                 .to(at::kLong)
                                 .reshape({n, 1, count});
                auto wt = (weight * inside.to(at::kFloat)).reshape({n, 1, count});
                return at::gather(flat, 2, index.expand({n, c, count})) * wt;
            };
            at::Tensor out;
            if (interpolation == kNearest) {
                out = (sample(
                    at::round(ix), at::round(iy), at::round(iz), at::ones_like(ix)
                ));
            } else {
                auto x0 = at::floor(ix), y0 = at::floor(iy), z0 = at::floor(iz);
                auto tx = ix - x0, ty = iy - y0, tz = iz - z0;
                for (int64_t k = 0; k < 8; ++k) {
                    const bool bx = k & 1, by = k & 2, bz = k & 4;
                    auto wx = bx ? tx : 1.0 - tx, wy = by ? ty : 1.0 - ty;
                    auto wz = bz ? tz : 1.0 - tz;
                    auto term = (sample(
                        x0 + (bx ? 1.0 : 0.0),
                        y0 + (by ? 1.0 : 0.0),
                        z0 + (bz ? 1.0 : 0.0),
                        wx * wy * wz
                    ));
                    out = out.defined() ? out + term : term;
                }
            }
            return out.reshape({n, c, od, oh, ow}).to(input.scalar_type());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("grid_sampler_3d", &tt_grid_sampler_3d);
        m.impl("grid_sampler_2d", &tt_grid_sampler_2d);
        m.impl("grid_sampler_2d_backward", &tt_grid_sampler_2d_backward);
    }

}  // namespace ktt
