// aten grid_sampler_2d for tt (bilinear / nearest; zeros / border / reflection) and its
// backward: fp32 corner indices and weights on device, gathers / scatter_adds.
#include <torch/library.h>

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
                auto inside = (
                    at::logical_and(at::gt(c.value, 0), at::lt(c.value, size - 1))
                );
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

        void check_mode(const char* op, int64_t interpolation) {
            KTT_REQUIRE(
                interpolation == kBilinear || interpolation == kNearest,
                op,
                "bicubic interpolation is not supported yet"
            );
        }

        // input [N, C, H, W] sampled at grid [N, Ho, Wo, 2] (x, y) -> [N, C, Ho, Wo].
        at::Tensor tt_grid_sampler_2d(
            const at::Tensor& input,
            const at::Tensor& grid,
            int64_t interpolation,
            int64_t padding,
            bool align_corners
        ) {
            check_mode("grid_sampler_2d", interpolation);
            const int64_t n = input.size(0), c = input.size(1);
            const int64_t h = input.size(2), w = input.size(3);
            const int64_t ho = grid.size(1), wo = grid.size(2), count = ho * wo;
            if (n == 0 || c == 0 || count == 0) {
                return at::empty({n, c, ho, wo}, input.options());
            }
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

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("grid_sampler_2d", &tt_grid_sampler_2d);
        m.impl("grid_sampler_2d_backward", &tt_grid_sampler_2d_backward);
    }

}  // namespace ktt
