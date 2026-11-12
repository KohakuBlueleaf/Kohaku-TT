// Bridge group norm pieces: Kohaku-TT's per-plane moments and affine kernels
// (ktt_group_norm).
#include <array>
#include <optional>

#include "aten/nn/ktt_group_norm/program.hpp"
#include "core/device/handle.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ktt::bridge {

    std::pair<HandlePtr, HandlePtr> group_norm_moments(const HandlePtr& x) {
        const std::vector<int64_t> shape{x->shape[0], x->shape[1], 1, 1};
        std::pair<HandlePtr, HandlePtr> out{
            detail::make_handle(shape, DType::Float32),
            detail::make_handle(shape, DType::Float32)
        };
        detail::counters().ops++;
        detail::submit([out, x] {
            auto [s1, s2] = ktt_group_norm::moments(x->tensor);
            out.first->tensor = s1;
            out.second->tensor = s2;
            detail::check_shape(*out.first);
            detail::check_shape(*out.second);
        });
        return out;
    }

    std::array<HandlePtr, 4> group_norm_stats(
        const HandlePtr& s1,
        const HandlePtr& s2,
        const HandlePtr& weight,
        const HandlePtr& bias,
        int64_t groups,
        double count,
        double eps,
        DType out
    ) {
        const int64_t n = s1->shape[0], c = s1->shape[1];
        const std::vector<int64_t> planes{n, c, 1, 1}, per_group{n, groups};
        std::array<HandlePtr, 4> r{
            detail::make_handle(planes, DType::Float32),
            detail::make_handle(planes, DType::Float32),
            detail::make_handle(per_group, out),
            detail::make_handle(per_group, out)
        };
        detail::counters().ops++;
        detail::submit([=] {
            using tt::tt_metal::Layout;
            const auto tiled = [](const HandlePtr& h) -> std::optional<ttnn::Tensor> {
                if (!h) {
                    return std::nullopt;
                }
                return (
                    h->tensor.layout() == Layout::TILE
                        ? h->tensor
                        : ttnn::to_layout(h->tensor, Layout::TILE)
                );
            };
            const auto w = tiled(weight), b = tiled(bias);
            auto st = (ktt_group_norm::stats(
                s1->tensor,
                s2->tensor,
                w ? &*w : nullptr,
                b ? &*b : nullptr,
                static_cast<uint32_t>(groups),
                static_cast<float>(count),
                static_cast<float>(eps),
                detail::to_ttnn(out)
            ));
            r[0]->tensor = st.a;
            r[1]->tensor = st.b;
            r[2]->tensor = st.mean;
            r[3]->tensor = st.rstd;
            for (const auto& h : r) {
                detail::check_shape(*h);
            }
        });
        return r;
    }

    HandlePtr group_norm_affine(
        const HandlePtr& x, const HandlePtr& a, const HandlePtr& b
    ) {
        const int64_t bytes = x->bytes;
        return detail::timed_op(x->dtype, x->shape, [=] {
            return (ktt_group_norm::affine(
                x->tensor, a->tensor, b->tensor, detail::activation_config(bytes)
            ));
        });
    }

    bool group_norm_rows_supported(const HandlePtr& x, int64_t groups) {
        if (x->dtype != DType::BFloat16 && x->dtype != DType::Float32) {
            return false;
        }
        const bool nhwc = x->format == Format::Nhwc;
        if (!nhwc && (x->format != Format::Nchw || x->shape.size() != 3)) {
            return false;
        }
        // [N, C, H, W] stored channels-last, or a tile-padding-free [N, L, C].
        const int64_t n = x->shape[0], c = nhwc ? x->shape[1] : x->shape[2];
        const int64_t rows = nhwc ? x->shape[2] * x->shape[3] : x->shape[1];
        if (!nhwc && (rows % 32 != 0 || c % 32 != 0)) {
            return false;
        }
        return n > 0 && c > 0 && rows > 0 && groups > 0 &&
            ktt_group_norm::rows_supported(
                   {static_cast<uint32_t>(n),
                    static_cast<uint32_t>(rows),
                    static_cast<uint32_t>(c),
                    static_cast<uint32_t>(groups),
                    0.0f}
            );
    }

    bool group_norm_twin_supported(const HandlePtr& x, int64_t groups) {
        const auto& s = x->shape;
        if (x->format != Format::Nchw || s.size() != 4 ||
            !x->placement.is_replicated() ||
            (x->dtype != DType::BFloat16 && x->dtype != DType::Float32)) {
            return false;
        }
        return s[0] > 0 && s[1] > 0 && s[2] * s[3] > 0 && groups > 0 &&
            ktt_group_norm::rows_supported(
                   {static_cast<uint32_t>(s[0]),
                    static_cast<uint32_t>(s[2] * s[3]),
                    static_cast<uint32_t>(s[1]),
                    static_cast<uint32_t>(groups),
                    0.0f}
            );
    }

    std::array<HandlePtr, 3> group_norm_rows_sums(
        const HandlePtr& x,
        const HandlePtr& weight,
        const HandlePtr& bias,
        int64_t groups,
        double eps
    ) {
        const bool nhwc = x->format == Format::Nhwc;
        const ktt_group_norm::Rows r{
            static_cast<uint32_t>(x->shape[0]),
            static_cast<uint32_t>(nhwc ? x->shape[2] * x->shape[3] : x->shape[1]),
            static_cast<uint32_t>(nhwc ? x->shape[1] : x->shape[2]),
            static_cast<uint32_t>(groups),
            static_cast<float>(eps)
        };
        const std::vector<int64_t> per_channel{
            x->shape[0], static_cast<int64_t>(r.channels)
        };
        std::array<HandlePtr, 3> out{
            detail::make_handle(x->shape, x->dtype, x->format),
            detail::make_handle(per_channel, DType::Float32),
            detail::make_handle(per_channel, DType::Float32)
        };
        const int64_t bytes = x->bytes;
        detail::counters().ops++;
        detail::submit([=] {
            using tt::tt_metal::Layout;
            const auto tiled = [](const HandlePtr& h) -> std::optional<ttnn::Tensor> {
                if (!h) {
                    return std::nullopt;
                }
                return h->tensor.layout() == Layout::TILE
                    ? h->tensor
                    : ttnn::to_layout(h->tensor, Layout::TILE);
            };
            const auto w = tiled(weight), b = tiled(bias);
            std::pair<ttnn::Tensor, ttnn::Tensor> sums;
            out[0]->tensor = ktt_group_norm::rows_norm(
                x->tensor,
                w ? &*w : nullptr,
                b ? &*b : nullptr,
                r,
                detail::activation_config(bytes),
                &sums
            );
            out[1]->tensor = sums.first;
            out[2]->tensor = sums.second;
            for (const auto& h : out) {
                detail::check_shape(*h);
            }
        });
        return out;
    }

    std::array<HandlePtr, 3> group_norm_rows_backward(
        const HandlePtr& x,
        const HandlePtr& dy,
        const HandlePtr& mean,
        const HandlePtr& rstd,
        const HandlePtr& weight,
        int64_t groups
    ) {
        const ktt_group_norm::Rows r{
            static_cast<uint32_t>(x->shape[0]),
            static_cast<uint32_t>(x->shape[2] * x->shape[3]),
            static_cast<uint32_t>(x->shape[1]),
            static_cast<uint32_t>(groups),
            0.0f
        };
        const std::vector<int64_t> per_channel{x->shape[0], x->shape[1]};
        std::array<HandlePtr, 3> out{
            detail::make_handle(x->shape, x->dtype, Format::Nhwc),
            detail::make_handle(per_channel, DType::Float32),
            detail::make_handle(per_channel, DType::Float32)
        };
        const int64_t bytes = x->bytes;
        detail::counters().ops++;
        detail::submit([=] {
            using tt::tt_metal::Layout;
            const auto tiled = [](const HandlePtr& h) {
                return h->tensor.layout() == Layout::TILE
                    ? h->tensor
                    : ttnn::to_layout(h->tensor, Layout::TILE);
            };
            std::optional<ttnn::Tensor> w;
            if (weight) {
                w = tiled(weight);
            }
            std::pair<ttnn::Tensor, ttnn::Tensor> terms;
            out[0]->tensor = ktt_group_norm::rows_backward(
                x->tensor,
                dy->tensor,
                tiled(mean),
                tiled(rstd),
                w ? &*w : nullptr,
                r,
                detail::activation_config(bytes),
                terms
            );
            out[1]->tensor = terms.first;
            out[2]->tensor = terms.second;
            for (const auto& h : out) {
                detail::check_shape(*h);
            }
        });
        return out;
    }

    HandlePtr group_norm_rows(
        const HandlePtr& x,
        const HandlePtr& weight,
        const HandlePtr& bias,
        int64_t groups,
        double eps
    ) {
        const bool nhwc = x->format == Format::Nhwc;
        const ktt_group_norm::Rows r{
            static_cast<uint32_t>(x->shape[0]),
            static_cast<uint32_t>(nhwc ? x->shape[2] * x->shape[3] : x->shape[1]),
            static_cast<uint32_t>(nhwc ? x->shape[1] : x->shape[2]),
            static_cast<uint32_t>(groups),
            static_cast<float>(eps)
        };
        const int64_t bytes = x->bytes;
        return detail::timed_op(
            x->dtype,
            x->shape,
            [=] {
                using tt::tt_metal::Layout;
                const auto tiled =
                    [](const HandlePtr& h) -> std::optional<ttnn::Tensor> {
                    if (!h) {
                        return std::nullopt;
                    }
                    return h->tensor.layout() == Layout::TILE
                        ? h->tensor
                        : ttnn::to_layout(h->tensor, Layout::TILE);
                };
                const auto w = tiled(weight), b = tiled(bias);
                return ktt_group_norm::rows_norm(
                    x->tensor,
                    w ? &*w : nullptr,
                    b ? &*b : nullptr,
                    r,
                    detail::activation_config(bytes)
                );
            },
            x->format
        );
    }

}  // namespace ktt::bridge
