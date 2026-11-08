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
            auto st = (
                ktt_group_norm::stats(
                    s1->tensor,
                    s2->tensor,
                    w ? &*w : nullptr,
                    b ? &*b : nullptr,
                    static_cast<uint32_t>(groups),
                    static_cast<float>(count),
                    static_cast<float>(eps),
                    detail::to_ttnn(out)
                )
            );
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
            return (
                ktt_group_norm::affine(
                    x->tensor, a->tensor, b->tensor, detail::activation_config(bytes)
                )
            );
        });
    }

}  // namespace ktt::bridge
