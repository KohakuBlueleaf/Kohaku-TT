// Bridge nearest upsampling: Kohaku-TT's row-copy kernel (ktt_upsample/rows.cpp) on
// Nchw and Nhwc storage; pixel shuffle.
#include "aten/nn/ktt_upsample/program.hpp"
#include "core/device/handle.hpp"

namespace ktt::bridge {

    HandlePtr upsample_nearest(const HandlePtr& x, int64_t kh, int64_t kw) {
        const auto u32 = [](int64_t v) {
            return static_cast<uint32_t>(v);
        };
        auto shape = x->shape;
        shape[shape.size() - 2] *= kh;
        shape.back() *= kw;
        if (x->format == Format::Nhwc) {
            // Channels last: each output pixel row a copy of its source pixel's row.
            const int64_t n = x->shape[0], c = x->shape[1], h = x->shape[2],
                          w = x->shape[3];
            const int64_t bytes = detail::padded_bytes(
                {1, 1, n * h * kh * w * kw, c}, x->dtype
            );
            return detail::timed_op(
                x->dtype,
                shape,
                [=] {
                    return ktt_upsample::rows_nhwc(
                        x->tensor,
                        u32(n),
                        u32(h),
                        u32(w),
                        u32(kh),
                        u32(kw),
                        detail::activation_config(bytes)
                    );
                },
                Format::Nhwc
            );
        }
        if (x->format != Format::Nchw) {
            return upsample_nearest(to_nchw(x), kh, kw);
        }
        const int64_t bytes = detail::padded_bytes(shape, x->dtype);
        return detail::timed_op(x->dtype, shape, [=] {
            return ktt_upsample::rows(
                x->tensor, u32(kh), u32(kw), detail::activation_config(bytes)
            );
        });
    }

    HandlePtr pixel_shuffle(const HandlePtr& x, int64_t r) {
        auto shape = x->shape;
        const size_t rank = shape.size();
        shape[rank - 3] /= r * r;
        shape[rank - 2] *= r;
        shape[rank - 1] *= r;
        const int64_t bytes = detail::padded_bytes(shape, x->dtype);
        return detail::timed_op(x->dtype, shape, [=] {
            return (ktt_upsample::shuffle(
                x->tensor, static_cast<uint32_t>(r), detail::activation_config(bytes)
            ));
        });
    }

}  // namespace ktt::bridge
