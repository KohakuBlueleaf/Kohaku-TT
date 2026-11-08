// Bridge nearest upsampling: Kohaku-TT's gather kernel (ktt_upsample).
#include "aten/nn/ktt_upsample/program.hpp"
#include "core/device/handle.hpp"

namespace ktt::bridge {

    HandlePtr upsample_nearest(const HandlePtr& x, int64_t kh, int64_t kw) {
        auto shape = x->shape;
        shape[shape.size() - 2] *= kh;
        shape.back() *= kw;
        const int64_t bytes = detail::padded_bytes(shape, x->dtype);
        return detail::timed_op(x->dtype, shape, [=] {
            return (
                ktt_upsample::run(
                    x->tensor,
                    static_cast<uint32_t>(kh),
                    static_cast<uint32_t>(kw),
                    detail::activation_config(bytes)
                )
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
            return (
                ktt_upsample::shuffle(
                    x->tensor,
                    static_cast<uint32_t>(r),
                    detail::activation_config(bytes)
                )
            );
        });
    }

}  // namespace ktt::bridge
