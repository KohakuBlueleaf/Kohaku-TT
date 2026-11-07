// Bridge random ops on device (no host draw, no upload): uniform, normal, bernoulli.
#include "core/device/handle.hpp"
#include "ttnn/operations/bernoulli/bernoulli.hpp"
#include "ttnn/operations/randn/randn.hpp"
#include "ttnn/operations/uniform/uniform.hpp"

namespace ktt::bridge {

    HandlePtr uniform(const HandlePtr& like, double from, double to, uint32_t seed) {
        if (detail::is_empty(like)) {
            return detail::empty_of(like, like->dtype);
        }
        return detail::timed_op(like->dtype, like->shape, [=] {
            return (
                ttnn::uniform(
                    like->tensor, static_cast<float>(from), static_cast<float>(to), seed
                )
            );
        });
    }

    HandlePtr normal(const std::vector<int64_t>& shape, DType dtype, uint32_t seed) {
        return detail::timed_op(dtype, detail::device_dims(shape), [=] {
            return (
                ttnn::randn(
                    detail::device_shape(shape),
                    *detail::device(),
                    detail::to_ttnn(dtype),
                    tt::tt_metal::Layout::TILE,
                    ttnn::DRAM_MEMORY_CONFIG,
                    std::nullopt,
                    seed
                )
            );
        });
    }

    HandlePtr bernoulli(const HandlePtr& p, uint32_t seed) {
        if (detail::is_empty(p)) {
            return detail::empty_of(p, p->dtype);
        }
        return detail::timed_op(p->dtype, p->shape, [=] {
            return ttnn::bernoulli(p->tensor, seed);
        });
    }

}  // namespace ktt::bridge
