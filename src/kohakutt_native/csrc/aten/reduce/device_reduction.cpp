// Bridge reductions over device dims, reshaped to the logical output's device shape;
// softmax along one device dim.
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"

namespace ktt::bridge {
    namespace {

        // Reductions and softmax: fp32 accumulation, fidelity floor HiFi4 (memory-bound
        // ops; fp32 inputs lose their low mantissa bits below it). Call where the op is
        // queued.
        ttnn::ComputeKernelConfig reduce_config() {
            return numerics::compute_config(
                numerics::effective({.floor = tt::tt_metal::MathFidelity::HiFi4})
            );
        }

    }  // namespace

    HandlePtr sum(
        const HandlePtr& a,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& out
    ) {
        if (detail::is_empty(a)) {
            return full(out, 0.0, a->dtype);
        }
        const auto config = reduce_config();
        return detail::timed_op(a->dtype, detail::device_dims(out), [=] {
            ttsl::SmallVector<int> d(dims.begin(), dims.end());
            const bool keepdim = true;
            auto reduced = ttnn::sum(a->tensor, d, keepdim, std::nullopt, config);
            return ttnn::reshape(reduced, detail::device_shape(out));
        });
    }

    HandlePtr amax(
        const HandlePtr& a,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& out
    ) {
        if (detail::is_empty(a)) {
            return empty(out, a->dtype);
        }
        const auto config = reduce_config();
        return detail::timed_op(a->dtype, detail::device_dims(out), [=] {
            ttsl::SmallVector<int> d(dims.begin(), dims.end());
            const bool keepdim = true;
            auto reduced = ttnn::max(a->tensor, d, keepdim, std::nullopt, config);
            return ttnn::reshape(reduced, detail::device_shape(out));
        });
    }

    HandlePtr prod(
        const HandlePtr& a, std::optional<int64_t> dim, const std::vector<int64_t>& out
    ) {
        return detail::timed_op(a->dtype, detail::device_dims(out), [=] {
            const bool keepdim = dim.has_value();
            auto reduced = ttnn::prod(a->tensor, dim, keepdim);
            return ttnn::reshape(reduced, detail::device_shape(out));
        });
    }

    HandlePtr softmax(const HandlePtr& a, int64_t dim) {
        if (detail::is_empty(a)) {
            return detail::empty_of(a, a->dtype);
        }
        const auto config = reduce_config();
        return detail::timed_op(a->dtype, a->shape, [=] {
            return (ttnn::softmax(
                a->tensor, static_cast<int>(dim), std::nullopt, config, true
            ));
        });
    }

}  // namespace ktt::bridge
