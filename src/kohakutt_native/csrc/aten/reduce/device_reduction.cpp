// Bridge reductions over device dims, reshaped to the logical output's device shape;
// softmax along one device dim (the last: Kohaku-TT's ktt_softmax).
#include <cstdlib>
#include <string>

#include "aten/nn/ktt_softmax/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
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

        // True when ktt_softmax takes `a` along its last dim `dim` (bf16 / fp32):
        // KTT_OWN_SOFTMAX 1 always, 0 never, auto when that dim is ragged.
        bool own_softmax(const HandlePtr& a, int64_t dim) {
            const char* v = std::getenv("KTT_OWN_SOFTMAX");
            const std::string mode = v ? v : "auto";
            const auto& s = a->shape;
            const bool fp32 = a->dtype == DType::Float32;
            if (mode == "0" || !program::own_kernels_enabled() ||
                dim + 1 != static_cast<int64_t>(s.size()) || s.size() < 2 ||
                !(fp32 || a->dtype == DType::BFloat16)) {
                return false;
            }
            return mode == "1" || s.back() % 32 != 0;
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

    HandlePtr mean(
        const HandlePtr& a,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& out
    ) {
        const auto config = reduce_config();
        return detail::timed_op(a->dtype, detail::device_dims(out), [=] {
            ttsl::SmallVector<int> d(dims.begin(), dims.end());
            const bool keepdim = true;
            auto reduced = ttnn::mean(a->tensor, d, keepdim, std::nullopt, config);
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
        if (own_softmax(a, dim)) {
            const int64_t bytes = a->bytes;
            return detail::timed_op(a->dtype, a->shape, [=] {
                const auto memory = detail::activation_config(bytes);
                return ktt_softmax::forward(a->tensor, memory);
            });
        }
        const auto config = reduce_config();
        return detail::timed_op(a->dtype, a->shape, [=] {
            // The padded columns of a ragged last dim are read: zero them.
            const auto& s = a->tensor.logical_shape();
            const bool ragged_last = dim + 1 == static_cast<int64_t>(s.rank()) &&
                s[-1] % 32 != 0;
            return (ttnn::softmax(
                ragged_last ? ttnn::fill_implicit_tile_padding(a->tensor, 0.0f)
                            : a->tensor,
                static_cast<int>(dim),
                std::nullopt,
                config,
                true
            ));
        });
    }

}  // namespace ktt::bridge
