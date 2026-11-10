// Bridge index-like ops: argmax, sort, topk, tril, cumsum and general arange.
#include <map>
#include <optional>
#include <tuple>

#include "core/device/handle.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/reduction/accumulation/cumprod/cumprod.hpp"
#include "ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp"
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "aten/reduce/ktt_argmax/program.hpp"
#include "aten/reduce/ktt_cumsum/program.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/reduction/topk/topk.hpp"

namespace ktt::bridge {

    HandlePtr argmax_last(const HandlePtr& a, const std::vector<int64_t>& out) {
        // Outputs of rank <= 2 ([P, R] with P, R from `out`): Kohaku-TT's argmax,
        // unless fewer than 8 units (32 P x 32 R) each scan more than 64 W tiles.
        const int64_t w = a->shape.back();
        const int64_t r = out.empty() ? 1 : out.back();
        const int64_t p = out.size() == 2 ? out[0] : 1;
        const int64_t units = ((p + 31) / 32) * ((r + 31) / 32);
        const bool own =
            (program::own_kernels_enabled() && out.size() <= 2 &&
             (a->dtype == DType::Float32 || a->dtype == DType::BFloat16) &&
             !(units < 8 && (w + 31) / 32 > 64));
        if (own) {
            return detail::timed_op(DType::Int32, detail::device_dims(out), [=] {
                const auto mc = (detail::activation_config(
                    detail::padded_bytes(detail::device_dims(out), DType::Int32)
                ));
                const auto u32 = [](int64_t v) {
                    return static_cast<uint32_t>(v);
                };
                auto x = ttnn::reshape(
                    a->tensor, ttnn::Shape({u32(p), u32(r), u32(w)})
                );
                auto idx = ktt_argmax::run(x, mc);
                return ttnn::reshape(idx, detail::device_shape(out));
            });
        }
        return detail::timed_op(DType::Int32, detail::device_dims(out), [=] {
            // TT-NN argmax reads row-major input along the last dim and returns UINT32.
            auto rm = ttnn::to_layout(a->tensor, tt::tt_metal::Layout::ROW_MAJOR);
            const bool keepdim = true;
            auto idx = ttnn::argmax(rm, -1, keepdim);
            auto tiled = ttnn::to_layout(idx, tt::tt_metal::Layout::TILE);
            auto ints = ttnn::typecast(tiled, tt::tt_metal::DataType::INT32);
            return ttnn::reshape(ints, detail::device_shape(out));
        });
    }

    namespace {

        // (values, int32 indices) of `shape` from a TT-NN op returning [values,
        // indices]; one stream item fills both handles.
        template <typename F>
        std::pair<HandlePtr, HandlePtr> values_indices(
            const HandlePtr& a, std::vector<int64_t> shape, F call
        ) {
            auto values = detail::make_handle(shape, a->dtype);
            auto indices = detail::make_handle(shape, DType::Int32);
            detail::counters().ops++;
            detail::submit([values, indices, call = std::move(call)]() mutable {
                auto parts = call();
                values->tensor = parts[0];
                indices->tensor = (ttnn::typecast(
                    parts[1], tt::tt_metal::DataType::INT32
                ));
                detail::check_shape(*values);
                detail::check_shape(*indices);
            });
            return {values, indices};
        }

    }  // namespace

    std::pair<HandlePtr, HandlePtr> sort(
        const HandlePtr& a, int64_t dim, bool descending, bool stable
    ) {
        return values_indices(a, a->shape, [=] {
            return ttnn::sort(a->tensor, static_cast<int8_t>(dim), descending, stable);
        });
    }

    std::pair<HandlePtr, HandlePtr> topk(
        const HandlePtr& a, int64_t k, int64_t dim, bool largest, bool sorted
    ) {
        auto shape = a->shape;
        shape[dim < 0 ? shape.size() + dim : dim] = k;
        return values_indices(a, shape, [=] {
            const auto count = static_cast<uint32_t>(k);
            const auto axis = static_cast<int8_t>(dim);
            return ttnn::topk(a->tensor, count, axis, largest, sorted);
        });
    }

    HandlePtr tril(const HandlePtr& a, int64_t diagonal) {
        return detail::timed_op(a->dtype, a->shape, [=] {
            return ttnn::tril(a->tensor, static_cast<int32_t>(diagonal));
        });
    }

    HandlePtr cumprod(const HandlePtr& a, int64_t dim) {
        if (detail::is_empty(a)) {
            return detail::empty_of(a, a->dtype);
        }
        return detail::timed_op(a->dtype, a->shape, [=] {
            return ttnn::cumprod(a->tensor, static_cast<int32_t>(dim));
        });
    }

    HandlePtr cumsum(const HandlePtr& a, int64_t dim) {
        // Along the last dim of a bf16 tensor of rank >= 2: Kohaku-TT's cumsum (fp32
        // keeps TT-NN's, which carries fp32 sums exactly).
        const int64_t rank = static_cast<int64_t>(a->shape.size());
        const bool own =
            (program::own_kernels_enabled() && rank >= 2 &&
             (dim == rank - 1 || dim == -1) && a->dtype == DType::BFloat16);
        return detail::timed_op(a->dtype, a->shape, [=] {
            if (own && a->tensor.layout() == tt::tt_metal::Layout::TILE) {
                return (ktt_cumsum::run(
                    a->tensor,
                    detail::activation_config(detail::padded_bytes(a->shape, a->dtype))
                ));
            }
            return ttnn::cumsum(a->tensor, static_cast<int32_t>(dim));
        });
    }

    namespace {

        // arange handles per range (TT-NN builds them on host: created eagerly, reused
        // inside captures). Never destroyed at exit (after TT-Metal's teardown);
        // close() clears it.
        using RangeKey = std::tuple<int64_t, int64_t, int64_t, DType>;

        std::map<RangeKey, HandlePtr>& ranges() {
            static auto* map = new std::map<RangeKey, HandlePtr>();
            return *map;
        }

    }  // namespace

    namespace detail {

        void clear_ranges() {
            ranges().clear();
        }

    }  // namespace detail

    HandlePtr arange_range(int64_t start, int64_t end, int64_t step, DType dtype) {
        auto& slot = ranges()[{start, end, step, dtype}];
        if (slot) {
            return slot;
        }
        const int64_t n = (end - start + step - (step > 0 ? 1 : -1)) / step;
        slot = detail::timed_op(dtype, detail::device_dims({n}), [=] {
            auto row = (ttnn::arange(
                start,
                end,
                step,
                detail::to_ttnn(dtype),
                std::ref(*detail::device()),
                ttnn::DRAM_MEMORY_CONFIG,
                tt::tt_metal::Layout::TILE
            ));
            return ttnn::reshape(row, detail::device_shape({n}));
        });
        return slot;
    }

}  // namespace ktt::bridge
