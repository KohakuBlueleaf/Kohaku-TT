// Bridge data movement: reshape, permute, slice, concat, repeat, gather, scatter.
#include <algorithm>
#include <stdexcept>

#include "core/device/handle.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/gather/gather.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/scatter/scatter.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

namespace ktt::bridge {

    // Zero-volume inputs take the bridge's zero-size path (TT-NN program factories
    // divide by tile counts): the result is an empty handle of the output shape.
    namespace {

        int64_t volume(const std::vector<int64_t>& shape) {
            int64_t n = 1;
            for (auto s : shape) {
                n *= s;
            }
            return n;
        }

    }  // namespace

    HandlePtr reshape(const HandlePtr& a, const std::vector<int64_t>& sizes) {
        if (detail::is_empty(a) || volume(sizes) == 0) {
            return empty(sizes, a->dtype);
        }
        auto shape = detail::device_shape(sizes);
        auto dims = detail::device_dims(sizes);

        // Same last dim, second-last equal or tile-aligned on both sides: a TT-NN
        // metadata view, no device program.
        const int64_t r = a->shape.size(), n = dims.size();
        const bool view =
            (a->shape[r - 1] == dims[n - 1] &&
             (a->shape[r - 2] == dims[n - 2] ||
              (a->shape[r - 2] % 32 == 0 && dims[n - 2] % 32 == 0)));
        if (view) {
            auto out = detail::make_handle(dims, a->dtype);
            detail::submit([out, a, shape] {
                out->tensor = ttnn::reshape(a->tensor, shape);
            });
            return out;
        }
        return detail::timed_op(a->dtype, dims, [=] {
            return ttnn::reshape(a->tensor, shape);
        });
    }

    HandlePtr permute(const HandlePtr& a, const std::vector<int64_t>& dims) {
        std::vector<int64_t> out;
        for (auto d : dims) {
            out.push_back(a->shape[d]);
        }
        if (detail::is_empty(a)) {
            return empty(out, a->dtype);
        }
        ttsl::SmallVector<int64_t> order(dims.begin(), dims.end());
        const int64_t bytes = a->bytes;
        return detail::timed_op(a->dtype, out, [=] {
            return ttnn::permute(a->tensor, order, detail::activation_config(bytes));
        });
    }

    HandlePtr slice(
        const HandlePtr& a,
        const std::vector<int64_t>& begins,
        const std::vector<int64_t>& ends,
        const std::vector<int64_t>& steps
    ) {
        std::vector<int64_t> out(begins.size());
        for (size_t k = 0; k < begins.size(); ++k) {
            const int64_t span = std::max<int64_t>(0, ends[k] - begins[k]);
            out[k] = (span + steps[k] - 1) / steps[k];
        }
        if (detail::is_empty(a) || volume(out) == 0) {
            return empty(out, a->dtype);
        }
        ttsl::SmallVector<int32_t> b(begins.begin(), begins.end());
        ttsl::SmallVector<int32_t> e(ends.begin(), ends.end());
        ttsl::SmallVector<int32_t> s(steps.begin(), steps.end());

        // TT-NN's slice is wrong for large fp32 tensors cut at unaligned offsets in
        // both tile dims at once: such a cut runs as two slices (the last dim
        // separately).
        const size_t r = b.size();
        auto cut = [&](size_t k) {
            return b[k] != 0 || e[k] != a->shape[k] || s[k] != 1;
        };
        const bool split =
            (r >= 2 && cut(r - 1) && cut(r - 2) &&
             (b[r - 1] % 32 != 0 || b[r - 2] % 32 != 0));
        if (split) {
            auto first_e = e;
            auto first_b = b;
            auto first_s = s;
            first_b[r - 1] = 0;
            first_e[r - 1] = static_cast<int32_t>(a->shape[r - 1]);
            first_s[r - 1] = 1;
            auto rows = out;
            rows[r - 1] = a->shape[r - 1];
            auto head = detail::timed_op(a->dtype, rows, [=] {
                return ttnn::slice(a->tensor, first_b, first_e, first_s);
            });
            ttsl::SmallVector<int32_t> zero(r, 0), whole(rows.begin(), rows.end()),
                one(r, 1);
            zero[r - 1] = b[r - 1];
            whole[r - 1] = e[r - 1];
            one[r - 1] = s[r - 1];
            return detail::timed_op(a->dtype, out, [=] {
                return ttnn::slice(head->tensor, zero, whole, one);
            });
        }
        return detail::timed_op(a->dtype, out, [=] {
            return ttnn::slice(a->tensor, b, e, s);
        });
    }

    HandlePtr concat(const std::vector<HandlePtr>& parts, int64_t dim) {
        std::vector<HandlePtr> live;
        auto shape = parts.front()->shape;
        shape[dim] = 0;
        for (const auto& p : parts) {
            shape[dim] += p->shape[dim];
            if (!detail::is_empty(p)) {
                live.push_back(p);
            }
        }
        if (live.empty()) {
            return empty(shape, parts.front()->dtype);
        }
        if (live.size() == 1) {
            // A storage never shares another's handle: copy the one non-empty part.
            return clone(live.front());
        }
        return detail::timed_op(parts.front()->dtype, shape, [=] {
            std::vector<ttnn::Tensor> tensors;
            for (const auto& p : live) {
                tensors.push_back(p->tensor);
            }
            return ttnn::concat(tensors, static_cast<int>(dim));
        });
    }

    HandlePtr gather(const HandlePtr& a, int64_t dim, const HandlePtr& index) {
        if (detail::is_empty(index)) {
            return empty(index->shape, a->dtype);
        }
        if (detail::is_empty(a)) {
            throw std::invalid_argument("kohakutt_native: gather from an empty tensor");
        }
        return detail::timed_op(a->dtype, index->shape, [=] {
            // TT-NN gather takes UINT32 / UINT16 indices; ours are non-negative int32.
            auto idx = ttnn::typecast(index->tensor, tt::tt_metal::DataType::UINT32);
            return (
                ttnn::gather(
                    a->tensor, static_cast<int8_t>(dim), idx, false, std::nullopt
                )
            );
        });
    }

    HandlePtr scatter(
        const HandlePtr& a,
        int64_t dim,
        const HandlePtr& index,
        const HandlePtr& src,
        const std::string& reduce
    ) {
        if (detail::is_empty(a) || detail::is_empty(index)) {
            return detail::is_empty(a) ? empty(a->shape, a->dtype) : clone(a);
        }
        return detail::timed_op(a->dtype, a->shape, [=] {
            using tt::tt_metal::Layout;
            auto idx = ttnn::typecast(index->tensor, tt::tt_metal::DataType::UINT32);
            idx = ttnn::to_layout(idx, Layout::ROW_MAJOR);
            const auto d = static_cast<int32_t>(dim);

            // TT-NN scatter rejects tiled fp32 and tiled 32-bit rows > 256 (data and
            // index): the index is always row-major, fp32 / int32 data too.
            const bool row_major = a->dtype != DType::BFloat16;
            auto input =
                (row_major ? ttnn::to_layout(a->tensor, Layout::ROW_MAJOR) : a->tensor);
            auto source =
                (row_major ? ttnn::to_layout(src->tensor, Layout::ROW_MAJOR)
                           : src->tensor);
            auto out =
                (reduce.empty()
                     ? ttnn::scatter(input, d, idx, source)
                     : ttnn::scatter(input, d, idx, source, std::nullopt, reduce));
            return row_major ? ttnn::to_layout(out, Layout::TILE) : out;
        });
    }

    HandlePtr repeat(const HandlePtr& a, const std::vector<int64_t>& repeats) {
        auto shape = a->shape;
        for (size_t k = 0; k < shape.size() && k < repeats.size(); ++k) {
            shape[shape.size() - 1 - k] *= repeats[repeats.size() - 1 - k];
        }
        if (detail::is_empty(a) || volume(repeats) == 0) {
            return empty(shape, a->dtype);
        }
        ttsl::SmallVector<uint32_t> r(repeats.begin(), repeats.end());
        return detail::timed_op(a->dtype, shape, [=] {
            return ttnn::repeat(a->tensor, r);
        });
    }

}  // namespace ktt::bridge
