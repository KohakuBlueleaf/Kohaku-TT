// Bridge data movement: reshape, permute, slice, concat, repeat, gather, scatter.
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "aten/index/ktt_gather/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/program.hpp"
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

        [[noreturn]] void bad(const char* op, const std::string& what) {
            throw std::invalid_argument(
                std::string("kohakutt_native: ") + op + ": " + what
            );
        }

    }  // namespace

    HandlePtr reshape(const HandlePtr& a, const std::vector<int64_t>& sizes) {
        if (volume(sizes) != volume(a->shape)) {
            bad("reshape",
                "cannot reshape " + detail::dims_str(a->shape) + " (" +
                    std::to_string(volume(a->shape)) + " elements) to " +
                    detail::dims_str(sizes) + " (" + std::to_string(volume(sizes)) +
                    " elements)");
        }
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
        // A re-tiled output can be far larger than its input (e.g. [O, C] -> [O, C,
        // 1]): placed by its own size, not the input's memory config.
        const int64_t bytes = detail::padded_bytes(dims, a->dtype);
        return detail::timed_op(a->dtype, dims, [=] {
            return ttnn::reshape(a->tensor, shape, detail::activation_config(bytes));
        });
    }

    HandlePtr permute(const HandlePtr& a, const std::vector<int64_t>& dims) {
        std::vector<bool> seen(a->shape.size(), false);
        bool valid = dims.size() == a->shape.size();
        for (size_t k = 0; valid && k < dims.size(); ++k) {
            valid = dims[k] >= 0 && dims[k] < static_cast<int64_t>(seen.size()) &&
                !seen[dims[k]];
            if (valid) {
                seen[dims[k]] = true;
            }
        }
        if (!valid) {
            bad("permute",
                "dims " + detail::dims_str(dims) + " are not a permutation of the " +
                    std::to_string(a->shape.size()) + " dims of " +
                    detail::dims_str(a->shape));
        }
        std::vector<int64_t> out;
        for (auto d : dims) {
            out.push_back(a->shape[d]);
        }
        if (detail::is_empty(a)) {
            return empty(out, a->dtype);
        }
        ttsl::SmallVector<int64_t> order(dims.begin(), dims.end());
        // Placed by the output's bytes: other last two dims pad to other tile counts.
        const int64_t bytes = detail::padded_bytes(out, a->dtype);
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
        const size_t rank = a->shape.size();
        if (begins.size() != rank || ends.size() != rank || steps.size() != rank) {
            bad("slice",
                "begins / ends / steps must have the rank of " +
                    detail::dims_str(a->shape) + ", got " + detail::dims_str(begins) +
                    " / " + detail::dims_str(ends) + " / " + detail::dims_str(steps));
        }
        for (size_t k = 0; k < rank; ++k) {
            if (begins[k] < 0 || begins[k] > a->shape[k] || ends[k] < 0 ||
                ends[k] > a->shape[k] || steps[k] < 1) {
                bad("slice",
                    "out of bounds at dim " + std::to_string(k) + " of " +
                        detail::dims_str(a->shape) + ": begin " +
                        std::to_string(begins[k]) + ", end " + std::to_string(ends[k]) +
                        ", step " + std::to_string(steps[k]));
            }
        }
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
        if (parts.empty()) {
            bad("concat", "no tensors");
        }
        const auto& first = parts.front()->shape;
        if (dim < 0 || dim >= static_cast<int64_t>(first.size())) {
            bad("concat",
                "dim " + std::to_string(dim) + " out of range for " +
                    detail::dims_str(first));
        }
        for (const auto& p : parts) {
            bool same = p->shape.size() == first.size();
            for (size_t k = 0; same && k < first.size(); ++k) {
                same = static_cast<int64_t>(k) == dim || p->shape[k] == first[k];
            }
            if (!same) {
                bad("concat",
                    "shapes " + detail::dims_str(first) + " and " +
                        detail::dims_str(p->shape) + " differ outside dim " +
                        std::to_string(dim));
            }
        }
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

    namespace {

        // False when KTT_OWN_GATHER=0: every gather runs TT-NN's.
        bool own_gather() {
            const char* v = std::getenv("KTT_OWN_GATHER");
            return v == nullptr || std::string(v) != "0";
        }

    }  // namespace

    HandlePtr gather(const HandlePtr& a, int64_t dim, const HandlePtr& index) {
        if (detail::is_empty(index)) {
            return empty(index->shape, a->dtype);
        }
        if (detail::is_empty(a)) {
            throw std::invalid_argument("kohakutt_native: gather from an empty tensor");
        }
        const auto ad = detail::device_dims(a->shape),
                   id = detail::device_dims(index->shape);
        const int64_t rank = static_cast<int64_t>(ad.size());
        const int64_t d = dim < 0 ? dim + rank : dim;
        // Along the last dim with matching leading dims: Kohaku-TT's gather.
        bool own =
            (program::own_kernels_enabled() && own_gather() &&
             static_cast<int64_t>(id.size()) == rank && d == rank - 1 &&
             index->dtype == DType::Int32 &&
             (a->dtype == DType::BFloat16 || a->dtype == DType::Float32 ||
              a->dtype == DType::Int32) &&
             id[rank - 2] <= ad[rank - 2]);
        for (int64_t k = 0; own && k + 2 < rank; ++k) {
            own = id[k] == ad[k];
        }
        if (own) {
            return detail::timed_op(a->dtype, index->shape, [=] {
                using tt::tt_metal::Layout;
                const auto tiled = [](const ttnn::Tensor& t) {
                    return t.layout() == Layout::TILE
                        ? t
                        : ttnn::to_layout(t, Layout::TILE);
                };
                const auto mc = (detail::activation_config(
                    detail::padded_bytes(index->shape, a->dtype)
                ));
                return ktt_gather::run(tiled(a->tensor), tiled(index->tensor), mc);
            });
        }
        int64_t elements = 1;
        for (const int64_t s : a->shape) {
            elements *= s;
        }
        const char* allow = std::getenv("KTT_TTNN_GATHER_LARGE");
        if (elements > 262144 && !(allow != nullptr && std::string(allow) == "1")) {
            throw std::invalid_argument(
                "kohakutt_native: gather of " + std::to_string(elements) +
                " input elements along dim " + std::to_string(d) +
                " is not covered by Kohaku-TT's gather (last dim, matching leading "
                "dims) and would run TT-NN's gather, which takes minutes at this size; "
                "set KTT_TTNN_GATHER_LARGE=1 to run it anyway"
            );
        }
        return detail::timed_op(a->dtype, index->shape, [=] {
            // TT-NN gather takes UINT32 / UINT16 indices; ours are non-negative int32.
            auto idx = ttnn::typecast(index->tensor, tt::tt_metal::DataType::UINT32);
            return (ttnn::gather(
                a->tensor, static_cast<int8_t>(dim), idx, false, std::nullopt
            ));
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
