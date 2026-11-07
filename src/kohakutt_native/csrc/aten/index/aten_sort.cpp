// aten ordering kernels for tt: sort, topk, kthvalue, searchsorted / bucketize,
// unique / unique_consecutive (host).

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // (values, int64 indices) of `self` sorted along `dim`.
        std::tuple<at::Tensor, at::Tensor> tt_sort(
            const at::Tensor& self,
            std::optional<bool> stable,
            int64_t dim,
            bool descending
        ) {
            const char* op = "sort.stable";
            auto long_opts = self.options().dtype(at::kLong);
            if (self.dim() == 0 || self.numel() <= 1) {
                return {self.clone(), at::zeros(self.sizes(), long_opts)};
            }
            dim = c10::maybe_wrap_dim(dim, self.dim());
            const int64_t pad = std::max<int64_t>(0, 2 - self.dim());
            auto [values, indices] = (
                bridge::sort(
                    dense(op, self), dim + pad, descending, stable.value_or(false)
                )
            );
            return {
                wrap_output(values, self.sizes(), self.scalar_type()),
                wrap_output(indices, self.sizes(), at::kLong)
            };
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_sort_out(
            const at::Tensor& self,
            std::optional<bool> stable,
            int64_t dim,
            bool descending,
            at::Tensor& values,
            at::Tensor& indices
        ) {
            auto [v, i] = tt_sort(self, stable, dim, descending);
            into("sort.values_stable", values, v);
            into("sort.values_stable", indices, i);
            return {values, indices};
        }

        // (values, int64 indices) of the k largest (or smallest) of `self` along `dim`.
        std::tuple<at::Tensor, at::Tensor> tt_topk(
            const at::Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted
        ) {
            const char* op = "topk";
            auto long_opts = self.options().dtype(at::kLong);
            dim = c10::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
            auto sizes = self.sizes().vec();
            if (self.dim() > 0) {
                sizes[dim] = k;
            }
            if (k == 0 || self.numel() == 0) {
                return {at::empty(sizes, self.options()), at::empty(sizes, long_opts)};
            }
            if (self.dim() == 0) {
                return {self.clone(), at::zeros({}, long_opts)};
            }
            const int64_t pad = std::max<int64_t>(0, 2 - self.dim());
            auto [values, indices] = (
                bridge::topk(dense(op, self), k, dim + pad, largest, sorted)
            );
            return {
                wrap_output(values, sizes, self.scalar_type()),
                wrap_output(indices, sizes, at::kLong)
            };
        }

        std::tuple<at::Tensor&, at::Tensor&> tt_topk_out(
            const at::Tensor& self,
            int64_t k,
            int64_t dim,
            bool largest,
            bool sorted,
            at::Tensor& values,
            at::Tensor& indices
        ) {
            auto [v, i] = tt_topk(self, k, dim, largest, sorted);
            into("topk.values", values, v);
            into("topk.values", indices, i);
            return {values, indices};
        }

        // The k-th smallest element along dim (1-based) and its index, via a stable
        // sort.
        std::tuple<at::Tensor&, at::Tensor&> tt_kthvalue_out(
            const at::Tensor& self,
            int64_t k,
            int64_t dim,
            bool keepdim,
            at::Tensor& values,
            at::Tensor& indices
        ) {
            const char* op = "kthvalue.values";
            auto x = self.dim() == 0 ? self.reshape({1}) : self;
            dim = c10::maybe_wrap_dim(dim, x.dim());
            KTT_REQUIRE(k >= 1 && k <= x.size(dim), op, "k out of range");
            const bool stable = true;
            const bool descending = false;
            auto [sorted, order] = at::sort(x, stable, dim, descending);
            auto v = sorted.narrow(dim, k - 1, 1), i = order.narrow(dim, k - 1, 1);
            if (!keepdim || self.dim() == 0) {
                v = v.squeeze(dim);
                i = i.squeeze(dim);
            }
            for (auto* t : {&values, &indices}) {
                if (t->sizes() != v.sizes()) {
                    t->resize_(v.sizes());
                }
            }
            write_into(op, values, v.to(values.scalar_type()));
            write_into(op, indices, i.to(indices.scalar_type()));
            return {values, indices};
        }

        // Insertion index of each value into sorted `boundaries` (last dim; batched
        // when boundaries has more dims): count of boundaries < value (right: <=).
        at::Tensor search_sorted(
            const at::Tensor& boundaries,
            const at::Tensor& values,
            bool out_int32,
            bool right
        ) {
            auto b = boundaries.to(at::kFloat).unsqueeze(-2);  // [..., 1, M]
            auto v = values.to(at::kFloat).unsqueeze(-1);      // [..., K, 1]
            auto below = right ? at::le(b, v) : at::lt(b, v);
            auto count = below.to(at::kFloat).sum(-1);
            return count.to(out_int32 ? at::kInt : at::kLong);
        }

        at::Tensor tt_bucketize(
            const at::Tensor& self,
            const at::Tensor& boundaries,
            bool out_int32,
            bool right
        ) {
            KTT_REQUIRE(boundaries.dim() == 1, "bucketize", "boundaries must be 1-D");
            auto flat = search_sorted(boundaries, self.reshape({-1}), out_int32, right);
            return flat.reshape(self.sizes());
        }

        at::Tensor tt_searchsorted(
            const at::Tensor& sorted,
            const at::Tensor& self,
            bool out_int32,
            bool right,
            std::optional<c10::string_view> side,
            const std::optional<at::Tensor>& sorter
        ) {
            const bool use_right = side ? *side == "right" : right;
            // With a sorter, sorted_sequence[sorter] is the sorted sequence.
            auto seq =
                (sorter && sorter->defined() ? at::gather(sorted, -1, *sorter)
                                             : sorted);
            if (seq.dim() == 1) {
                auto flat = search_sorted(
                    seq, self.reshape({-1}), out_int32, use_right
                );
                return flat.reshape(self.sizes());
            }
            return search_sorted(seq, self, out_int32, use_right);
        }

        // unique (sorted) with inverse and counts, on host (data-dependent size).
        std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_unique2(
            const at::Tensor& self, bool sorted, bool return_inverse, bool return_counts
        ) {
            auto [values, inverse, counts] = (
                at::_unique2(self.cpu(), sorted, return_inverse, return_counts)
            );
            auto device = self.device();
            return {values.to(device), inverse.to(device), counts.to(device)};
        }

        // unique_consecutive with inverse and counts, on host (data-dependent size).
        std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_unique_consecutive(
            const at::Tensor& self,
            bool return_inverse,
            bool return_counts,
            std::optional<int64_t> dim
        ) {
            auto [values, inverse, counts] = (
                at::unique_consecutive(self.cpu(), return_inverse, return_counts, dim)
            );
            auto device = self.device();
            return {values.to(device), inverse.to(device), counts.to(device)};
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("sort.stable", &tt_sort);
        m.impl("sort.values_stable", &tt_sort_out);
        m.impl("topk", &tt_topk);
        m.impl("topk.values", &tt_topk_out);
        m.impl("kthvalue.values", &tt_kthvalue_out);
        m.impl("bucketize.Tensor", &tt_bucketize);
        m.impl("searchsorted.Tensor", &tt_searchsorted);
        m.impl("_unique2", &tt_unique2);
        m.impl("unique_consecutive", &tt_unique_consecutive);
    }

}  // namespace ktt
