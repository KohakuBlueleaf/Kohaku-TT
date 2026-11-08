// aten scatter kernels for tt: scatter (src / value, add / multiply), scatter_add,
// scatter_reduce (sum / prod / mean / amax / amin), index_put, index_fill,
// masked_scatter.

#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/accumulate.h>
#include <torch/library.h>

#include <algorithm>
#include <limits>
#include <string>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // self with src[index-shaped part] scattered along `dim`; reduce "", "add",
        // "multiply".
        at::Tensor scatter_impl(
            const char* op,
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src,
            const std::string& reduce
        ) {
            if (index.numel() == 0 || self.numel() == 0) {
                return self.clone();
            }
            const int64_t rank = std::max<int64_t>(self.dim(), 1);
            dim = c10::maybe_wrap_dim(dim, rank);
            KTT_REQUIRE(
                index.dim() == self.dim(), op, "index rank must match input rank"
            );

            // Only the index-shaped corner of src is scattered.
            auto part = src.to(self.scalar_type());
            for (int64_t d = 0; d < index.dim(); ++d) {
                if (part.size(d) != index.size(d)) {
                    part = part.narrow(d, 0, index.size(d));
                }
            }
            // Device dim: rank < 2 tensors are padded to 2-D (a 0-d tensor is (1, 1)).
            const int64_t pad = std::max<int64_t>(0, 2 - self.dim());
            const int64_t device_dim = self.dim() == 0 ? 1 : dim + pad;
            auto h = (
                bridge::scatter(
                    dense(op, self),
                    device_dim,
                    dense_as(op, index, at::kInt),
                    dense(op, part),
                    reduce
                )
            );
            return wrap_output(h, self.sizes(), self.scalar_type());
        }

        std::string reduce_name(const char* op, c10::string_view reduce) {
            if (reduce == "add" || reduce == "sum") {
                return "add";
            }
            if (reduce == "multiply" || reduce == "prod") {
                return "multiply";
            }
            unsupported(op, c10::str("reduce '", reduce, "'"));
            return "";
        }

        at::Tensor tt_scatter_src(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src
        ) {
            return scatter_impl("scatter.src", self, dim, index, src, "");
        }

        at::Tensor tt_scatter_value(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Scalar& v
        ) {
            auto src = at::full(index.sizes(), v, self.options());
            return scatter_impl("scatter.value", self, dim, index, src, "");
        }

        at::Tensor tt_scatter_reduce_src(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src,
            c10::string_view reduce
        ) {
            const char* op = "scatter.reduce";
            return scatter_impl(op, self, dim, index, src, reduce_name(op, reduce));
        }

        at::Tensor tt_scatter_add(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src
        ) {
            return scatter_impl("scatter_add", self, dim, index, src, "add");
        }

        // scatter_reduce: sum / prod (with or without self), mean = sum / count.
        at::Tensor tt_scatter_reduce_two(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src,
            c10::string_view reduce,
            bool include_self
        ) {
            const char* op = "scatter_reduce.two";
            if (reduce == "amin") {
                return (
                    -tt_scatter_reduce_two(
                        -self, dim, index, -src, "amax", include_self
                    )
                );
            }
            if ((reduce == "amax" || reduce == "amin") && self.dim() == 0) {
                return tt_scatter_reduce_two(
                           self.reshape({1}),
                           0,
                           index.reshape({1}),
                           src.reshape({1}),
                           reduce,
                           include_self
                )
                    .reshape({});
            }
            if (reduce == "amax") {
                const double lowest =
                    (at::isFloatingType(self.scalar_type())
                         ? -std::numeric_limits<double>::infinity()
                         : static_cast<double>(std::numeric_limits<int32_t>::min()));
                auto t = self;
                if (!include_self) {
                    t = (
                        scatter_impl(
                            op,
                            self,
                            dim,
                            index,
                            at::full(index.sizes(), lowest, self.options()),
                            ""
                        )
                    );
                }
                auto s = src;
                for (int64_t d = 0; d < index.dim(); ++d) {
                    s = s.slice(d, 0, index.size(d));
                }
                // Each element takes the max over its row's elements with the same
                // index (pairwise [n, n] mask), so duplicate writes carry one value.
                const int64_t n = index.size(dim);
                const int64_t rows = n > 0 ? index.numel() / n : 0;
                if (rows * n * n <= (int64_t{1} << 24)) {
                    auto idx = index.movedim(dim, -1);
                    auto values = s.movedim(dim, -1);
                    auto same = at::eq(idx.unsqueeze(-1), idx.unsqueeze(-2));
                    auto masked = at::where(same, values.unsqueeze(-2), lowest);
                    auto group = std::get<0>(masked.max(-1)).movedim(-1, dim);
                    auto current = t.gather(dim, index);
                    return (
                        scatter_impl(op, t, dim, index, at::maximum(current, group), "")
                    );
                }
                // Large rows: running max by repeated gather / max / scatter until no
                // src exceeds its target.
                for (int pass = 0; pass < 64; ++pass) {
                    auto current = t.gather(dim, index);
                    auto higher = at::gt(s, current);
                    if (!higher.any().item<bool>()) {
                        break;
                    }
                    t = scatter_impl(op, t, dim, index, at::maximum(current, s), "");
                }
                return t;
            }
            const bool mean = reduce == "mean";
            KTT_REQUIRE(
                reduce == "sum" || reduce == "prod" || mean,
                op,
                "reduce '",
                reduce,
                "' (TT-NN scatter reduces by add / multiply only)"
            );
            const bool prod = reduce == "prod";
            auto base = self;
            if (!include_self) {
                // Positions that receive values start from the reduction's identity.
                auto identity = (
                    at::full(index.sizes(), prod ? 1.0 : 0.0, self.options())
                );
                base = scatter_impl(op, self, dim, index, identity, "");
            }
            auto total = (
                scatter_impl(op, base, dim, index, src, prod ? "multiply" : "add")
            );
            if (!mean) {
                return total;
            }
            KTT_REQUIRE(at::isFloatingType(self.scalar_type()), op, "mean of integers");
            auto counts = (
                at::full(self.sizes(), include_self ? 1.0 : 0.0, self.options())
            );
            auto ones = at::ones(index.sizes(), self.options());
            counts = scatter_impl(op, counts, dim, index, ones, "add");
            auto divisor = at::clamp(counts, 1, std::nullopt);
            return at::where(at::gt(counts, 0), total / divisor, self);
        }

        at::Tensor& tt_scatter_src_out(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src,
            at::Tensor& out
        ) {
            return into("scatter.src_out", out, tt_scatter_src(self, dim, index, src));
        }

        at::Tensor& tt_scatter_value_out(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Scalar& value,
            at::Tensor& out
        ) {
            return (
                into(
                    "scatter.value_out", out, tt_scatter_value(self, dim, index, value)
                )
            );
        }

        at::Tensor& tt_scatter_reduce_out(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src,
            c10::string_view reduce,
            at::Tensor& out
        ) {
            auto result = tt_scatter_reduce_src(self, dim, index, src, reduce);
            return into("scatter.reduce_out", out, result);
        }

        at::Tensor& tt_scatter_add_out(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src,
            at::Tensor& out
        ) {
            return into("scatter_add.out", out, tt_scatter_add(self, dim, index, src));
        }

        at::Tensor& tt_scatter_reduce_two_out(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& src,
            c10::string_view reduce,
            bool include_self,
            at::Tensor& out
        ) {
            auto result = (
                tt_scatter_reduce_two(self, dim, index, src, reduce, include_self)
            );
            return into("scatter_reduce.two_out", out, result);
        }

        // self[indices] = values (+= with accumulate) for indices on a leading block of
        // dims (bool masks via nonzero): flat element indices on device, scattered into
        // flat self.
        at::Tensor& tt_index_put_impl_(
            at::Tensor& self,
            const c10::List<std::optional<at::Tensor>>& indices,
            const at::Tensor& values,
            bool accumulate,
            bool unsafe
        ) {
            const char* op = "_index_put_impl_";
            std::vector<at::Tensor> idx;
            bool gap = false;
            for (const auto& maybe : indices) {
                const auto& t = static_cast<std::optional<at::Tensor>>(maybe);
                if (!t || !t->defined()) {
                    gap = true;
                    continue;
                }
                KTT_REQUIRE(
                    !gap, op, "indices after a skipped dim are not supported yet"
                );
                if (t->scalar_type() == at::kBool || t->scalar_type() == at::kByte) {
                    auto nz = at::nonzero(*t);
                    for (int64_t k = 0; k < nz.size(1); ++k) {
                        idx.push_back(nz.select(1, k));
                    }
                } else {
                    idx.push_back(*t);
                }
            }
            const int64_t m = static_cast<int64_t>(idx.size());
            KTT_REQUIRE(m >= 1 && m <= self.dim(), op, "needs 1..dim index tensors");

            // Broadcast index shape B, trailing (un-indexed) sizes and their element
            // count T.
            auto shape = at::infer_size_dimvector(idx[0].sizes(), idx[0].sizes());
            for (const auto& t : idx) {
                shape = at::infer_size_dimvector(shape, t.sizes());
            }
            std::vector<int64_t> trailing(self.sizes().begin() + m, self.sizes().end());
            const int64_t inner = c10::multiply_integers(trailing);
            if (c10::multiply_integers(shape) == 0 || inner == 0) {
                return self;
            }

            // Row index over the indexed dims (negative indices wrap), then element
            // index.
            at::Tensor row;
            int64_t row_stride = 1;
            for (int64_t k = m - 1; k >= 0; --k) {
                auto i = idx[k].expand(shape).to(at::kInt);
                i = at::where(at::lt(i, 0), i + self.size(k), i);
                auto term = row_stride == 1 ? i : i * row_stride;
                row = row.defined() ? row + term : term;
                row_stride *= self.size(k);
            }
            std::vector<int64_t> full(shape.begin(), shape.end());
            full.insert(full.end(), trailing.begin(), trailing.end());
            std::vector<int64_t> row_shape(shape.begin(), shape.end());
            row_shape.insert(row_shape.end(), trailing.size(), 1);
            auto flat_index = row.reshape(row_shape) * inner;
            if (inner > 1) {
                auto offsets = at::arange(inner, self.options().dtype(at::kInt));
                flat_index = (flat_index + offsets.reshape(trailing)).expand(full);
            }
            const int64_t count = c10::multiply_integers(full);
            auto src = values.to(self.scalar_type()).expand(full).reshape({1, count});
            auto flat_self = self.reshape({1, self.numel()});
            auto result = scatter_impl(
                op,
                flat_self,
                1,
                flat_index.reshape({1, count}),
                src,
                accumulate ? "add" : ""
            );
            return write_into(op, self, result.reshape(self.sizes()));
        }

        // self[indices]: indexed dims moved first, index tensors linearized to row ids
        // on device, rows gathered; index dims placed as PyTorch does.
        at::Tensor tt_index(
            const at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices
        ) {
            const char* op = "index.Tensor";
            std::vector<int64_t> indexed;
            std::vector<at::Tensor> idx;
            int64_t d = 0;
            for (const auto& maybe : indices) {
                const auto& t = static_cast<std::optional<at::Tensor>>(maybe);
                if (!t || !t->defined()) {
                    ++d;
                    continue;
                }
                if (t->scalar_type() == at::kBool || t->scalar_type() == at::kByte) {
                    auto nz = at::nonzero(*t);
                    for (int64_t k = 0; k < nz.size(1); ++k) {
                        indexed.push_back(d++);
                        idx.push_back(nz.select(1, k));
                    }
                } else {
                    indexed.push_back(d++);
                    idx.push_back(*t);
                }
            }
            KTT_REQUIRE(!idx.empty(), op, "no index tensors");
            KTT_REQUIRE(d <= self.dim(), op, "too many indices");
            const bool adjacent =
                (indexed.back() - indexed.front() + 1 ==
                 static_cast<int64_t>(indexed.size()));

            // Indexed dims first, the rest in order.
            std::vector<int64_t> perm(indexed.begin(), indexed.end()), rest;
            for (int64_t k = 0; k < self.dim(); ++k) {
                if (std::find(indexed.begin(), indexed.end(), k) == indexed.end()) {
                    perm.push_back(k);
                    rest.push_back(k);
                }
            }
            auto front = self.permute(perm);
            const int64_t m = static_cast<int64_t>(idx.size());

            auto shape = idx[0].sizes().vec();
            for (const auto& t : idx) {
                shape = at::infer_size(shape, t.sizes());
            }
            std::vector<int64_t> trailing(
                front.sizes().begin() + m, front.sizes().end()
            );
            std::vector<int64_t> out_sizes(shape.begin(), shape.end());
            out_sizes.insert(out_sizes.end(), trailing.begin(), trailing.end());
            const int64_t rows = c10::multiply_integers(shape);
            const int64_t inner = c10::multiply_integers(trailing);
            at::Tensor out;
            if (rows == 0 || inner == 0 || self.numel() == 0) {
                out = at::empty(out_sizes, self.options());
            } else {
                at::Tensor row;
                int64_t row_stride = 1;
                for (int64_t k = m - 1; k >= 0; --k) {
                    auto i = idx[k].expand(shape).to(at::kLong);
                    i = at::where(at::lt(i, 0), i + front.size(k), i);
                    auto term = row_stride == 1 ? i : i * row_stride;
                    row = row.defined() ? row + term : term;
                    row_stride *= front.size(k);
                }
                auto table = front.reshape({row_stride, inner});
                out = at::index_select(table, 0, row.reshape({rows}))
                          .reshape(out_sizes);
            }
            if (!adjacent || indexed.front() == 0) {
                return out;
            }
            // Move the index dims (now first) to the first indexed position.
            const int64_t b = static_cast<int64_t>(shape.size());
            const int64_t at_dim = indexed.front();
            std::vector<int64_t> order;
            for (int64_t k = 0; k < at_dim; ++k) {
                order.push_back(b + k);
            }
            for (int64_t k = 0; k < b; ++k) {
                order.push_back(k);
            }
            for (int64_t k = b + at_dim; k < out.dim(); ++k) {
                order.push_back(k);
            }
            return out.permute(order).contiguous();
        }

        at::Tensor& tt_index_out(
            const at::Tensor& self,
            const c10::List<std::optional<at::Tensor>>& indices,
            at::Tensor& out
        ) {
            return into("index.Tensor_out", out, tt_index(self, indices));
        }

        // self's slices at `index` along `dim` set to `value`: index_put on a view with
        // `dim` moved to the front.
        at::Tensor& tt_index_fill_(
            at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Scalar& value
        ) {
            dim = c10::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
            auto front = self.dim() == 0 ? self.reshape({1}) : self.transpose(0, dim);
            c10::List<std::optional<at::Tensor>> indices;
            indices.push_back(index.reshape({-1}));
            auto filler = at::scalar_tensor(value, self.options());
            tt_index_put_impl_(front, indices, filler, false, false);
            return self;
        }

        at::Tensor& tt_index_fill_tensor_(
            at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& value
        ) {
            KTT_REQUIRE(
                value.dim() == 0, "index_fill_.int_Tensor", "value must be 0-d"
            );
            auto v = value.is_cpu() ? value.item() : value.cpu().item();
            return tt_index_fill_(self, dim, index, v);
        }

        // self with alpha * source[i] added at self's index[i] along `dim`: a
        // scatter_add with the 1-D index spread over source's shape.
        at::Tensor& tt_index_add_out(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& source,
            const at::Scalar& alpha,
            at::Tensor& out
        ) {
            const char* op = "index_add.out";
            if (self.dim() == 0) {
                auto result = self + source.reshape({}) * alpha;
                return into(op, out, result);
            }
            dim = c10::maybe_wrap_dim(dim, self.dim());
            auto src = source.dim() == 0 ? source.reshape({1}) : source;
            std::vector<int64_t> view(src.dim(), 1);
            view[dim] = index.numel();
            auto spread = index.reshape(view).expand(src.sizes());
            auto scaled = alpha.toDouble() == 1.0 ? src : src * alpha;
            return into(op, out, at::scatter_add(self, dim, spread, scaled));
        }

        // Reduction (sum / mean / max / min / prod) of consecutive segments of `data`
        // along `axis`, segments from 1-D `lengths` or `offsets`: scatter_reduce of
        // each position into its segment; an empty segment holds `initial`, else 0 /
        // NaN / -inf / inf / 1.
        at::Tensor tt_segment_reduce(
            const at::Tensor& data,
            c10::string_view reduce,
            const std::optional<at::Tensor>& lengths,
            const std::optional<at::Tensor>& indices,
            const std::optional<at::Tensor>& offsets,
            int64_t axis,
            bool unsafe,
            const std::optional<at::Scalar>& initial
        ) {
            const char* op = "segment_reduce";
            KTT_REQUIRE(
                !indices || !indices->defined(), op, "indices are not supported"
            );
            at::Tensor counts;
            if (lengths && lengths->defined()) {
                counts = lengths->cpu().to(at::kLong);
            } else {
                KTT_REQUIRE(
                    offsets && offsets->defined(), op, "needs lengths or offsets"
                );
                auto o = offsets->cpu().to(at::kLong);
                KTT_REQUIRE(o.dim() == 1, op, "batched offsets are not supported");
                counts = o.narrow(0, 1, o.numel() - 1) - o.narrow(0, 0, o.numel() - 1);
            }
            KTT_REQUIRE(counts.dim() == 1, op, "batched lengths are not supported");
            axis = c10::maybe_wrap_dim(axis, data.dim());
            const int64_t segments = counts.numel();
            auto ids = at::repeat_interleave(at::arange(segments, at::kLong), counts);
            const int64_t used = ids.numel();
            auto x = data.narrow(axis, 0, used).to(at::kFloat);
            std::vector<int64_t> view(x.dim(), 1);
            view[axis] = used;
            auto index = (
                ids.to(data.device()).reshape(view).expand(x.sizes()).contiguous()
            );
            auto out_sizes = x.sizes().vec();
            out_sizes[axis] = segments;
            const std::string kind(reduce);
            double fill = 0.0;
            std::string scatter = kind;
            if (kind == "max") {
                fill = -std::numeric_limits<double>::infinity();
                scatter = "amax";
            } else if (kind == "min") {
                fill = std::numeric_limits<double>::infinity();
                scatter = "amin";
            } else if (kind == "prod") {
                fill = 1.0;
            } else if (kind == "mean") {
                fill = std::numeric_limits<double>::quiet_NaN();
            } else {
                KTT_REQUIRE(kind == "sum", op, "reduce '", kind, "'");
            }
            const double empty = initial ? initial->toDouble() : fill;
            auto base = at::full(out_sizes, kind == "mean" ? 0.0 : empty, x.options());
            if (used == 0) {
                return at::full(out_sizes, empty, data.options());
            }
            // `initial` enters every segment (mean: added to the sum, divided by the
            // segment length).
            const bool include_self = initial.has_value();
            const std::string scatter_kind = kind == "mean" ? "sum" : scatter;
            if (initial) {
                base = at::full(out_sizes, initial->toDouble(), x.options());
            }
            auto result = (
                tt_scatter_reduce_two(base, axis, index, x, scatter_kind, include_self)
            );
            auto size_view = std::vector<int64_t>(x.dim(), 1);
            size_view[axis] = segments;
            auto sizes = counts.to(at::kFloat).to(data.device()).reshape(size_view);
            if (kind == "mean") {
                result = result / at::where(at::gt(sizes, 0.0), sizes, 1.0);
            }
            result = at::where(at::eq(sizes, 0.0), empty, result);
            return result.to(data.scalar_type());
        }

        // index_reduce = scatter_reduce with the 1-D index spread over source's shape.
        at::Tensor& tt_index_reduce_out(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& source,
            c10::string_view reduce,
            bool include_self,
            at::Tensor& out
        ) {
            // 0-d tensors run as 1-D and are shaped back.
            auto base = self.dim() == 0 ? self.reshape({1}) : self;
            auto src = source.dim() == 0 ? source.reshape({1}) : source;
            dim = c10::maybe_wrap_dim(dim, base.dim());
            std::vector<int64_t> view(src.dim(), 1);
            view[dim] = index.numel();
            auto spread = index.reshape(view).expand(src.sizes());
            auto result = (
                tt_scatter_reduce_two(base, dim, spread, src, reduce, include_self)
            );
            return into("index_reduce.out", out, result.reshape(self.sizes()));
        }

        // self with masked positions filled, in order, from the elements of source.
        at::Tensor& tt_masked_scatter_(
            at::Tensor& self, const at::Tensor& mask, const at::Tensor& source
        ) {
            const char* op = "masked_scatter_";
            auto m = mask.expand(self.sizes());
            if (self.numel() == 0 || source.numel() == 0) {
                return self;
            }
            // Source position of each element: running count of true mask entries - 1.
            auto flat_mask = m.reshape({1, self.numel()}).to(at::kInt);
            auto position = (
                at::clamp(
                    at::cumsum(flat_mask, 1) - 1,
                    0,
                    static_cast<int64_t>(source.numel()) - 1
                )
            );
            auto flat_source = source.to(self.scalar_type())
                                   .reshape({1, source.numel()});
            auto picked = at::gather(flat_source, 1, position.to(at::kLong));
            auto out = at::where(m, picked.reshape(self.sizes()), self);
            return write_into(op, self, out);
        }

        // out[i][j] = self[index[i][j]][j] (dim 0) etc.: TT-NN gather along `dim`.
        at::Tensor tt_gather(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            bool sparse_grad
        ) {
            const char* op = "gather";
            KTT_REQUIRE(
                self.dim() == index.dim(), op, "index rank must match input rank"
            );
            KTT_REQUIRE(self.dim() >= 1, op, "0-d input");
            dim = c10::maybe_wrap_dim(dim, self.dim());
            auto pad = std::max<int64_t>(0, 2 - self.dim());
            auto h = (
                bridge::gather(
                    dense(op, self), dim + pad, dense_as(op, index, at::kInt)
                )
            );
            return wrap_output(h, index.sizes(), self.scalar_type());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("scatter.src", &tt_scatter_src);
        m.impl("scatter.src_out", &tt_scatter_src_out);
        m.impl("scatter.value", &tt_scatter_value);
        m.impl("scatter.value_out", &tt_scatter_value_out);
        m.impl("scatter.reduce", &tt_scatter_reduce_src);
        m.impl("scatter.reduce_out", &tt_scatter_reduce_out);
        m.impl("scatter_add", &tt_scatter_add);
        m.impl("scatter_add.out", &tt_scatter_add_out);
        m.impl("scatter_reduce.two", &tt_scatter_reduce_two);
        m.impl("scatter_reduce.two_out", &tt_scatter_reduce_two_out);
        m.impl("_index_put_impl_", &tt_index_put_impl_);
        m.impl("index.Tensor", &tt_index);
        m.impl("index.Tensor_out", &tt_index_out);
        m.impl("masked_scatter_", &tt_masked_scatter_);
        m.impl("index_add.out", &tt_index_add_out);
        m.impl("index_reduce.out", &tt_index_reduce_out);
        m.impl("segment_reduce", &tt_segment_reduce);
        m.impl("index_fill_.int_Scalar", &tt_index_fill_);
        m.impl("index_fill_.int_Tensor", &tt_index_fill_tensor_);
        m.impl("gather", &tt_gather);
    }

}  // namespace ktt
