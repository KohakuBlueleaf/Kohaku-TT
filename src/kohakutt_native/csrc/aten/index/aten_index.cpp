// aten indexing kernels for tt: index_select / index_copy, take / put, masked_select,
// nonzero, isin, repeat_interleave.

#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>

#include "core/torch/contract.hpp"
#include "core/torch/mesh_rules.hpp"

namespace ktt {
    namespace {

        // self rows (along dim) at `index`, by gather over the dim moved to the front.
        at::Tensor tt_index_select(
            const at::Tensor& self, int64_t dim, const at::Tensor& index
        ) {
            const char* op = "index_select";
            KTT_REQUIRE(index.dim() <= 1, op, "index must be 0-d or 1-d");
            dim = c10::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));

            // Rows of a 2-D bf16 table (embedding lookups), or along the second-to-last
            // dim of an fp32 / bf16 tensor: the row gather.
            const bool bf16_table =
                (dim == 0 && self.dim() == 2 && self.scalar_type() == at::kBFloat16);
            const bool fp32_rows =
                (self.dim() >= 2 && dim == self.dim() - 2 &&
                 (self.scalar_type() == at::kFloat ||
                  self.scalar_type() == at::kBFloat16) &&
                 bridge::row_gather());
            if ((bf16_table || fp32_rows) && index.numel() > 0 && self.numel() > 0) {
                const int64_t n = index.numel();
                // A mesh-placed table: rows per chip (vocab- or hidden-sharded).
                if (dim == 0 && self.dim() == 2) {
                    if (auto table = mesh::placed_base(self)) {
                        auto idx = dense_as(op, index.reshape({1, n}), at::kInt);
                        if (auto rows = mesh::placed_rows(table, idx, n)) {
                            if (mesh::recording()) {
                                mesh::record(
                                    op,
                                    {bridge::placement(table), bridge::placement(idx)},
                                    bridge::placement(*rows),
                                    {},
                                    "rows of a sharded table"
                                );
                            }
                            return wrap_output(
                                *rows, {n, self.size(1)}, self.scalar_type()
                            );
                        }
                    } else if (auto idx = mesh::placed_base(index.reshape({1, n}))) {
                        // Mesh-placed indices into a replicated table: rows per chip.
                        const int64_t version =
                            (self.is_inference()
                                 ? 0
                                 : static_cast<int64_t>(self._version()));
                        if (auto rows = mesh::placed_index_rows(
                                dense(op, self), version, idx, n
                            )) {
                            if (mesh::recording()) {
                                mesh::record(
                                    op,
                                    {bridge::Placement{}, bridge::placement(idx)},
                                    bridge::placement(*rows),
                                    {},
                                    "rows at sharded indices"
                                );
                            }
                            return wrap_output(
                                *rows, {n, self.size(1)}, self.scalar_type()
                            );
                        }
                    }
                }
                auto h = bridge::embedding_rows(
                    dense(op, self),
                    self.is_inference() ? 0 : static_cast<int64_t>(self._version()),
                    dense_as(op, index.reshape({1, n}), at::kInt),
                    n
                );
                auto sizes = to_vector(self.sizes());
                sizes[dim] = n;
                return wrap_output(h, sizes, self.scalar_type());
            }
            // Along the last dim of an fp32 / bf16 tensor: the row gather between two
            // transposes of the last two dims.
            if (self.dim() >= 2 && dim == self.dim() - 1 &&
                (self.scalar_type() == at::kFloat ||
                 self.scalar_type() == at::kBFloat16) &&
                bridge::row_gather() && index.numel() > 0 && self.numel() > 0) {
                auto t = self.transpose(-1, -2).contiguous();
                return (tt_index_select(t, self.dim() - 2, index)
                            .transpose(-1, -2)
                            .contiguous());
            }
            auto moved = self.dim() == 0 ? self.reshape({1}) : self.movedim(dim, 0);
            auto rows = moved.reshape({moved.size(0), -1});
            const int64_t n = index.numel(), width = rows.size(1);
            // Other dims of an fp32 / bf16 tensor: the row gather of the [D, rest]
            // rows.
            if (self.dim() > 2 && rows.dim() == 2 &&
                (self.scalar_type() == at::kFloat ||
                 self.scalar_type() == at::kBFloat16) &&
                bridge::row_gather() && n > 0 && self.numel() > 0) {
                auto out_sizes = to_vector(moved.sizes());
                out_sizes[0] = n;
                return (tt_index_select(rows, 0, index)
                            .reshape(out_sizes)
                            .movedim(0, dim)
                            .contiguous());
            }
            auto idx = index.reshape({n, 1}).expand({n, width});
            auto picked = (bridge::gather(
                dense(op, rows), 0, dense_as(op, idx, at::kInt)
            ));
            auto out_sizes = to_vector(moved.sizes());
            out_sizes[0] = n;
            auto flat = wrap_output(picked, {n, width}, self.scalar_type());
            auto result = flat.reshape(out_sizes);
            return self.dim() == 0 ? result.reshape({})
                                   : result.movedim(0, dim).contiguous();
        }

        // A [B, H, T, D] head view over its whole [B, T, C] Nchw storage: the storage's
        // base, with the heads' column offset and stride in `view`; else null.
        bridge::HandlePtr head_view_base(const at::Tensor& t, bridge::HeadView& view) {
            if (t.dim() != 4) {
                return nullptr;
            }
            const auto& st = storage_of(t);
            if (!st.base || st.blank || st.dtype != t.scalar_type() ||
                st.sizes.size() != 3 || bridge::is_block(bridge::dtype(st.base)) ||
                bridge::format(st.base) != bridge::Format::Nchw ||
                !bridge::placement(st.base).is_replicated()) {
                return nullptr;
            }
            const int64_t b = st.sizes[0], seq = st.sizes[1], c = st.sizes[2];
            const int64_t heads = t.size(1), d = t.size(3), off = t.storage_offset();
            const int64_t stride = heads > 1 ? t.stride(1) : 0;
            if (t.size(0) != b || t.size(2) != seq || t.stride(3) != 1 ||
                (b > 1 && t.stride(0) != seq * c) || (seq > 1 && t.stride(2) != c) ||
                off % 32 != 0 || stride % 32 != 0 || d % 32 != 0 ||
                off + (heads - 1) * stride + d > c) {
                return nullptr;
            }
            view = bridge::HeadView{heads, off, stride};
            return st.base;
        }

        // index_copy_ of a whole bf16 / fp32 / int32 storage along dim -2 or a leading
        // dim: one program writing the touched tiles in place; false for other forms.
        bool index_copy_in_place(
            at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& source
        ) {
            const char* op = "index_copy_";
            const int64_t rank = self.dim(), n = index.numel();
            if (!index.is_privateuseone() || !source.is_privateuseone() || rank < 2 ||
                dim >= rank - 1 || n == 0 || self.numel() == 0 ||
                source.scalar_type() != self.scalar_type() ||
                (index.scalar_type() != at::kLong && index.scalar_type() != at::kInt) ||
                source.storage().is_alias_of(self.storage())) {
                return false;
            }
            auto expected = to_vector(self.sizes());
            expected[dim] = n;
            auto& st = storage_of(self);
            const auto type = self.scalar_type();
            if (to_vector(source.sizes()) != expected || !self.is_contiguous() ||
                self.storage_offset() != 0 || !st.base || st.blank ||
                st.dtype != self.scalar_type() || st.sizes != to_vector(self.sizes()) ||
                (type != at::kBFloat16 && type != at::kFloat && type != at::kInt &&
                 type != at::kLong && type != at::kBool) ||
                bridge::is_block(bridge::dtype(st.base)) ||
                bridge::format(st.base) != bridge::Format::Nchw ||
                !bridge::placement(st.base).is_replicated()) {
                return false;
            }
            bridge::HeadView view;
            auto src = dim == rank - 2 ? head_view_base(source, view) : nullptr;
            if (!src) {
                view = {};
                src = dense(op, source);
            }
            auto idx = dense_as(op, index.reshape({1, n}), at::kInt);
            if (!bridge::index_copy_rows(st.base, dim, idx, n, src, view, false)) {
                return false;
            }
            // Outside traces the first in-place write goes to a private copy (a base
            // can be shared); while traces are alive bases are written in place.
            if (!bridge::traces_alive() && st.owned.lock() != st.base) {
                auto copy = bridge::clone(st.base);
                bridge::keep_in_dram(copy);
                note_rebind(st, self.storage());
                st.base = copy;
                st.owned = copy;
            }
            bridge::index_copy_rows(st.base, dim, idx, n, src, view);
            st.heads_of.reset();
            st.heads.clear();
            st.scalar.reset();
            return true;
        }

        // self[.., index[i], ..] = source[.., i, ..] on `dim`: index_copy_in_place,
        // else self * (1 - hit) + onehot(L, T) @ source with the dim moved to -2.
        at::Tensor& tt_index_copy_(
            at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& source
        ) {
            const char* op = "index_copy_";
            KTT_REQUIRE(index.dim() <= 1, op, "index must be 0-d or 1-d");
            if (self.dim() < 2) {
                // 0-d / 1-d: run as [L, 1] along dim 0.
                auto as2d = self.reshape({-1, 1}).clone();
                tt_index_copy_(as2d, 0, index.reshape({-1}), source.reshape({-1, 1}));
                return write_into(op, self, as2d.reshape(self.sizes()));
            }
            dim = c10::maybe_wrap_dim(dim, self.dim());
            if (index_copy_in_place(self, dim, index, source)) {
                return self;
            }
            const int64_t length = self.size(dim), count = index.numel();
            auto idx = index.reshape({1, count});
            auto classes = at::arange(length, idx.options()).reshape({length, 1});
            auto onehot = at::eq(classes, idx).to(at::kFloat);
            const bool keepdim = true;
            auto hit = onehot.amax(std::vector<int64_t>{1}, keepdim);
            auto dst = self.movedim(dim, -2).to(at::kFloat);
            auto src = source.movedim(dim, -2).to(at::kFloat);
            auto updated = dst * (1.0 - hit) + exact_select(onehot, src, false);
            return (
                write_into(op, self, updated.movedim(-2, dim).to(self.scalar_type()))
            );
        }

        at::Tensor& tt_index_copy_out(
            const at::Tensor& self,
            int64_t dim,
            const at::Tensor& index,
            const at::Tensor& source,
            at::Tensor& out
        ) {
            auto result = self.clone();
            // A fresh clone is this tensor's alone: written in place without a copy.
            auto& st = storage_of(result);
            st.owned = st.base;
            tt_index_copy_(result, dim, index, source);
            return into("index_copy.out", out, result);
        }

        // self.flatten()[index] with negative indices wrapped, shaped as index.
        at::Tensor tt_take(const at::Tensor& self, const at::Tensor& index) {
            const int64_t n = self.numel();
            if (index.numel() == 0) {
                return at::empty(index.sizes(), self.options());
            }
            auto idx = index.reshape({-1}).to(at::kLong);
            idx = at::where(at::lt(idx, 0), idx + n, idx);
            return self.reshape({-1}).index_select(0, idx).reshape(index.sizes());
        }

        // self.flatten()[index] = source (or += with accumulate), negative indices
        // wrapped.
        at::Tensor& tt_put_(
            at::Tensor& self,
            const at::Tensor& index,
            const at::Tensor& source,
            bool accumulate
        ) {
            if (index.numel() == 0) {
                return self;
            }
            const int64_t n = self.numel();
            auto idx = index.reshape({-1}).to(at::kLong);
            idx = at::where(at::lt(idx, 0), idx + n, idx);
            auto flat = self.reshape({-1}).clone();
            auto values = source.reshape({-1}).to(self.scalar_type());
            auto result = at::index_put(flat, {idx}, values, accumulate);
            return write_into("put_", self, result.reshape(self.sizes()));
        }

        // Elements of self (broadcast with mask) where mask is true, flattened.
        at::Tensor tt_masked_select(const at::Tensor& self, const at::Tensor& mask) {
            auto shape = at::infer_size(self.sizes(), mask.sizes());
            auto flat = self.expand(shape).reshape({-1});
            auto picks = at::nonzero(mask.expand(shape).reshape({-1}).cpu())
                             .reshape({-1});
            return flat.index_select(0, picks.to(self.device()));
        }

        // The first `size` nonzero indices (padded with fill_value), on host like
        // nonzero.
        at::Tensor tt_nonzero_static(
            const at::Tensor& self, int64_t size, int64_t fill
        ) {
            return at::nonzero_static(self.cpu(), size, fill).to(self.device());
        }

        // Indices of nonzero elements, computed on host (data-dependent size) and
        // uploaded.
        at::Tensor tt_nonzero(const at::Tensor& self) {
            return at::nonzero(self.cpu()).to(self.device());
        }

        // isin: any(elements[..., None] == test_elements.flatten()) over the last dim.
        at::Tensor& tt_isin_out(
            const at::Tensor& elements,
            const at::Tensor& test_elements,
            bool assume_unique,
            bool invert,
            at::Tensor& out
        ) {
            auto hits = at::eq(elements.unsqueeze(-1), test_elements.reshape({-1}));
            auto found = at::any(hits, -1);
            return (into(
                "isin.Tensor_Tensor_out", out, invert ? at::logical_not(found) : found
            ));
        }

        // Indices 0..n-1, index i repeated repeats[i] times; host (data-dependent
        // size).
        at::Tensor tt_repeat_interleave(
            const at::Tensor& repeats, std::optional<c10::SymInt> output_size
        ) {
            auto size =
                (output_size ? std::optional<int64_t>(output_size->expect_int())
                             : std::nullopt);
            return at::repeat_interleave(repeats.cpu(), size).to(repeats.device());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("index_select", &tt_index_select);
        m.impl("index_copy_", &tt_index_copy_);
        m.impl("index_copy.out", &tt_index_copy_out);
        m.impl("take", &tt_take);
        m.impl("put_", &tt_put_);
        m.impl("masked_select", &tt_masked_select);
        m.impl("nonzero_static", &tt_nonzero_static);
        m.impl("nonzero", &tt_nonzero);
        m.impl("isin.Tensor_Tensor_out", &tt_isin_out);
        m.impl("repeat_interleave.Tensor", &tt_repeat_interleave);
    }

}  // namespace ktt
