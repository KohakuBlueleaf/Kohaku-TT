// aten shape kernels for tt: cat, repeat, flip, roll, tril / triu, unfold (view),
// channel_shuffle, diagonal_scatter, diag_embed.

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"
#include "core/torch/mesh_rules.hpp"
#include "core/torch/constants.hpp"

namespace ktt {
    namespace {

        // self with its (offset, dim1, dim2) diagonal replaced by src (shape of
        // self.diagonal(offset, dim1, dim2)).
        at::Tensor tt_diagonal_scatter(
            const at::Tensor& self,
            const at::Tensor& src,
            int64_t offset,
            int64_t dim1,
            int64_t dim2
        ) {
            const char* op = "diagonal_scatter";
            dim1 = c10::maybe_wrap_dim(dim1, self.dim());
            dim2 = c10::maybe_wrap_dim(dim2, self.dim());
            KTT_REQUIRE(dim1 != dim2, op, "dim1 and dim2 must differ");
            auto x = self.movedim({dim1, dim2}, {-2, -1});
            const int64_t m = x.size(-2), n = x.size(-1);
            const int64_t length = src.size(-1);
            if (length == 0 || self.numel() == 0) {
                return self.clone();
            }
            auto opts = self.options().dtype(at::kInt);
            auto rows = at::arange(m, opts).reshape({m, 1});
            auto cols = at::arange(n, opts).reshape({1, n});
            auto mask = at::eq(cols - rows, offset);
            auto s = src.to(self.scalar_type());
            // Diagonal element k at row k (offset >= 0) or column k (offset < 0).
            auto values = offset >= 0
                ? at::constant_pad_nd(s, {0, m - length}).unsqueeze(-1)
                : at::constant_pad_nd(s, {0, n - length}).unsqueeze(-2);
            auto out = at::where(mask, values, x);
            return out.movedim({-2, -1}, {dim1, dim2}).contiguous();
        }

        // Tensor with self's last dim on its (offset, dim1, dim2) diagonal, zeros
        // elsewhere.
        at::Tensor tt_diag_embed(
            const at::Tensor& self, int64_t offset, int64_t dim1, int64_t dim2
        ) {
            const int64_t rank = self.dim() + 1;
            dim1 = c10::maybe_wrap_dim(dim1, rank);
            dim2 = c10::maybe_wrap_dim(dim2, rank);
            const int64_t size = self.size(-1) + std::abs(offset);
            std::vector<int64_t> sizes;
            int64_t k = 0;
            for (int64_t d = 0; d < rank; ++d) {
                sizes.push_back(d == dim1 || d == dim2 ? size : self.size(k++));
            }
            auto zeros = at::zeros(sizes, self.options());
            return tt_diagonal_scatter(zeros, self, offset, dim1, dim2);
        }

        // 2-D tensor with self's elements (flattened) on its `offset` diagonal.
        at::Tensor tt_diagflat(const at::Tensor& self, int64_t offset) {
            return tt_diag_embed(self.reshape({-1}), offset, 0, 1);
        }

        // 1-D: a 2-D tensor with self on its `diagonal`; 2-D: that diagonal of self.
        at::Tensor tt_diag(const at::Tensor& self, int64_t diagonal) {
            KTT_REQUIRE(self.dim() == 1 || self.dim() == 2, "diag", "needs 1-D or 2-D");
            if (self.dim() == 1) {
                return tt_diag_embed(self, diagonal, 0, 1);
            }
            return self.diagonal(diagonal).clone();
        }

        at::Tensor& tt_diag_out(
            const at::Tensor& self, int64_t diagonal, at::Tensor& out
        ) {
            return into("diag.out", out, tt_diag(self, diagonal));
        }

        // Concatenation along `dim`; empty (numel 0, 1-D) inputs are skipped as in
        // PyTorch.
        at::Tensor tt_cat(const at::ITensorListRef& tensors, int64_t dim) {
            std::vector<at::Tensor> parts;
            for (const auto& t : tensors) {
                if (!(t.dim() == 1 && t.numel() == 0)) {
                    parts.push_back(t);
                }
            }
            if (parts.empty()) {
                KTT_REQUIRE(tensors.size() > 0, "cat", "no inputs");
                auto dtype = tensors.front().scalar_type();
                for (const auto& t : tensors) {
                    dtype = at::promote_types(dtype, t.scalar_type());
                }
                return at::empty({0}, tensors.front().options().dtype(dtype));
            }
            const int64_t rank = parts[0].dim();
            KTT_REQUIRE(rank >= 1, "cat", "0-d inputs cannot be concatenated");
            dim = c10::maybe_wrap_dim(dim, rank);

            // Complex inputs: the fp32 pairs concatenate along the same dim.
            if (std::any_of(parts.begin(), parts.end(), [](const auto& t) {
                    return holds_complex(t);
                })) {
                std::vector<at::Tensor> pairs;
                for (const auto& t : parts) {
                    pairs.push_back(as_real(t.to(at::kComplexFloat)));
                }
                return as_complex(at::cat(pairs, dim));
            }

            auto dtype = parts[0].scalar_type();
            for (const auto& t : parts) {
                dtype = at::promote_types(dtype, t.scalar_type());
            }
            auto pad = std::max<int64_t>(0, 2 - rank);

            // A mesh-placed part: concatenated per chip (the cat dim whole on each
            // chip); a single placed part is copied.
            if (parts.size() == 1 && mesh::placed_base(parts[0])) {
                return parts[0].to(dtype, false, true, at::MemoryFormat::Contiguous);
            }
            {
                std::vector<bridge::HandlePtr> inputs;
                bool any = false;
                for (const auto& t : parts) {
                    auto p = mesh::placed_base(t);
                    any = any || p;
                    inputs.push_back(p);
                }
                if (any && parts.size() > 1) {
                    auto out_sizes = to_vector(parts[0].sizes());
                    out_sizes[dim] = 0;
                    for (size_t i = 0; i < parts.size(); ++i) {
                        if (!inputs[i]) {
                            inputs[i] = dense("cat", parts[i]);
                        }
                        out_sizes[dim] += parts[i].size(dim);
                    }
                    const auto dims = bridge::device_sizes_of(out_sizes);
                    auto f = (mesh::elementwise(
                        static_cast<int>(parts.size()), dims.size()
                    ));
                    f.kinds[dim + pad] = mesh::FactorKind::Replicate;
                    f.partial = mesh::Factors::Partial::All;
                    const auto to = to_bridge(dtype);
                    auto out = mesh::run_placed(
                        "cat", f, inputs, dims, [&](const auto& in) {
                            std::vector<bridge::HandlePtr> cast;
                            for (const auto& h : in) {
                                cast.push_back(
                                    bridge::dtype(h) == to ? h : bridge::typecast(h, to)
                                );
                            }
                            return bridge::concat(cast, dim + pad);
                        }
                    );
                    if (out) {
                        return wrap_output(*out, out_sizes, dtype);
                    }
                }
            }
            std::vector<bridge::HandlePtr> handles;
            auto sizes = to_vector(parts[0].sizes());
            sizes[dim] = 0;
            for (const auto& t : parts) {
                sizes[dim] += t.size(dim);
            }

            // Head-major parts: concatenated as stored, the result head-major.
            if (parts.size() > 1 && rank >= 4) {
                const auto cast = to_bridge(dtype);
                for (const auto& t : parts) {
                    auto h = head_stored(t);
                    if (!h) {
                        break;
                    }
                    h = bridge::dtype(h) == cast ? h : bridge::typecast(h, cast);
                    handles.push_back(bridge::as_stored(h));
                }
                if (handles.size() == parts.size()) {
                    const int64_t stored_dim = dim == rank - 3 ? rank - 2
                        : dim == rank - 2                      ? rank - 3
                                                               : dim;
                    auto h = bridge::view_as(
                        bridge::concat(handles, stored_dim),
                        sizes,
                        bridge::Format::HeadMajor
                    );
                    return wrap_output(h, sizes, dtype);
                }
                handles.clear();
            }
            for (const auto& t : parts) {
                handles.push_back(dense_as("cat", t, dtype));
            }
            auto h =
                (handles.size() == 1 ? handles[0] : bridge::concat(handles, dim + pad));
            return wrap_output(h, sizes, dtype);
        }

        // self reversed along `dims`: index_select with a descending device arange per
        // dim.
        at::Tensor tt_flip(const at::Tensor& self, at::IntArrayRef dims) {
            if (self.numel() == 0) {
                return at::empty(self.sizes(), self.options());
            }
            auto out = self;
            for (auto d : dims) {
                d = c10::maybe_wrap_dim(d, std::max<int64_t>(self.dim(), 1));
                const int64_t n = self.dim() == 0 ? 1 : self.size(d);
                if (n <= 1) {
                    continue;
                }
                auto index = at::arange(n - 1, -1, -1, self.options().dtype(at::kLong));
                out = at::index_select(out, d, index);
            }
            return out.is_same(self) ? self.clone() : out;
        }

        // self shifted cyclically by `shifts` along `dims` (flattened when dims is
        // empty).
        at::Tensor tt_roll(
            const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims
        ) {
            if (self.numel() == 0) {
                return self.clone();
            }
            if (dims.empty()) {
                KTT_REQUIRE(shifts.size() == 1, "roll", "one shift without dims");
                return tt_roll(self.flatten(), shifts, {0}).reshape(self.sizes());
            }
            KTT_REQUIRE(shifts.size() == dims.size(), "roll", "shifts / dims length");
            auto out = self;
            for (size_t k = 0; k < dims.size(); ++k) {
                const int64_t d = c10::maybe_wrap_dim(dims[k], self.dim());
                const int64_t n = self.size(d);
                const int64_t s = ((shifts[k] % n) + n) % n;
                if (s == 0) {
                    continue;
                }
                out = at::cat({out.narrow(d, n - s, s), out.narrow(d, 0, n - s)}, d);
            }
            return out.is_same(self) ? self.clone() : out;
        }

        // [N, C, ...] -> [N, g, C / g, ...] -> swap the group dims -> [N, C, ...].
        at::Tensor tt_channel_shuffle(const at::Tensor& self, c10::SymInt groups) {
            const char* op = "channel_shuffle";
            const int64_t g = groups.expect_int();
            KTT_REQUIRE(
                self.dim() >= 3 && g > 0 && self.size(1) % g == 0, op, "bad groups"
            );
            auto shape = self.sizes().vec();
            std::vector<int64_t> split{shape[0], g, shape[1] / g};
            split.insert(split.end(), shape.begin() + 2, shape.end());
            return self.reshape(split).transpose(1, 2).contiguous().reshape(shape);
        }

        // self tiled `repeats` times per dim (leading new dims allowed).
        at::Tensor tt_repeat(const at::Tensor& self, c10::IntArrayRef repeats) {
            KTT_REQUIRE(
                static_cast<int64_t>(repeats.size()) >= self.dim(),
                "repeat",
                "too few repeats"
            );
            std::vector<int64_t> sizes(repeats.size() - self.dim(), 1);
            sizes.insert(sizes.end(), self.sizes().begin(), self.sizes().end());
            auto base = self.reshape(sizes);
            std::vector<int64_t> out_sizes(sizes.size());
            for (size_t i = 0; i < sizes.size(); ++i) {
                out_sizes[i] = sizes[i] * repeats[i];
            }
            if (base.numel() == 0) {
                return at::empty(out_sizes, self.options());
            }
            std::vector<int64_t> device_repeats(repeats.vec());
            while (device_repeats.size() < 2) {
                device_repeats.insert(device_repeats.begin(), 1);
            }
            auto h = bridge::repeat(dense("repeat", base), device_repeats);
            return wrap_output(h, out_sizes, self.scalar_type());
        }

        // where(mask, self, 0) with a cached bool triangle mask [rows, cols] (lower:
        // col - row
        // <= diagonal; upper: col - row >= diagonal).
        at::Tensor triangle(const at::Tensor& self, int64_t diagonal, bool lower) {
            KTT_REQUIRE(self.dim() >= 2, lower ? "tril" : "triu", "needs >= 2-D input");
            const int64_t rows = self.size(-2), cols = self.size(-1);
            auto key = c10::str("tri|", lower, "|", rows, "|", cols, "|", diagonal);
            auto keep = device_constant(key, [&] {
                auto r = at::arange(rows).reshape({rows, 1});
                auto c = at::arange(cols).reshape({1, cols});
                return lower ? at::le(c - r, diagonal) : at::ge(c - r, diagonal);
            });
            // where, not a multiply: masked-out infinities / NaNs become exact zeros.
            return at::where(keep, self, at::zeros({}, self.options()));
        }

        at::Tensor tt_tril(const at::Tensor& self, int64_t diagonal) {
            const bool lower = true;
            return triangle(self, diagonal, lower);
        }

        at::Tensor tt_triu(const at::Tensor& self, int64_t diagonal) {
            const bool lower = false;
            return triangle(self, diagonal, lower);
        }

        at::Tensor& tt_triu_out(
            const at::Tensor& self, int64_t diagonal, at::Tensor& out
        ) {
            return into("triu.out", out, tt_triu(self, diagonal));
        }

        at::Tensor& tt_tril_out(
            const at::Tensor& self, int64_t diagonal, at::Tensor& out
        ) {
            return into("tril.out", out, tt_tril(self, diagonal));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("cat", &tt_cat);
        m.impl("diagonal_scatter", &tt_diagonal_scatter);
        m.impl("diag_embed", &tt_diag_embed);
        m.impl("diagflat", &tt_diagflat);
        m.impl("diag", &tt_diag);
        m.impl("diag.out", &tt_diag_out);
        m.impl("flip", &tt_flip);
        m.impl("roll", &tt_roll);
        m.impl("channel_shuffle", &tt_channel_shuffle);
        m.impl("repeat", &tt_repeat);
        m.impl("tril", &tt_tril);
        m.impl("triu", &tt_triu);
        m.impl("triu.out", &tt_triu_out);
        m.impl("tril.out", &tt_tril_out);
        m.impl("unfold", &at::native::unfold);
    }

}  // namespace ktt
