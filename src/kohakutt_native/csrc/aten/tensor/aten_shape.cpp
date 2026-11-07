// aten shape kernels for tt: cat, repeat, flip, roll, tril / triu, unfold (view),
// channel_shuffle.

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"
#include "core/torch/constants.hpp"

namespace ktt {
    namespace {

        // Concatenation along `dim`; empty (numel 0, 1-D) inputs are skipped as in
        // PyTorch.
        at::Tensor tt_cat(const at::ITensorListRef& tensors, int64_t dim) {
            std::vector<at::Tensor> parts;
            for (const auto& t : tensors) {
                if (!(t.dim() == 1 && t.numel() == 0)) {
                    parts.push_back(t);
                }
            }
            KTT_REQUIRE(!parts.empty(), "cat", "all inputs are empty");
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
            std::vector<bridge::HandlePtr> handles;
            auto sizes = to_vector(parts[0].sizes());
            sizes[dim] = 0;
            for (const auto& t : parts) {
                handles.push_back(dense_as("cat", t, dtype));
                sizes[dim] += t.size(dim);
            }
            auto pad = std::max<int64_t>(0, 2 - rank);
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
