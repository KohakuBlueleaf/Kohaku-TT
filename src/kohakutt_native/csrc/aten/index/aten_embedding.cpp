// aten embedding kernels for tt: embedding backward, embedding_renorm_, embedding_bag
// (sum / mean, + backward); rows by index_select, sums by scatter_add.
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        constexpr int64_t kSum = 0, kMean = 1;

        // Bag id of each of the L indices: running count of bag starts
        // (offsets[1:bags]).
        at::Tensor offset_to_bag(
            const at::Tensor& offsets, int64_t length, int64_t bags
        ) {
            auto marks = at::zeros({length}, offsets.options().dtype(at::kFloat));
            if (bags > 1) {
                auto starts = offsets.narrow(0, 1, bags - 1);
                auto inside = at::lt(starts, length);
                auto ones = inside.to(at::kFloat);
                auto where = at::clamp(starts, 0, length - 1);
                marks = at::scatter_add(marks, 0, where, ones);
            }
            return at::cumsum(marks, 0).to(at::kLong);
        }

        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tt_embedding_bag(
            const at::Tensor& weight,
            const at::Tensor& indices,
            const at::Tensor& offsets,
            bool scale_grad_by_freq,
            int64_t mode,
            bool sparse,
            const std::optional<at::Tensor>& per_sample_weights,
            bool include_last_offset,
            int64_t padding_idx
        ) {
            const char* op = "_embedding_bag";
            KTT_REQUIRE(
                mode == kSum || mode == kMean, op, "mode 'max' is not supported yet"
            );
            KTT_REQUIRE(indices.dim() == 1, op, "indices must be 1-D (with offsets)");
            const int64_t length = indices.numel(), dim = weight.size(1);
            const int64_t bags = offsets.numel() - (include_last_offset ? 1 : 0);
            auto long_opts = indices.options().dtype(at::kLong);
            if (length == 0 || bags <= 0) {
                const int64_t b = std::max<int64_t>(bags, 0);
                return {
                    at::zeros({b, dim}, weight.options()),
                    at::empty({0}, long_opts),
                    at::zeros({b}, long_opts),
                    at::zeros({b}, long_opts)
                };
            }
            auto bag = offset_to_bag(offsets, length, bags);
            auto rows = at::index_select(weight, 0, indices).to(at::kFloat);
            auto valid = padding_idx >= 0 ? at::ne(indices, padding_idx).to(at::kFloat)
                                          : at::ones({length}, rows.options());
            auto scale = valid;
            if (per_sample_weights && per_sample_weights->defined()) {
                KTT_REQUIRE(mode == kSum, op, "per_sample_weights needs mode 'sum'");
                scale = scale * per_sample_weights->to(at::kFloat);
            }
            auto spread = bag.reshape({length, 1}).expand({length, dim});
            auto zeros = at::zeros({bags, dim}, rows.options());
            auto out = at::scatter_add(
                zeros, 0, spread, rows * scale.reshape({length, 1})
            );
            auto counts = at::scatter_add(
                at::zeros({bags}, rows.options()), 0, bag, valid
            );
            if (mode == kMean) {
                out = out / at::clamp(counts, 1, std::nullopt).reshape({bags, 1});
            }
            return {
                out.to(weight.scalar_type()),
                bag,
                counts.to(at::kLong),
                at::zeros({bags}, long_opts)
            };
        }

        // grad_weight [num_weights, D]: each index's bag gradient (mean: / bag size,
        // sum:
        // * its sample weight) scatter-added at its row; padding rows get none.
        at::Tensor tt_embedding_bag_dense_backward(
            const at::Tensor& grad,
            const at::Tensor& indices,
            const at::Tensor& offset2bag,
            const at::Tensor& bag_size,
            const at::Tensor& maximum_indices,
            int64_t num_weights,
            bool scale_grad_by_freq,
            int64_t mode,
            const std::optional<at::Tensor>& per_sample_weights,
            int64_t padding_idx
        ) {
            const char* op = "_embedding_bag_dense_backward";
            KTT_REQUIRE(
                mode == kSum || mode == kMean, op, "mode 'max' is not supported yet"
            );
            KTT_REQUIRE(
                !scale_grad_by_freq, op, "scale_grad_by_freq is not supported yet"
            );
            const int64_t length = indices.numel(), dim = grad.size(1);
            auto out = at::zeros({num_weights, dim}, grad.options().dtype(at::kFloat));
            if (length == 0) {
                return out.to(grad.scalar_type());
            }
            auto g = at::index_select(grad.to(at::kFloat), 0, offset2bag);  // [L, D]
            auto scale = padding_idx >= 0 ? at::ne(indices, padding_idx).to(at::kFloat)
                                          : at::ones({length}, g.options());
            if (mode == kMean) {
                auto sizes = at::index_select(bag_size, 0, offset2bag).to(at::kFloat);
                scale = scale / at::clamp(sizes, 1, std::nullopt);
            }
            if (per_sample_weights && per_sample_weights->defined()) {
                scale = scale * per_sample_weights->to(at::kFloat);
            }
            auto spread = indices.reshape({length, 1}).expand({length, dim});
            out = at::scatter_add(out, 0, spread, g * scale.reshape({length, 1}));
            return out.to(grad.scalar_type());
        }

        at::Tensor tt_embedding_bag_backward(
            const at::Tensor& grad,
            const at::Tensor& indices,
            const at::Tensor& offsets,
            const at::Tensor& offset2bag,
            const at::Tensor& bag_size,
            const at::Tensor& maximum_indices,
            int64_t num_weights,
            bool scale_grad_by_freq,
            int64_t mode,
            bool sparse,
            const std::optional<at::Tensor>& per_sample_weights,
            int64_t padding_idx
        ) {
            const char* op = "_embedding_bag_backward";
            KTT_REQUIRE(!sparse, op, "sparse gradients are not supported");
            return (tt_embedding_bag_dense_backward(
                grad,
                indices,
                offset2bag,
                bag_size,
                maximum_indices,
                num_weights,
                scale_grad_by_freq,
                mode,
                per_sample_weights,
                padding_idx
            ));
        }

        // d(output) / d(per_sample_weight[l]) = grad[bag(l)] . weight[index(l)] (sum
        // mode).
        at::Tensor tt_embedding_bag_per_sample_weights_backward(
            const at::Tensor& grad,
            const at::Tensor& weight,
            const at::Tensor& indices,
            const at::Tensor& offsets,
            const at::Tensor& offset2bag,
            int64_t mode,
            int64_t padding_idx
        ) {
            const char* op = "_embedding_bag_per_sample_weights_backward";
            KTT_REQUIRE(mode == kSum, op, "per_sample_weights needs mode 'sum'");
            auto g = at::index_select(grad.to(at::kFloat), 0, offset2bag);
            auto rows = at::index_select(weight.to(at::kFloat), 0, indices);
            auto dot = (g * rows).sum(1);
            if (padding_idx >= 0) {
                dot = dot * at::ne(indices, padding_idx).to(at::kFloat);
            }
            return dot.to(grad.scalar_type());
        }

        // Rows of self at `indices` whose p-norm exceeds max_norm rescaled to max_norm
        // (in place; repeated indices write the same row).
        at::Tensor& tt_embedding_renorm_(
            at::Tensor& self,
            const at::Tensor& indices,
            double max_norm,
            double norm_type
        ) {
            if (indices.numel() == 0) {
                return self;
            }
            auto flat = indices.reshape({-1});
            auto rows = at::index_select(self, 0, flat).to(at::kFloat);
            auto a = at::abs(rows);
            auto norm =
                (norm_type == 2.0
                     ? at::sqrt((a * a).sum(1, true))
                     : at::pow(at::pow(a, norm_type).sum(1, true), 1.0 / norm_type));
            auto scale = (at::where(
                at::gt(norm, max_norm), max_norm / (norm + 1e-7), 1.0
            ));
            c10::List<std::optional<at::Tensor>> index;
            index.push_back(flat);
            self.index_put_(index, (rows * scale).to(self.scalar_type()));
            return self;
        }

        // d(weight) of embedding: one-hot(indices)^T @ grad (rows of padding_idx
        // zeroed).
        at::Tensor tt_embedding_backward(
            const at::Tensor& grad,
            const at::Tensor& indices,
            int64_t num_weights,
            int64_t padding_idx,
            bool scale_grad_by_freq
        ) {
            KTT_REQUIRE(
                !scale_grad_by_freq, "embedding_dense_backward", "scale_grad_by_freq"
            );
            // Each token's gradient row scatter-added at its index in an fp32 table.
            const int64_t dim = grad.size(-1);
            auto g = grad.reshape({-1, dim}).to(at::kFloat);
            const int64_t length = g.size(0);
            auto out = at::zeros({num_weights, dim}, g.options());
            if (length == 0) {
                return out.to(grad.scalar_type());
            }
            auto idx = indices.reshape({length});
            if (padding_idx >= 0) {
                g = g * at::ne(idx, padding_idx).to(at::kFloat).reshape({length, 1});
            }
            auto spread = idx.to(at::kLong).reshape({length, 1}).expand({length, dim});
            return at::scatter_add(out, 0, spread, g).to(grad.scalar_type());
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("embedding_dense_backward", &tt_embedding_backward);
        m.impl("embedding_renorm_", &tt_embedding_renorm_);
        m.impl("_embedding_bag", &tt_embedding_bag);
        m.impl("_embedding_bag_forward_only", &tt_embedding_bag);
        m.impl("_embedding_bag_backward", &tt_embedding_bag_backward);
        m.impl("_embedding_bag_dense_backward", &tt_embedding_bag_dense_backward);
        m.impl(
            "_embedding_bag_per_sample_weights_backward",
            &tt_embedding_bag_per_sample_weights_backward
        );
    }

}  // namespace ktt
