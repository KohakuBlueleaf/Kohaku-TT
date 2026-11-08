// TT block-float storage and scaled matmul: ktt::block_float_ stores a bf16 tensor as
// BFLOAT8_B / BFLOAT4_B, aten::_scaled_mm_v2 (F.scaled_mm) multiplies with it as
// stored.
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // torch.nn.functional.ScalingType / SwizzleType values.
        constexpr int64_t kTensorWise = 0;
        constexpr int64_t kRowWise = 1;
        constexpr int64_t kNoSwizzle = 0;

        bridge::DType block_dtype(const std::string& format) {
            if (format == "bfp8") {
                return bridge::DType::BFloat8B;
            }
            if (format == "bfp4") {
                return bridge::DType::BFloat4B;
            }
            unsupported(
                "ktt::block_float_",
                "format must be \"bfp8\" or \"bfp4\", got " + format
            );
        }

        // Stores bf16 tt tensor `self` (a whole storage) as block floats in place; it
        // still reads as bf16 (the rounded values).
        at::Tensor& block_float_(at::Tensor& self, c10::string_view format) {
            const char* op = "ktt::block_float_";
            KTT_REQUIRE(self.is_privateuseone(), op, "expected a tt tensor");
            KTT_REQUIRE(self.scalar_type() == at::kBFloat16, op, "needs a bf16 tensor");
            KTT_REQUIRE(
                self.dim() >= 1 && self.numel() > 0, op, "needs a non-empty tensor"
            );
            KTT_REQUIRE(
                self.storage_offset() == 0 && self.is_contiguous() &&
                    self.numel() * self.element_size() ==
                        static_cast<int64_t>(self.storage().nbytes()),
                op,
                "needs a tensor covering its whole storage"
            );
            KTT_REQUIRE(!bridge::capturing(), op, "not inside a trace capture");
            const auto dtype = block_dtype(std::string(format));
            auto h = dense(op, self);
            auto& st = storage_of(self);
            st.heads_of.reset();
            st.heads.clear();
            st.major_of.reset();
            st.major.reset();
            st.base = bridge::typecast(h, dtype);
            st.sizes = to_vector(self.sizes());
            return self;
        }

        // "bfp8" / "bfp4" for block-float storage, else the storage's dtype name.
        std::string storage_format(const at::Tensor& self) {
            KTT_REQUIRE(
                self.is_privateuseone(), "ktt::storage_format", "expected a tt tensor"
            );
            auto& st = storage_of(self);
            if (st.base && bridge::dtype(st.base) == bridge::DType::BFloat8B) {
                return "bfp8";
            }
            if (st.base && bridge::dtype(st.base) == bridge::DType::BFloat4B) {
                return "bfp4";
            }
            return c10::toString(st.dtype);
        }

        // One scale of `recipe` for an operand with `rows` (scale_a: M) or `cols`
        // (scale_b: N): the fp32 factor to broadcast over the [M, N] result, undefined
        // when it is a host 1.
        at::Tensor scale_factor(
            const char* op,
            const at::Tensor& scale,
            int64_t recipe,
            int64_t n,
            bool is_a
        ) {
            KTT_REQUIRE(
                recipe == kTensorWise || recipe == kRowWise,
                op,
                "scale recipes TensorWise / RowWise are supported (block scales live "
                "in "
                "block-float storage, see kohakutt.block_float_), got ",
                recipe
            );
            if (recipe == kTensorWise) {
                KTT_REQUIRE(
                    scale.numel() == 1, op, "a TensorWise scale has one element"
                );
                if (!scale.is_privateuseone() && scale.item<double>() == 1.0) {
                    return {};
                }
                return scale.reshape({}).to(at::kFloat);
            }
            KTT_REQUIRE(scale.numel() == n, op, "a RowWise scale has ", n, " elements");
            auto f = scale.to(at::kFloat);
            return is_a ? f.reshape({n, 1}) : f.reshape({1, n});
        }

        // mat_a [M, K] @ mat_b [K, N] * fp32 scales (TensorWise / RowWise) + bias, in
        // out_dtype (bf16 default); block-float operands as stored; use_fast_accum runs
        // LoFi.
        at::Tensor scaled_mm_v2(
            const at::Tensor& self,
            const at::Tensor& mat2,
            const at::ITensorListRef& scale_a,
            at::IntArrayRef recipe_a,
            at::IntArrayRef swizzle_a,
            const at::ITensorListRef& scale_b,
            at::IntArrayRef recipe_b,
            at::IntArrayRef swizzle_b,
            const std::optional<at::Tensor>& bias,
            std::optional<at::ScalarType> out_dtype,
            at::IntArrayRef contraction_dim,
            bool use_fast_accum
        ) {
            const char* op = "_scaled_mm_v2";
            KTT_REQUIRE(self.dim() == 2 && mat2.dim() == 2, op, "needs 2-D operands");
            KTT_REQUIRE(self.size(1) == mat2.size(0), op, "inner sizes differ");
            KTT_REQUIRE(
                contraction_dim.empty() ||
                    (contraction_dim.size() == 2 &&
                     (contraction_dim[0] == 1 || contraction_dim[0] == -1) &&
                     (contraction_dim[1] == 0 || contraction_dim[1] == -2)),
                op,
                "contraction_dim must be empty or (1, 0)"
            );
            for (auto s : {swizzle_a, swizzle_b}) {
                for (auto v : s) {
                    KTT_REQUIRE(
                        v == kNoSwizzle, op, "scale swizzles are not supported"
                    );
                }
            }
            KTT_REQUIRE(
                scale_a.size() == 1 && recipe_a.size() == 1 && scale_b.size() == 1 &&
                    recipe_b.size() == 1,
                op,
                "one scale and recipe per operand"
            );
            for (const auto* t : {&self, &mat2}) {
                KTT_REQUIRE(
                    t->scalar_type() == at::kBFloat16 || t->scalar_type() == at::kFloat,
                    op,
                    "operands are bf16 / fp32 (block floats via "
                    "kohakutt.block_float_), got ",
                    t->scalar_type()
                );
            }
            const int64_t m = self.size(0), n = mat2.size(1);
            const auto dtype = out_dtype.value_or(at::kBFloat16);
            auto fa = scale_factor(op, *scale_a.begin(), recipe_a[0], m, true);
            auto fb = scale_factor(op, *scale_b.begin(), recipe_b[0], n, false);
            const bool has_bias = bias && bias->defined();

            // Operands as stored: block floats and transposes of contiguous tensors
            // pass as is.
            auto operand = [&](const at::Tensor& t, bool& transposed) {
                if (auto block = block_base(t, transposed)) {
                    return block;
                }
                transposed =
                    (t.size(0) > 1 && t.size(1) > 1 && t.stride(0) == 1 &&
                     t.stride(1) == t.size(0));
                return dense(op, transposed ? t.t() : t);
            };
            bool ta = false, tb = false;
            auto a = operand(self, ta);
            auto b = operand(mat2, tb);

            // Unscaled, no bias: one matmul in the output dtype; else fp32, then the
            // epilogue.
            if (!fa.defined() && !fb.defined() && !has_bias) {
                return wrap_output(
                    bridge::mm(a, b, ta, tb, to_bridge(dtype), use_fast_accum),
                    {m, n},
                    dtype
                );
            }
            auto out = wrap_output(
                bridge::mm(a, b, ta, tb, bridge::DType::Float32, use_fast_accum),
                {m, n},
                at::kFloat
            );
            if (fa.defined()) {
                out = out * fa;
            }
            if (fb.defined()) {
                out = out * fb;
            }
            if (has_bias) {
                out = out + bias->to(at::kFloat);
            }
            return out.to(dtype);
        }

    }  // namespace

    TORCH_LIBRARY_FRAGMENT(ktt, m) {
        m.def("block_float_(Tensor(a!) self, str format) -> Tensor(a!)", &block_float_);
        m.def("storage_format(Tensor self) -> str", &storage_format);
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("_scaled_mm_v2", &scaled_mm_v2);
    }

}  // namespace ktt
