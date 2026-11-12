// torch.ops.ktt._debug_*: internal hooks for testing the Nhwc storage format.
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // Stores the storage of `self` (whole, contiguous, rank 4, bf16 / fp32,
        // replicated) as Nhwc in place; the values are unchanged.
        void debug_to_nhwc(const at::Tensor& self) {
            const char* op = "ktt::_debug_to_nhwc";
            KTT_REQUIRE(
                self.is_privateuseone() && self.dim() == 4 && self.numel() > 0,
                op,
                "needs a non-empty rank-4 tt tensor"
            );
            KTT_REQUIRE(
                self.scalar_type() == at::kBFloat16 || self.scalar_type() == at::kFloat,
                op,
                "needs bfloat16 or float32"
            );
            KTT_REQUIRE(
                self.is_contiguous() && self.storage_offset() == 0 &&
                    self.numel() * self.element_size() ==
                        static_cast<int64_t>(self.storage().nbytes()),
                op,
                "needs a tensor covering its whole storage"
            );
            KTT_REQUIRE(!bridge::traces_alive(), op, "not while traces are alive");
            auto h = dense(op, self);
            auto& st = storage_of(self);
            KTT_REQUIRE(
                bridge::placement(st.base).is_replicated(),
                op,
                "needs a replicated tensor"
            );
            st.heads_of.reset();
            st.heads.clear();
            st.major_of.reset();
            st.major.reset();
            st.base = bridge::to_nhwc(h);
            st.sizes = to_vector(self.sizes());
        }

        // "nchw", "nhwc", "flat", "head_major" or "fold": the format of `self`'s
        // storage; "none" before its first use.
        std::string debug_layout(const at::Tensor& self) {
            KTT_REQUIRE(
                self.is_privateuseone(), "ktt::_debug_layout", "needs a tt tensor"
            );
            const auto& st = storage_of(self);
            if (!st.base) {
                return "none";
            }
            switch (bridge::format(st.base)) {
                case bridge::Format::Nhwc:
                    return "nhwc";
                case bridge::Format::Flat:
                    return "flat";
                case bridge::Format::HeadMajor:
                    return "head_major";
                case bridge::Format::Fold:
                    return "fold";
                default:
                    return "nchw";
            }
        }

        int64_t debug_layout_conversions() {
            return bridge::layout_conversions();
        }

        // Channels-last convs run since load (own-program launches count once they ran:
        // synchronize first).
        int64_t debug_nhwc_conv_runs() {
            return bridge::nhwc_conv_runs();
        }

        // Channels-last convs sent to conv2d()'s route since load (synchronize first).
        int64_t debug_nhwc_conv_fallbacks() {
            return bridge::nhwc_conv_fallbacks();
        }

        // Inputs the channels-last conv moved from L1 to DRAM since load (synchronize
        // first).
        int64_t debug_nhwc_conv_input_moves() {
            return bridge::nhwc_conv_input_moves();
        }

        // TT-NN conv route (synchronize first): {host weight preparations, device
        // weight preparations, geometries verified for the device build, geometries
        // rejected}.
        std::vector<int64_t> debug_conv_weight_preps() {
            return bridge::conv_weight_preps();
        }

        int64_t debug_layout_expansions() {
            return bridge::layout_expansions();
        }

        // Conv weight `self` [O, C, KH, KW] as the [rows, O] matrix a conv reads:
        // tap-major, or Cin-block-major when `blocked` (bridge::tap_major).
        at::Tensor debug_tap_major(const at::Tensor& self, bool blocked) {
            auto w = dense_any_layout("ktt::_debug_tap_major", self);
            auto h = bridge::tap_major(w, blocked);
            const auto sizes = bridge::device_sizes(h);
            return wrap_output(h, sizes, self.scalar_type());
        }

        // With go > 0 the channels-last conv's own program takes only the plan (go,
        // bm_b, npb, bn_b), when it fits; go 0: the planner's choice. `iters`: passes
        // of its kernels' loops per launch (1 unless timing). Synchronize first.
        void debug_nhwc_conv_plan(
            int64_t go, int64_t bm_b, int64_t npb, int64_t bn_b, int64_t iters
        ) {
            const char* op = "ktt::_debug_nhwc_conv_plan";
            KTT_REQUIRE(
                go >= 0 && bm_b >= 0 && npb >= 0 && bn_b >= 0 && iters >= 1,
                op,
                "needs non-negative block sizes and iters >= 1"
            );
            bridge::nhwc_conv_debug_plan(go, bm_b, npb, bn_b, iters);
        }

        // The plan of the last own-program launch of the channels-last conv
        // (synchronize first).
        std::string debug_nhwc_conv_last_plan() {
            return bridge::nhwc_conv_last_plan();
        }

        // Launches of the channels-last conv's weight-gradient program since load
        // (synchronize first).
        int64_t debug_nhwc_wgrad_runs() {
            return bridge::nhwc_wgrad_runs();
        }

        // Launches of the X^T program since load: weight gradients whose forward kept
        // no X^T (synchronize first).
        int64_t debug_nhwc_xt_rotates() {
            return bridge::nhwc_xt_rotates();
        }

        // With go > 0 the channels-last conv's weight-gradient program takes only the
        // plan (go, bc, kb, bo), when it fits; go 0: the planner's choice. `iters`:
        // passes of its kernels' loops per launch (1 unless timing). Synchronize first.
        void debug_nhwc_wgrad_plan(
            int64_t go, int64_t bc, int64_t kb, int64_t bo, int64_t iters
        ) {
            const char* op = "ktt::_debug_nhwc_wgrad_plan";
            KTT_REQUIRE(
                go >= 0 && bc >= 0 && kb >= 0 && bo >= 0 && iters >= 1,
                op,
                "needs non-negative block sizes and iters >= 1"
            );
            bridge::nhwc_wgrad_debug_plan(go, bc, kb, bo, iters);
        }

        // The plan of the last launch of the channels-last conv's weight-gradient
        // program (synchronize first).
        std::string debug_nhwc_wgrad_last_plan() {
            return bridge::nhwc_wgrad_last_plan();
        }

        // The rows the channels-last conv's weight-gradient program writes for
        // grad_output [N, O, H, W], input [N, C, H, W] and a K x K kernel: [Ct K K 32,
        // Nt 32], row (cb K K + t) 32 + c and column o holding dW[o, 32 cb + c, t].
        at::Tensor debug_nhwc_wgrad_rows(
            const at::Tensor& grad_output, const at::Tensor& input, int64_t k
        ) {
            const char* op = "ktt::_debug_nhwc_wgrad_rows";
            KTT_REQUIRE(
                grad_output.is_privateuseone() && input.is_privateuseone() &&
                    grad_output.dim() == 4 && input.dim() == 4,
                op,
                "needs rank-4 tt tensors"
            );
            const int64_t o = grad_output.size(1), c = input.size(1);
            auto out = bridge::conv2d_nhwc_weight_grad(
                dense_any_layout(op, grad_output),
                dense_any_layout(op, input),
                {o, c, k, k},
                false  // the program's own rows
            );
            KTT_REQUIRE(
                out != nullptr, op, "the weight-gradient route does not take these"
            );
            return wrap_output(
                out,
                std::vector<int64_t>{(c + 31) / 32 * k * k * 32, (o + 31) / 32 * 32},
                at::kBFloat16
            );
        }

        // `self` [O, taps C] (column t C + c) as a Flat tensor of `sizes` [O, C, ...]
        // (bridge::flat_from_taps).
        at::Tensor debug_flat_from_taps(const at::Tensor& self, at::IntArrayRef sizes) {
            auto h = bridge::flat_from_taps(
                dense("ktt::_debug_flat_from_taps", self),
                bridge::device_sizes_of(to_vector(sizes))
            );
            return wrap_output(h, sizes, self.scalar_type());
        }

        // The largest |value| in the tile padding of `self`'s Nhwc or Nchw storage.
        double debug_padding_max(const at::Tensor& self) {
            const char* op = "ktt::_debug_padding_max";
            auto h = dense_any_layout(op, self);
            KTT_REQUIRE(
                bridge::format(h) == bridge::Format::Nhwc ||
                    bridge::format(h) == bridge::Format::Nchw,
                op,
                "needs a whole Nhwc or Nchw storage"
            );
            return bridge::padding_max(h);
        }

        // A new tensor of `self`'s values whose storage is `self`'s (whole) storage
        // converted to the other format now, uncached.
        at::Tensor debug_convert(const at::Tensor& self) {
            const char* op = "ktt::_debug_convert";
            auto h = dense_any_layout(op, self);
            const auto to = bridge::format(h) != bridge::Format::Nchw
                ? bridge::Format::Nchw
                : bridge::Format::Nhwc;
            return wrap_output(
                bridge::convert_layout(h, to), self.sizes(), self.scalar_type()
            );
        }

    }  // namespace

    TORCH_LIBRARY_FRAGMENT(ktt, m) {
        m.def("_debug_to_nhwc(Tensor self) -> ()", &debug_to_nhwc);
        m.def("_debug_layout(Tensor self) -> str", &debug_layout);
        m.def("_debug_layout_conversions() -> int", &debug_layout_conversions);
        m.def("_debug_padding_max(Tensor self) -> float", &debug_padding_max);
        m.def("_debug_convert(Tensor self) -> Tensor", &debug_convert);
        m.def("_debug_nhwc_conv_runs() -> int", &debug_nhwc_conv_runs);
        m.def("_debug_nhwc_conv_fallbacks() -> int", &debug_nhwc_conv_fallbacks);
        m.def("_debug_nhwc_conv_input_moves() -> int", &debug_nhwc_conv_input_moves);
        m.def("_debug_layout_expansions() -> int", &debug_layout_expansions);
        m.def("_debug_conv_weight_preps() -> int[]", &debug_conv_weight_preps);
        m.def(
            "_debug_tap_major(Tensor self, bool blocked) -> Tensor", &debug_tap_major
        );
        m.def(
            "_debug_flat_from_taps(Tensor self, int[] sizes) -> Tensor",
            &debug_flat_from_taps
        );
        m.def(
            "_debug_nhwc_conv_plan(int go, int bm_b, int npb, int bn_b, int iters) -> "
            "()",
            &debug_nhwc_conv_plan
        );
        m.def("_debug_nhwc_conv_last_plan() -> str", &debug_nhwc_conv_last_plan);
        m.def("_debug_nhwc_wgrad_runs() -> int", &debug_nhwc_wgrad_runs);
        m.def("_debug_nhwc_xt_rotates() -> int", &debug_nhwc_xt_rotates);
        m.def(
            "_debug_nhwc_wgrad_plan(int go, int bc, int kb, int bo, int iters) -> ()",
            &debug_nhwc_wgrad_plan
        );
        m.def("_debug_nhwc_wgrad_last_plan() -> str", &debug_nhwc_wgrad_last_plan);
        m.def(
            "_debug_nhwc_wgrad_rows(Tensor grad_output, Tensor input, int k) -> Tensor",
            &debug_nhwc_wgrad_rows
        );
    }

}  // namespace ktt
