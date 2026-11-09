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
                bridge::placement(st.base).is_replicated(), op, "needs a replicated tensor"
            );
            st.heads_of.reset();
            st.heads.clear();
            st.major_of.reset();
            st.major.reset();
            st.base = bridge::to_nhwc(h);
            st.sizes = to_vector(self.sizes());
        }

        // "nhwc" or "nchw": the format of `self`'s storage; "none" before its first use.
        std::string debug_layout(const at::Tensor& self) {
            KTT_REQUIRE(self.is_privateuseone(), "ktt::_debug_layout", "needs a tt tensor");
            const auto& st = storage_of(self);
            if (!st.base) {
                return "none";
            }
            return bridge::format(st.base) == bridge::Format::Nhwc ? "nhwc" : "nchw";
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

        // The largest |value| in the tile padding of `self`'s Nhwc storage.
        double debug_padding_max(const at::Tensor& self) {
            const char* op = "ktt::_debug_padding_max";
            auto h = dense_any_layout(op, self);
            KTT_REQUIRE(
                bridge::format(h) == bridge::Format::Nhwc,
                op,
                "needs a whole Nhwc storage"
            );
            return bridge::padding_max(h);
        }

        // A new tensor of `self`'s values whose storage is `self`'s (whole) storage
        // converted to the other format now, uncached.
        at::Tensor debug_convert(const at::Tensor& self) {
            const char* op = "ktt::_debug_convert";
            auto h = dense_any_layout(op, self);
            const auto to = bridge::format(h) == bridge::Format::Nhwc
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
    }

}  // namespace ktt
