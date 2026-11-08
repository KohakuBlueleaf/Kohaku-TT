// aten copy kernels for tt: copy_, _copy_from, _to_copy (CPU <-> tt, tt <-> tt;
// complex64 through its fp32 pairs), item().
#include <torch/library.h>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"

namespace ktt {
    namespace {

        // The elements of tt view `src` as `dtype` in a buffer no other storage shares
        // (dense_as may return src's own base handle, which a trace replay can
        // overwrite).
        bridge::HandlePtr private_copy(
            const std::string& op, const at::Tensor& src, at::ScalarType dtype
        ) {
            auto h = dense_as(op, src, dtype);
            return h == storage_of(src).base ? bridge::clone(h) : h;
        }

        // True for tensors covering their whole contiguous storage.
        bool whole(const at::Tensor& t) {
            return (
                t.is_contiguous() && t.storage_offset() == 0 &&
                t.numel() * t.element_size() ==
                    static_cast<int64_t>(t.storage().nbytes())
            );
        }

        // Most heads merged by one nlp_concat_heads program (its circular buffers grow
        // with the head count).
        constexpr int64_t kMaxMergedHeads = 128;

        // Heads merged as [B, S, H D] when `src` is x.transpose(1, 2) of a whole [B, H,
        // S, D] storage (D tile-aligned, H <= kMaxMergedHeads), in one program; null
        // otherwise.
        bridge::HandlePtr merged_heads(const at::Tensor& src) {
            if (src.dim() != 4 || src.size(3) % 32 != 0 || bridge::traces_alive() ||
                src.size(2) > kMaxMergedHeads ||
                (src.scalar_type() != at::kBFloat16 &&
                 src.scalar_type() != at::kFloat)) {
                return nullptr;
            }
            auto base = src.transpose(1, 2);
            if (!whole(base) || storage_of(base).sizes != to_vector(base.sizes())) {
                return nullptr;
            }
            return bridge::concat_heads(dense("copy_", base));
        }

        // True for a uint8 view of a storage: storage-level byte access
        // (serialization).
        bool is_byte_view(const at::Tensor& t) {
            return t.scalar_type() == at::kByte;
        }

        // A contiguous host tensor (logical dtype) holding the elements of tt view
        // `src`.
        at::Tensor download(const at::Tensor& src, int chip = 0) {
            if (is_byte_view(src)) {
                KTT_REQUIRE(
                    src.is_contiguous(), "copy", "strided byte views are not supported"
                );
                auto bytes = (
                    storage_bytes(storage_of(src), src.storage_offset(), src.numel())
                );
                return bytes.view(src.sizes());
            }
            auto phys = physical(src.scalar_type());
            auto host = at::empty(src.sizes(), at::TensorOptions().dtype(phys));
            // A whole mesh-placed storage is assembled from its shards on the host.
            auto& st = storage_of(src);
            const bool placed = (
                st.base && whole(src) && !bridge::is_block(bridge::dtype(st.base)) &&
                !bridge::placement(st.base).is_replicated()
            );
            bridge::download(placed ? st.base : dense("copy", src), host.data_ptr(), chip);
            return phys == src.scalar_type() ? host : host.to(src.scalar_type());
        }

        // torch.ops.ktt.replica: the copy of tt tensor `self` on mesh chip `chip`, as a
        // CPU tensor.
        at::Tensor replica(const at::Tensor& self, int64_t chip) {
            TORCH_CHECK(self.is_privateuseone(), "ktt::replica: a tt tensor");
            TORCH_CHECK(!is_byte_view(self), "ktt::replica: not for byte views");
            return download(self, static_cast<int>(chip));
        }

        // A device handle holding host tensor `host` stored as logical dtype `dtype`.
        bridge::HandlePtr upload(const at::Tensor& host, at::ScalarType dtype) {
            auto phys = physical(dtype);
            auto values = host.to(dtype);
            if (dtype == at::kLong && values.numel() > 0) {
                KTT_REQUIRE(
                    values.abs().max().item<int64_t>() <=
                        std::numeric_limits<int32_t>::max(),
                    "copy",
                    "int64 values beyond int32 range cannot be stored on tt"
                );
            }
            values = values.to(phys).contiguous();
            return (
                bridge::upload(
                    values.data_ptr(), to_vector(values.sizes()), to_bridge(dtype)
                )
            );
        }

        // copy_ with a complex side, through the fp32 pairs: complex <- real sets im =
        // 0, real <- complex takes the real part (as PyTorch, which warns).
        at::Tensor& copy_complex(at::Tensor& self, const at::Tensor& src) {
            KTT_REQUIRE(!self.is_conj(), "copy_", "writing into a conjugated view");
            if (holds_complex(self) && holds_complex(src)) {
                as_real(self).copy_(as_real(src.to(at::kComplexFloat)));
            } else if (holds_complex(self)) {
                auto pairs = as_real(self);
                pairs.select(-1, 0).copy_(src.expand_as(self));
                pairs.select(-1, 1).fill_(0);
            } else {
                self.copy_(as_real(src.to(at::kComplexFloat)).select(-1, 0));
            }
            return self;
        }

        // copy_ into a tt view writes through commit(); into CPU it downloads.
        at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool) {
            if (holds_complex(self) || holds_complex(src)) {
                return copy_complex(self, src);
            }
            if (self.is_cpu() && src.is_privateuseone()) {
                self.copy_(download(src));
                return self;
            }
            KTT_REQUIRE(
                self.is_privateuseone(), "copy_", "unsupported target ", self.device()
            );
            if (src.is_cpu() && is_byte_view(self)) {
                KTT_REQUIRE(
                    self.storage_offset() == 0 && self.is_contiguous() &&
                        self.numel() == static_cast<int64_t>(self.storage().nbytes()),
                    "copy_",
                    "byte writes must cover the whole storage"
                );
                set_storage_bytes(
                    storage_of(self), src.to(at::kByte).expand_as(self).flatten()
                );
                return self;
            }
            if (src.is_cpu()) {
                commit("copy_", self, upload(src.expand_as(self), self.scalar_type()));
                return self;
            }
            KTT_REQUIRE(
                src.is_privateuseone(), "copy_", "unsupported source ", src.device()
            );

            // A merged-heads copy becomes the storage, laid out as [B, S, H D].
            if (src.scalar_type() == self.scalar_type() && whole(self) &&
                self.sizes() == src.sizes()) {
                if (auto merged = merged_heads(src)) {
                    auto& st = storage_of(self);
                    st.heads_of.reset();
                    st.heads.clear();
                    st.major_of.reset();
                    st.major.reset();
                    st.base = merged;
                    st.sizes = {src.size(0), src.size(1), src.size(2) * src.size(3)};
                    st.dtype = self.scalar_type();
                    return self;
                }
            }

            // Different dtypes: cast on device first (_to_copy), then copy in.
            auto source =
                (src.scalar_type() == self.scalar_type() ? src
                                                         : src.to(self.scalar_type()));
            commit(
                "copy_",
                self,
                private_copy("copy_", source.expand_as(self), source.scalar_type())
            );
            return self;
        }

        // dst.resize_(self.sizes()).copy_(self): writes a CPU fallback result into a tt
        // out.
        at::Tensor copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
            auto out = dst;
            if (out.sizes() != self.sizes()) {
                out.resize_(self.sizes());
            }
            return copy_(out, self, false);
        }

        at::Tensor copy_from(
            const at::Tensor& self, const at::Tensor& dst, bool non_blocking
        ) {
            auto out = dst;
            return copy_(out, self, non_blocking);
        }

        // CPU -> tt: one upload of the contiguous, cast host tensor.
        // tt -> CPU: one download into a contiguous host tensor, then a cast if asked.
        at::Tensor to_copy(
            const at::Tensor& self,
            std::optional<at::ScalarType> dtype,
            std::optional<at::Layout>,
            std::optional<at::Device> device,
            std::optional<bool>,
            bool,
            std::optional<at::MemoryFormat> memory_format
        ) {
            auto target = device.value_or(self.device());
            auto scalar = dtype.value_or(self.scalar_type());

            // Complex on either side: a fresh contiguous tensor filled by copy_complex.
            if (holds_complex(self) || at::isComplexType(scalar)) {
                auto out = (
                    at::empty(self.sizes(), self.options().dtype(scalar).device(target))
                );
                out.copy_(self);
                return out;
            }

            // preserve_format keeps a dense non-contiguous input's strides, as PyTorch
            // does (compiled graphs bake in views that rely on them).
            const auto format = memory_format.value_or(at::MemoryFormat::Preserve);
            if (format == at::MemoryFormat::Preserve && !self.is_contiguous() &&
                self.is_non_overlapping_and_dense()) {
                auto out = (
                    at::empty_strided(
                        self.sizes(),
                        self.strides(),
                        self.options().dtype(scalar).device(target)
                    )
                );
                // Cast first (contiguous) so copy_ never re-enters _to_copy.
                auto source = self;
                if (self.is_privateuseone() && scalar != self.scalar_type()) {
                    source = (
                        wrap_output(
                            private_copy("_to_copy", self, scalar), self.sizes(), scalar
                        )
                    );
                }
                out.copy_(source);
                return out;
            }
            if (self.is_cpu() && target.is_privateuseone()) {
                return make_tt(upload(self, scalar), self.sizes(), scalar);
            }

            // tt -> tt: a device cast (or a copy when the dtype is unchanged).
            if (self.is_privateuseone() && target.is_privateuseone()) {
                return (
                    wrap_output(
                        private_copy("_to_copy", self, scalar), self.sizes(), scalar
                    )
                );
            }
            KTT_REQUIRE(
                self.is_privateuseone() && target.is_cpu(),
                "_to_copy",
                "from ",
                self.device(),
                " to ",
                target,
                " is not supported yet"
            );
            auto host = download(self);
            return scalar != self.scalar_type() ? host.to(scalar) : host;
        }

        // tensor.item(): one download of a one-element tensor.
        at::Scalar local_scalar_dense(const at::Tensor& self) {
            KTT_REQUIRE(
                self.numel() == 1, "_local_scalar_dense", "needs a one-element tensor"
            );
            if (holds_complex(self)) {
                auto pair = download(as_real(self).contiguous()).flatten();
                return (
                    c10::complex<double>(pair[0].item<double>(), pair[1].item<double>())
                );
            }
            return download(self).item();
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("_local_scalar_dense", &local_scalar_dense);
        m.impl("copy_", &copy_);
        m.impl("_copy_from", &copy_from);
        m.impl("_copy_from_and_resize", &copy_from_and_resize);
        m.impl("_to_copy", &to_copy);
    }

    TORCH_LIBRARY_FRAGMENT(ktt, m) {
        m.def("replica(Tensor self, int chip) -> Tensor", &replica);
    }

}  // namespace ktt
