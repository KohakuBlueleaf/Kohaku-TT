// aten copy kernels for tt: copy_, _copy_from, _to_copy (CPU <-> tt, tt <-> tt;
// complex64 through its fp32 pairs), item().
#include <torch/library.h>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"
#include "core/torch/mesh_rules.hpp"

namespace ktt {
    namespace {

        // The elements of tt view `src` as `dtype` in a buffer no other storage shares
        // (dense_as may return src's own base handle, which a trace replay can
        // overwrite).
        bridge::HandlePtr private_copy(
            const std::string& op, const at::Tensor& src, at::ScalarType dtype
        ) {
            // A whole Flat or head-major storage is copied (or cast) as stored.
            auto h = dense_stored(op, src);
            const auto cast = to_bridge(dtype);
            if (bridge::dtype(h) != cast) {
                return bridge::typecast(h, cast);
            }
            // A head view reads its storage's cached head-major copy.
            const auto& base = storage_of(src).base;
            const bool cached =
                (bridge::format(h) == bridge::Format::HeadMajor &&
                 bridge::format(base) == bridge::Format::Nchw);
            return cached || bridge::aliases(h, base) ? bridge::clone(h) : h;
        }

        // True for tensors covering their whole contiguous storage.
        bool whole(const at::Tensor& t) {
            return (
                t.is_contiguous() && t.storage_offset() == 0 &&
                t.numel() * t.element_size() ==
                    static_cast<int64_t>(t.storage().nbytes())
            );
        }

        // The mesh placement of the value whole tt tensor `dst` holds (read at its own
        // sizes); nullopt for a view, a never-written storage or a partial sum.
        std::optional<bridge::Placement> held_placement(const at::Tensor& dst) {
            const auto& st = storage_of(dst);
            if (!st.base || st.blank || !whole(dst) ||
                st.sizes != to_vector(dst.sizes()) ||
                bridge::is_block(bridge::dtype(st.base))) {
                return std::nullopt;
            }
            auto p = bridge::placement(st.base);
            if (p.has_partial()) {
                return std::nullopt;
            }
            return p;
        }

        // `value` converted to `held` (the destination's placement) when there is one.
        bridge::HandlePtr to_held(
            bridge::HandlePtr value, const std::optional<bridge::Placement>& held
        ) {
            if (!held || bridge::placement(value) == *held) {
                return value;
            }
            return bridge::distribute(value, *held);
        }

        // src = x.movedim(1, -2) of a whole head-major x [B, H, S..., D] as [B, S...,
        // H D] (its sizes into `sizes`), from one concat-heads program; else null.
        bridge::HandlePtr merged_heads(
            const at::Tensor& src, std::vector<int64_t>& sizes
        ) {
            if (src.dim() < 4 || src.numel() == 0 || src.size(-1) % 32 != 0 ||
                (src.scalar_type() != at::kBFloat16 &&
                 src.scalar_type() != at::kFloat)) {
                return nullptr;
            }
            const int64_t tile_bytes = 1024 * src.element_size();
            if (2 * src.size(-2) * (src.size(-1) / 32) * tile_bytes >
                kMaxMergedHeadBytes) {
                return nullptr;
            }
            auto base = src.movedim(-2, 1);
            if (!whole(base)) {
                return nullptr;
            }
            const int64_t b = src.size(0), heads = src.size(-2), d = src.size(-1);
            const int64_t s = base.numel() / (b * heads * d);
            // The storage reads as [B, H, S, D] by a metadata reshape.
            const auto& stored = storage_of(base).sizes;
            const size_t n = stored.size();
            if (n < 2 || stored[n - 1] != d ||
                !(stored[n - 2] == s || (stored[n - 2] % 32 == 0 && s % 32 == 0))) {
                return nullptr;
            }
            auto merged = (bridge::concat_heads(
                dense("copy_", base.reshape({b, heads, s, d}))
            ));
            sizes = to_vector(src.sizes());
            sizes.pop_back();
            sizes.back() = heads * d;
            return src.dim() == 4 ? merged : bridge::reshape(merged, sizes);
        }

        // True for a uint8 view of a storage of another dtype, or of one not yet
        // typed: storage-level byte access (serialization). uint8 data is not one.
        bool is_byte_view(const at::Tensor& t) {
            if (t.scalar_type() != at::kByte) {
                return false;
            }
            const auto& st = storage_of(t);
            return !(st.base && st.dtype == at::kByte);
        }

        // A contiguous host tensor (logical dtype) holding the elements of tt view
        // `src`.
        at::Tensor download(const at::Tensor& src, int chip = 0) {
            if (is_byte_view(src)) {
                KTT_REQUIRE(
                    src.is_contiguous(), "copy", "strided byte views are not supported"
                );
                auto bytes = (storage_bytes(
                    storage_of(src), src.storage_offset(), src.numel()
                ));
                return bytes.view(src.sizes());
            }
            auto phys = physical(src.scalar_type());
            auto host = at::empty(src.sizes(), at::TensorOptions().dtype(phys));
            if (src.numel() == 0) {
                return host.to(src.scalar_type());
            }
            // A whole mesh-placed storage is assembled from its shards on the host.
            auto& st = storage_of(src);
            const bool placed =
                (st.base && whole(src) && !bridge::is_block(bridge::dtype(st.base)) &&
                 !bridge::placement(st.base).is_replicated());
            bridge::download(
                placed ? st.base : dense_stored("copy", src), host.data_ptr(), chip
            );
            return phys == src.scalar_type() ? host : host.to(src.scalar_type());
        }

        // torch.ops.ktt.replica: the copy of tt tensor `self` on mesh chip `chip`, as a
        // CPU tensor.
        at::Tensor replica(const at::Tensor& self, int64_t chip) {
            TORCH_CHECK(self.is_privateuseone(), "ktt::replica: a tt tensor");
            TORCH_CHECK(!is_byte_view(self), "ktt::replica: not for byte views");
            return download(self, static_cast<int>(chip));
        }

        // A device handle holding host tensor `host` stored as logical dtype `dtype`,
        // at placement `held` when given (else the automatic placement).
        bridge::HandlePtr upload(
            const at::Tensor& host,
            at::ScalarType dtype,
            const std::optional<bridge::Placement>& held = std::nullopt
        ) {
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
            if (held && !held->is_replicated()) {
                return (bridge::upload_placed(
                    values.data_ptr(),
                    to_vector(values.sizes()),
                    to_bridge(dtype),
                    *held
                ));
            }
            return to_held(
                bridge::upload_auto(
                    values.data_ptr(), to_vector(values.sizes()), to_bridge(dtype)
                ),
                held
            );
        }

        // copy_ with a complex side, through the fp32 pairs: complex <- real sets im =
        // 0, real <- complex takes the real part and warns once, as PyTorch does.
        at::Tensor& copy_complex(at::Tensor& self, const at::Tensor& src) {
            KTT_REQUIRE(!self.is_conj(), "copy_", "writing into a conjugated view");
            if (holds_complex(self) && holds_complex(src)) {
                copy_pairs(self, src);
            } else if (holds_complex(self)) {
                auto pairs = as_real(self);
                pairs.select(-1, 0).copy_(src.expand_as(self));
                pairs.select(-1, 1).fill_(0);
            } else {
                TORCH_WARN_ONCE(
                    "Casting complex values to real discards the imaginary part"
                );
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
            // A destination that holds a value keeps its placement.
            const auto held = held_placement(self);
            if (src.is_cpu()) {
                commit(
                    "copy_", self, upload(src.expand_as(self), self.scalar_type(), held)
                );
                return self;
            }
            KTT_REQUIRE(
                src.is_privateuseone(), "copy_", "unsupported source ", src.device()
            );

            // A merged-heads copy becomes the storage, as [B, S..., H D]; while traces
            // are alive only a never-written one (others keep their buffers).
            if (src.scalar_type() == self.scalar_type() && whole(self) &&
                self.sizes() == src.sizes() && (!held || held->is_replicated()) &&
                (!bridge::traces_alive() || storage_of(self).blank)) {
                std::vector<int64_t> sizes;
                if (auto merged = merged_heads(src, sizes)) {
                    auto& st = storage_of(self);
                    st.heads_of.reset();
                    st.heads.clear();
                    st.major_of.reset();
                    st.major.reset();
                    st.base = merged;
                    st.blank = false;
                    st.sizes = std::move(sizes);
                    st.dtype = self.scalar_type();
                    return self;
                }
            }

            // A mesh-placed source (whole storage or a permutation of one) into a whole
            // destination: copied per chip, then converted to the destination's.
            if (src.scalar_type() == self.scalar_type() && whole(self) &&
                self.sizes() == src.sizes()) {
                if (auto base = mesh::placed_base(src)) {
                    const auto dims = bridge::device_sizes(base);
                    auto f = mesh::elementwise(1, dims.size());
                    f.partial = mesh::Factors::Partial::All;
                    auto out = mesh::run_placed(
                        "copy_", f, {base}, dims, [](const auto& in) {
                            return bridge::clone(in[0]);
                        }
                    );
                    if (out) {
                        commit("copy_", self, to_held(*out, held));
                        return self;
                    }
                }
            }

            // Different dtypes: cast on device first (_to_copy), then copy in.
            auto source =
                (src.scalar_type() == self.scalar_type() ? src
                                                         : src.to(self.scalar_type()));
            commit(
                "copy_",
                self,
                to_held(
                    private_copy("copy_", source.expand_as(self), source.scalar_type()),
                    held
                )
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
                auto out = (at::empty(
                    self.sizes(), self.options().dtype(scalar).device(target)
                ));
                out.copy_(self);
                return out;
            }

            // preserve_format keeps a dense non-contiguous input's strides, as PyTorch
            // does (compiled graphs bake in views that rely on them).
            const auto format = memory_format.value_or(at::MemoryFormat::Preserve);
            if (format == at::MemoryFormat::Preserve && !self.is_contiguous() &&
                self.is_non_overlapping_and_dense()) {
                auto out = (at::empty_strided(
                    self.sizes(),
                    self.strides(),
                    self.options().dtype(scalar).device(target)
                ));
                // Cast first (contiguous) so copy_ never re-enters _to_copy.
                auto source = self;
                if (self.is_privateuseone() && scalar != self.scalar_type()) {
                    source = (wrap_output(
                        private_copy("_to_copy", self, scalar), self.sizes(), scalar
                    ));
                }
                out.copy_(source);
                return out;
            }
            if (self.is_cpu() && target.is_privateuseone()) {
                return make_tt(upload(self, scalar), self.sizes(), scalar);
            }

            // tt -> tt: a device cast (or a copy when the dtype is unchanged).
            if (self.is_privateuseone() && target.is_privateuseone()) {
                // Mesh-placed input: cast per chip on its shard; P(sum) passes through
                // copies and widening casts.
                if (auto base = mesh::placed_base(self)) {
                    const auto to = to_bridge(scalar);
                    const auto from = bridge::dtype(base);
                    const auto dims = bridge::device_sizes(base);
                    auto f = mesh::elementwise(1, dims.size());
                    if (to == from ||
                        (from == bridge::DType::BFloat16 &&
                         to == bridge::DType::Float32)) {
                        f.partial = mesh::Factors::Partial::All;
                    }
                    f.dtype = to;
                    auto out = mesh::run_placed(
                        "_to_copy", f, {base}, dims, [&](const auto& in) {
                            return to == from ? bridge::clone(in[0])
                                              : bridge::typecast(in[0], to);
                        }
                    );
                    if (out) {
                        return wrap_output(*out, self.sizes(), scalar);
                    }
                }
                return (wrap_output(
                    private_copy("_to_copy", self, scalar), self.sizes(), scalar
                ));
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
