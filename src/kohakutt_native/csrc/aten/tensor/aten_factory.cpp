// aten factory kernels for tt: empty.memory_format, empty_strided, fill_, zero_,
// resize_ (complex64 through its fp32 pairs).
#include <ATen/ops/as_strided_native.h>
#include <c10/util/accumulate.h>
#include <torch/library.h>

#include <algorithm>
#include <numeric>

#include "core/torch/contract.hpp"
#include "core/torch/complex.hpp"
#include "core/torch/constants.hpp"

namespace ktt {
    namespace {

        at::Tensor empty_memory_format(
            c10::IntArrayRef sizes,
            std::optional<at::ScalarType> dtype,
            std::optional<at::Layout>,
            std::optional<at::Device>,
            std::optional<bool>,
            std::optional<at::MemoryFormat>
        ) {
            auto scalar = dtype.value_or(at::kFloat);
            if (scalar == at::kComplexFloat) {
                return complex_empty(sizes);
            }

            // Zero-element tensors of a dtype without a device type (e.g. batch norm's
            // uint8 reserve) have no buffer; using their data raises the dtype error.
            const bool device_dtype =
                (scalar == at::kBFloat16 || scalar == at::kFloat ||
                 scalar == at::kInt || scalar == at::kLong || scalar == at::kBool);
            if (!device_dtype && c10::multiply_integers(sizes) == 0) {
                return make_tt(nullptr, sizes, scalar);
            }
            return (
                make_tt(
                    bridge::empty(to_vector(sizes), to_bridge(scalar)), sizes, scalar
                )
            );
        }

        // A dense base in stride order, viewed with the requested (permuted) strides.
        at::Tensor empty_strided(
            c10::IntArrayRef sizes,
            c10::IntArrayRef strides,
            std::optional<at::ScalarType> dtype,
            std::optional<at::Layout> layout,
            std::optional<at::Device> device,
            std::optional<bool> pin
        ) {
            std::vector<int64_t> order(sizes.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                return strides[a] > strides[b];
            });
            std::vector<int64_t> base_sizes;
            for (int64_t d : order) {
                base_sizes.push_back(sizes[d]);
            }
            auto base = (
                empty_memory_format(
                    base_sizes, dtype, layout, device, pin, std::nullopt
                )
            );
            if (base_sizes == to_vector(sizes)) {
                return base;
            }
            return at::native::as_strided_tensorimpl(base, sizes, strides, 0);
        }

        // Filled on device from self's own buffer (no host write; trace-safe).
        at::Tensor& fill_scalar(at::Tensor& self, const at::Scalar& value) {
            if (holds_complex(self)) {
                auto z = value.toComplexDouble();
                auto pairs = as_real(self);
                pairs.select(-1, 0).fill_(z.real());
                pairs.select(-1, 1).fill_(z.imag());
                return self;
            }
            auto h = bridge::full_like(dense("fill_.Scalar", self), value.toDouble());
            commit("fill_.Scalar", self, std::move(h));
            return self;
        }

        // Filled with a 0-d tensor's value (read to host).
        at::Tensor& fill_tensor(at::Tensor& self, const at::Tensor& value) {
            KTT_REQUIRE(value.dim() == 0, "fill_.Tensor", "value must be 0-d");
            return (
                fill_scalar(self, value.is_cpu() ? value.item() : value.cpu().item())
            );
        }

        at::Tensor& zero_(at::Tensor& self) {
            return fill_scalar(self, 0);
        }

        // Resize to new sizes: a fresh contiguous storage with undefined contents
        // (PyTorch's resize_ semantics when the element count changes); same sizes keep
        // the data.
        const at::Tensor& resize_(
            const at::Tensor& self,
            c10::IntArrayRef size,
            std::optional<at::MemoryFormat>
        ) {
            if (self.sizes() == size) {
                return self;
            }
            auto fresh = (
                empty_memory_format(
                    size,
                    self.scalar_type(),
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt
                )
            );
            auto* impl = self.unsafeGetTensorImpl();
            impl->set_storage_keep_dtype(fresh.storage());
            impl->set_storage_offset(0);
            impl->set_sizes_contiguous(size);
            return self;
        }

        // Identity [n, m] in out's dtype: a cached device constant.
        at::Tensor& eye_m_out(int64_t n, int64_t m, at::Tensor& out) {
            out.resize_({n, m});
            if (n == 0 || m == 0) {
                return out;
            }
            const auto dtype = out.scalar_type();
            auto ident = device_constant(c10::str("eye|", n, "|", m, "|", dtype), [&] {
                return at::eye(n, m, at::TensorOptions().dtype(dtype));
            });
            return into("eye.m_out", out, ident);
        }

        at::Tensor& eye_out(int64_t n, at::Tensor& out) {
            return eye_m_out(n, n, out);
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("eye.m_out", &eye_m_out);
        m.impl("eye.out", &eye_out);
        m.impl("empty.memory_format", &empty_memory_format);
        m.impl("empty_strided", &empty_strided);
        m.impl("fill_.Scalar", &fill_scalar);
        m.impl("fill_.Tensor", &fill_tensor);
        m.impl("zero_", &zero_);
        m.impl("resize_", &resize_);
    }

}  // namespace ktt
