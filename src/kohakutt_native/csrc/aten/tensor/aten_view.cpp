// aten view kernels for tt: PyTorch's own metadata-only implementations over tt
// storage.
#include <algorithm>
#include <vector>

#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/SymInt.h>
#include <torch/library.h>

#include "core/torch/tt_tensor.hpp"

namespace ktt {

    namespace {

        // set_ to (part of) an existing tt storage; never grows it (tt storage is not
        // resized on the host).
        at::Tensor& set_storage(
            at::Tensor& self,
            c10::Storage source,
            c10::SymInt offset,
            c10::SymIntArrayRef size,
            c10::SymIntArrayRef stride
        ) {
            TORCH_CHECK(
                stride.empty() || stride.size() == size.size(),
                "set_: sizes and strides differ in length"
            );
            const int64_t off = offset.guard_int(__FILE__, __LINE__);
            TORCH_CHECK(off >= 0, "set_ on tt: negative storage offset");
            int64_t numel = 1, end = off + 1;
            for (size_t i = 0; i < size.size(); ++i) {
                const int64_t n = size[i].guard_int(__FILE__, __LINE__);
                numel *= n;
                if (!stride.empty()) {
                    const int64_t s = stride[i].guard_int(__FILE__, __LINE__);
                    TORCH_CHECK(s >= 0, "set_ on tt: negative strides");
                    end += (n - 1) * s;
                }
            }
            if (numel == 0) {
                end = off;
            } else if (stride.empty()) {
                end = off + numel;
            }
            TORCH_CHECK(
                end * static_cast<int64_t>(self.dtype().itemsize()) <=
                    static_cast<int64_t>(source.nbytes()),
                "set_ on tt: the view does not fit the source storage"
            );
            TORCH_CHECK(
                source.device() == self.device(),
                "set_: the storage is on another device"
            );
            // The storage keeps one logical dtype (kByte: never written typed).
            if (auto* st = static_cast<TtStorage*>(source.data_ptr().get_context())) {
                TORCH_CHECK(
                    st->dtype == at::kByte || st->dtype == self.scalar_type(),
                    "set_ on tt: the storage holds ",
                    st->dtype,
                    ", the tensor is ",
                    self.scalar_type()
                );
            }
            std::vector<int64_t> sizes(size.size()), strides(size.size());
            int64_t step = 1;
            for (size_t i = size.size(); i-- > 0;) {
                sizes[i] = size[i].guard_int(__FILE__, __LINE__);
                strides[i] = stride.empty() ? step
                                            : stride[i].guard_int(__FILE__, __LINE__);
                step *= std::max<int64_t>(sizes[i], 1);
            }
            auto* impl = self.unsafeGetTensorImpl();
            if (!self.storage().is_alias_of(source)) {
                impl->set_storage_keep_dtype(std::move(source));
            }
            impl->set_sizes_and_strides(sizes, strides, off);
            return self;
        }

        void unsupported(const char* form) {
            TORCH_CHECK(
                false,
                "set_",
                form,
                " is not supported on tt; use t.set_(other) or t.set_(storage, offset, "
                "size, stride)"
            );
        }

        at::Tensor& set_whole_storage(at::Tensor& self, c10::Storage) {
            unsupported("(storage)");
            return self;
        }

        at::Tensor& set_empty(at::Tensor& self) {
            unsupported("()");
            return self;
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("view", &at::native::view);
        m.impl("as_strided", &at::native::as_strided_tensorimpl);
        m.impl("_reshape_alias", &at::native::_reshape_alias);
        m.impl("set_.source_Tensor", &at::native::set_tensor_);
        m.impl("set_.source_Storage_storage_offset", &set_storage);
        m.impl("set_.source_Storage", &set_whole_storage);
        m.impl("set_", &set_empty);
    }

}  // namespace ktt
