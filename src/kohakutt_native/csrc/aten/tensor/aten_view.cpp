// aten view kernels for tt: PyTorch's own metadata-only implementations over tt
// storage.
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <torch/library.h>

namespace ktt {

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("view", &at::native::view);
        m.impl("as_strided", &at::native::as_strided_tensorimpl);
        m.impl("_reshape_alias", &at::native::_reshape_alias);
    }

}  // namespace ktt
