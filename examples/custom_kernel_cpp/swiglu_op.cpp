// Torch side of the SwiGLU extension: the op kohakutt_example_cpp::swiglu on tt
// tensors, running the registered program (torch headers only).
#include <torch/library.h>

#include "ktt/extension/torch.hpp"

namespace {

    at::Tensor swiglu(const at::Tensor& a, const at::Tensor& b) {
        TORCH_CHECK(a.sizes() == b.sizes(), "swiglu: a and b differ in size");
        TORCH_CHECK(
            a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
            "swiglu: bf16 inputs only"
        );
        return ktt::extension::run(
            "kohakutt_example_cpp::swiglu", {a, b}, {{a.sizes().vec(), a.scalar_type()}}
        )[0];
    }

}  // namespace

TORCH_LIBRARY(kohakutt_example_cpp, m) {
    m.def("swiglu(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(kohakutt_example_cpp, PrivateUse1, m) {
    m.impl("swiglu", &swiglu);
}

// Shapes only (torch.compile tracing).
TORCH_LIBRARY_IMPL(kohakutt_example_cpp, Meta, m) {
    m.impl("swiglu", [](const at::Tensor& a, const at::Tensor&) {
        return at::empty_like(a);
    });
}
