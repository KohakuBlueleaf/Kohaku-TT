// complex64 on tt: a complex tensor views an fp32 storage of interleaved (re, im)
// pairs; complex-aware kernels compute on that fp32 view and wrap results back.
#pragma once

#include <ATen/ATen.h>

#include "core/device/tt_bridge.hpp"

namespace ktt {

    // True for complex dtypes (only complex64 is stored on tt; others raise).
    bool holds_complex(const at::Tensor& t);

    // The [..., 2] fp32 view of complex tensor `t` (a conjugated view is resolved
    // first).
    at::Tensor as_real(const at::Tensor& t);

    // The complex64 tensor of fp32 pairs `pairs` [..., 2] (made contiguous when
    // needed).
    at::Tensor as_complex(const at::Tensor& pairs);

    // (re, im) fp32 parts of `t`: complex tensors split, real ones with a zero im.
    std::pair<at::Tensor, at::Tensor> parts(const at::Tensor& t);

    // The complex64 tensor re + i im (re and im broadcast to one shape).
    at::Tensor from_parts(const at::Tensor& re, const at::Tensor& im);

    // An uninitialized complex64 tt tensor of `sizes`.
    at::Tensor complex_empty(c10::IntArrayRef sizes);

    // Complex paths of the generic table kernels (add / sub / mul / div, neg / abs /
    // exp); others raise unsupported.
    at::Tensor complex_binary(
        const char* op,
        bridge::BinaryOp id,
        const at::Tensor& self,
        const at::Tensor& other,
        const at::Scalar& alpha
    );
    at::Tensor complex_binary_scalar(
        const char* op, bridge::BinaryOp id, const at::Tensor& self, const at::Scalar& s
    );
    at::Tensor complex_unary(
        const char* op, bridge::UnaryOp id, const at::Tensor& self
    );

}  // namespace ktt
