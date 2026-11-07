// Kohaku-TT extension API, torch side (torch headers only): runs a registered program
// (ktt/extension/device.hpp) on tt tensors from an op implementation.
#pragma once

#include <string>
#include <vector>

#include <ATen/ATen.h>

#define KTT_EXTENSION_API __attribute__((visibility("default")))

namespace ktt::extension {

    // An output of a program: logical sizes and dtype (bf16, fp32 or int32).
    struct Output {
        std::vector<int64_t> sizes;
        at::ScalarType dtype;
    };

    // Runs program `name` on the default stream over the dense device tensors of
    // `inputs` and new tensors of `outputs`; returns the output tt tensors.
    KTT_EXTENSION_API std::vector<at::Tensor> run(
        const std::string& name,
        at::TensorList inputs,
        const std::vector<Output>& outputs
    );

}  // namespace ktt::extension
