// Device constants built from host data (index / interpolation tables), uploaded once
// per key and reused: no per-call uploads, and usable inside trace captures once built.
#pragma once

#include <ATen/ATen.h>

#include <functional>
#include <string>

namespace ktt {

    // The tt tensor for `key`; `build` makes its CPU value on first use only.
    at::Tensor device_constant(
        const std::string& key, const std::function<at::Tensor()>& build
    );

    // True when the constant for `key` is built.
    bool has_device_constant(const std::string& key);

    // Drops every constant (before the device closes).
    void clear_device_constants();

}  // namespace ktt
