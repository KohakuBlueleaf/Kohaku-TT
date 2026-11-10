// Mesh placement rules on tt tensors: placed views of a tensor's storage and the
// persistent-tensor migration (the engine is core/mesh/rules.hpp).
#pragma once

#include <ATen/ATen.h>

#include "core/mesh/rules.hpp"

namespace ktt::mesh {

    // Persistent tensors re-placed for their consumers since load.
    int64_t migrations();

    // The storage base of tt tensor `t` as placed when `t` is a view the placement
    // survives and the base is mesh-placed; null otherwise.
    bridge::HandlePtr placed_base(const at::Tensor& t);

}  // namespace ktt::mesh
