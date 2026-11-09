// Kohaku-TT's own fp32 reduce-scatter on a 1x2 mesh (TT-NN headers; device side only).
#pragma once

#include <optional>

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::mesh_rs {

    // The fp32 sum over the two chips of `t` (each chip's partial value, rank >= 2,
    // leading dims 1, TILE, DRAM interleaved), scattered on the rows: chip k gets rows
    // [k H / 2, (k + 1) H / 2). fp32 adds on the SFPU (round to nearest even); tiles
    // cross every fabric link between the chips. nullopt when the mesh is not 1x2 over
    // `axis` 1, `t` is not fp32 TILE in DRAM, or H is not a multiple of 64.
    std::optional<ttnn::Tensor> reduce_scatter_rows(const ttnn::Tensor& t, uint32_t axis);

    // Frees the global semaphores (before the device closes).
    void clear();

}  // namespace ktt::bridge::mesh_rs
