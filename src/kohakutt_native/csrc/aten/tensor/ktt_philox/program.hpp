// Kohaku-TT's counter-based uniforms (aten/tensor/ktt_philox/writer.cpp): Philox4x32-10
// on the data-movement cores, one draw per global element index.
#pragma once

#include <cstdint>
#include <vector>

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_philox {

    // The largest device rank run() takes.
    constexpr int64_t kMaxRank = 8;

    // A TILE `dtype` (bf16 / fp32) tensor of device shape `local`: this chip's box of a
    // tensor of device shape `global`, its uniforms in [0, 1) (24 bits; bf16 truncated)
    // at their global row-major indices. `params` (int32 TILE, per chip [1, 3 + rank]):
    // key low / high word, Philox word, the box's global offsets per dim.
    ttnn::Tensor run(
        const std::vector<int64_t>& local,
        const std::vector<int64_t>& global,
        tt::tt_metal::DataType dtype,
        const ttnn::Tensor& params,
        const tt::tt_metal::MemoryConfig& memory
    );

}  // namespace ktt::bridge::ktt_philox
