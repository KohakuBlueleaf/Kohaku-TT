// Kohaku-TT's in-place index copy (aten/index/ktt_index_copy/reader.cpp): every size a
// runtime argument, the indices read on the device.
#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ktt::bridge::ktt_index_copy {

    // Tiles. Rows: dst [batch, lt, dt], row k of b at tile ((b / heads) tt + k / 32)
    // src_wt + src_coff + (b % heads) src_hstride of src; planes: [batch, length, dt].
    struct Plan {
        uint32_t planes = 0, batch = 1, dt = 1, lt = 1, length = 1;
        uint32_t tt = 1, src_wt = 1, src_coff = 0, src_hstride = 0, heads = 1;
    };

    // dst: TILE bf16 / fp32 / int32, written in place; src: TILE, dst's dtype; index:
    // TILE int32 [1, n]. Indices outside [0, length) are skipped. Returns dst.
    ttnn::Tensor run(
        const ttnn::Tensor& dst,
        const ttnn::Tensor& src,
        const ttnn::Tensor& index,
        uint32_t n,
        const Plan& plan
    );

}  // namespace ktt::bridge::ktt_index_copy
