// 1-D conv input layout: NCL TILE bf16 x [N, C, L] -> zero-padded channels-last rows
// [N Wp, Cp] (Cp: C rounded up to 32, channels from C on zero). Per unit (image,
// channel tile, length tile): the tile into L1, transposed, each position's 64-byte
// channel segment written into its row. Border rows (left: the padding; right: the rest
// of Wp) are written as zeros.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

namespace {

    constexpr uint32_t kTileBytes = 2048;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of x, then of the rows, then the
// argument-layout version. Runtime args (per core: start; the rest common): version, x
// address, rows address, first unit, unit step, N, C, Cp, L, Wp, pad left, pixels per
// rows page (P).
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        rows_args = (TensorAccessorArgs<x_args.next_compile_time_args_offset()>());
    if (get_compile_time_arg_val(rows_args.next_compile_time_args_offset()) !=
            ktt_conv::kArgsVersion ||
        get_common_arg_val<uint32_t>(0) != ktt_conv::kArgsVersion) {
        return;
    }
    uint32_t a = 1;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t rows_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t n = get_common_arg_val<uint32_t>(a++);
    const uint32_t c = get_common_arg_val<uint32_t>(a++);
    const uint32_t cp = get_common_arg_val<uint32_t>(a++);
    const uint32_t len = get_common_arg_val<uint32_t>(a++);
    const uint32_t wp = get_common_arg_val<uint32_t>(a++);
    const uint32_t pw = get_common_arg_val<uint32_t>(a++);
    const uint32_t per_page = get_common_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_tile = 0, cb_stage = 1, cb_zero = 2;
    const uint32_t row_bytes = cp * 2;
    const auto x = TensorAccessor(x_args, x_addr, kTileBytes);
    const auto rows = TensorAccessor(rows_args, rows_addr, per_page * row_bytes);
    // NOC address of byte `offset` of pixel row `pixel` (P is a power of two).
    const uint32_t page_shift = __builtin_ctz(per_page), page_mask = per_page - 1;
    const auto pixel_addr = [&](uint32_t pixel, uint32_t offset) {
        return rows.get_noc_addr(
            pixel >> page_shift, (pixel & page_mask) * row_bytes + offset
        );
    };

    cb_reserve_back(cb_tile, 1);
    cb_reserve_back(cb_stage, 1);
    cb_reserve_back(cb_zero, 1);
    const uint32_t tile_l1 = get_write_ptr(cb_tile);
    auto* stage = reinterpret_cast<uint16_t*>(get_write_ptr(cb_stage));
    const uint32_t zero = get_write_ptr(cb_zero);
    for (uint32_t i = 0; i < row_bytes / 4; ++i) {
        reinterpret_cast<uint32_t*>(zero)[i] = 0;
    }

    const uint32_t ct = cp / 32, lt = (len + 31) / 32;
    const uint32_t units = n * ct * lt;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t img = u / (ct * lt), g = (u / lt) % ct, tl = u % lt;
        const uint32_t page = (img * ct + g) * lt + tl;
        noc_async_read_page(page, x, tile_l1);
        noc_async_read_barrier();
        const auto* t = reinterpret_cast<const uint16_t*>(tile_l1);
        const uint32_t channels = c - g * 32 < 32 ? c - g * 32 : 32;
        const uint32_t cols = len - tl * 32 < 32 ? len - tl * 32 : 32;
        // stage[position][channel], channels past C zero.
        for (uint32_t l = 0; l < cols; ++l) {
            for (uint32_t k = 0; k < 32; ++k) {
                stage[l * 32 + k] = k < channels ? t[face_offset(k, l)] : 0;
            }
        }
        const uint32_t base = img * wp + pw + tl * 32;
        for (uint32_t l = 0; l < cols; ++l) {
            noc_async_write(
                reinterpret_cast<uint32_t>(stage + l * 32),
                pixel_addr(base + l, g * 64),
                64
            );
        }
        noc_async_write_barrier();
    }

    // Border: zero pages for the padded positions of each image.
    for (uint32_t img = start; img < n; img += step) {
        for (uint32_t col = 0; col < wp; ++col) {
            if (col < pw || col >= len + pw) {
                noc_async_write(zero, pixel_addr(img * wp + col, 0), row_bytes);
            }
        }
        noc_async_write_barrier();
    }
}
