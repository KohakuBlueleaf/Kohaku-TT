// Group norm moments reader: per plane (n, c) of a TILE [N, C, H, W] tensor its tiles
// in order, the padding rows / columns of the last row / column tile zeroed in L1.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // Zeroes rows >= rows and columns >= cols of the tile at addr (T: element type).
    template <typename T>
    void zero_padding(uint32_t addr, uint32_t rows, uint32_t cols) {
        auto* t = reinterpret_cast<volatile T*>(addr);
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = 0; c < 32; ++c) {
                if (r >= rows || c >= cols) {
                    t[face_offset(r, c)] = 0;
                }
            }
        }
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of x. Runtime args (per core: start; the rest
// common): x address, first plane, plane step, planes, row tiles, column
// tiles, valid rows of the last row tile, valid columns of the last column tile,
// element bytes (2 or 4).
void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t planes = get_common_arg_val<uint32_t>(a++);
    const uint32_t ht = get_common_arg_val<uint32_t>(a++);
    const uint32_t wt = get_common_arg_val<uint32_t>(a++);
    const uint32_t h_last = get_common_arg_val<uint32_t>(a++);
    const uint32_t w_last = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_x = 0;
    const uint32_t tile_bytes = 1024 * elem;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, x_addr, tile_bytes);

    for (uint32_t p = start; p < planes; p += step) {
        for (uint32_t th = 0; th < ht; ++th) {
            for (uint32_t tw = 0; tw < wt; ++tw) {
                cb_reserve_back(cb_x, 1);
                const uint32_t addr = get_write_ptr(cb_x);
                noc_async_read_page((p * ht + th) * wt + tw, x, addr);
                noc_async_read_barrier();
                const uint32_t rows = th == ht - 1 ? h_last : 32;
                const uint32_t cols = tw == wt - 1 ? w_last : 32;
                if (rows < 32 || cols < 32) {
                    if (elem == 4) {
                        zero_padding<uint32_t>(addr, rows, cols);
                    } else {
                        zero_padding<uint16_t>(addr, rows, cols);
                    }
                }
                cb_push_back(cb_x, 1);
            }
        }
    }
}
