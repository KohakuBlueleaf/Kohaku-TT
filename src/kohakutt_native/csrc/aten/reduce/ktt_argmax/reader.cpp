// Argmax reader: a column-index tile (fp32, element (r, c) = c) once; per unit (32
// leading rows, one row tile) and leading row p, the row tile's W tiles of a TILE
// [P, R, W] tensor in order, the last tile's padding columns set to -inf.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    template <typename T>
    void fill_columns(uint32_t addr, uint32_t first, T value) {
        auto* t = reinterpret_cast<volatile T*>(addr);
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = first; c < 32; ++c) {
                t[face_offset(r, c)] = value;
            }
        }
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of x. Common runtime args: x address, unit
// step, P, row tiles, W tiles, valid columns of the last W tile, element bytes (2 / 4);
// per core: first unit.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t p_count = get_common_arg_val<uint32_t>(a++);
    const uint32_t rt = get_common_arg_val<uint32_t>(a++);
    const uint32_t wt = get_common_arg_val<uint32_t>(a++);
    const uint32_t w_last = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x = 0, cb_col = 3;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, x_addr, 1024 * elem);

    cb_reserve_back(cb_col, 1);
    auto* col = reinterpret_cast<volatile float*>(get_write_ptr(cb_col));
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            col[face_offset(r, c)] = static_cast<float>(c);
        }
    }
    cb_push_back(cb_col, 1);

    const uint32_t units = (p_count + 31) / 32 * rt;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t p0 = (u / rt) * 32, r_tile = u % rt;
        const uint32_t p1 = p0 + 32 < p_count ? p0 + 32 : p_count;
        for (uint32_t p = p0; p < p1; ++p) {
            for (uint32_t t = 0; t < wt; ++t) {
                cb_reserve_back(cb_x, 1);
                const uint32_t addr = get_write_ptr(cb_x);
                noc_async_read_page((p * rt + r_tile) * wt + t, x, addr);
                noc_async_read_barrier();
                if (t == wt - 1 && w_last < 32) {
                    if (elem == 4) {
                        fill_columns<uint32_t>(addr, w_last, 0xFF800000u);
                    } else {
                        fill_columns<uint16_t>(addr, w_last, 0xFF80u);
                    }
                }
                cb_push_back(cb_x, 1);
            }
        }
    }
}
