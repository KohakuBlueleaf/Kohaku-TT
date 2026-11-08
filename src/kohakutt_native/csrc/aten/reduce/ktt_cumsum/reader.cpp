// Cumsum reader: per row tile (p, r) of a TILE [P, R, W] tensor its W tiles in order,
// each transposed in L1 (element (i, j) to (j, i)).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    template <typename T>
    void transpose(uint32_t src, uint32_t dst) {
        const auto* s = reinterpret_cast<const volatile T*>(src);
        auto* d = reinterpret_cast<volatile T*>(dst);
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = 0; c < 32; ++c) {
                d[face_offset(c, r)] = s[face_offset(r, c)];
            }
        }
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of x. Runtime args: x address, first row tile,
// row tile step, row tiles (P R tiles), W tiles, element bytes (2 / 4).
void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t step = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t wt = get_arg_val<uint32_t>(a++);
    const uint32_t elem = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_x = 0, cb_raw = 4;
    const auto x = TensorAccessor(TensorAccessorArgs<0>(), x_addr, 1024 * elem);
    cb_reserve_back(cb_raw, 1);
    const uint32_t raw = get_write_ptr(cb_raw);

    for (uint32_t u = start; u < rows; u += step) {
        for (uint32_t t = 0; t < wt; ++t) {
            noc_async_read_page(u * wt + t, x, raw);
            noc_async_read_barrier();
            cb_reserve_back(cb_x, 1);
            if (elem == 4) {
                transpose<uint32_t>(raw, get_write_ptr(cb_x));
            } else {
                transpose<uint16_t>(raw, get_write_ptr(cb_x));
            }
            cb_push_back(cb_x, 1);
        }
    }
}
