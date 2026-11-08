// Cumsum writer: per row tile its W output tiles (CB 16, transposed) transposed back in
// L1 and written in order.
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

// Compile-time args: TensorAccessorArgs of the output. Common runtime args: output
// address, row tile step, row tiles, W tiles, element bytes (2 / 4); per core: first
// row tile.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t out_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t rows = get_common_arg_val<uint32_t>(a++);
    const uint32_t wt = get_common_arg_val<uint32_t>(a++);
    const uint32_t elem = get_common_arg_val<uint32_t>(a++);
    const uint32_t start = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = 16, cb_back = 5;
    const auto out = TensorAccessor(TensorAccessorArgs<0>(), out_addr, 1024 * elem);
    cb_reserve_back(cb_back, 1);
    const uint32_t back = get_write_ptr(cb_back);

    for (uint32_t u = start; u < rows; u += step) {
        for (uint32_t t = 0; t < wt; ++t) {
            cb_wait_front(cb_out, 1);
            if (elem == 4) {
                transpose<uint32_t>(get_read_ptr(cb_out), back);
            } else {
                transpose<uint16_t>(get_read_ptr(cb_out), back);
            }
            cb_pop_front(cb_out, 1);
            noc_async_write_page(u * wt + t, out, back);
            noc_async_write_barrier();
        }
    }
}
