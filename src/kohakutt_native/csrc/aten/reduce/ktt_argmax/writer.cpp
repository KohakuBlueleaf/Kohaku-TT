// Argmax writer: per unit (32 leading rows, one row tile), per leading row p its B / I
// tiles (CB 16 / 17, fp32); per row the column of the largest B (lowest index on ties,
// floats ordered as integers) gives the index; the 32 x 32 indices (rows p, columns r)
// written as one int32 tile of the [P, R] output.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // fp32 bits as an unsigned key ordered like the floats.
    inline uint32_t order_key(uint32_t bits) {
        return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    }

    // The non-negative integer value of fp32 bits (an exact integer below 2^24).
    inline uint32_t to_uint(uint32_t bits) {
        const uint32_t e = (bits >> 23) & 0xFF;
        if (e < 127) {
            return 0;
        }
        const uint32_t m = (bits & 0x7FFFFF) | 0x800000;
        return e >= 150 ? m << (e - 150) : m >> (150 - e);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of the output. Runtime args: output address,
// first unit, unit step, P, row tiles.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t step = get_arg_val<uint32_t>(2);
    const uint32_t p_count = get_arg_val<uint32_t>(3);
    const uint32_t rt = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_b = 16, cb_i = 17, cb_out = 18, kTile = 4096;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, kTile);

    cb_reserve_back(cb_out, 1);
    auto* staged = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_out));
    const uint32_t units = (p_count + 31) / 32 * rt;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t p0 = (u / rt) * 32, r_tile = u % rt;
        const uint32_t p1 = p0 + 32 < p_count ? p0 + 32 : p_count;
        for (uint32_t i = 0; i < 1024; ++i) {
            staged[i] = 0;
        }
        for (uint32_t p = p0; p < p1; ++p) {
            cb_wait_front(cb_b, 1);
            cb_wait_front(cb_i, 1);
            const auto* b = reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_b));
            const auto* idx = reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_i));
            for (uint32_t r = 0; r < 32; ++r) {
                uint32_t best_key = 0, best = 0xFFFFFFFFu;
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t off = face_offset(r, c);
                    const uint32_t key = order_key(b[off]);
                    const uint32_t index = to_uint(idx[off]);
                    if (key > best_key || (key == best_key && index < best)) {
                        best_key = key;
                        best = index;
                    }
                }
                staged[face_offset(p - p0, r)] = best;
            }
            cb_pop_front(cb_b, 1);
            cb_pop_front(cb_i, 1);
        }
        noc_async_write_page(u, out, get_write_ptr(cb_out));
        noc_async_write_barrier();
    }
}
