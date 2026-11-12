// Argmax writer: per item and leading row, the column of the largest B (lowest index on
// ties, NaN above all) per row; W chunks of a unit combined on its chunk-0 core.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // fp32 bits as an unsigned key ordered like the floats: NaN above all, zeros and
    // subnormals equal (as the compute's compare).
    inline uint32_t order_key(uint32_t bits) {
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
            return 0xFFFFFFFFu;
        }
        if ((bits & 0x7F800000u) == 0) {
            return 0x80000000u;
        }
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

// Compile-time args: TensorAccessorArgs of the output. Common runtime args: output
// address, item step, P, row tiles, split, semaphore id, R; per core: first item, the
// NoC (x, y) of its unit's chunk-0 core.
void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(1);
    const uint32_t p_count = get_common_arg_val<uint32_t>(2);
    const uint32_t rt = get_common_arg_val<uint32_t>(3);
    const uint32_t split = get_common_arg_val<uint32_t>(4);
    const uint32_t sem = get_semaphore(get_common_arg_val<uint32_t>(5));
    const uint32_t r_count = get_common_arg_val<uint32_t>(6);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t head_x = get_arg_val<uint32_t>(1);
    const uint32_t head_y = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_b = 16, cb_i = 17, cb_out = 18, cb_parts = 19, kTile = 4096;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, kTile);

    cb_reserve_back(cb_out, 1);
    cb_reserve_back(cb_parts, 1);
    auto* staged = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_out));
    const uint32_t parts = get_write_ptr(cb_parts);
    // A chunk's part: (key, index) per (leading row, row) of its unit.
    const uint32_t rows_max = p_count < 32 ? p_count : 32;
    const uint32_t part_bytes = rows_max * 32 * 8;
    const uint32_t items = (p_count + 31) / 32 * rt * split;
    for (uint32_t item = start; item < items; item += step) {
        const uint32_t u = item / split, j = item % split;
        const uint32_t p0 = (u / rt) * 32;
        const uint32_t p1 = p0 + 32 < p_count ? p0 + 32 : p_count;
        // Rows of the row tile inside R (the rest is padding).
        const uint32_t r0 = (u % rt) * 32;
        const uint32_t rows = r_count - r0 < 32 ? r_count - r0 : 32;
        auto* part = reinterpret_cast<volatile uint32_t*>(parts + j * part_bytes);
        for (uint32_t i = 0; i < 1024; ++i) {
            staged[i] = 0;
        }
        for (uint32_t p = p0; p < p1; ++p) {
            cb_wait_front(cb_b, 1);
            cb_wait_front(cb_i, 1);
            const auto* b = reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_b));
            const auto* idx = reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_i));
            for (uint32_t r = 0; r < rows; ++r) {
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
                if (split > 1) {
                    part[((p - p0) * 32 + r) * 2] = best_key;
                    part[((p - p0) * 32 + r) * 2 + 1] = best;
                }
            }
            cb_pop_front(cb_b, 1);
            cb_pop_front(cb_i, 1);
        }
        if (split > 1 && j > 0) {
            // The part to the same place on the unit's chunk-0 core, then its count.
            const uint32_t at = parts + j * part_bytes;
            noc_async_write(at, get_noc_addr(head_x, head_y, at), part_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(head_x, head_y, sem), 1);
            noc_async_atomic_barrier();
            continue;
        }
        if (split > 1) {
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), split - 1
            );
            invalidate_l1_cache();
            for (uint32_t k = 0; k < (p1 - p0) * 32; ++k) {
                if (k % 32 >= rows) {
                    continue;
                }
                uint32_t best_key = 0, best = 0xFFFFFFFFu;
                for (uint32_t q = 0; q < split; ++q) {
                    const auto* w = reinterpret_cast<volatile uint32_t*>(
                        parts + q * part_bytes
                    );
                    const uint32_t key = w[k * 2], index = w[k * 2 + 1];
                    if (key > best_key || (key == best_key && index < best)) {
                        best_key = key;
                        best = index;
                    }
                }
                staged[face_offset(k / 32, k % 32)] = best;
            }
        }
        noc_async_write_page(u, out, get_write_ptr(cb_out));
        noc_async_write_barrier();
    }
}
