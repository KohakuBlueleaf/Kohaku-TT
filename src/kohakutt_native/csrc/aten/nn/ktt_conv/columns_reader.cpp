// Columns reader: per unit (row tile m, tap) the input rows the tap reads for output
// pixels first + 32 m .. + 31 (row (image OH + y) OW + x) into CB 0; past end, zero.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time: x_rows' accessor. Per core: first, end unit; common: x address, rows
// per page, row bytes, Ct, KH, KW, Sh, Sw, dh, dw, Hp, Wp, OH, OW, first, end row.
void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    uint32_t a = 0;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t per_page = get_common_arg_val<uint32_t>(a++);
    const uint32_t row_bytes = get_common_arg_val<uint32_t>(a++);
    const uint32_t ct = get_common_arg_val<uint32_t>(a++);
    const uint32_t kh = get_common_arg_val<uint32_t>(a++);
    const uint32_t kw = get_common_arg_val<uint32_t>(a++);
    const uint32_t sh = get_common_arg_val<uint32_t>(a++);
    const uint32_t sw = get_common_arg_val<uint32_t>(a++);
    const uint32_t dh = get_common_arg_val<uint32_t>(a++);
    const uint32_t dw = get_common_arg_val<uint32_t>(a++);
    const uint32_t hp = get_common_arg_val<uint32_t>(a++);
    const uint32_t wp = get_common_arg_val<uint32_t>(a++);
    const uint32_t oh = get_common_arg_val<uint32_t>(a++);
    const uint32_t ow = get_common_arg_val<uint32_t>(a++);
    const uint32_t row0 = get_common_arg_val<uint32_t>(a++);
    const uint32_t row_end = get_common_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_rows = 0;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, x_addr, per_page * row_bytes);
    const uint32_t taps = kh * kw;

    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t m = u / taps, tap = u % taps;
        const uint32_t shift = (tap / kw) * dh * wp + (tap % kw) * dw;
        const uint32_t first = row0 + m * 32;
        const uint32_t valid = row_end - first < 32 ? row_end - first : 32;
        // Pixel of row `first`, then walked along x.
        uint32_t img = first / (oh * ow), y = (first / ow) % oh, xx = first % ow;
        cb_reserve_back(cb_rows, ct);
        const uint32_t dst = get_write_ptr(cb_rows);
        // One read per run of consecutive source rows within one page.
        uint32_t start = 0, len = 0, at = dst;
        for (uint32_t q = 0; q < valid; ++q) {
            const uint32_t src = img * hp * wp + y * sh * wp + xx * sw + shift;
            if (len > 0 && src == start + len && src / per_page == start / per_page) {
                ++len;
            } else {
                if (len > 0) {
                    noc_async_read(
                        x.get_noc_addr(
                            start / per_page, (start % per_page) * row_bytes
                        ),
                        at,
                        len * row_bytes
                    );
                    at += len * row_bytes;
                }
                start = src;
                len = 1;
            }
            if (++xx == ow) {
                xx = 0;
                if (++y == oh) {
                    y = 0;
                    ++img;
                }
            }
        }
        if (len > 0) {
            noc_async_read(
                x.get_noc_addr(start / per_page, (start % per_page) * row_bytes),
                at,
                len * row_bytes
            );
        }
        // Rows valid .. 31 (last tile only): zero.
        auto* tail = reinterpret_cast<uint32_t*>(dst + valid * row_bytes);
        for (uint32_t e = 0; e < (32 - valid) * row_bytes / 4; ++e) {
            tail[e] = 0;
        }
        noc_async_read_barrier();
        cb_push_back(cb_rows, ct);
    }
}
