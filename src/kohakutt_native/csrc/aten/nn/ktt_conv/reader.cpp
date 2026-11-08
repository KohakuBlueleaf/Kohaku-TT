// Conv reader: this core's weight block (all K = taps x Ct weight rows of its output
// column tiles) into CB 1 once; then per output row tile m and tap (i, j), the 32 input
// rows the tap reads (shifted by i dh Wp + j dw within the image) into CB 0, row-major,
// for the compute kernel to tilize. Window mode: per m and kernel row i, the 32 + (KW -
// 1) dw rows of all its taps read once into CB 3 (two sets: m + 1 read while m's taps
// are consumed), each tap's 32 rows copied from there. Input rows are read as runs of
// consecutive rows within a page (P rows per page).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: TensorAccessorArgs of x_rows, then of w. Runtime args: x address,
// w address, first / end output row tile, first output column tile, column tiles,
// output column tiles in all (Ot), Ct, KH, KW, dh, dw, Wp, input rows per image (Hp
// Wp), output rows per image (OH Wp), input rows in all, output rows in all, x row
// bytes, window mode (0 / 1), rows per x page (P).
void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t m0 = get_arg_val<uint32_t>(a++);
    const uint32_t m1 = get_arg_val<uint32_t>(a++);
    const uint32_t n0 = get_arg_val<uint32_t>(a++);
    const uint32_t nt = get_arg_val<uint32_t>(a++);
    const uint32_t ot = get_arg_val<uint32_t>(a++);
    const uint32_t ct = get_arg_val<uint32_t>(a++);
    const uint32_t kh = get_arg_val<uint32_t>(a++);
    const uint32_t kw = get_arg_val<uint32_t>(a++);
    const uint32_t dh = get_arg_val<uint32_t>(a++);
    const uint32_t dw = get_arg_val<uint32_t>(a++);
    const uint32_t wp = get_arg_val<uint32_t>(a++);
    const uint32_t in_per_image = get_arg_val<uint32_t>(a++);
    const uint32_t out_per_image = get_arg_val<uint32_t>(a++);
    const uint32_t in_rows = get_arg_val<uint32_t>(a++);
    const uint32_t out_rows = get_arg_val<uint32_t>(a++);
    const uint32_t row_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t window = get_arg_val<uint32_t>(a++);
    const uint32_t per_page = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_rows = 0, cb_w = 1, cb_window = 3;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto w_args = (
        TensorAccessorArgs<x_args.next_compile_time_args_offset()>()
    );
    const auto x = TensorAccessor(x_args, x_addr, per_page * row_bytes);
    const uint32_t tile = get_tile_size(cb_w);
    const auto w = TensorAccessor(w_args, w_addr, tile);

    // Weight block: tile (k, n0 + j) at CB position k nt + j.
    const uint32_t k_tiles = kh * kw * ct;
    cb_reserve_back(cb_w, k_tiles * nt);
    uint32_t addr = get_write_ptr(cb_w);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t j = 0; j < nt; ++j) {
            noc_async_read_page(k * ot + n0 + j, w, addr);
            addr += tile;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, k_tiles * nt);

    // Input row of output row tile m, row q (clamped into the tensor), tap shift.
    const auto source = [&](uint32_t m, uint32_t q, uint32_t shift) {
        uint32_t r = m * 32 + q;
        r = r < out_rows ? r : out_rows - 1;
        const uint32_t image = r / out_per_image;
        const uint32_t src = image * in_per_image + r % out_per_image + shift;
        return src < in_rows ? src : in_rows - 1;
    };
    // Rows q = 0 .. count - 1 of output row tile m at `shift` into dst (row_bytes
    // each), one read per run of consecutive rows in one page.
    const auto read_rows =
        [&](uint32_t m, uint32_t count, uint32_t shift, uint32_t dst) {
            uint32_t start = source(m, 0, shift), len = 1, at = dst;
            for (uint32_t q = 1; q <= count; ++q) {
                const uint32_t s = q < count ? source(m, q, shift) : 0xFFFFFFFFu;
                if (s == start + len && s / per_page == start / per_page) {
                    ++len;
                    continue;
                }
                noc_async_read(
                    x.get_noc_addr(start / per_page, (start % per_page) * row_bytes),
                    at,
                    len * row_bytes
                );
                at += len * row_bytes;
                start = s;
                len = 1;
            }
        };

    if (window == 0) {
        for (uint32_t m = m0; m < m1; ++m) {
            for (uint32_t i = 0; i < kh; ++i) {
                for (uint32_t j = 0; j < kw; ++j) {
                    cb_reserve_back(cb_rows, ct);
                    read_rows(m, 32, i * dh * wp + j * dw, get_write_ptr(cb_rows));
                    noc_async_read_barrier();
                    cb_push_back(cb_rows, ct);
                }
            }
        }
        return;
    }

    // Windows: set s, kernel row i at base + (s kh + i) span row_bytes.
    const uint32_t span = 32 + (kw - 1) * dw;
    cb_reserve_back(cb_window, 1);
    const uint32_t base = get_write_ptr(cb_window);
    const auto fetch = [&](uint32_t m, uint32_t set) {
        for (uint32_t i = 0; i < kh; ++i) {
            read_rows(m, span, i * dh * wp, base + (set * kh + i) * span * row_bytes);
        }
    };
    if (m0 < m1) {
        fetch(m0, 0);
    }
    for (uint32_t m = m0; m < m1; ++m) {
        const uint32_t set = (m - m0) & 1;
        noc_async_read_barrier();
        for (uint32_t i = 0; i < kh; ++i) {
            const uint32_t row0 = base + (set * kh + i) * span * row_bytes;
            for (uint32_t j = 0; j < kw; ++j) {
                cb_reserve_back(cb_rows, ct);
                noc_async_read(
                    get_noc_addr(row0 + j * dw * row_bytes),
                    get_write_ptr(cb_rows),
                    32 * row_bytes
                );
                noc_async_read_barrier();
                cb_push_back(cb_rows, ct);
            }
        }
        if (m + 1 < m1) {
            fetch(m + 1, set ^ 1);
        }
    }
    noc_async_read_barrier();
}
