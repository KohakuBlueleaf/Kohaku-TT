// Weight-gradient reader. Per unit (column group of nt output tiles from n0, chunk of
// output row tiles [m0, m1)): the chunk's grad rows (the group's output channels of
// each output pixel row, [32 pixels][n 32 channels], CB 1, nt pages per row tile; rows
// past the last pixel row zero), then per tap (i, j) and row tile m the tap's 32 input
// rows (the group's input channel tiles, as the forward reader gathers them, CB 0, K
// pages per row tile). Window mode (sw = 1): per tap row, each row tile's 32 + spread
// rows read once into CB 3, the row's taps copied from there.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "taps.hpp"

// Compile-time args: TensorAccessorArgs of the input rows, then of the grad rows, then
// the argument-layout version. Common runtime args: version, input rows address, grad
// rows address, Wp, Hp Wp, input rows in all, input row bytes, input rows per page,
// grad row bytes, grad rows per page, pixel rows (N OH pitch), rows per image (OH
// pitch), pitch, sh, sw, window mode (0 / 1), window span (rows), then the split (unit
// step, units, chunks, row tiles per chunk, row tiles, Ot, Kmax, Cg, Og, O, KH, KW, K,
// nt), then axis h and axis w (taps.hpp). Per core: first unit.
void kernel_main() {
    using namespace ktt_conv2;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    if (!args_match<g_args.next_compile_time_args_offset()>()) {
        return;
    }
    uint32_t a = 1;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t wp = get_common_arg_val<uint32_t>(a++);
    const uint32_t in_per_image = get_common_arg_val<uint32_t>(a++);
    const uint32_t in_rows = get_common_arg_val<uint32_t>(a++);
    const uint32_t x_row_bytes = get_common_arg_val<uint32_t>(a++);
    const uint32_t x_per_page = get_common_arg_val<uint32_t>(a++);
    const uint32_t g_row_bytes = get_common_arg_val<uint32_t>(a++);
    const uint32_t g_per_page = get_common_arg_val<uint32_t>(a++);
    const uint32_t valid_rows = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_per_image = get_common_arg_val<uint32_t>(a++);
    const uint32_t pitch = get_common_arg_val<uint32_t>(a++);
    const uint32_t sh = get_common_arg_val<uint32_t>(a++);
    const uint32_t sw = get_common_arg_val<uint32_t>(a++);
    const uint32_t window = get_common_arg_val<uint32_t>(a++);
    const uint32_t span_max = get_common_arg_val<uint32_t>(a++);
    const uint32_t step = get_common_arg_val<uint32_t>(a++);
    const uint32_t units = get_common_arg_val<uint32_t>(a++);
    const uint32_t chunks = get_common_arg_val<uint32_t>(a++);
    const uint32_t per_chunk = get_common_arg_val<uint32_t>(a++);
    const uint32_t mt = get_common_arg_val<uint32_t>(a++);
    Shape s{};
    s.ot = get_common_arg_val<uint32_t>(a++);
    s.kmax = get_common_arg_val<uint32_t>(a++);
    s.cg = get_common_arg_val<uint32_t>(a++);
    s.og = get_common_arg_val<uint32_t>(a++);
    s.o = get_common_arg_val<uint32_t>(a++);
    s.kh = get_common_arg_val<uint32_t>(a++);
    s.kw = get_common_arg_val<uint32_t>(a++);
    s.kpages = get_common_arg_val<uint32_t>(a++);
    s.nt_max = get_common_arg_val<uint32_t>(a++);
    s.axis_h = a;
    s.axis_w = a + axis_args(a);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t rows_per_image = out_per_image / pitch;
    const uint32_t x_shift = __builtin_ctz(x_per_page),
                   g_shift = __builtin_ctz(g_per_page);

    constexpr uint32_t cb_rows = 0, cb_g = 1, cb_window = 3;
    uint32_t window_base = 0;
    if (window) {
        cb_reserve_back(cb_window, 1);
        window_base = get_write_ptr(cb_window);
    }
    const auto x = TensorAccessor(x_args, x_addr, x_per_page * x_row_bytes);
    const auto gy = TensorAccessor(g_args, g_addr, g_per_page * g_row_bytes);
    uint32_t roff[kMaxTaps], ridx[kMaxTaps], coff[kMaxTaps], cidx[kMaxTaps];
    const uint32_t nh = axis_taps(s.axis_h, 0, roff, ridx);
    const uint32_t nw = axis_taps(s.axis_w, 0, coff, cidx);

    for (uint32_t u = start; u < units; u += step) {
        const uint32_t n0 = (u / chunks) * s.nt_max, sc = u % chunks;
        const uint32_t nt = s.ot - n0 < s.nt_max ? s.ot - n0 : s.nt_max;
        const uint32_t m0 = sc * per_chunk;
        const uint32_t m1 = m0 + per_chunk < mt ? m0 + per_chunk : mt;
        uint32_t kc0, kc1, unused;
        channel_tiles(n0, s, kc0, unused);
        channel_tiles(n0 + nt - 1, s, unused, kc1);

        // Grad rows: [32 pixels][nt 32 channels] per row tile.
        const uint32_t g_seg = nt * 64, g_offset = n0 * 64;
        const bool g_full = g_seg == g_row_bytes;
        for (uint32_t m = m0; m < m1; ++m) {
            cb_reserve_back(cb_g, s.nt_max);
            const uint32_t dst = get_write_ptr(cb_g);
            uint32_t q = 0;
            while (q < 32) {
                const uint32_t r = m * 32 + q;
                if (r >= valid_rows) {
                    auto* z = reinterpret_cast<volatile uint32_t*>(dst + q * g_seg);
                    for (uint32_t e = 0; e < (32 - q) * g_seg / 4; ++e) {
                        z[e] = 0;
                    }
                    break;
                }
                // A run of rows within one page (full rows), else one row.
                uint32_t len = 1;
                if (g_full) {
                    const uint32_t page_end = ((r >> g_shift) + 1) << g_shift;
                    len = 32 - q;
                    len = r + len > page_end ? page_end - r : len;
                    len = r + len > valid_rows ? valid_rows - r : len;
                }
                noc_async_read(
                    gy.get_noc_addr(
                        r >> g_shift, (r & (g_per_page - 1)) * g_row_bytes + g_offset
                    ),
                    dst + q * g_seg,
                    len * g_seg
                );
                q += len;
            }
            noc_async_read_barrier();
            cb_push_back(cb_g, s.nt_max);
        }

        // Input rows per tap and row tile.
        const uint32_t seg = (kc1 - kc0) * 64, seg_offset = kc0 * 64;
        const bool full = seg == x_row_bytes;
        // The segment of rows q = 0 .. count - 1 of row tile m at (ro, co) into dst.
        // Output row r = (image, y, x) reads input row image Hp Wp + (y sh + ro) Wp + x
        // sw + co, walked incrementally; rows past the last pixel row repeat it,
        // sources past the input clamp to its last row. Full rows: one read per run of
        // consecutive rows in a page.
        const auto read_rows =
            [&](uint32_t m, uint32_t count, uint32_t ro, uint32_t co, uint32_t dst) {
                uint32_t r = m * 32;
                r = r < valid_rows ? r : valid_rows - 1;
                uint32_t image = r / out_per_image, y = (r % out_per_image) / pitch;
                uint32_t xo = r % pitch;
                uint32_t base = image * in_per_image + (y * sh + ro) * wp + co;
                uint32_t head = 0, len = 0, at = dst;
                const auto issue = [&] {
                    noc_async_read(
                        x.get_noc_addr(
                            head >> x_shift,
                            (head & (x_per_page - 1)) * x_row_bytes + seg_offset
                        ),
                        at,
                        len * seg
                    );
                    at += len * seg;
                };
                for (uint32_t q = 0; q < count; ++q) {
                    uint32_t src = base + xo * sw;
                    src = src < in_rows ? src : in_rows - 1;
                    if (full && len > 0 && src == head + len &&
                        (src >> x_shift) == (head >> x_shift)) {
                        ++len;
                    } else {
                        if (len > 0) {
                            issue();
                        }
                        head = src;
                        len = 1;
                    }
                    if (r + 1 < valid_rows) {
                        ++r;
                        if (++xo == pitch) {
                            xo = 0;
                            if (++y == rows_per_image) {
                                y = 0;
                                ++image;
                            }
                            base = image * in_per_image + (y * sh + ro) * wp + co;
                        }
                    }
                }
                if (len > 0) {
                    issue();
                }
            };
        uint32_t lo = 0xFFFFFFFFu, hi = 0;
        for (uint32_t tb = 0; tb < nw; ++tb) {
            lo = coff[tb] < lo ? coff[tb] : lo;
            hi = coff[tb] > hi ? coff[tb] : hi;
        }
        for (uint32_t ta = 0; ta < nh; ++ta) {
            if (window) {
                // Row tile m's window of tap row ta: 32 + hi - lo rows from column
                // offset lo, at window_base + (m - m0) span_max seg.
                for (uint32_t m = m0; m < m1; ++m) {
                    read_rows(
                        m,
                        32 + hi - lo,
                        roff[ta],
                        lo,
                        window_base + (m - m0) * span_max * seg
                    );
                }
                noc_async_read_barrier();
            }
            for (uint32_t tb = 0; tb < nw; ++tb) {
                for (uint32_t m = m0; m < m1; ++m) {
                    cb_reserve_back(cb_rows, s.kpages);
                    if (window) {
                        noc_async_read(
                            get_noc_addr(
                                window_base +
                                ((m - m0) * span_max + coff[tb] - lo) * seg
                            ),
                            get_write_ptr(cb_rows),
                            32 * seg
                        );
                    } else {
                        read_rows(m, 32, roff[ta], coff[tb], get_write_ptr(cb_rows));
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_rows, s.kpages);
                }
            }
        }
    }
}
