// Gather conv reader. Per work item (row tiles [m0, m1), column tiles [n0, n0 + nt)):
// the item's sparse weight block (per output column tile, per tap, the channel tiles
// of its groups) into CB 1 (packed weights: the writer fills CB 1); then per output
// row tile m (phase f = m / phase tiles) and
// tap (a, b) of f, the 32 input rows the tap reads, only the item's channel tiles of
// each, into CB 0 (K pages per tap), row-major, for the compute kernel to tilize.
// Output row r = (image, y, x) of phase f reads input row image Hp Wp + (y sh +
// roff[a]) Wp + x sw + coff[b]. Window mode: per m and tap row a, the Sw rows of each
// of 32 + spread / Sw output positions read once into a CB 3 page, which the compute
// tilizes.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "taps.hpp"

// Compile-time args: TensorAccessorArgs of x_rows, then of w, then the argument-layout
// version. Common runtime args: version, x address, w address, Wp, Hp Wp, input rows in
// all, x row bytes, rows per x page, window mode (0 / 1), rows per phase (N OHq pitch),
// rows per image (OHq pitch), pitch, sh, sw, window span, packed weights (0 / 1: the
// writer loads them), taps per read barrier (G; CB 0 holds 2 G taps), then the shape
// (see taps.hpp). Per core: items, then an Item (taps.hpp) per item.
void kernel_main() {
    using namespace ktt_conv2;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        w_args = (TensorAccessorArgs<x_args.next_compile_time_args_offset()>());
    if (!args_match<w_args.next_compile_time_args_offset()>()) {
        return;
    }
    uint32_t a = 1;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(a++);
    const uint32_t wp = get_common_arg_val<uint32_t>(a++);
    const uint32_t in_per_image = get_common_arg_val<uint32_t>(a++);
    const uint32_t in_rows = get_common_arg_val<uint32_t>(a++);
    const uint32_t row_bytes = get_common_arg_val<uint32_t>(a++);
    const uint32_t per_page = get_common_arg_val<uint32_t>(a++);
    const uint32_t window = get_common_arg_val<uint32_t>(a++);
    const uint32_t valid_rows = get_common_arg_val<uint32_t>(a++);
    const uint32_t out_per_image = get_common_arg_val<uint32_t>(a++);
    const uint32_t pitch = get_common_arg_val<uint32_t>(a++);
    const uint32_t sh = get_common_arg_val<uint32_t>(a++);
    const uint32_t sw = get_common_arg_val<uint32_t>(a++);
    const uint32_t span_max = get_common_arg_val<uint32_t>(a++);
    const uint32_t packed = get_common_arg_val<uint32_t>(a++);
    const uint32_t group = get_common_arg_val<uint32_t>(a++);
    const Shape s = read_shape(a);
    const uint32_t rows_per_image = out_per_image / pitch;
    const uint32_t page_shift = __builtin_ctz(per_page);
    // Window mode: Sw input rows per output position (a tap's rows are every Sw-th row
    // of its window from its column offset on); a window page of span_max rows of K
    // tiles.
    const uint32_t per = window ? sw : 1;
    const uint32_t window_page = span_max * s.kpages * 64;

    constexpr uint32_t cb_rows = 0, cb_w = 1, cb_window = 3;
    const uint32_t kTileBytes = get_tile_size(cb_rows);
    const auto x = TensorAccessor(x_args, x_addr, per_page * row_bytes);
    const uint32_t tile = get_tile_size(cb_w);
    const uint32_t taps = s.kh * s.kw;
    const auto w = TensorAccessor(w_args, w_addr, tile);

    Phase ph;
    const uint32_t items = get_arg_val<uint32_t>(0);
    for (uint32_t it = 0; it < items; ++it) {
        const Item item = read_item(it);
        const uint32_t m0 = item.m0, m1 = item.m1, n0 = item.n0, nt = item.nt;

        // Weight block: tile (jl, t, k) at CB position (jl taps + t) Kmax + k.
        if (!packed) {
            // Only the taps of the item's tap rows (a split item: kernel rows a0 .. a1
            // - 1).
            const bool rows = item.a0 > 0 || item.a1 < s.rows_h;
            const uint32_t t0 = rows ? item.a0 * s.kw : 0;
            const uint32_t t1 = rows ? item.a1 * s.kw : taps;
            cb_reserve_back(cb_w, s.nt_max * taps * s.kmax);
            const uint32_t w_base = get_write_ptr(cb_w);
            for (uint32_t jl = 0; jl < nt; ++jl) {
                uint32_t k0, k1;
                channel_tiles(n0 + jl, s, k0, k1);
                for (uint32_t t = t0; t < t1; ++t) {
                    for (uint32_t k = 0; k < k1 - k0; ++k) {
                        noc_async_read_page(
                            ((n0 + jl) * taps + t) * s.kmax + k,
                            w,
                            w_base + ((jl * taps + t) * s.kmax + k) * tile
                        );
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, s.nt_max * taps * s.kmax);
        }

        // The item's channel tiles: the union over its output column tiles.
        uint32_t kc0, kc1, unused;
        channel_tiles(n0, s, kc0, unused);
        channel_tiles(n0 + nt - 1, s, unused, kc1);
        const uint32_t seg = (kc1 - kc0) * 64, seg_offset = kc0 * 64;
        const bool full = seg == row_bytes;

        // The item's segment of `per` input rows for each output position q = 0 ..
        // count - 1 from phase-local output row tile `local`, at tap row offset ro,
        // column offset co, into dst (seg bytes each). Output row r = (image, y, x)
        // reads input rows image Hp Wp + (y sh + ro) Wp + x sw + co + e, e < per,
        // walked incrementally; rows past the last output row repeat it, sources past
        // the input clamp to its last row. Full rows: one read per run of consecutive
        // rows in a page.
        const auto read_rows = [&](uint32_t local,
                                   uint32_t count,
                                   uint32_t ro,
                                   uint32_t co,
                                   uint32_t dst,
                                   uint32_t per) {
            uint32_t r = local * 32;
            r = r < valid_rows ? r : valid_rows - 1;
            uint32_t image = r / out_per_image, y = (r % out_per_image) / pitch;
            uint32_t xo = r % pitch;
            uint32_t base = image * in_per_image + (y * sh + ro) * wp + co;
            uint32_t start = 0, len = 0, at = dst;
            const auto issue = [&] {
                noc_async_read(
                    x.get_noc_addr(
                        start >> page_shift,
                        (start & (per_page - 1)) * row_bytes + seg_offset
                    ),
                    at,
                    len * seg
                );
                at += len * seg;
            };
            for (uint32_t q = 0; q < count; ++q) {
                for (uint32_t e = 0; e < per; ++e) {
                    uint32_t src = base + xo * sw + e;
                    src = src < in_rows ? src : in_rows - 1;
                    if (full && len > 0 && src == start + len &&
                        (src >> page_shift) == (start >> page_shift)) {
                        ++len;
                    } else {
                        if (len > 0) {
                            issue();
                        }
                        start = src;
                        len = 1;
                    }
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

        if (window == 0) {
            for (uint32_t m = m0; m < m1; ++m) {
                load_phase(s, m, ph, item.a0, item.a1);
                const uint32_t local = m % s.phase_tiles;
                for (uint32_t ta = 0; ta < ph.nh; ++ta) {
                    // One barrier per group of taps (a tap row in one-phase gathers).
                    for (uint32_t tb = 0; tb < ph.nw; tb += group) {
                        const uint32_t n = ph.nw - tb < group ? ph.nw - tb : group;
                        cb_reserve_back(cb_rows, n * s.kpages);
                        const uint32_t dst = get_write_ptr(cb_rows);
                        for (uint32_t i = 0; i < n; ++i) {
                            read_rows(
                                local,
                                32,
                                ph.roff[ta],
                                ph.coff[tb + i],
                                dst + i * s.kpages * kTileBytes,
                                1
                            );
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_rows, n * s.kpages);
                    }
                }
            }
            continue;
        }

        // Windows: per row tile with taps, one CB 3 page per tap row (the page's rows
        // seg bytes apart), pushed together; the compute tilizes the taps from them.
        for (uint32_t m = m0; m < m1; ++m) {
            load_phase(s, m, ph, item.a0, item.a1);
            if (ph.nh * ph.nw == 0) {
                continue;
            }
            const uint32_t local = m % s.phase_tiles;
            const uint32_t positions = 32 + (ph.chi - ph.clo) / per;
            // rows_h pages per row tile (tap rows past nh unused), so pages never wrap.
            {
                DeviceZoneScopedN("r-free");
                cb_reserve_back(cb_window, s.rows_h);
            }
            DeviceZoneScopedN("r-win");
            const uint32_t base = get_write_ptr(cb_window);
            for (uint32_t ta = 0; ta < ph.nh; ++ta) {
                read_rows(
                    local, positions, ph.roff[ta], ph.clo, base + ta * window_page, per
                );
            }
            noc_async_read_barrier();
            cb_push_back(cb_window, s.rows_h);
        }
    }
}
