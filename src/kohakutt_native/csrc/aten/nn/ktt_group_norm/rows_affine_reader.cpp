// Row-form group norm affine reader. Per piece (rows.hpp), the inputs of its
// statistics: for each channel tile its groups reach, that segment's two partial pages
// (rows past its core count zeroed) and group mask; its own mask and valid-channel
// tile; the weight and bias tiles when given. A piece ending in a row tile shared with
// the next image gets the same for the next image's segment, then two row-mask tiles.
// Then the piece's x tiles, `block` per read barrier; every x push is a whole block.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "rows.hpp"
#include "rows_dm.hpp"

// Compile-time args: TensorAccessorArgs of x, the partials, weight, bias (of x when
// absent). Common runtime args: x, partials, weight, bias addresses, channel tiles Ct,
// rows per image, images, block, has weight, has bias, channels per group, channels C.
// Per core: q0, q1, then the core count of each segment whose partial pages it reads,
// in reading order.
void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t part_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(3);
    const ktt_group_norm::Geometry g{
        get_common_arg_val<uint32_t>(4),
        get_common_arg_val<uint32_t>(5),
        get_common_arg_val<uint32_t>(6)
    };
    const uint32_t block = get_common_arg_val<uint32_t>(7);
    const bool has_w = get_common_arg_val<uint32_t>(8) != 0;
    const bool has_b = get_common_arg_val<uint32_t>(9) != 0;
    const uint32_t cg = get_common_arg_val<uint32_t>(10);
    const uint32_t c = get_common_arg_val<uint32_t>(11);
    const uint32_t q1 = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = 0, cb_w = 1, cb_b = 2, cb_gather = 3, cb_mask = 4,
                       cb_back = 8;
    constexpr uint32_t cb_rows = 10;
    constexpr uint32_t kOne = 0x3f800000u, kTile = ktt_group_norm::kF32Tile;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        p_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto
        w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto
        b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t tile = get_tile_size(cb_x);
    const auto x = TensorAccessor(x_args, x_addr, tile);
    const auto part = TensorAccessor(p_args, part_addr, kTile);
    const auto w = TensorAccessor(w_args, w_addr, get_tile_size(cb_w));
    const auto b = TensorAccessor(b_args, b_addr, get_tile_size(cb_b));

    uint32_t item = 2;
    // The statistics inputs of segment (image, cb).
    const auto statistics = [&](uint32_t image, uint32_t cb) {
        const ktt_group_norm::Span span(cb, cg, c);
        // Mask of channel tile t at `addr`: 1 at (channel, index of its group among the
        // groups of channel tile cb).
        const auto mask_of = [&](uint32_t t, uint32_t addr) {
            ktt_group_norm::zero_rows(addr, 0, 32, 4);
            auto* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
            for (uint32_t ch = 0; ch < 32 && t * 32 + ch < c; ++ch) {
                const uint32_t group = (t * 32 + ch) / cg;
                if (group >= span.g_lo && group <= span.g_hi) {
                    m[ktt_group_norm::at(ch, group - span.g_lo)] = kOne;
                }
            }
        };
        for (uint32_t t = span.t_lo; t <= span.t_hi; ++t) {
            const uint32_t cores = get_arg_val<uint32_t>(item++);
            const uint32_t seg = image * g.ct + t;
            cb_reserve_back(cb_gather, 2);
            const uint32_t gather = get_write_ptr(cb_gather);
            noc_async_read_page(2 * seg, part, gather);
            noc_async_read_page(2 * seg + 1, part, gather + kTile);
            noc_async_read_barrier();
            ktt_group_norm::zero_rows(gather, cores, 32, 4);
            ktt_group_norm::zero_rows(gather + kTile, cores, 32, 4);
            cb_push_back(cb_gather, 2);
            cb_reserve_back(cb_mask, 1);
            mask_of(t, get_write_ptr(cb_mask));
            cb_push_back(cb_mask, 1);
        }
        // The tile's own mask again, and the valid tile: 1 in row 0 at the valid
        // channels.
        cb_reserve_back(cb_back, 2);
        const uint32_t back = get_write_ptr(cb_back);
        mask_of(cb, back);
        ktt_group_norm::zero_rows(back + kTile, 0, 32, 4);
        auto* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(back + kTile);
        for (uint32_t ch = 0; ch < 32 && cb * 32 + ch < c; ++ch) {
            v[ktt_group_norm::at(0, ch)] = kOne;
        }
        cb_push_back(cb_back, 2);
        if (has_w) {
            cb_reserve_back(cb_w, 1);
            noc_async_read_page(cb, w, get_write_ptr(cb_w));
            noc_async_read_barrier();
            cb_push_back(cb_w, 1);
        }
        if (has_b) {
            cb_reserve_back(cb_b, 1);
            noc_async_read_page(cb, b, get_write_ptr(cb_b));
            noc_async_read_barrier();
            cb_push_back(cb_b, 1);
        }
    };

    for (uint32_t q = get_arg_val<uint32_t>(0); q < q1;) {
        const ktt_group_norm::Piece p(g, q, q1);
        statistics(p.n, p.cb);
        if (g.shared(p.n) && p.last) {
            statistics(p.n + 1, p.cb);
            // Row masks as columns (the compute broadcasts column 0): 1 at the image's
            // rows of the shared tile; 1 at the next image's.
            const uint32_t own = g.tail_rows(p.n);
            cb_reserve_back(cb_rows, 2);
            const uint32_t masks = get_write_ptr(cb_rows);
            ktt_group_norm::zero_rows(masks, 0, 32, 4);
            ktt_group_norm::zero_rows(masks + kTile, 0, 32, 4);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t tile_of = r < own ? masks : masks + kTile;
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    tile_of
                )[ktt_group_norm::at(r, 0)] = kOne;
            }
            cb_push_back(cb_rows, 2);
        }

        for (uint32_t t = 0; t < p.len; t += block) {
            const uint32_t n = p.len - t < block ? p.len - t : block;
            cb_reserve_back(cb_x, block);
            const uint32_t base = get_write_ptr(cb_x);
            for (uint32_t k = 0; k < n; ++k) {
                noc_async_read_page(p.page(g, t + k), x, base + k * tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, block);
        }
        q += p.len;
    }
}
