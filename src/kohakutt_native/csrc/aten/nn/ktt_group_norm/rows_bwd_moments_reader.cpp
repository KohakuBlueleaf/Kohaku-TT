// Row-form group norm backward, sums reader: the core's x and dy tiles in pieces
// (rows.hpp), `block` tiles of each per read barrier; every push is a whole block. An
// image's last row tile that also holds rows of the next image has those rows of dy
// zeroed, and both tiles follow once more after the piece, as their own block, with the
// image's rows of dy zeroed.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "rows.hpp"
#include "rows_dm.hpp"

// Compile-time args: TensorAccessorArgs of x, then of dy. Common runtime args: x
// address, dy address, channel tiles Ct, rows per image, images, block. Per core: q0,
// q1.
void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(1);
    const ktt_group_norm::Geometry g{
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3),
        get_common_arg_val<uint32_t>(4)
    };
    const uint32_t block = get_common_arg_val<uint32_t>(5);
    const uint32_t q1 = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = 0, cb_g = 1;
    const uint32_t tile = get_tile_size(cb_x);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, x_addr, tile);
    const auto dy = TensorAccessor(g_args, g_addr, tile);

    for (uint32_t q = get_arg_val<uint32_t>(0); q < q1;) {
        const ktt_group_norm::Piece p(g, q, q1);
        const bool shared = g.shared(p.n);
        const uint32_t own = g.tail_rows(p.n);
        for (uint32_t t = 0; t < p.len; t += block) {
            const uint32_t n = p.len - t < block ? p.len - t : block;
            cb_reserve_back(cb_x, block);
            cb_reserve_back(cb_g, block);
            const uint32_t xs = get_write_ptr(cb_x), gs = get_write_ptr(cb_g);
            for (uint32_t k = 0; k < n; ++k) {
                noc_async_read_page(p.page(g, t + k), x, xs + k * tile);
                noc_async_read_page(p.page(g, t + k), dy, gs + k * tile);
            }
            noc_async_read_barrier();
            if (shared && p.ends_image(g, t + n - 1)) {
                ktt_group_norm::zero_rows(gs + (n - 1) * tile, own, 32, tile / 1024);
            }
            cb_push_back(cb_x, block);
            cb_push_back(cb_g, block);
        }
        if (shared && p.last) {
            cb_reserve_back(cb_x, block);
            cb_reserve_back(cb_g, block);
            const uint32_t xs = get_write_ptr(cb_x), gs = get_write_ptr(cb_g);
            noc_async_read_page(p.page(g, p.len - 1), x, xs);
            noc_async_read_page(p.page(g, p.len - 1), dy, gs);
            noc_async_read_barrier();
            ktt_group_norm::zero_rows(gs, 0, own, tile / 1024);
            cb_push_back(cb_x, block);
            cb_push_back(cb_g, block);
        }
        q += p.len;
    }
}
