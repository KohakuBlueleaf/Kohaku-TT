// Row-form group norm moments reader: the core's x tiles in pieces (rows.hpp), `block`
// tiles per read barrier; every push is a whole block. An image's last row tile that
// also holds rows of the next image is read with those rows zeroed, and once more after
// the piece, as its own block, with the image's rows zeroed.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "rows.hpp"
#include "rows_dm.hpp"

// Compile-time args: TensorAccessorArgs of x. Common runtime args: x address, channel
// tiles Ct, rows per image, images, block. Per core: q0, q1.
void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const ktt_group_norm::Geometry g{
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3)
    };
    const uint32_t block = get_common_arg_val<uint32_t>(4);
    const uint32_t q1 = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = 0;
    const uint32_t tile = get_tile_size(cb_x);
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, x_addr, tile);

    for (uint32_t q = get_arg_val<uint32_t>(0); q < q1;) {
        const ktt_group_norm::Piece p(g, q, q1);
        const bool shared = g.shared(p.n);
        const uint32_t own = g.tail_rows(p.n);
        for (uint32_t t = 0; t < p.len; t += block) {
            const uint32_t n = p.len - t < block ? p.len - t : block;
            cb_reserve_back(cb_x, block);
            const uint32_t base = get_write_ptr(cb_x);
            for (uint32_t k = 0; k < n; ++k) {
                noc_async_read_page(p.page(g, t + k), x, base + k * tile);
            }
            noc_async_read_barrier();
            if (shared && p.ends_image(g, t + n - 1)) {
                ktt_group_norm::zero_rows(base + (n - 1) * tile, own, 32, tile / 1024);
            }
            cb_push_back(cb_x, block);
        }
        if (shared && p.last) {
            cb_reserve_back(cb_x, block);
            const uint32_t base = get_write_ptr(cb_x);
            noc_async_read_page(p.page(g, p.len - 1), x, base);
            noc_async_read_barrier();
            ktt_group_norm::zero_rows(base, 0, own, tile / 1024);
            cb_push_back(cb_x, block);
        }
        q += p.len;
    }
}
