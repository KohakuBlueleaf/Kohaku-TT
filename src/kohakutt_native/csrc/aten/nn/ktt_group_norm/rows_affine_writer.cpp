// Row-form group norm affine writer: per piece (rows.hpp) and block, the piece's y
// tiles (CB 16) to their pages; the last image's last row tile first gets its padding
// rows zeroed.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "rows.hpp"
#include "rows_dm.hpp"

// Compile-time args: TensorAccessorArgs of y, then of the two row outputs (of y when
// not wanted). Common runtime args: y address, channel tiles Ct, rows per image,
// images, block, rows wanted, the two row outputs' addresses (fp32 TILE [images, C]:
// row n of the pair of tiles in CB 12, from the piece starting each segment). Per core:
// q0, q1.
void kernel_main() {
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const ktt_group_norm::Geometry g{
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3)
    };
    const uint32_t block = get_common_arg_val<uint32_t>(4);
    const bool sums_out = get_common_arg_val<uint32_t>(5) != 0;
    const uint32_t q1 = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_stats = 12, cb_y = 16;
    const uint32_t tile = get_tile_size(cb_y);
    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto
        s1_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto
        s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    const auto y = TensorAccessor(y_args, y_addr, tile);
    const auto s1 = TensorAccessor(
        s1_args, get_common_arg_val<uint32_t>(6), ktt_group_norm::kF32Tile
    );
    const auto s2 = TensorAccessor(
        s2_args, get_common_arg_val<uint32_t>(7), ktt_group_norm::kF32Tile
    );
    const uint32_t last_rows = g.tail_rows(g.images - 1);

    for (uint32_t q = get_arg_val<uint32_t>(0); q < q1;) {
        const ktt_group_norm::Piece p(g, q, q1);
        if (sums_out && p.i == 0) {
            // Row 0 of the segment's two sum tiles to row n of the outputs' tile (n /
            // 32, cb).
            cb_wait_front(cb_stats, 2);
            const uint32_t src = get_read_ptr(cb_stats), r = p.n % 32;
            const uint32_t row = (r >> 4) * 2048 + (r & 15) * 64,
                           page = (p.n / 32) * g.ct + p.cb;
            noc_async_write(src, s1.get_noc_addr(page, row), 64);
            noc_async_write(src + 1024, s1.get_noc_addr(page, row + 1024), 64);
            noc_async_write(
                src + ktt_group_norm::kF32Tile, s2.get_noc_addr(page, row), 64
            );
            noc_async_write(
                src + ktt_group_norm::kF32Tile + 1024,
                s2.get_noc_addr(page, row + 1024),
                64
            );
            noc_async_write_barrier();
            cb_pop_front(cb_stats, 2);
        }
        for (uint32_t t = 0; t < p.len; t += block) {
            const uint32_t n = p.len - t < block ? p.len - t : block;
            cb_wait_front(cb_y, block);
            const uint32_t base = get_read_ptr(cb_y);
            for (uint32_t k = 0; k < n; ++k) {
                if (last_rows < 32 && p.n + 1 == g.images && p.ends_image(g, t + k)) {
                    ktt_group_norm::zero_rows(
                        base + k * tile, last_rows, 32, tile / 1024
                    );
                }
                noc_async_write_page(p.page(g, t + k), y, base + k * tile);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_y, block);
        }
        q += p.len;
    }
}
