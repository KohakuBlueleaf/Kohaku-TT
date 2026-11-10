// Row-form group norm moments writer: per piece, row 0 of its two sum tiles (CB 16: sum
// x, sum x^2 per channel) to row k of pages 2 s and 2 s + 1 of the partials (fp32
// tiles), k the core's index among the cores of segment s. The sums of a row tile
// shared with the next image go to that image's segment, in the row after its own
// cores'.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "rows.hpp"

// Compile-time args: TensorAccessorArgs of the partials. Common runtime args: partials
// address, channel tiles Ct, rows per image, images. Per core: q0, q1, the core's index
// in its first segment, then per shared row tile the core count of the next image's
// segment.
void kernel_main() {
    const uint32_t part_addr = get_common_arg_val<uint32_t>(0);
    const ktt_group_norm::Geometry g{
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        get_common_arg_val<uint32_t>(3)
    };
    const uint32_t q1 = get_arg_val<uint32_t>(1);
    uint32_t k = get_arg_val<uint32_t>(2), item = 3;

    constexpr uint32_t cb_part = 16;
    constexpr auto part_args = TensorAccessorArgs<0>();
    const auto part = TensorAccessor(part_args, part_addr, ktt_group_norm::kF32Tile);

    // Row 0 of the two front tiles of CB 16 to row `index` of segment `segment`'s pages
    // (row r of an fp32 tile: columns 0-15 at 64 r, rows 16 on 2048 further; columns
    // 16-31 1024 bytes on).
    const auto send = [&](uint32_t segment, uint32_t index) {
        cb_wait_front(cb_part, 2);
        const uint32_t src = get_read_ptr(cb_part);
        const uint32_t row = (index >> 4) * 2048 + (index & 15) * 64;
        for (uint32_t t = 0; t < 2; ++t) {
            const uint32_t from = src + t * ktt_group_norm::kF32Tile;
            noc_async_write(from, part.get_noc_addr(2 * segment + t, row), 64);
            noc_async_write(
                from + 1024, part.get_noc_addr(2 * segment + t, row + 1024), 64
            );
        }
        noc_async_write_barrier();
        cb_pop_front(cb_part, 2);
    };

    for (uint32_t q = get_arg_val<uint32_t>(0); q < q1;) {
        const ktt_group_norm::Piece p(g, q, q1);
        send(p.segment(g), k);
        if (g.shared(p.n) && p.last) {
            send(p.segment(g) + g.ct, get_arg_val<uint32_t>(item++));
        }
        k = 0;
        q += p.len;
    }
}
