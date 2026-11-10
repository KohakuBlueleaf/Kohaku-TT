// Softmax writer: per tile row (index from the reader) the group's (max, sum) exchange
// when g > 1, then the core's y tiles from the row's rotation.
#include <stdint.h>

#include "../ktt_norm/dm.hpp"
#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

// Compile-time: y's accessor, common arg count. Common: y, Wt, tile rows, g, data /
// ready semaphores, tiles per flush. Per core: group, index, first, seg, peers.
void kernel_main() {
    constexpr auto y_args = TensorAccessorArgs<0>();
    if (!ktt_softmax::args_ok<y_args.next_compile_time_args_offset()>(8)) {
        return;
    }
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t wt = get_common_arg_val<uint32_t>(1);
    const uint32_t rows = get_common_arg_val<uint32_t>(2);
    const uint32_t gsize = get_common_arg_val<uint32_t>(3);
    const uint32_t block = get_common_arg_val<uint32_t>(6);
    const uint32_t index = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t seg = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_gather = 4, cb_y = 16, cb_part = 17, cb_row = 25;
    const uint32_t y_tile = get_tile_size(cb_y);
    const auto y = TensorAccessor(y_args, y_addr, y_tile);

    const ktt_norm::Exchange ex{
        gsize,
        index,
        get_arg_addr(4),
        get_semaphore(get_common_arg_val<uint32_t>(4)),
        get_semaphore(get_common_arg_val<uint32_t>(5))
    };
    const uint32_t gather = get_write_ptr(cb_gather);
    if (gsize > 1) {
        // Receive tiles zeroed before any peer writes.
        ktt_norm::zero_l1(gather, 4 * ktt_norm::kF32Tile);
        ex.ready();
    }

    for (uint32_t k = 0;; ++k) {
        const uint32_t r = ktt_norm::take_word(cb_row);
        if (r >= rows) {
            break;
        }
        if (gsize > 1) {
            cb_wait_front(cb_part, 2);
            ex.send(
                get_read_ptr(cb_part), gather + (k & 1) * 2 * ktt_norm::kF32Tile, 2
            );
            cb_pop_front(cb_part, 2);
            cb_reserve_back(cb_gather, 2);
            ex.wait(k + 1);
            cb_push_back(cb_gather, 2);
        }
        const uint32_t page = r * wt + first;
        const auto& fifo = get_local_cb_interface(cb_y);
        // L1 address of the t-th unpopped y tile (the ring wraps).
        const auto at = [&](uint32_t t) {
            uint32_t a = get_read_ptr(cb_y) + t * y_tile;
            return a >= fifo.fifo_limit ? a - fifo.fifo_size : a;
        };
        uint32_t pending = 0, j = r % seg;
        for (uint32_t i = 0; i < seg; ++i) {
            cb_wait_front(cb_y, pending + 1);
            noc_async_write_page(page + j, y, at(pending));
            ++pending;
            if (pending >= block || i + 1 == seg) {
                noc_async_writes_flushed();
                // One page per pop: a pop wraps only exactly at the end of the ring.
                for (; pending > 0; --pending) {
                    cb_pop_front(cb_y, 1);
                }
            }
            j = j + 1 == seg ? 0 : j + 1;
        }
    }
    noc_async_write_barrier();
}
