// Norm forward writer: per tile row (index from the reader, until one >= the tile rows)
// the group's statistics exchange (groups of more than one core), the row's mean / rstd
// tiles (first core of the group), then the core's y tiles in the compute's pair order,
// flushed every `block` tiles.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"
#include "dm.hpp"

// Compile-time args: TensorAccessorArgs of y, mean, rstd (of y when absent), then the
// common arg count. Common runtime args: y, mean, rstd addresses, Wt, tile rows,
// groups, group size g, stats (bit 0 mean, bit 1 rstd), data / ready semaphore ids,
// tiles per flush. Per core: group, index in group, first column tile, column tiles,
// then g peer (x, y) NoC coordinates.
void kernel_main() {
    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto
        m_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto
        r_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    if (!ktt_norm::args_ok<r_args.next_compile_time_args_offset()>(12)) {
        return;
    }
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t mean_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t rstd_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t wt = get_common_arg_val<uint32_t>(3);
    const uint32_t rows = get_common_arg_val<uint32_t>(4);
    const uint32_t gsize = get_common_arg_val<uint32_t>(6);
    const uint32_t stats = get_common_arg_val<uint32_t>(7);
    const uint32_t block = get_common_arg_val<uint32_t>(10);
    const uint32_t index = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t seg = get_arg_val<uint32_t>(3);
    const uint32_t pairs = (seg + 1) / 2;

    constexpr uint32_t cb_gather = 4, cb_count = 7, cb_y = 16, cb_part = 17, cb_n = 18;
    constexpr uint32_t cb_so = 19, cb_row = 25;
    const uint32_t y_tile = get_tile_size(cb_y);
    const auto y = TensorAccessor(y_args, y_addr, y_tile);
    const auto mean = TensorAccessor(m_args, mean_addr, ktt_norm::kF32Tile);
    const auto rstd = TensorAccessor(r_args, rstd_addr, ktt_norm::kF32Tile);

    const ktt_norm::Exchange ex{
        gsize,
        index,
        get_arg_addr(4),
        get_semaphore(get_common_arg_val<uint32_t>(8)),
        get_semaphore(get_common_arg_val<uint32_t>(9))
    };
    const uint32_t gather = get_write_ptr(cb_gather);
    if (gsize > 1) {
        // Receive tiles zeroed before any peer writes; then the count rows, once.
        ktt_norm::zero_l1(gather, 4 * ktt_norm::kF32Tile);
        cb_reserve_back(cb_count, 1);
        const uint32_t count = get_write_ptr(cb_count);
        ktt_norm::zero_l1(count, ktt_norm::kF32Tile);
        ex.ready();
        cb_wait_front(cb_n, 1);
        ex.send(get_read_ptr(cb_n), count, 1);
        cb_pop_front(cb_n, 1);
        ex.wait(1);
        cb_push_back(cb_count, 1);
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
            ex.wait(k + 2);
            cb_push_back(cb_gather, 2);
        }
        if (stats != 0) {
            cb_wait_front(cb_so, 2);
            if (index == 0) {
                if (stats & 1) {
                    noc_async_write_page(r, mean, get_read_ptr(cb_so));
                }
                if (stats & 2) {
                    noc_async_write_page(
                        r, rstd, get_read_ptr(cb_so) + ktt_norm::kF32Tile
                    );
                }
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_so, 2);
        }
        const uint32_t page = r * wt + first;
        const auto& fifo = get_local_cb_interface(cb_y);
        // L1 address of the t-th unpopped y tile (the ring wraps).
        const auto at = [&](uint32_t t) {
            uint32_t a = get_read_ptr(cb_y) + t * y_tile;
            return a >= fifo.fifo_limit ? a - fifo.fifo_size : a;
        };
        uint32_t pending = 0, p = r % pairs;
        for (uint32_t i = 0; i < pairs; ++i) {
            const uint32_t j = 2 * p;
            const uint32_t n = j + 1 < seg ? 2 : 1;
            cb_wait_front(cb_y, pending + n);
            noc_async_write_page(page + j, y, at(pending));
            if (n == 2) {
                noc_async_write_page(page + j + 1, y, at(pending + 1));
            }
            pending += n;
            if (pending >= block || i + 1 == pairs) {
                noc_async_writes_flushed();
                // One page per pop: a pop wraps only exactly at the end of the ring.
                for (; pending > 0; --pending) {
                    cb_pop_front(cb_y, 1);
                }
            }
            p = p + 1 == pairs ? 0 : p + 1;
        }
    }
    noc_async_write_barrier();
}
