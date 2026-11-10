// Norm backward writer: per tile row, the group's exchange of the row sums A, B (groups
// of more than one core) and the dx tiles in the compute's order, flushed every `block`
// tiles; at the end the core's dgamma / dbeta partial tiles (rows of the core's tile
// rows summed).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"
#include "dm.hpp"

// Compile-time args: TensorAccessorArgs of dx, dgamma partials, dbeta partials (of an
// output present when absent), then the common arg count. Common runtime args: dx,
// dgamma partials, dbeta partials addresses, Wt, tile rows, groups, group size g, want
// dx, want dgamma, want dbeta, data / ready semaphore ids, tiles per flush. Per core:
// group, index in group, first column tile, column tiles, write rotation, then g peer
// (x, y) NoC coordinates.
void kernel_main() {
    constexpr auto dx_args = TensorAccessorArgs<0>();
    constexpr auto
        pg_args = TensorAccessorArgs<dx_args.next_compile_time_args_offset()>();
    constexpr auto
        pb_args = TensorAccessorArgs<pg_args.next_compile_time_args_offset()>();
    if (!ktt_norm::args_ok<pb_args.next_compile_time_args_offset()>(14)) {
        return;
    }
    const uint32_t dx_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t pg_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t pb_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t wt = get_common_arg_val<uint32_t>(3);
    const uint32_t rows = get_common_arg_val<uint32_t>(4);
    const uint32_t groups = get_common_arg_val<uint32_t>(5);
    const uint32_t gsize = get_common_arg_val<uint32_t>(6);
    const bool want_dx = get_common_arg_val<uint32_t>(7) != 0;
    const bool want_dg = get_common_arg_val<uint32_t>(8) != 0;
    const bool want_db = get_common_arg_val<uint32_t>(9) != 0;
    const uint32_t block = get_common_arg_val<uint32_t>(12);
    const uint32_t gid = get_arg_val<uint32_t>(0);
    const uint32_t index = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t seg = get_arg_val<uint32_t>(3);
    const uint32_t rot = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_gather = 4, cb_dx = 16, cb_part = 17, cb_dg = 21, cb_db = 22;
    const uint32_t dx_tile = get_tile_size(cb_dx);
    const auto dx = TensorAccessor(dx_args, dx_addr, dx_tile);
    const auto pg = TensorAccessor(pg_args, pg_addr, ktt_norm::kF32Tile);
    const auto pb = TensorAccessor(pb_args, pb_addr, ktt_norm::kF32Tile);

    const ktt_norm::Exchange ex{
        gsize,
        index,
        get_arg_addr(5),
        get_semaphore(get_common_arg_val<uint32_t>(10)),
        get_semaphore(get_common_arg_val<uint32_t>(11))
    };
    const uint32_t gather = get_write_ptr(cb_gather);
    if (gsize > 1 && want_dx) {
        ktt_norm::zero_l1(gather, 4 * ktt_norm::kF32Tile);
        ex.ready();
    }

    const auto& fifo = get_local_cb_interface(cb_dx);
    // L1 address of the t-th unpopped dx tile (the ring wraps).
    const auto at = [&](uint32_t t) {
        uint32_t a = get_read_ptr(cb_dx) + t * dx_tile;
        return a >= fifo.fifo_limit ? a - fifo.fifo_size : a;
    };
    uint32_t k = 0;
    for (uint32_t r = gid; r < rows; r += groups, ++k) {
        if (!want_dx) {
            continue;
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
        uint32_t pending = 0, j = rot;
        for (uint32_t i = 0; i < seg; ++i) {
            cb_wait_front(cb_dx, pending + 1);
            noc_async_write_page(page + j, dx, at(pending));
            ++pending;
            if (pending == block || i + 1 == seg) {
                noc_async_writes_flushed();
                // One page per pop: a pop wraps only exactly at the end of the ring.
                for (; pending > 0; --pending) {
                    cb_pop_front(cb_dx, 1);
                }
            }
            j = j + 1 == seg ? 0 : j + 1;
        }
    }
    // Partial sums: page gid Wt + first + j of the [groups * 32, W] fp32 tensors.
    const uint32_t page = gid * wt + first;
    if (want_dg) {
        cb_wait_front(cb_dg, seg);
        for (uint32_t j = 0; j < seg; ++j) {
            noc_async_write_page(
                page + j, pg, get_read_ptr(cb_dg) + j * ktt_norm::kF32Tile
            );
        }
    }
    if (want_db) {
        cb_wait_front(cb_db, seg);
        for (uint32_t j = 0; j < seg; ++j) {
            noc_async_write_page(
                page + j, pb, get_read_ptr(cb_db) + j * ktt_norm::kF32Tile
            );
        }
    }
    noc_async_write_barrier();
}
