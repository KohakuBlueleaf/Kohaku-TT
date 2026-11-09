// Norm forward reader: once, the core's gamma / beta column tiles; then per tile row the
// row's index to the compute and the writer, and the core's segment of x, read starting at
// the row's rotation (DRAM bank spread). Rows: the group's first, then either every
// groups-th (static) or the next one of a counter shared by all cores (dynamic); an index
// >= the tile rows ends the compute and the writer.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"
#include "dm.hpp"

// Compile-time args: TensorAccessorArgs of x, gamma, beta (of x when absent), then the
// common arg count. Common runtime args: x, gamma, beta addresses, Wt, tile rows, groups,
// has gamma, has beta, the x ring's pages per row, dynamic rows, the counter core's NoC
// (x, y), the counter semaphore id. Per core: group, first column tile, column tiles.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    if (!ktt_norm::args_ok<b_args.next_compile_time_args_offset()>(14)) {
        return;
    }
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t wt = get_common_arg_val<uint32_t>(3);
    const uint32_t rows = get_common_arg_val<uint32_t>(4);
    const uint32_t groups = get_common_arg_val<uint32_t>(5);
    const bool has_gamma = get_common_arg_val<uint32_t>(6) != 0;
    const bool has_beta = get_common_arg_val<uint32_t>(7) != 0;
    const uint32_t stride = get_common_arg_val<uint32_t>(8);
    const bool dynamic = get_common_arg_val<uint32_t>(9) != 0;
    const uint64_t counter = get_noc_addr(
        get_common_arg_val<uint32_t>(10), get_common_arg_val<uint32_t>(11),
        get_semaphore(get_common_arg_val<uint32_t>(12))
    );
    const uint32_t gid = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t seg = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_x = 0, cb_g = 1, cb_b = 2, cb_row_c = 24, cb_row_w = 25;
    const uint32_t x_tile = get_tile_size(cb_x);
    const auto x = TensorAccessor(x_args, x_addr, x_tile);

    const auto load = [&](const auto& args, uint32_t addr, uint32_t cb) {
        const uint32_t tile = get_tile_size(cb);
        const auto t = TensorAccessor(args, addr, tile);
        cb_reserve_back(cb, seg);
        const uint32_t base = get_write_ptr(cb);
        for (uint32_t j = 0; j < seg; ++j) {
            noc_async_read_page(first + j, t, base + j * tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb, seg);
    };
    if (has_gamma) {
        load(g_args, g_addr, cb_g);
    }
    if (has_beta) {
        load(b_args, b_addr, cb_b);
    }

    uint32_t r = gid;
    while (true) {
        ktt_norm::put_word(cb_row_c, r);
        ktt_norm::put_word(cb_row_w, r);
        if (r >= rows) {
            break;
        }
        {
            DeviceZoneScopedN("slot-wait");
            cb_reserve_back(cb_x, stride);
        }
        const uint32_t base = get_write_ptr(cb_x);
        const uint32_t page = r * wt + first;
        uint32_t j = r % seg;
        for (uint32_t i = 0; i < seg; ++i) {
            noc_async_read_page(page + j, x, base + j * x_tile);
            j = j + 1 == seg ? 0 : j + 1;
        }
        const uint32_t next = dynamic ? ktt_norm::fetch_add(counter) : r + groups;
        {
            DeviceZoneScopedN("read-wait");
            noc_async_read_barrier();
        }
        cb_push_back(cb_x, stride);
        r = next;
    }
}
