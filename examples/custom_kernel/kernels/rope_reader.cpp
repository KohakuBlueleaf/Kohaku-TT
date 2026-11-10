// RoPE reader: streams this core's tiles of x into CB 0 and the table tile of each into
// CB 1, BLOCK tiles per NoC barrier.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Runtime args: x address, table address, first tile, tile count, table tiles. Tile t of
// x takes table tile t % (table tiles): the table repeats over x's leading dims.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto table_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t period = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = 0, cb_table = 1;
    const uint32_t x_page = get_tile_size(cb_x), table_page = get_tile_size(cb_table);
    const auto x = TensorAccessor(x_args, x_addr, x_page);
    const auto table = TensorAccessor(table_args, table_addr, table_page);

    for (uint32_t done = 0; done < count; done += BLOCK) {
        const uint32_t n = count - done < BLOCK ? count - done : BLOCK;
        cb_reserve_back(cb_x, BLOCK);
        cb_reserve_back(cb_table, BLOCK);
        const uint32_t x_l1 = get_write_ptr(cb_x), table_l1 = get_write_ptr(cb_table);
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t t = start + done + k;
            noc_async_read_page(t, x, x_l1 + k * x_page);
            noc_async_read_page(t % period, table, table_l1 + k * table_page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, BLOCK);
        cb_push_back(cb_table, BLOCK);
    }
}
