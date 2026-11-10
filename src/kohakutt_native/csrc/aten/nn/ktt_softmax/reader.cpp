// Softmax reader: per tile row the row's index to the compute and the writer, then the
// core's segment of x from the row's rotation; rows static or from a shared counter.
#include <stdint.h>

#include "../ktt_norm/dm.hpp"
#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

// Compile-time: x's accessor, common arg count. Common: x, Wt, tile rows, groups, pages
// per row, dynamic, counter core (x, y), its semaphore. Per core: group, first, seg.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    if (!ktt_softmax::args_ok<x_args.next_compile_time_args_offset()>(10)) {
        return;
    }
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t wt = get_common_arg_val<uint32_t>(1);
    const uint32_t rows = get_common_arg_val<uint32_t>(2);
    const uint32_t groups = get_common_arg_val<uint32_t>(3);
    const uint32_t stride = get_common_arg_val<uint32_t>(4);
    const bool dynamic = get_common_arg_val<uint32_t>(5) != 0;
    const uint64_t counter = (
        get_noc_addr(
            get_common_arg_val<uint32_t>(6),
            get_common_arg_val<uint32_t>(7),
            get_semaphore(get_common_arg_val<uint32_t>(8))
        )
    );
    const uint32_t gid = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t seg = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_x = 0, cb_row_c = 24, cb_row_w = 25;
    const uint32_t x_tile = get_tile_size(cb_x);
    const auto x = TensorAccessor(x_args, x_addr, x_tile);

    uint32_t r = gid;
    while (true) {
        ktt_norm::put_word(cb_row_c, r);
        ktt_norm::put_word(cb_row_w, r);
        if (r >= rows) {
            break;
        }
        cb_reserve_back(cb_x, stride);
        const uint32_t base = get_write_ptr(cb_x);
        const uint32_t page = r * wt + first;
        uint32_t j = r % seg;
        for (uint32_t i = 0; i < seg; ++i) {
            noc_async_read_page(page + j, x, base + j * x_tile);
            j = j + 1 == seg ? 0 : j + 1;
        }
        const uint32_t next = dynamic ? ktt_norm::fetch_add(counter) : r + groups;
        noc_async_read_barrier();
        cb_push_back(cb_x, stride);
        r = next;
    }
}
