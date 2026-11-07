// Fused-chain writer: this core's output tiles from CB 16, column by column (the
// reader's order).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: the output's TensorAccessorArgs. Runtime args: address, first
// tile, tile count, ct (tile columns).
void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t ct = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = 16;
    const uint32_t page = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, addr, page);

    const uint32_t end = start + count;
    const uint32_t first_row = start / ct, last_row = (end - 1) / ct;
    for (uint32_t c = 0; c < ct; ++c) {
        const uint32_t gr0 = first_row * ct + c >= start ? first_row : first_row + 1;
        const uint32_t gr1 = last_row * ct + c < end ? last_row : last_row - 1;
        if (gr1 < gr0 || gr1 == UINT32_MAX) {
            continue;
        }
        for (uint32_t gr = gr0; gr <= gr1; ++gr) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(gr * ct + c, out, get_read_ptr(cb_out));
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
