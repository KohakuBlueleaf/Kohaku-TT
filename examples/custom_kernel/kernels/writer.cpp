// SwiGLU writer: writes this core's output tiles from CB 16.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Writes tiles [start, start + count) of the output.
void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = 16;
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t t = start; t < start + count; t++) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(t, out, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
