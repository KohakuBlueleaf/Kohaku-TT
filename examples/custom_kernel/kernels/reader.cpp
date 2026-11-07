// SwiGLU reader: streams this core's tiles of A and B into CB 0 and CB 1, one tile
// each per step.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Reads tiles [start, start + count) of A and B.
void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = 0, cb_b = 1;
    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t t = start; t < start + count; t++) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        noc_async_read_page(t, a, get_write_ptr(cb_a));
        noc_async_read_page(t, b, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
