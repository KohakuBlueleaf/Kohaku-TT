// Group norm moments writer: per plane its s1 / s2 tiles (CB 16 / 17) to page `plane`
// of the [N, C, 1, 1] fp32 outputs.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: TensorAccessorArgs of s1, then of s2. Runtime args: s1 address, s2
// address, first plane, plane step, planes.
void kernel_main() {
    const uint32_t s1_addr = get_arg_val<uint32_t>(0);
    const uint32_t s2_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t step = get_arg_val<uint32_t>(3);
    const uint32_t planes = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_s1 = 16, cb_s2 = 17, kTile = 4096;
    constexpr auto s1_args = TensorAccessorArgs<0>();
    constexpr auto s2_args = (
        TensorAccessorArgs<s1_args.next_compile_time_args_offset()>()
    );
    const auto s1 = TensorAccessor(s1_args, s1_addr, kTile);
    const auto s2 = TensorAccessor(s2_args, s2_addr, kTile);

    for (uint32_t p = start; p < planes; p += step) {
        cb_wait_front(cb_s1, 1);
        cb_wait_front(cb_s2, 1);
        noc_async_write_page(p, s1, get_read_ptr(cb_s1));
        noc_async_write_page(p, s2, get_read_ptr(cb_s2));
        noc_async_write_barrier();
        cb_pop_front(cb_s1, 1);
        cb_pop_front(cb_s2, 1);
    }
}
