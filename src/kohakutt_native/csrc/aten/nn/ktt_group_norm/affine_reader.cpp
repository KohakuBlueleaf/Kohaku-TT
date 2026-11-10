// Group norm affine reader: per plane (n, c), its a / b scalars (page `plane` of the
// [N, C, 1, 1] fp32 inputs) as full tiles, then the plane's x tiles in order.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: TensorAccessorArgs of x, a, b. Runtime args (per core: start; the
// rest common): x address, a address, b address, first plane, plane step,
// planes, tiles per plane, x element bytes (2 / 4).
void kernel_main() {
    uint32_t i = 0;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(i++);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(i++);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(i++);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(i++);
    const uint32_t planes = get_common_arg_val<uint32_t>(i++);
    const uint32_t tiles = get_common_arg_val<uint32_t>(i++);
    const uint32_t elem = get_common_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = 0, cb_a = 1, cb_b = 2, kScalarTile = 4096;
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        a_args = (TensorAccessorArgs<x_args.next_compile_time_args_offset()>());
    constexpr auto
        b_args = (TensorAccessorArgs<a_args.next_compile_time_args_offset()>());
    const auto x = TensorAccessor(x_args, x_addr, 1024 * elem);
    const auto a = TensorAccessor(a_args, a_addr, kScalarTile);
    const auto b = TensorAccessor(b_args, b_addr, kScalarTile);

    for (uint32_t p = start; p < planes; p += step) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        noc_async_read_page(p, a, get_write_ptr(cb_a));
        noc_async_read_page(p, b, get_write_ptr(cb_b));
        noc_async_read_barrier();
        // Each scalar ([0, 0]) replicated over its tile.
        for (const uint32_t cb : {cb_a, cb_b}) {
            auto* t = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb));
            const uint32_t v = t[0];
            for (uint32_t k = 1; k < 1024; ++k) {
                t[k] = v;
            }
        }
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
        for (uint32_t t = 0; t < tiles; ++t) {
            cb_reserve_back(cb_x, 1);
            noc_async_read_page(p * tiles + t, x, get_write_ptr(cb_x));
            noc_async_read_barrier();
            cb_push_back(cb_x, 1);
        }
    }
}
