// Flash attention backward dK / dV writer: per key row of each job, DT dK tiles then DV
// dV tiles.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "flash_bw.hpp"

// Compile-time args: DT, DV, then TensorAccessorArgs of dK, dV. Runtime args: dK, dV
// addresses, iterations, jobs, then per job: first key tile row (global), key tile
// rows, row stride.
void kernel_main() {
    using namespace ktt_flash_bw::dkv;
    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr auto adk = TensorAccessorArgs<2>();
    constexpr auto adv = TensorAccessorArgs<adk.next_compile_time_args_offset()>();
    const uint32_t page = get_tile_size(cb_out);
    const auto dk = TensorAccessor(adk, get_arg_val<uint32_t>(0), page);
    const auto dv = TensorAccessor(adv, get_arg_val<uint32_t>(1), page);
    const uint32_t iters = get_arg_val<uint32_t>(2);
    const uint32_t jobs = get_arg_val<uint32_t>(3);
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t krow0 = get_arg_val<uint32_t>(4 + 3 * job);
            const uint32_t rows = get_arg_val<uint32_t>(5 + 3 * job);
            const uint32_t stride = get_arg_val<uint32_t>(6 + 3 * job);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t row = krow0 + r * stride;
                cb_wait_front(cb_out, DT + DV);
                const uint32_t base = get_read_ptr(cb_out);
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_write_page(row * DT + d, dk, base + d * page);
                }
                for (uint32_t u = 0; u < DV; ++u) {
                    noc_async_write_page(row * DV + u, dv, base + (DT + u) * page);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, DT + DV);
            }
        }
    }
}
