// Attention writer: per unit (batch-head, q chunk), per Q row tile, the Dv output tiles
// (CB 16) to out [B, H, Sq, Dv].
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: TensorAccessorArgs of the output. Common runtime args: output
// address, unit step, units, Sq tiles, Dv tiles, q chunk tiles; per core: first unit.
void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t step = get_common_arg_val<uint32_t>(1);
    const uint32_t units = get_common_arg_val<uint32_t>(2);
    const uint32_t sqt = get_common_arg_val<uint32_t>(3);
    const uint32_t dvt = get_common_arg_val<uint32_t>(4);
    const uint32_t qc = get_common_arg_val<uint32_t>(5);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile);

    const uint32_t q_chunks = (sqt + qc - 1) / qc;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t bh = u / q_chunks, chunk = u % q_chunks;
        const uint32_t q0 = chunk * qc, nq = sqt - q0 < qc ? sqt - q0 : qc;
        for (uint32_t i = 0; i < nq; ++i) {
            cb_wait_front(cb_out, dvt);
            uint32_t src = get_read_ptr(cb_out);
            const uint32_t first = (bh * sqt + q0 + i) * dvt;
            for (uint32_t n = 0; n < dvt; ++n) {
                noc_async_write_page(first + n, out, src);
                src += tile;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, dvt);
        }
    }
}
