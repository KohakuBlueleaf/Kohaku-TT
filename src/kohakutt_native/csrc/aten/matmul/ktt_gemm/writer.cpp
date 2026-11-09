// Block matmul writer (RISCV_0, dedicated NoC, the NoC the core's reader does not use): per
// output block, the core's sh x sw output subblocks from CB 16 to DRAM in compute's order
// (tiles outside the core's range dropped). Repeats `iters` times.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

// Compile-time args: TensorAccessorArgs of the output.
// Common runtime args: output address, N tiles, bm, bn, blocks per core (rows, columns), sh,
// sw, iterations, sentinel.
// Per-core runtime args: first row tile, first column tile, end row tile, end column tile,
// sentinel.
void kernel_main() {
    if (get_common_arg_val<uint32_t>(9) != ktt_gemm::kArgSentinel ||
        get_arg_val<uint32_t>(4) != ktt_gemm::kArgSentinel) {
        return;
    }
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t nt = get_common_arg_val<uint32_t>(1);
    const uint32_t bm = get_common_arg_val<uint32_t>(2);
    const uint32_t bn = get_common_arg_val<uint32_t>(3);
    const uint32_t nbh = get_common_arg_val<uint32_t>(4);
    const uint32_t nbw = get_common_arg_val<uint32_t>(5);
    const uint32_t sh = get_common_arg_val<uint32_t>(6);
    const uint32_t sw = get_common_arg_val<uint32_t>(7);
    const uint32_t iters = get_common_arg_val<uint32_t>(8);
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const uint32_t m_end = get_arg_val<uint32_t>(2);
    const uint32_t n_end = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile);
    const uint32_t sub = sh * sw;

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t bh = 0; bh < nbh; ++bh) {
            for (uint32_t bw = 0; bw < nbw; ++bw) {
                for (uint32_t i = 0; i < bm; i += sh) {
                    for (uint32_t j = 0; j < bn; j += sw) {
                        cb_wait_front(cb_out, sub);
                        uint32_t src = get_read_ptr(cb_out);
                        for (uint32_t r = 0; r < sh; ++r) {
                            const uint32_t row = m0 + bh * bm + i + r;
                            for (uint32_t c = 0; c < sw; ++c) {
                                const uint32_t col = n0 + bw * bn + j + c;
                                if (row < m_end && col < n_end) {
                                    noc_async_write_page(row * nt + col, out, src);
                                }
                                src += tile;
                            }
                        }
                        noc_async_writes_flushed();
                        cb_pop_front(cb_out, sub);
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
}
