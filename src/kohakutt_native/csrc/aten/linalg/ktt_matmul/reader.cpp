// Matmul reader: this core's weight block (all K tiles of its nt output column tiles,
// tile (k, j) at CB 1 position k nt + j) once; then per output row tile m the K tiles
// of row m of a, in chunks of kc tiles, into CB 0.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Compile-time args: TensorAccessorArgs of a, then of w. Runtime args (per core: m0,
// m1, n0, nt; the rest common): a address, w address, first / end row tile, first
// column tile, column tiles, K tiles, column tiles in all (Nt), chunk (kc), w stored
// [N, K] (1) or [K, N] (0).
void kernel_main() {
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t m1 = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);
    const uint32_t nt = get_arg_val<uint32_t>(3);
    const uint32_t kt = get_common_arg_val<uint32_t>(2);
    const uint32_t ntot = get_common_arg_val<uint32_t>(3);
    const uint32_t kc = get_common_arg_val<uint32_t>(4);
    const uint32_t w_nk = get_common_arg_val<uint32_t>(5);

    constexpr uint32_t cb_a = 0, cb_w = 1;
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto w_args = (
        TensorAccessorArgs<a_args.next_compile_time_args_offset()>()
    );
    const uint32_t a_tile = get_tile_size(cb_a), w_tile = get_tile_size(cb_w);
    const auto a = TensorAccessor(a_args, a_addr, a_tile);
    const auto w = TensorAccessor(w_args, w_addr, w_tile);

    cb_reserve_back(cb_w, kt * nt);
    uint32_t addr = get_write_ptr(cb_w);
    for (uint32_t k = 0; k < kt; ++k) {
        for (uint32_t j = 0; j < nt; ++j) {
            const uint32_t n = n0 + j;
            noc_async_read_page(w_nk ? n * kt + k : k * ntot + n, w, addr);
            addr += w_tile;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, kt * nt);

    for (uint32_t m = m0; m < m1; ++m) {
        for (uint32_t k0 = 0; k0 < kt; k0 += kc) {
            const uint32_t count = kt - k0 < kc ? kt - k0 : kc;
            cb_reserve_back(cb_a, kc);
            uint32_t dst = get_write_ptr(cb_a);
            for (uint32_t k = k0; k < k0 + count; ++k) {
                noc_async_read_page(m * kt + k, a, dst);
                dst += a_tile;
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, kc);
        }
    }
}
