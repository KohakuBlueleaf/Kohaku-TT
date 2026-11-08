// Matmul compute: per output row tile, out[j] = sum_k a[k] @ w[k, j] for the nt column
// tiles, accumulated in DST (fp32) over the K chunks, packed into CB 16.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"

// Runtime args (per core: rows, nt; the rest common): row tiles, column tiles (nt <=
// 4), K tiles, chunk (kc), w transposed within its tiles (1 for w stored [N, K]).
void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t nt = get_arg_val<uint32_t>(1);
    const uint32_t kt = get_common_arg_val<uint32_t>(0);
    const uint32_t kc = get_common_arg_val<uint32_t>(1);
    const uint32_t transpose = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t cb_a = 0, cb_w = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_a, cb_w, cb_out);
    matmul_block_init(cb_a, cb_w, transpose, nt, 1, kc);
    cb_wait_front(cb_w, kt * nt);
    for (uint32_t m = 0; m < rows; ++m) {
        cb_reserve_back(cb_out, nt);
        tile_regs_acquire();
        for (uint32_t k0 = 0; k0 < kt; k0 += kc) {
            const uint32_t count = kt - k0 < kc ? kt - k0 : kc;
            cb_wait_front(cb_a, kc);
            for (uint32_t k = 0; k < count; ++k) {
                matmul_block(cb_a, cb_w, k, (k0 + k) * nt, 0, transpose, nt, 1, kc);
            }
            cb_pop_front(cb_a, kc);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nt; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, nt);
    }
    cb_pop_front(cb_w, kt * nt);
}
