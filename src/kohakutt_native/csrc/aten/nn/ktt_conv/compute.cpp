// Conv compute: per output row tile, every tap's 32 row-major input rows (CB 0, Ct
// tiles wide) tilized into CB 2 (K = taps x Ct tiles), then out[j] = sum_k act[k] @
// w[k, j] accumulated in DST (fp32) for the nt column tiles and packed into CB 16.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"

// Runtime args (per core: rows, nt; the rest common): output row tiles, column tiles
// (nt <= 4), taps (KH KW), Ct.
void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t nt = get_arg_val<uint32_t>(1);
    const uint32_t taps = get_common_arg_val<uint32_t>(0);
    const uint32_t ct = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t cb_rows = 0, cb_w = 1, cb_act = 2, cb_out = 16;
    const uint32_t k_tiles = taps * ct;

    compute_kernel_hw_startup(cb_rows, cb_w, cb_out);
    cb_wait_front(cb_w, k_tiles * nt);
    for (uint32_t m = 0; m < rows; ++m) {
        tilize_init(cb_rows, ct, cb_act);
        for (uint32_t t = 0; t < taps; ++t) {
            cb_wait_front(cb_rows, ct);
            cb_reserve_back(cb_act, ct);
            tilize_block(cb_rows, ct, cb_act);
            cb_push_back(cb_act, ct);
            cb_pop_front(cb_rows, ct);
        }
        tilize_uninit(cb_rows, cb_act);

        matmul_init(cb_act, cb_w);
        cb_wait_front(cb_act, k_tiles);
        cb_reserve_back(cb_out, nt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < k_tiles; ++k) {
            for (uint32_t j = 0; j < nt; ++j) {
                matmul_tiles(cb_act, cb_w, k, k * nt + j, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nt; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, nt);
        cb_pop_front(cb_act, k_tiles);
    }
    cb_pop_front(cb_w, k_tiles * nt);
}
