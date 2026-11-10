// SwiGLU compute: out = silu(a) * b, tile by tile, on the SFPU in fp32 DST.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

// Processes `count` tiles (runtime arg 0) from CB 0 / CB 1 into CB 16.
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    for (uint32_t t = 0; t < count; t++) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        tile_regs_acquire();

        // a in DST[0], b in DST[1]; DST[0] = silu(DST[0]) * DST[1] on the SFPU.
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 1);
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
