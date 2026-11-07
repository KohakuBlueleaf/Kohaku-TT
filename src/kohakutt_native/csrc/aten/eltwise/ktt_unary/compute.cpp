// Blocked unary SFPU compute: `block` tiles of CB 0 per DST acquire, the op chain
// SFPU_OP_CHAIN_0 (TT-NN's unary defines, destination index `j`) on each, into CB 16.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/tile_move_copy.h"

// Runtime args: tile count, block size (<= the DST capacity).
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    const uint32_t block = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_in = 0, cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    for (uint32_t t = 0; t < count; t += block) {
        const uint32_t n = count - t < block ? count - t : block;
        cb_wait_front(cb_in, n);
        cb_reserve_back(cb_out, n);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            copy_tile(cb_in, j, j);
        }
        for (uint32_t j = 0; j < n; ++j) {
            SFPU_OP_CHAIN_0
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_pop_front(cb_in, n);
        cb_push_back(cb_out, n);
    }
}
