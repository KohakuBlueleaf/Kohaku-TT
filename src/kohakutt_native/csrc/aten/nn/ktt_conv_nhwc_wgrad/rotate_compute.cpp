// X^T compute: each tile of CB 0 transposed by the unpacker (faces and rows within a
// face), copied through DST and packed into CB 16; `block` tiles per DST acquire.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/transpose.h"

// Common runtime arg: block size (<= the DST capacity); per core: tile count.
void kernel_main() {
    const uint32_t block = get_common_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_out);
    transpose_init(cb_in);
    for (uint32_t t = 0; t < count; t += block) {
        const uint32_t n = count - t < block ? count - t : block;
        cb_wait_front(cb_in, n);
        cb_reserve_back(cb_out, n);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            transpose_tile(cb_in, j, j);
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
