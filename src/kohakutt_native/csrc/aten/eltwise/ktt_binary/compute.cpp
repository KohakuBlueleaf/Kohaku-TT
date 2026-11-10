// Broadcast binary compute: per block, a's tiles (CB 0) and the current b tile (CB 1)
// into DST, BINARY_OP on the SFPU (fp32 DST), into CB 16. b is popped at the end of
// each j group.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

#include "complex.hpp"

// Common runtime args: repeats per j (nr), block size (< DST size); per core: first
// pair, pair count.
void kernel_main() {
    const uint32_t nr = get_common_arg_val<uint32_t>(0);
    const uint32_t block = get_common_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    const uint32_t end = start + count;
    uint32_t idx = start;
    while (idx < end) {
        const uint32_t r = idx % nr;
        if (r == 0 || idx == start) {
            cb_wait_front(cb_b, 1);
        }
        uint32_t n = nr - r;
        n = n < block ? n : block;
        n = n < end - idx ? n : end - idx;

        cb_wait_front(cb_a, n);
        cb_reserve_back(cb_out, n);
        tile_regs_acquire();
        copy_init(cb_a);
        for (uint32_t k = 0; k < n; ++k) {
            copy_tile(cb_a, k, k);
        }
        copy_init(cb_b);
        copy_tile(cb_b, 0, n);
        BINARY_OP_INIT
        for (uint32_t k = 0; k < n; ++k) {
            BINARY_OP(k, n, k);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t k = 0; k < n; ++k) {
            pack_tile(k, cb_out);
        }
        tile_regs_release();
        cb_pop_front(cb_a, n);
        cb_push_back(cb_out, n);

        idx += n;
        if (idx % nr == 0 || idx == end) {
            cb_pop_front(cb_b, 1);
        }
    }
}
