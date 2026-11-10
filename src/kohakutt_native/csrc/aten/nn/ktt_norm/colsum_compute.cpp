// Column-sum compute (fp32 SFPU): per column tile, the groups' partial tiles summed,
// then the 32 rows summed into row 0; packed in the output format.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "args.hpp"

// Common runtime args: Wt, groups, cores. Per core: first column tile.
void kernel_main() {
    if (!ktt_norm::args_ok<0>(4)) {
        return;
    }
    const uint32_t wt = get_common_arg_val<uint32_t>(0);
    const uint32_t groups = get_common_arg_val<uint32_t>(1);
    const uint32_t cores = get_common_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_out = 16;

    // Partial g accumulates into D(g % 3) (D3 takes the loaded tile), then D0 + D1 +
    // D2.
    constexpr uint32_t kAcc = 3;
    compute_kernel_hw_startup(cb_in, cb_out);
    for (uint32_t t = start; t < wt; t += cores) {
        tile_regs_acquire();
        for (uint32_t g = 0; g < groups; ++g) {
            cb_wait_front(cb_in, 1);
            reconfig_data_format_srca(cb_in);
            copy_init(cb_in);
            copy_tile(cb_in, 0, g < kAcc ? g : kAcc);
            cb_pop_front(cb_in, 1);
            if (g >= kAcc) {
                add_binary_tile_init();
                add_binary_tile(g % kAcc, kAcc, g % kAcc);
            }
        }
        for (uint32_t a = 1; a < kAcc && a < groups; ++a) {
            add_binary_tile_init();
            add_binary_tile(0, a, 0);
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
