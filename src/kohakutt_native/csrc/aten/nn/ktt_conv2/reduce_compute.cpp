// Chunk-sum compute: per block, the chunks' 1024 fp32 values (CB 0, unpacked to DST as
// fp32) summed on the SFPU in fp32 and packed as fp32 into CB 16.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "../ktt_conv/args.hpp"

// Compile-time args: the argument-layout version. Common runtime args: version, block
// step, blocks, chunks. Per core: first block.
void kernel_main() {
    if (get_compile_time_arg_val(0) != ktt_conv::kArgsVersion ||
        get_common_arg_val<uint32_t>(0) != ktt_conv::kArgsVersion) {
        return;
    }
    const uint32_t step = get_common_arg_val<uint32_t>(1);
    const uint32_t blocks = get_common_arg_val<uint32_t>(2);
    const uint32_t chunks = get_common_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_out = 16;
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    add_binary_tile_init();
    for (uint32_t b = start; b < blocks; b += step) {
        tile_regs_acquire();
        for (uint32_t s = 0; s < chunks; ++s) {
            cb_wait_front(cb_in, 1);
            copy_tile(cb_in, 0, s == 0 ? 0 : 1);
            if (s > 0) {
                add_binary_tile(0, 1, 0);
            }
            cb_pop_front(cb_in, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
