// Block matmul compute: per output block, the K blocks' products summed into the fp32
// partials CB 24 by packer accumulation; the last K block reloads them and packs bf16 to CB
// 16, one sh x sw subblock at a time. Repeats `iters` times (1 in normal use).
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "args.hpp"

// Common runtime args: bm, bn, kb, K blocks, K tiles of the last block, blocks per core
// (rows, columns), sh, sw, iterations, w transposed within its tiles, sentinel.
void kernel_main() {
    if (get_common_arg_val<uint32_t>(11) != ktt_gemm::kArgSentinel) {
        return;
    }
    const uint32_t bm = get_common_arg_val<uint32_t>(0);
    const uint32_t bn = get_common_arg_val<uint32_t>(1);
    const uint32_t kb = get_common_arg_val<uint32_t>(2);
    const uint32_t nk = get_common_arg_val<uint32_t>(3);
    const uint32_t kb_last = get_common_arg_val<uint32_t>(4);
    const uint32_t nbh = get_common_arg_val<uint32_t>(5);
    const uint32_t nbw = get_common_arg_val<uint32_t>(6);
    const uint32_t sh = get_common_arg_val<uint32_t>(7);
    const uint32_t sw = get_common_arg_val<uint32_t>(8);
    const uint32_t iters = get_common_arg_val<uint32_t>(9);
    const uint32_t transpose = get_common_arg_val<uint32_t>(10);
    constexpr uint32_t cb_a = 0, cb_w = 1, cb_out = 16, cb_part = 24;

    const uint32_t a_block = bm * kb, w_block = kb * bn, sub = sh * sw;
    const uint32_t block_tiles = bm * bn;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, nk > 1 ? cb_part : cb_out);
    matmul_block_init(cb_a, cb_w, transpose, sw, sh, kb);

    for (uint32_t blk = 0; blk < iters * nbh * nbw; ++blk) {
        if (nk > 1) {
            pack_reconfig_data_format(cb_part);
        }
        for (uint32_t kblk = 0; kblk < nk; ++kblk) {
            const bool last = kblk == nk - 1;
            const uint32_t inner = last ? kb_last : kb;
            cb_wait_front(cb_a, a_block);
            cb_wait_front(cb_w, w_block);
            for (uint32_t i = 0; i < bm; i += sh) {
                for (uint32_t j = 0; j < bn; j += sw) {
                    tile_regs_acquire();
                    if (last && nk > 1) {
                        reconfig_data_format_srca(cb_w, cb_part);
                        copy_init(cb_part);
                        cb_wait_front(cb_part, sub);
                        copy_block(cb_part, 0, 0, sub);
                        cb_pop_front(cb_part, sub);
                        reconfig_data_format_srca(cb_part, cb_w);
                        matmul_block_init(cb_a, cb_w, transpose, sw, sh, kb);
                    }
                    uint32_t a_index = i * kb, w_index = j;
                    for (uint32_t k = 0; k < inner; ++k) {
                        matmul_block(cb_a, cb_w, a_index, w_index, 0, transpose, sw, sh, kb);
                        ++a_index;
                        w_index += bn;
                    }
                    tile_regs_commit();
                    if (last) {
                        cb_reserve_back(cb_out, sub);
                        tile_regs_wait();
                        if (nk > 1) {
                            pack_reconfig_data_format(cb_out);
                            pack_reconfig_l1_acc(0);
                        }
                        for (uint32_t t = 0; t < sub; ++t) {
                            pack_tile(t, cb_out);
                        }
                        tile_regs_release();
                        cb_push_back(cb_out, sub);
                    } else {
                        cb_reserve_back(cb_part, sub);
                        tile_regs_wait();
                        pack_reconfig_l1_acc(kblk > 0 ? 1 : 0);
                        for (uint32_t t = 0; t < sub; ++t) {
                            pack_tile(t, cb_part);
                        }
                        tile_regs_release();
                        cb_push_back(cb_part, sub);
                    }
                }
            }
            if (!last && kblk + 2 < nk) {
                cb_wait_front(cb_part, block_tiles);
                cb_pop_front(cb_part, block_tiles);
            }
            cb_pop_front(cb_a, a_block);
            cb_pop_front(cb_w, w_block);
        }
        if (nk > 1) {
            pack_reconfig_l1_acc(0);
        }
    }
}
