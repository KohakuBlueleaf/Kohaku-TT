// Nhwc conv weight-gradient compute. Per (Cout tile, Cin block of bc tiles) and pixel
// block of kb tiles, per tap: the products X^T (in0: tile (j, m) at j kb + m of the
// block, bc rows with stride kb) x G (in1: tile m taps + tap from the even / odd G CBs)
// over the block's valid pixel tiles into bc fp32 Dst tiles. Partials (bc x taps tiles,
// order (tap, j)) are summed across pixel blocks in the fp32 CB 24 by packer L1
// accumulation; the last block reloads them and packs to CB 16 (bf16, or fp32 for
// partial sums over a pixel range).
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// Common runtime args: taps, bc, kb, nq, npb, bo, iters, even G tiles per step, odd G
// tiles per step. Per core: pixel tiles of its range.
void kernel_main() {
    const uint32_t taps = get_common_arg_val<uint32_t>(0);
    const uint32_t bc = get_common_arg_val<uint32_t>(1);
    const uint32_t kb = get_common_arg_val<uint32_t>(2);
    const uint32_t nq = get_common_arg_val<uint32_t>(3);
    const uint32_t npb = get_common_arg_val<uint32_t>(4);
    const uint32_t bo = get_common_arg_val<uint32_t>(5);
    const uint32_t iters = get_common_arg_val<uint32_t>(6);
    const uint32_t n_even = get_common_arg_val<uint32_t>(7);
    const uint32_t n_odd = get_common_arg_val<uint32_t>(8);
    const uint32_t m_cnt = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_x = 1, cb_a0 = 2, cb_a1 = 3, cb_out = 16, cb_part = 24;
    const uint32_t block_tiles = bc * taps;
    const uint32_t groups = iters * bo * nq;

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_x, cb_a0, npb > 1 ? cb_part : cb_out
    );
    matmul_block_init(cb_x, cb_a0, 0, 1, bc, kb);
    for (uint32_t g = 0; g < groups; ++g) {
        if (npb > 1) {
            pack_reconfig_data_format(cb_part);
        }
        for (uint32_t pb = 0; pb < npb; ++pb) {
            const bool last = pb == npb - 1;
            const uint32_t first = pb * kb;
            const uint32_t valid = first >= m_cnt
                ? 0
                : (m_cnt - first < kb ? m_cnt - first : kb);
            cb_wait_front(cb_x, bc * kb);
            if (n_even > 0) {
                cb_wait_front(cb_a0, n_even);
            }
            if (n_odd > 0) {
                cb_wait_front(cb_a1, n_odd);
            }
            for (uint32_t tap = 0; tap < taps; ++tap) {
                tile_regs_acquire();
                if (last && npb > 1) {
                    reconfig_data_format_srca(cb_a0, cb_part);
                    copy_init(cb_part);
                    cb_wait_front(cb_part, bc);
                    copy_block(cb_part, 0, 0, bc);
                    cb_pop_front(cb_part, bc);
                    reconfig_data_format_srca(cb_part, cb_a0);
                    matmul_block_init(cb_x, cb_a0, 0, 1, bc, kb);
                }
                for (uint32_t m = 0; m < valid; ++m) {
                    const uint32_t i = m * taps + tap;
                    if (i & 1) {
                        matmul_block(cb_x, cb_a1, m, i >> 1, 0, 0, 1, bc, kb);
                    } else {
                        matmul_block(cb_x, cb_a0, m, i >> 1, 0, 0, 1, bc, kb);
                    }
                }
                tile_regs_commit();
                if (last) {
                    cb_reserve_back(cb_out, bc);
                    tile_regs_wait();
                    if (npb > 1) {
                        pack_reconfig_data_format(cb_out);
                        pack_reconfig_l1_acc(0);
                    }
                    for (uint32_t j = 0; j < bc; ++j) {
                        pack_tile(j, cb_out);
                    }
                    tile_regs_release();
                    cb_push_back(cb_out, bc);
                    if (npb > 1) {
                        pack_reconfig_data_format(cb_part);
                    }
                } else {
                    cb_reserve_back(cb_part, bc);
                    tile_regs_wait();
                    pack_reconfig_l1_acc(pb > 0 ? 1 : 0);
                    for (uint32_t j = 0; j < bc; ++j) {
                        pack_tile(j, cb_part);
                    }
                    tile_regs_release();
                    cb_push_back(cb_part, bc);
                }
            }
            if (!last && pb + 2 < npb) {
                cb_wait_front(cb_part, block_tiles);
                cb_pop_front(cb_part, block_tiles);
            }
            if (n_even > 0) {
                cb_pop_front(cb_a0, n_even);
            }
            if (n_odd > 0) {
                cb_pop_front(cb_a1, n_odd);
            }
            cb_pop_front(cb_x, bc * kb);
        }
        if (npb > 1) {
            pack_reconfig_l1_acc(0);
        }
    }
}
