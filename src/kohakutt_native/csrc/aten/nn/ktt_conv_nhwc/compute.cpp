// Nhwc conv compute. Per pixel block (bm_b output tiles) and Cin step, per Cout
// sub-block (bn_b tiles, one W' block), per output tile m and subblock of `sub` Cout
// tiles: the taps' products (A tile m x taps + t from the even / odd A CBs, W' tile t
// bn_b + j) into fp32 Dst. The pixel block's partials (bm_b x nnb bn_b tiles, order
// (nb, m, j)) are summed across Cin steps in the fp32 CB 24 by packer L1 accumulation;
// the last step reloads them and packs bf16 to CB 16. With `side`: before a step's
// products, its center-tap A tiles (the input's tiles of that Cin block) transposed by
// the unpacker and packed to CB 17, one per output tile.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"

// Common runtime args: Ct, taps, bn_b, nnb, npb, subblock width, iters, bm_b, even A
// tiles per step, odd A tiles per step. Per core: side (1 / 0).
void kernel_main() {
    const uint32_t ct = get_common_arg_val<uint32_t>(0);
    const uint32_t taps = get_common_arg_val<uint32_t>(1);
    const uint32_t bn_b = get_common_arg_val<uint32_t>(2);
    const uint32_t nnb = get_common_arg_val<uint32_t>(3);
    const uint32_t npb = get_common_arg_val<uint32_t>(4);
    const uint32_t sub = get_common_arg_val<uint32_t>(5);
    const uint32_t iters = get_common_arg_val<uint32_t>(6);
    const uint32_t bm_b = get_common_arg_val<uint32_t>(7);
    const uint32_t n_even = get_common_arg_val<uint32_t>(8);
    const uint32_t n_odd = get_common_arg_val<uint32_t>(9);
    const bool side = get_arg_val<uint32_t>(0) != 0;
    constexpr uint32_t cb_w = 1, cb_a0 = 2, cb_a1 = 3, cb_out = 16, cb_side = 17,
                       cb_part = 24;
    const uint32_t block_tiles = bm_b * bn_b * nnb;

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        cb_a0, cb_w, ct > 1 ? cb_part : cb_out
    );
    matmul_block_init(cb_a0, cb_w, 0, sub, 1, 1);
    for (uint32_t pb = 0; pb < iters * npb; ++pb) {
        if (ct > 1) {
            pack_reconfig_data_format(cb_part);
        }
        for (uint32_t k = 0; k < ct; ++k) {
            const bool last = k == ct - 1;
            if (n_even > 0) {
                cb_wait_front(cb_a0, n_even);
            }
            if (n_odd > 0) {
                cb_wait_front(cb_a1, n_odd);
            }
            if (side) {
                transpose_init(cb_a0);
                pack_reconfig_data_format(cb_side);
                if (ct > 1) {
                    pack_reconfig_l1_acc(0);
                }
                for (uint32_t m = 0; m < bm_b; ++m) {
                    const uint32_t i = m * taps + taps / 2;
                    tile_regs_acquire();
                    transpose_tile((i & 1) ? cb_a1 : cb_a0, i >> 1, 0);
                    tile_regs_commit();
                    cb_reserve_back(cb_side, 1);
                    tile_regs_wait();
                    pack_tile(0, cb_side);
                    tile_regs_release();
                    cb_push_back(cb_side, 1);
                }
                pack_reconfig_data_format(ct > 1 ? cb_part : cb_out);
                matmul_block_init(cb_a0, cb_w, 0, sub, 1, 1);
            }
            for (uint32_t nb = 0; nb < nnb; ++nb) {
                cb_wait_front(cb_w, taps * bn_b);
                for (uint32_t m = 0; m < bm_b; ++m) {
                    for (uint32_t j0 = 0; j0 < bn_b; j0 += sub) {
                        tile_regs_acquire();
                        if (last && ct > 1) {
                            reconfig_data_format_srca(cb_w, cb_part);
                            copy_init(cb_part);
                            cb_wait_front(cb_part, sub);
                            copy_block(cb_part, 0, 0, sub);
                            cb_pop_front(cb_part, sub);
                            reconfig_data_format_srca(cb_part, cb_w);
                            matmul_block_init(cb_a0, cb_w, 0, sub, 1, 1);
                        }
                        for (uint32_t t = 0; t < taps; ++t) {
                            const uint32_t i = m * taps + t;
                            if (i & 1) {
                                matmul_block(
                                    cb_a1, cb_w, i >> 1, t * bn_b + j0, 0, 0, sub, 1, 1
                                );
                            } else {
                                matmul_block(
                                    cb_a0, cb_w, i >> 1, t * bn_b + j0, 0, 0, sub, 1, 1
                                );
                            }
                        }
                        tile_regs_commit();
                        if (last) {
                            cb_reserve_back(cb_out, sub);
                            tile_regs_wait();
                            if (ct > 1) {
                                pack_reconfig_data_format(cb_out);
                                pack_reconfig_l1_acc(0);
                            }
                            for (uint32_t t = 0; t < sub; ++t) {
                                pack_tile(t, cb_out);
                            }
                            tile_regs_release();
                            cb_push_back(cb_out, sub);
                            if (ct > 1) {
                                pack_reconfig_data_format(cb_part);
                            }
                        } else {
                            cb_reserve_back(cb_part, sub);
                            tile_regs_wait();
                            pack_reconfig_l1_acc(k > 0 ? 1 : 0);
                            for (uint32_t t = 0; t < sub; ++t) {
                                pack_tile(t, cb_part);
                            }
                            tile_regs_release();
                            cb_push_back(cb_part, sub);
                        }
                    }
                }
                cb_pop_front(cb_w, taps * bn_b);
            }
            if (!last && k + 2 < ct) {
                cb_wait_front(cb_part, block_tiles);
                cb_pop_front(cb_part, block_tiles);
            }
            if (n_even > 0) {
                cb_pop_front(cb_a0, n_even);
            }
            if (n_odd > 0) {
                cb_pop_front(cb_a1, n_odd);
            }
        }
        if (ct > 1) {
            pack_reconfig_l1_acc(0);
        }
    }
}
