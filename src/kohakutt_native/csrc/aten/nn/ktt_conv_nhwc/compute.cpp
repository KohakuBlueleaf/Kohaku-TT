// Nhwc conv compute. Per Cin block and output tile m, per subblock of `sub` Cout tiles: the taps'
// products (A tile m x taps + t from the even / odd A CBs, W' tile t bn + j) into fp32 Dst; partials
// summed across Cin blocks in the fp32 CB 24 by packer L1 accumulation; the last block reloads
// them and packs bf16 to CB 16.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// Common runtime args: Ct, taps, bn, subblock width, iters. Per core: m0, m1.
void kernel_main() {
    const uint32_t ct = get_common_arg_val<uint32_t>(0);
    const uint32_t taps = get_common_arg_val<uint32_t>(1);
    const uint32_t bn = get_common_arg_val<uint32_t>(2);
    const uint32_t sub = get_common_arg_val<uint32_t>(3);
    const uint32_t iters = get_common_arg_val<uint32_t>(4);
    const uint32_t m0 = get_arg_val<uint32_t>(0), m1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_w = 1, cb_a0 = 2, cb_a1 = 3, cb_out = 16, cb_part = 24;
    const uint32_t nm = m1 - m0;
    const uint32_t n_even = (nm * taps + 1) / 2, n_odd = nm * taps / 2;
    const uint32_t block_tiles = nm * bn;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a0, cb_w, ct > 1 ? cb_part : cb_out);
    matmul_block_init(cb_a0, cb_w, 0, sub, 1, 1);
    for (uint32_t it = 0; it < iters; ++it) {
        if (ct > 1) {
            pack_reconfig_data_format(cb_part);
        }
        for (uint32_t cb = 0; cb < ct; ++cb) {
            const bool last = cb == ct - 1;
            cb_wait_front(cb_w, taps * bn);
            if (n_even > 0) {
                cb_wait_front(cb_a0, n_even);
            }
            if (n_odd > 0) {
                cb_wait_front(cb_a1, n_odd);
            }
            for (uint32_t m = 0; m < nm; ++m) {
                for (uint32_t j0 = 0; j0 < bn; j0 += sub) {
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
                            matmul_block(cb_a1, cb_w, i >> 1, t * bn + j0, 0, 0, sub, 1, 1);
                        } else {
                            matmul_block(cb_a0, cb_w, i >> 1, t * bn + j0, 0, 0, sub, 1, 1);
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
                        pack_reconfig_l1_acc(cb > 0 ? 1 : 0);
                        for (uint32_t t = 0; t < sub; ++t) {
                            pack_tile(t, cb_part);
                        }
                        tile_regs_release();
                        cb_push_back(cb_part, sub);
                    }
                }
            }
            if (!last && cb + 2 < ct) {
                cb_wait_front(cb_part, block_tiles);
                cb_pop_front(cb_part, block_tiles);
            }
            if (n_even > 0) {
                cb_pop_front(cb_a0, n_even);
            }
            if (n_odd > 0) {
                cb_pop_front(cb_a1, n_odd);
            }
            cb_pop_front(cb_w, taps * bn);
        }
        if (ct > 1) {
            pack_reconfig_l1_acc(0);
        }
    }
}
