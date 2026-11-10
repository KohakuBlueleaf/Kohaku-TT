// Row-form group norm backward, affine compute (fp32 SFPU). With per channel Sg = sum
// dy and Sgx = sum dy x (the partial pages summed over their rows), mean m, rstd r,
// weight w, and D elements per group: per group A = sum w Sg and B = sum w r (Sgx - m
// Sg) through the mask tile; per channel P = r w, Q = -r^2 B / D, R = -r A / D - m Q;
// then dx = dy P + x Q + R for the piece's tiles, one per acquire. The piece starting a
// segment also packs the rows r (Sgx - m Sg) and Sg (the image's weight and bias
// gradient terms) to CB 12. A row tile shared with the next image takes P / Q / R tiles
// mixed by rows from both images.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "rows.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

    // One face: dst[2] = dst[3] dst[0] + dst[2] dst[1] (fp32; x in DST 2, dy in DST 3).
    inline void ktt_rows_bwd_affine(uint32_t, uint32_t, uint32_t) {
        for (int d = 0; d < 8; ++d) {
            sfpi::dst_reg[64] = sfpi::dst_reg[96] * sfpi::dst_reg[0] +
                sfpi::dst_reg[64] * sfpi::dst_reg[32];
            sfpi::dst_reg++;
        }
    }

}  // namespace ckernel::sfpu
#endif

namespace {

    constexpr uint32_t cb_x = 0, cb_w = 1, cb_g = 2, cb_gather = 3, cb_mask = 4,
                       cb_sums = 5;
    constexpr uint32_t cb_groups = 6, cb_pqr = 7, cb_back = 8, cb_next = 9,
                       cb_rows = 10;
    constexpr uint32_t cb_mix = 11, cb_stats = 12, cb_stat = 13, cb_terms = 14,
                       cb_y = 16;
    constexpr uint32_t kNone = 0xFFFFFFFFu, kMinusOne = 0xbf800000u;

    inline void copy(uint32_t cb, uint32_t tile, uint32_t dst) {
        reconfig_data_format_srca(cb);
        copy_init(cb);
        copy_tile(cb, tile, dst);
    }

    inline void row_bcast(uint32_t cb, uint32_t tile, uint32_t dst) {
        reconfig_data_format(cb, cb);
        unary_bcast_init<BroadcastType::ROW>(cb);
        unary_bcast<BroadcastType::ROW>(cb, tile, dst);
    }

    inline void sum_rows(uint32_t dst) {
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(dst);
    }

    inline void mul(uint32_t a, uint32_t b, uint32_t out) {
        mul_binary_tile_init();
        mul_binary_tile(a, b, out);
    }

    inline void scale(uint32_t dst, uint32_t bits) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, bits);
    }

    // DST tiles `first`, `second` packed to the two pages of fp32 CB `cb`, and of
    // `copy` too when it is a CB (not kNone).
    inline void pack_pair(
        uint32_t cb, uint32_t first, uint32_t second, uint32_t copy = kNone
    ) {
        tile_regs_commit();
        cb_reserve_back(cb, 2);
        if (copy != kNone) {
            cb_reserve_back(copy, 2);
        }
        tile_regs_wait();
        pack_reconfig_data_format(cb);
        pack_tile(first, cb);
        pack_tile(second, cb);
        if (copy != kNone) {
            pack_tile(first, copy);
            pack_tile(second, copy);
        }
        tile_regs_release();
        cb_push_back(cb, 2);
        if (copy != kNone) {
            cb_push_back(copy, 2);
        }
    }

}  // namespace

// Common runtime args: channel tiles Ct, rows per image, images, block, has weight, 1 /
// D (fp32 bits), channels per group, channels C, gradient terms wanted. Per core: q0,
// q1.
void kernel_main() {
    const ktt_group_norm::Geometry g{
        get_common_arg_val<uint32_t>(0),
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2)
    };
    const uint32_t block = get_common_arg_val<uint32_t>(3);
    const bool has_w = get_common_arg_val<uint32_t>(4) != 0;
    const uint32_t inv_count = get_common_arg_val<uint32_t>(5);
    const uint32_t cg = get_common_arg_val<uint32_t>(6);
    const uint32_t c = get_common_arg_val<uint32_t>(7);
    const bool terms_out = get_common_arg_val<uint32_t>(8) != 0;
    const uint32_t q1 = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_x, cb_y);
    for (uint32_t q = get_arg_val<uint32_t>(0); q < q1;) {
        const ktt_group_norm::Piece p(g, q, q1);
        const bool shared = g.shared(p.n) && p.last;
        const ktt_group_norm::Span span(p.cb, cg, c);
        for (uint32_t pass = 0; pass < (shared ? 2u : 1u); ++pass) {
            // Group sums A, B (row 0 of the two cb_groups tiles, one column per group
            // of the tile), added up in L1 over the channel tiles the groups reach.
            cb_reserve_back(cb_groups, 2);
            for (uint32_t t = 0; t < span.tiles(); ++t) {
                // Row 0 of DST 0: Sg; of DST 1: r (Sgx - m Sg).
                cb_wait_front(cb_gather, 2);
                cb_wait_front(cb_stat, 2);
                tile_regs_acquire();
                copy(cb_gather, 0, 0);
                copy_tile(cb_gather, 1, 1);
                sum_rows(0);
                sum_rows(1);
                copy(cb_stat, 0, 2);
                sum_rows(2);
                mul(2, 0, 3);
                sub_binary_tile_init();
                sub_binary_tile(1, 3, 1);
                copy(cb_stat, 1, 2);
                sum_rows(2);
                mul(1, 2, 1);
                const bool out = terms_out && pass == 0 && p.i == 0 &&
                    span.t_lo + t == p.cb;
                pack_pair(cb_terms, 1, 0, out ? cb_stats : kNone);
                cb_pop_front(cb_gather, 2);
                cb_pop_front(cb_stat, 2);

                // Times the weights: w Sg and w r (Sgx - m Sg).
                cb_wait_front(cb_terms, 2);
                tile_regs_acquire();
                copy(cb_terms, 0, 1);
                copy_tile(cb_terms, 1, 0);
                if (has_w) {
                    cb_wait_front(cb_w, 1);
                    copy(cb_w, 0, 2);
                    mul(0, 2, 0);
                    mul(1, 2, 1);
                }
                pack_pair(cb_sums, 0, 1);
                cb_pop_front(cb_terms, 2);
                if (has_w) {
                    cb_pop_front(cb_w, 1);
                }

                // Channel c's terms to (c, group), summed over the channels.
                cb_wait_front(cb_sums, 2);
                cb_wait_front(cb_mask, 1);
                tile_regs_acquire();
                row_bcast(cb_sums, 0, 0);
                row_bcast(cb_sums, 1, 1);
                transpose_dest_init<DST_ACCUM_MODE>();
                transpose_dest<DST_ACCUM_MODE>(0);
                transpose_dest<DST_ACCUM_MODE>(1);
                copy(cb_mask, 0, 2);
                mul(0, 2, 0);
                mul(1, 2, 1);
                sum_rows(0);
                sum_rows(1);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_groups);
                pack_reconfig_l1_acc(t > 0 ? 1 : 0);
                pack_tile<true>(0, cb_groups, 0);
                pack_tile<true>(1, cb_groups, 1);
                tile_regs_release();
                pack_reconfig_l1_acc(0);
                cb_pop_front(cb_sums, 2);
                cb_pop_front(cb_mask, 1);
            }
            cb_push_back(cb_groups, 2);

            // Back to the channels (row 0 of DST 0: A / D, of DST 1: B / D), then P, Q,
            // R.
            cb_wait_front(cb_groups, 2);
            cb_wait_front(cb_back, 2);
            cb_wait_front(cb_stat, 2);
            tile_regs_acquire();
            row_bcast(cb_groups, 0, 0);
            row_bcast(cb_groups, 1, 1);
            copy(cb_back, 0, 2);
            mul(0, 2, 0);
            mul(1, 2, 1);
            transpose_dest_init<DST_ACCUM_MODE>();
            transpose_dest<DST_ACCUM_MODE>(0);
            transpose_dest<DST_ACCUM_MODE>(1);
            sum_rows(0);
            sum_rows(1);
            scale(0, inv_count);
            scale(1, inv_count);
            // DST 2: r; DST 1: Q = -r^2 B / D; DST 0: R = -(r A / D + m Q); DST 2: P =
            // r w.
            copy(cb_stat, 1, 2);
            sum_rows(2);
            mul(2, 2, 3);
            mul(1, 3, 1);
            scale(1, kMinusOne);
            mul(0, 2, 0);
            copy(cb_stat, 0, 3);
            sum_rows(3);
            mul(3, 1, 3);
            add_binary_tile_init();
            add_binary_tile(0, 3, 0);
            scale(0, kMinusOne);
            if (has_w) {
                cb_wait_front(cb_w, 1);
                copy(cb_w, 0, 3);
                mul(2, 3, 2);
            }
            copy(cb_back, 1, 3);
            mul(0, 3, 0);
            mul(1, 3, 1);
            mul(2, 3, 2);
            const uint32_t to = pass == 0 ? cb_pqr : cb_next;
            tile_regs_commit();
            cb_reserve_back(to, 3);
            tile_regs_wait();
            pack_reconfig_data_format(to);
            pack_tile(2, to);
            pack_tile(1, to);
            pack_tile(0, to);
            tile_regs_release();
            cb_push_back(to, 3);
            cb_pop_front(cb_groups, 2);
            cb_pop_front(cb_back, 2);
            cb_pop_front(cb_stat, 2);
            if (has_w) {
                cb_pop_front(cb_w, 1);
            }
        }

        cb_wait_front(cb_pqr, 3);
        if (shared) {
            // The shared tile's P, Q, R as whole tiles: the image's rows from cb_pqr,
            // the next image's from cb_next.
            cb_wait_front(cb_next, 3);
            cb_wait_front(cb_rows, 2);
            cb_reserve_back(cb_mix, 3);
            for (uint32_t which = 0; which < 3; ++which) {
                tile_regs_acquire();
                row_bcast(cb_pqr, which, 0);
                row_bcast(cb_next, which, 1);
                reconfig_data_format(cb_rows, cb_rows);
                unary_bcast_init<BroadcastType::COL>(cb_rows);
                unary_bcast<BroadcastType::COL>(cb_rows, 0, 2);
                unary_bcast<BroadcastType::COL>(cb_rows, 1, 3);
                mul(0, 2, 0);
                mul(1, 3, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_mix);
                pack_tile<true>(0, cb_mix, which);
                tile_regs_release();
            }
            cb_push_back(cb_mix, 3);
            cb_pop_front(cb_next, 3);
            cb_pop_front(cb_rows, 2);
            cb_wait_front(cb_mix, 3);
        }
        for (uint32_t t = 0; t < p.len; t += block) {
            const uint32_t n = p.len - t < block ? p.len - t : block;
            cb_wait_front(cb_x, block);
            cb_wait_front(cb_g, block);
            cb_reserve_back(cb_y, block);
            for (uint32_t k = 0; k < n; ++k) {
                const bool mixed = shared && p.ends_image(g, t + k);
                tile_regs_acquire();
                if (mixed) {
                    copy(cb_mix, 0, 0);
                    copy_tile(cb_mix, 1, 1);
                } else {
                    row_bcast(cb_pqr, 0, 0);
                    row_bcast(cb_pqr, 1, 1);
                }
                copy(cb_x, k, 2);
                copy_tile(cb_g, k, 3);
                mul_binary_tile_init();
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    ktt_rows_bwd_affine,
                    2,
                    0,
                    1,
                    VectorMode::RC
                )));
                if (mixed) {
                    copy(cb_mix, 2, 0);
                } else {
                    row_bcast(cb_pqr, 2, 0);
                }
                add_binary_tile_init();
                add_binary_tile(2, 0, 2);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_y);
                pack_tile<true>(2, cb_y, k);
                tile_regs_release();
            }
            cb_push_back(cb_y, block);
            cb_pop_front(cb_x, block);
            cb_pop_front(cb_g, block);
        }
        cb_pop_front(cb_pqr, 3);
        if (shared) {
            cb_pop_front(cb_mix, 3);
        }
        q += p.len;
    }
}
