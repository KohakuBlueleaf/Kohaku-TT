// Attention compute: per unit and K chunk, S = scale Q K^T (-inf past the last valid
// key and, causal, past the diagonal), m' = max(m, rowmax S), P = exp(S - m'), a =
// exp(m - m'), l' = l a + rowsum P, O' = O a + P V; per unit out = O / l.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {

    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_mask = 3, cb_ones = 4;
    constexpr uint32_t cb_s = 5, cb_p = 6, cb_alpha = 11, cb_linv = 14, cb_out = 16;
    // DST tiles per acquire (half of DST: 8 at 16 bits, 4 at fp32).
    constexpr uint32_t kDst = DST_ACCUM_MODE ? 4 : 8;

    // Packs DST tiles 0 .. n - 1 into `cb` at tile indices first .. first + n - 1 and
    // releases the registers.
    void pack_block(uint32_t cb, uint32_t first, uint32_t n) {
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < n; ++t) {
            pack_tile<true>(t, cb, first + t);
        }
        tile_regs_release();
    }

}  // namespace

// Common runtime args: unit step, units, Sq tiles, Sk tiles, D tiles, Dv tiles, q chunk
// tiles, k chunk tiles, valid keys of the last K tile (32 when full), scale (fp32
// bits), causal; per core: first unit. K chunks arrive transposed: tile (key j, d) at
// d kc + j.
void kernel_main() {
    const uint32_t step = get_common_arg_val<uint32_t>(0);
    const uint32_t units = get_common_arg_val<uint32_t>(1);
    const uint32_t sqt = get_common_arg_val<uint32_t>(2);
    const uint32_t skt = get_common_arg_val<uint32_t>(3);
    const uint32_t dt = get_common_arg_val<uint32_t>(4);
    const uint32_t dvt = get_common_arg_val<uint32_t>(5);
    const uint32_t qc = get_common_arg_val<uint32_t>(6);
    const uint32_t kc = get_common_arg_val<uint32_t>(7);
    const uint32_t k_valid = get_common_arg_val<uint32_t>(8);
    const uint32_t scale = get_common_arg_val<uint32_t>(9);
    const bool causal = get_common_arg_val<uint32_t>(10) != 0;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t cb_m[2] = {7, 8}, cb_l[2] = {9, 10}, cb_o[2] = {12, 13};

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
    cb_wait_front(cb_mask, 3);
    cb_wait_front(cb_ones, 1);

    const uint32_t q_chunks = (sqt + qc - 1) / qc;
    for (uint32_t u = start; u < units; u += step) {
        const uint32_t q0 = (u % q_chunks) * qc;
        const uint32_t nq = sqt - q0 < qc ? sqt - q0 : qc;
        cb_wait_front(cb_q, qc * dt);
        uint32_t cur = 0;
        bool first = true;
        const uint32_t k_end = causal && q0 + nq < skt ? q0 + nq : skt;
        for (uint32_t k0 = 0; k0 < k_end; k0 += kc) {
            const uint32_t nk = k_end - k0 < kc ? k_end - k0 : kc;
            const bool ragged = k_valid < 32 && k0 + nk == skt;
            const uint32_t mo = cb_m[cur], mn = cb_m[1 - cur];
            const uint32_t lo = cb_l[cur], ln = cb_l[1 - cur];
            const uint32_t oo = cb_o[cur], on = cb_o[1 - cur];
            cb_wait_front(cb_k, kc * dt);
            cb_wait_front(cb_v, kc * dvt);

            // S = scale Q K^T in blocks of up to kDst tiles; tiles needing a mask (0
            // ragged keys, 1 causal diagonal, 2 all -inf) one at a time.
            pack_reconfig_data_format(cb_s);
            cb_reserve_back(cb_s, qc * kc);
            for (uint32_t i = 0; i < nq; ++i) {
                const uint32_t qi = q0 + i;
                uint32_t j0 = 0;
                while (j0 < nk) {
                    const uint32_t kj = k0 + j0;
                    const bool masked = (
                        (causal && kj >= qi) || (ragged && j0 == nk - 1)
                    );
                    uint32_t ct = 1;
                    if (!masked) {
                        // Unmasked run: up to the diagonal / ragged tile, at most kDst.
                        uint32_t end = nk;
                        if (causal && qi < k0 + nk) {
                            end = qi > k0 ? qi - k0 : 0;
                        }
                        if (ragged && end > nk - 1) {
                            end = nk - 1;
                        }
                        ct = end - j0 < kDst ? end - j0 : kDst;
                    }
                    reconfig_data_format(cb_k, cb_q);
                    matmul_block_init(cb_q, cb_k, 1, ct, 1, dt);
                    tile_regs_acquire();
                    for (uint32_t d = 0; d < dt; ++d) {
                        matmul_block(
                            cb_q, cb_k, i * dt + d, d * kc + j0, 0, 1, ct, 1, dt
                        );
                    }
                    binop_with_scalar_tile_init();
                    for (uint32_t t = 0; t < ct; ++t) {
                        mul_unary_tile(t, scale);
                    }
                    if (masked) {
                        reconfig_data_format_srca(cb_mask);
                        copy_init(cb_mask);
                        add_binary_tile_init();
                        if (causal && kj >= qi) {
                            copy_tile(cb_mask, kj > qi ? 2 : 1, 1);
                            add_binary_tile(0, 1, 0);
                        }
                        if (ragged && j0 == nk - 1) {
                            copy_tile(cb_mask, 0, 1);
                            add_binary_tile(0, 1, 0);
                        }
                    }
                    pack_block(cb_s, i * kc + j0, ct);
                    j0 += ct;
                }
            }
            cb_push_back(cb_s, qc * kc);
            cb_wait_front(cb_s, qc * kc);

            // m' = max(m, rowmax S).
            if (!first) {
                cb_wait_front(mo, qc);
            }
            pack_reconfig_data_format(mn);
            cb_reserve_back(mn, qc);
            for (uint32_t i = 0; i < nq; ++i) {
                reconfig_data_format(cb_s, cb_ones);
                reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_ones, mn);
                tile_regs_acquire();
                for (uint32_t j = 0; j < nk; ++j) {
                    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                        cb_s, cb_ones, i * kc + j, 0, 0
                    );
                }
                reduce_uninit();
                if (!first) {
                    reconfig_data_format_srca(mo);
                    copy_init(mo);
                    copy_tile(mo, i, 1);
                    binary_max_tile_init();
                    binary_max_tile(0, 1, 0);
                }
                pack_block(mn, i, 1);
            }
            cb_push_back(mn, qc);
            cb_wait_front(mn, qc);

            // P = exp(S - m').
            reconfig_data_format(cb_s, mn);
            pack_reconfig_data_format(cb_p);
            sub_bcast_cols_init(cb_s, mn);
            exp_tile_init();
            cb_reserve_back(cb_p, qc * kc);
            for (uint32_t i = 0; i < nq; ++i) {
                for (uint32_t j0 = 0; j0 < nk; j0 += kDst) {
                    const uint32_t ct = nk - j0 < kDst ? nk - j0 : kDst;
                    tile_regs_acquire();
                    for (uint32_t t = 0; t < ct; ++t) {
                        sub_tiles_bcast_cols(cb_s, mn, i * kc + j0 + t, i, t);
                    }
                    for (uint32_t t = 0; t < ct; ++t) {
                        exp_tile(t);
                    }
                    pack_block(cb_p, i * kc + j0, ct);
                }
            }
            cb_push_back(cb_p, qc * kc);
            cb_pop_front(cb_s, qc * kc);
            cb_wait_front(cb_p, qc * kc);

            // a = exp(m - m').
            if (!first) {
                reconfig_data_format(mo, mn);
                pack_reconfig_data_format(cb_alpha);
                sub_init(mo, mn);
                exp_tile_init();
                cb_reserve_back(cb_alpha, qc);
                for (uint32_t i0 = 0; i0 < nq; i0 += kDst) {
                    const uint32_t ct = nq - i0 < kDst ? nq - i0 : kDst;
                    tile_regs_acquire();
                    for (uint32_t t = 0; t < ct; ++t) {
                        sub_tiles(mo, mn, i0 + t, i0 + t, t);
                    }
                    for (uint32_t t = 0; t < ct; ++t) {
                        exp_tile(t);
                    }
                    pack_block(cb_alpha, i0, ct);
                }
                cb_push_back(cb_alpha, qc);
                cb_wait_front(cb_alpha, qc);
                cb_wait_front(lo, qc);
                cb_wait_front(oo, qc * dvt);
            }

            // l' = l a + rowsum P.
            pack_reconfig_data_format(ln);
            cb_reserve_back(ln, qc);
            for (uint32_t i = 0; i < nq; ++i) {
                reconfig_data_format(cb_ones, cb_p);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_ones, ln);
                tile_regs_acquire();
                for (uint32_t j = 0; j < nk; ++j) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                        cb_p, cb_ones, i * kc + j, 0, 0
                    );
                }
                reduce_uninit();
                if (!first) {
                    reconfig_data_format(lo, cb_alpha);
                    mul_init(lo, cb_alpha);
                    mul_tiles(lo, cb_alpha, i, i, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
                pack_block(ln, i, 1);
            }
            cb_push_back(ln, qc);

            // O' = O a + P V, in blocks of up to kDst output tiles.
            pack_reconfig_data_format(on);
            cb_reserve_back(on, qc * dvt);
            for (uint32_t i = 0; i < nq; ++i) {
                for (uint32_t n0 = 0; n0 < dvt; n0 += kDst) {
                    const uint32_t ct = dvt - n0 < kDst ? dvt - n0 : kDst;
                    tile_regs_acquire();
                    if (!first) {
                        reconfig_data_format(oo, cb_alpha);
                        mul_bcast_cols_init(oo, cb_alpha);
                        for (uint32_t t = 0; t < ct; ++t) {
                            mul_tiles_bcast_cols(oo, cb_alpha, i * dvt + n0 + t, i, t);
                        }
                    }
                    reconfig_data_format(cb_v, cb_p);
                    matmul_block_init(cb_p, cb_v, 0, ct, 1, kc);
                    for (uint32_t j = 0; j < nk; ++j) {
                        matmul_block(
                            cb_p, cb_v, i * kc + j, j * dvt + n0, 0, 0, ct, 1, kc
                        );
                    }
                    pack_block(on, i * dvt + n0, ct);
                }
            }
            cb_push_back(on, qc * dvt);

            cb_pop_front(cb_p, qc * kc);
            cb_pop_front(cb_k, kc * dt);
            cb_pop_front(cb_v, kc * dvt);
            if (!first) {
                cb_pop_front(mo, qc);
                cb_pop_front(lo, qc);
                cb_pop_front(oo, qc * dvt);
                cb_pop_front(cb_alpha, qc);
            }
            cur = 1 - cur;
            first = false;
        }

        // out = O / l.
        const uint32_t mf = cb_m[cur], lf = cb_l[cur], of = cb_o[cur];
        cb_wait_front(lf, qc);
        cb_wait_front(of, qc * dvt);
        reconfig_data_format_srca(lf);
        pack_reconfig_data_format(cb_linv);
        copy_init(lf);
        recip_tile_init();
        cb_reserve_back(cb_linv, qc);
        for (uint32_t i0 = 0; i0 < nq; i0 += kDst) {
            const uint32_t ct = nq - i0 < kDst ? nq - i0 : kDst;
            tile_regs_acquire();
            for (uint32_t t = 0; t < ct; ++t) {
                copy_tile(lf, i0 + t, t);
            }
            for (uint32_t t = 0; t < ct; ++t) {
                recip_tile(t);
            }
            pack_block(cb_linv, i0, ct);
        }
        cb_push_back(cb_linv, qc);
        cb_wait_front(cb_linv, qc);
        reconfig_data_format(of, cb_linv);
        pack_reconfig_data_format(cb_out);
        mul_bcast_cols_init(of, cb_linv);
        for (uint32_t i = 0; i < nq; ++i) {
            cb_reserve_back(cb_out, dvt);
            for (uint32_t n0 = 0; n0 < dvt; n0 += kDst) {
                const uint32_t ct = dvt - n0 < kDst ? dvt - n0 : kDst;
                tile_regs_acquire();
                for (uint32_t t = 0; t < ct; ++t) {
                    mul_tiles_bcast_cols(of, cb_linv, i * dvt + n0 + t, i, t);
                }
                pack_block(cb_out, n0, ct);
            }
            cb_push_back(cb_out, dvt);
        }
        cb_pop_front(cb_linv, qc);
        cb_pop_front(mf, qc);
        cb_pop_front(lf, qc);
        cb_pop_front(of, qc * dvt);
        cb_pop_front(cb_q, qc * dt);
    }
}
