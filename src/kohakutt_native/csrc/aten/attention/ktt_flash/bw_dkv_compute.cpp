// Flash attention backward compute, dK and dV per job of up to R key tile rows against
// streamed query blocks (design: docs/reference/kernels/flash.md, Backward). Default:
// scores as the forward computes them (S = Q K^T); FLASH_BW_KQ: the key-row orientation
// (S^T = K Q^T).
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "flash.hpp"
#include "flash_bw.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpu.hpp"
#endif
#ifdef FLASH_PROFILE
#include "api/debug/kernel_profiler.hpp"
#define ZONE(name) DeviceZoneScopedN(name)
#else
#define ZONE(name)
#endif

namespace {
    using namespace ktt_flash_bw::dkv;

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t QB = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4);  // query heads per K / V head
    constexpr uint32_t A_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(6);
    constexpr uint32_t SKT = get_compile_time_arg_val(7);  // key tiles per head
    constexpr uint32_t TAIL = get_compile_time_arg_val(
        8
    );  // keys in the last key tile; 0: 32
    constexpr uint32_t NO_MASK = 0xffffffff;
    constexpr uint32_t SC = ktt_flash_bw::kChunk;
    constexpr uint32_t DST = ktt_flash_bw::kDstTiles;
#ifdef FLASH_BW_KQ
    constexpr bool KQ = true;
    constexpr uint32_t NST = 2;  // statistics tiles per query tile row
    // Streamed operands of the dV and dK products.
    constexpr uint32_t cb_sv = cb_do, cb_sk = cb_q;
#else
    constexpr bool KQ = false;
    constexpr uint32_t NST = 1;
    constexpr uint32_t cb_sv = cb_dot, cb_sk = cb_qt;
#endif
#ifdef FLASH_VARLEN
    // The masks come per key row from the key-side plan (cb_um), not from constant
    // tiles.
    static_assert(
        !KQ && TAIL == 0,
        "the packed layout runs the default orientation with its plan's masks"
    );
#endif

    constexpr uint32_t width(uint32_t n, uint32_t i0) {
        return n - i0 < DST ? n - i0 : DST;
    }

    bool pack_fp32 = false;
    bool pack_acc = false;

    void pack_format(bool fp32) {
        if (fp32 != pack_fp32) {
            pack_reconfig_data_format(fp32 ? cb_p : cb_dk, fp32 ? cb_dk : cb_p);
            pack_fp32 = fp32;
        }
    }

    void pack_accumulate(bool acc) {
        if (acc != pack_acc) {
            pack_reconfig_l1_acc(acc ? 1 : 0);
            pack_acc = acc;
        }
    }

    void math_vector_begin() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH)));
        MATH((math::set_dst_write_addr<
              DstTileShape::Tile32x32,
              UnpackDestination::SrcRegs>(0)));
    }

    void math_vector_end() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::WAIT_SFPU)));
    }

    // Pack thread: waits for the math section, vector unit on the current Dst half.
    void vector_begin() {
        PACK((TTI_SEMWAIT(
            p_stall::STALL_SFPU | p_stall::STALL_SYNC,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_ZERO
        )));
        PACK((math::set_dst_write_addr<
              DstTileShape::Tile32x32,
              UnpackDestination::SrcRegs>(0)));
    }

    void vector_end() {
        PACK((
            TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_PACK, p_stall::WAIT_SFPU)
        ));
    }

    // Holds the unpacker until every pack issued so far is in L1.
    void pack_barrier() {
        cb_reserve_back(cb_tok, 1);
        cb_push_back(cb_tok, 1);
        cb_wait_front(cb_tok, 1);
        cb_pop_front(cb_tok, 1);
    }

#ifdef FLASH_BW_KQ
    static_assert(TAIL == 0, "the key-row orientation takes whole key tiles");
#endif

    // The mask tiles, built by the vector unit and packed once: with FLASH_CAUSAL the
    // diagonal tile's (cb_causal: keys after the query masked, in the score tile's
    // orientation); with TAIL the last key tile's (cb_tail) and with both the two
    // combined (cb_ctail).
    void constant_tiles() {
#ifndef FLASH_CAUSAL
        if constexpr (TAIL == 0) {
            return;
        }
#endif
        tile_regs_acquire_math_clear();
        math_vector_begin();
        MATH((math::clear_dst_reg_addr()));
        MATH((flash::const_lanes()));
#ifdef FLASH_CAUSAL
        MATH(
            (flash::const_tile(KQ ? flash::Const::CausalT : flash::Const::Causal, 0, 0))
        );
#endif
        if constexpr (TAIL != 0) {
            MATH((flash::const_tile(flash::Const::Tail, 1, TAIL)));
#ifdef FLASH_CAUSAL
            MATH((flash::const_tile(flash::Const::CausalTail, 2, TAIL)));
#endif
        }
        MATH((math::clear_dst_reg_addr()));
        math_vector_end();
        tile_regs_commit();
        tile_regs_wait();
        pack_format(false);
        pack_accumulate(false);
#ifdef FLASH_CAUSAL
        cb_reserve_back(cb_causal, 1);
        pack_tile(0, cb_causal);
        cb_push_back(cb_causal, 1);
#endif
        if constexpr (TAIL != 0) {
            cb_reserve_back(cb_tail, 1);
            pack_tile(1, cb_tail);
            cb_push_back(cb_tail, 1);
#ifdef FLASH_CAUSAL
            cb_reserve_back(cb_ctail, 1);
            pack_tile(2, cb_ctail);
            cb_push_back(cb_ctail, 1);
#endif
        }
        tile_regs_release_math_clear();
#ifdef FLASH_CAUSAL
        cb_wait_front(cb_causal, 1);
#endif
        if constexpr (TAIL != 0) {
            cb_wait_front(cb_tail, 1);
#ifdef FLASH_CAUSAL
            cb_wait_front(cb_ctail, 1);
#endif
        }
    }

    // Tiles [first, first + n) of `src` (bf16), each transposed by the unpacker on its
    // way to Dst, to slots [o0, o0 + n) of the reserved `dst` (bf16).
    [[maybe_unused]] void turn(
        uint32_t src, uint32_t first, uint32_t n, uint32_t dst, uint32_t o0
    ) {
        ZONE("T");
        for (uint32_t t0 = 0; t0 < n; t0 += DST) {
            const uint32_t w = width(n, t0);
            tile_regs_acquire_math_clear();
            copy_init(src, 1, 1);
            for (uint32_t t = 0; t < w; ++t) {
                copy_tile(src, first + t0 + t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_format(false);
            pack_accumulate(false);
            for (uint32_t t = 0; t < w; ++t) {
                pack_tile<true>(t, dst, o0 + t0 + t);
            }
            tile_regs_release_math_clear();
        }
    }

    // P and dS of query tile i (in the block) against key row j into slot t of the
    // reserved cb_p / cb_ds chunks. Default: S = Q_i K_j^T, dP = dO_i V_j^T, the query
    // tile's row statistics tile into Dst 3. FLASH_BW_KQ: S^T = K_j Q_i^T, dP^T = V_j
    // dO_i^T, the query tile's column statistics (D, w) into Dst 2 and 3. `mask`: the
    // buffer of the score tile's mask tile (its tile `mt`), or NO_MASK.
    void score(uint32_t i, uint32_t j, uint32_t t, uint32_t mask, uint32_t mt = 0) {
        ZONE("S");
        tile_regs_acquire_math_clear();
        if (mask != NO_MASK) {
            // The mask first; the score matmul accumulates onto it.
            copy_init(mask);
            copy_tile(mask, mt, 0);
        }
#ifdef FLASH_MASK_BCAST
        else {
            // The key mask's row of key row j (a mask tile holds it).
            unary_bcast_init<BroadcastType::ROW>(cb_mb);
            unary_bcast<BroadcastType::ROW>(cb_mb, j, 0);
            unary_bcast_uninit<BroadcastType::ROW>(cb_mb);
        }
#endif
#ifdef FLASH_BW_KQ
        mm_no_mop_init_short(cb_k, cb_q, true, 1, 1, DT);
        for (uint32_t d = 0; d < DT; ++d) {
            matmul_block_no_mop(cb_k, cb_q, j * DT + d, i * DT + d, 0, true, 1, 1, DT);
        }
        mm_no_mop_init_short(cb_v, cb_do, true, 1, 1, DV);
        for (uint32_t u = 0; u < DV; ++u) {
            matmul_block_no_mop(cb_v, cb_do, j * DV + u, i * DV + u, 1, true, 1, 1, DV);
        }
        reconfig_data_format_srca(cb_do, cb_wd);
        copy_init(cb_wd);
        copy_tile(cb_wd, 2 * i + 1, 2);
        copy_tile(cb_wd, 2 * i, 3);
        reconfig_data_format_srca(cb_wd, cb_do);
#else
        mm_no_mop_init_short(cb_q, cb_k, true, 1, 1, DT);
        for (uint32_t d = 0; d < DT; ++d) {
            matmul_block_no_mop(cb_q, cb_k, i * DT + d, j * DT + d, 0, true, 1, 1, DT);
        }
        mm_no_mop_init_short(cb_do, cb_v, true, 1, 1, DV);
        for (uint32_t u = 0; u < DV; ++u) {
            matmul_block_no_mop(cb_do, cb_v, i * DV + u, j * DV + u, 1, true, 1, 1, DV);
        }
        reconfig_data_format_srca(cb_v, cb_wd);
        copy_init(cb_wd);
        copy_tile(cb_wd, i, 3);
        reconfig_data_format_srca(cb_wd, cb_v);
#endif
        tile_regs_commit();

        tile_regs_wait();
        vector_begin();
#ifdef FLASH_BW_KQ
        PACK((flash::p_ds_step_cols()));
#else
        PACK((flash::p_ds_step<true>()));
#endif
        vector_end();
        pack_format(false);
        pack_accumulate(false);
        pack_tile<true>(0, cb_p, t);
        pack_tile<true>(1, cb_ds, t);
        tile_regs_release_math_clear();
    }

    // Dst tiles d0 .. d0 + w - 1 += the products for output tiles u0 .. u0 + w - 1 (of
    // n per row), summed over the nt chunk tiles of `chunk` against `stream` tiles
    // (first + t, u). Default: stream^T x chunk (outputs transposed); FLASH_BW_KQ:
    // chunk x stream.
    void products(
        uint32_t stream,
        uint32_t chunk,
        uint32_t first,
        uint32_t n,
        uint32_t u0,
        uint32_t w,
        uint32_t d0,
        uint32_t nt
    ) {
        if constexpr (KQ) {
            mm_no_mop_init_short(chunk, stream, false, 1, 1, nt);
        } else {
            mm_no_mop_init_short(stream, chunk, false, 1, 1, nt);
        }
        for (uint32_t u = 0; u < w; ++u) {
            for (uint32_t t = 0; t < nt; ++t) {
                const uint32_t s = (first + t) * n + u0 + u;
                if constexpr (KQ) {
                    matmul_block_no_mop(chunk, stream, t, s, d0 + u, false, 1, 1, nt);
                } else {
                    matmul_block_no_mop(stream, chunk, s, t, d0 + u, false, 1, 1, nt);
                }
            }
        }
    }

    // Row j of accumulator `acc` (n tiles per row) += `chunk` against `stream` from
    // stream tile row `first`, nt tiles.
    void accumulate_into(
        uint32_t stream,
        uint32_t chunk,
        uint32_t acc,
        uint32_t j,
        uint32_t first,
        uint32_t nt,
        uint32_t n,
        bool first_chunk
    ) {
        for (uint32_t u0 = 0; u0 < n; u0 += DST) {
            const uint32_t w = width(n, u0);
            tile_regs_acquire_math_clear();
            products(stream, chunk, first, n, u0, w, 0, nt);
            tile_regs_commit();
            tile_regs_wait();
            pack_format(true);
            pack_accumulate(!first_chunk);
            for (uint32_t u = 0; u < w; ++u) {
                pack_tile<true>(u, acc, j * n + u0 + u);
            }
            tile_regs_release_math_clear();
        }
    }

    // dV_j and dK_j accumulators += the chunk's P and dS (nt tiles) against the
    // streamed tiles from tile row `first`; one Dst section for both when DV + DT tiles
    // fit.
    void accumulate(uint32_t j, uint32_t first, uint32_t nt, bool first_chunk) {
        ZONE("A");
        cb_wait_front(cb_p, SC);
        cb_wait_front(cb_ds, SC);
        if constexpr (DV + DT <= DST) {
            tile_regs_acquire_math_clear();
            products(cb_sv, cb_p, first, DV, 0, DV, 0, nt);
            products(cb_sk, cb_ds, first, DT, 0, DT, DV, nt);
            tile_regs_commit();
            tile_regs_wait();
            pack_format(true);
            pack_accumulate(!first_chunk);
            for (uint32_t u = 0; u < DV; ++u) {
                pack_tile<true>(u, cb_dv, j * DV + u);
            }
            for (uint32_t d = 0; d < DT; ++d) {
                pack_tile<true>(DV + d, cb_dk, j * DT + d);
            }
            tile_regs_release_math_clear();
        } else {
            accumulate_into(cb_sv, cb_p, cb_dv, j, first, nt, DV, first_chunk);
            accumulate_into(cb_sk, cb_ds, cb_dk, j, first, nt, DT, first_chunk);
        }
        cb_pop_front(cb_p, SC);
        cb_pop_front(cb_ds, SC);
    }

    // Row j of fp32 accumulator `acc` (n tiles) to bf16 slots [o0, o0 + n) of the
    // reserved `out` (fp32 unpacked straight to Dst); with `scaled` times the softmax
    // scale.
    void row_out(
        uint32_t acc, uint32_t j, uint32_t n, bool scaled, uint32_t out, uint32_t o0
    ) {
        for (uint32_t u0 = 0; u0 < n; u0 += DST) {
            const uint32_t w = width(n, u0);
            tile_regs_acquire_math_clear();
            reconfig_data_format_srca(cb_k, acc);
            copy_init(acc);
            for (uint32_t u = 0; u < w; ++u) {
                copy_tile(acc, j * n + u0 + u, u);
            }
            reconfig_data_format_srca(acc, cb_k);
            tile_regs_commit();
            tile_regs_wait();
            if (scaled) {
                vector_begin();
                PACK((flash::scale_step(w, SCALE_BITS)));
                vector_end();
            }
            pack_format(false);
            pack_accumulate(false);
            for (uint32_t u = 0; u < w; ++u) {
                pack_tile<true>(u, out, o0 + u0 + u);
            }
            tile_regs_release_math_clear();
        }
    }

    // Row j of fp32 accumulator `acc` (n tiles) to zero.
    [[maybe_unused]] void clear_row(uint32_t acc, uint32_t j, uint32_t n) {
        for (uint32_t u0 = 0; u0 < n; u0 += DST) {
            const uint32_t w = width(n, u0);
            tile_regs_acquire_math_clear();
            tile_regs_commit();
            tile_regs_wait();
            pack_format(true);
            pack_accumulate(false);
            for (uint32_t u = 0; u < w; ++u) {
                pack_tile<true>(u, acc, j * n + u0 + u);
            }
            tile_regs_release_math_clear();
        }
    }

    // dK_j then dV_j to bf16; the default orientation's accumulators hold them
    // transposed.
    void finalize(uint32_t j) {
        ZONE("F");
        if constexpr (KQ) {
            cb_reserve_back(cb_out, DT + DV);
            row_out(cb_dk, j, DT, true, cb_out, 0);
            row_out(cb_dv, j, DV, false, cb_out, DT);
            cb_push_back(cb_out, DT + DV);
        } else {
            cb_reserve_back(cb_fin, DT + DV);
            row_out(cb_dk, j, DT, true, cb_fin, 0);
            row_out(cb_dv, j, DV, false, cb_fin, DT);
            cb_push_back(cb_fin, DT + DV);
            cb_wait_front(cb_fin, DT + DV);
            cb_reserve_back(cb_out, DT + DV);
            turn(cb_fin, 0, DT + DV, cb_out, 0);
            cb_push_back(cb_out, DT + DV);
            cb_pop_front(cb_fin, DT + DV);
        }
    }
}  // namespace

// Runtime args: iterations, jobs, then per job: key tile rows (at most R), the index of
// its first row within the head, query tiles streamed per head, the row step between
// its rows, the first streamed query tile (within the head).
void kernel_main() {
    const uint32_t iters = get_arg_val<uint32_t>(0);
    const uint32_t jobs = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_k, cb_q, cb_p);
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
    MATH((flash::init(__builtin_bit_cast(float, A_BITS))));
    PACK((flash::addr_init()));
    constant_tiles();
    cb_reserve_back(cb_dk, R * DT);
    cb_reserve_back(cb_dv, R * DV);

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t rows = get_arg_val<uint32_t>(2 + 5 * job);
            const uint32_t row0 = get_arg_val<uint32_t>(3 + 5 * job);
            const uint32_t nq = get_arg_val<uint32_t>(4 + 5 * job);
            const uint32_t stride = get_arg_val<uint32_t>(5 + 5 * job);
            const uint32_t q_first = get_arg_val<uint32_t>(6 + 5 * job);
#ifdef FLASH_VARLEN
            // row0: the job's first key tile row. The rows' records hold their query
            // tiles, here relative to the stream's first (q_first).
            if (rows == 0 && nq == 0) {
                continue;  // an empty job (the job list is padded to a fixed length)
            }
            using ktt_flash::kPlanPages;
            using ktt_flash::kRecordsPerPage;
            using ktt_flash::kRecordWords;
            ktt_flash::RowWindow win[R];
            if (rows > 0) {
                cb_wait_front(cb_plan, kPlanPages);
            }
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t idx = row0 % kRecordsPerPage + r;
                const uint32_t pg = idx / kRecordsPerPage,
                               w0 = idx % kRecordsPerPage * kRecordWords;
                win[r].first = read_tile_value(cb_plan, pg, w0) - q_first;
                win[r].last = read_tile_value(cb_plan, pg, w0 + 1) - q_first;
                win[r].a = read_tile_value(cb_plan, pg, w0 + 2);
                win[r].b = read_tile_value(cb_plan, pg, w0 + 3);
                win[r].u0 = 0;
            }
#endif
            {
                ZONE("WK");
                cb_wait_front(cb_k, R * DT);
                cb_wait_front(cb_v, R * DV);
#ifdef FLASH_MASK_BCAST
                cb_wait_front(cb_mb, R);
#endif
            }
            // First query tile key row r sees (causal: its own row).
            auto first_query = [&](uint32_t r) {
#ifdef FLASH_CAUSAL
                return row0 + r * stride;
#else
                (void)row0;
                (void)stride;
                (void)r;
                return q_first;
#endif
            };
            bool first_chunk[R];
            for (uint32_t r = 0; r < R; ++r) {
                first_chunk[r] = true;
            }
            for (uint32_t head = 0; head < G; ++head) {
                for (
                    uint32_t i_start = q_first; i_start < q_first + nq; i_start += QB
                ) {
                    const uint32_t qb = q_first + nq - i_start < QB
                        ? q_first + nq - i_start
                        : QB;
                    {
                        ZONE("WQ");
                        cb_wait_front(cb_q, QB * DT);
                        cb_wait_front(cb_do, QB * DV);
                        cb_wait_front(cb_wd, NST * QB);
                    }
                    for (uint32_t i0 = 0; i0 < qb; i0 += SC) {
                        const uint32_t nt_block = qb - i0 < SC ? qb - i0 : SC;
                        const uint32_t ig = i_start + i0;
#ifdef FLASH_VARLEN
                        // Chunk tiles [from(r), to(r)) key row r sees (to <= from:
                        // none).
                        (void)first_query;
                        const uint32_t igr = ig - q_first;
                        auto from = [&](uint32_t r) -> uint32_t {
                            return igr > win[r].first ? 0u : win[r].first - igr;
                        };
                        auto to = [&](uint32_t r) -> uint32_t {
                            if (win[r].empty() || win[r].last < igr) {
                                return 0u;
                            }
                            return win[r].last - igr + 1 < nt_block
                                ? win[r].last - igr + 1
                                : nt_block;
                        };
                        uint32_t c_lo = nt_block, c_hi = 0;
                        for (uint32_t r = 0; r < rows; ++r) {
                            if (from(r) < to(r)) {
                                c_lo = from(r) < c_lo ? from(r) : c_lo;
                                c_hi = to(r) > c_hi ? to(r) : c_hi;
                            }
                        }
                        if (c_lo >= c_hi) {
                            continue;
                        }
                        cb_reserve_back(cb_qt, SC * DT);
                        cb_reserve_back(cb_dot, SC * DV);
                        turn(cb_q, (i0 + c_lo) * DT, (c_hi - c_lo) * DT, cb_qt, 0);
                        turn(cb_do, (i0 + c_lo) * DV, (c_hi - c_lo) * DV, cb_dot, 0);
                        cb_push_back(cb_qt, SC * DT);
                        cb_push_back(cb_dot, SC * DV);
                        cb_wait_front(cb_qt, SC * DT);
                        cb_wait_front(cb_dot, SC * DV);
                        for (uint32_t r = 0; r < rows; ++r) {
                            const uint32_t s0 = from(r), e0 = to(r);
                            if (s0 >= e0) {
                                continue;
                            }
                            const ktt_flash::RowWindow& w = win[r];
                            const uint32_t rel0 = igr + s0 - w.first;
                            bool any = false;
                            for (uint32_t t = 0; t < e0 - s0; ++t) {
                                any = any || w.masked(rel0 + t);
                            }
                            if (any) {
                                cb_wait_front(cb_um, SC);
                            }
                            cb_reserve_back(cb_p, SC);
                            cb_reserve_back(cb_ds, SC);
                            for (uint32_t t = 0; t < e0 - s0; ++t) {
                                score(
                                    i0 + s0 + t,
                                    r,
                                    t,
                                    w.masked(rel0 + t) ? cb_um : NO_MASK,
                                    t
                                );
                            }
                            cb_push_back(cb_p, SC);
                            cb_push_back(cb_ds, SC);
                            if (any) {
                                cb_pop_front(cb_um, SC);
                            }
                            accumulate(r, s0 - c_lo, e0 - s0, first_chunk[r]);
                            first_chunk[r] = false;
                        }
                        cb_pop_front(cb_qt, SC * DT);
                        cb_pop_front(cb_dot, SC * DV);
#else
                        // Query tiles [ig + skip, ig + nt_block) row r sees.
                        auto skip = [&](uint32_t r) {
                            const uint32_t f = first_query(r);
                            return f <= ig ? 0u
                                           : (f - ig < nt_block ? f - ig : nt_block);
                        };
                        // The chunk's first tile any row sees (rows ascend).
                        const uint32_t lo = rows > 0 ? skip(0) : nt_block;
                        if (lo == nt_block) {
                            continue;
                        }
                        if constexpr (!KQ) {
                            cb_reserve_back(cb_qt, SC * DT);
                            cb_reserve_back(cb_dot, SC * DV);
                            turn(cb_q, (i0 + lo) * DT, (nt_block - lo) * DT, cb_qt, 0);
                            turn(
                                cb_do, (i0 + lo) * DV, (nt_block - lo) * DV, cb_dot, 0
                            );
                            cb_push_back(cb_qt, SC * DT);
                            cb_push_back(cb_dot, SC * DV);
                            cb_wait_front(cb_qt, SC * DT);
                            cb_wait_front(cb_dot, SC * DV);
                        }
                        for (uint32_t r = 0; r < rows; ++r) {
                            const uint32_t s0 = skip(r);
                            if (s0 == nt_block) {
                                continue;
                            }
                            cb_reserve_back(cb_p, SC);
                            cb_reserve_back(cb_ds, SC);
                            for (uint32_t t = s0; t < nt_block; ++t) {
#ifdef FLASH_CAUSAL
                                const bool diag = ig + t == row0 + r * stride;
#else
                                const bool diag = false;
#endif
                                const bool tail = TAIL != 0 &&
                                    row0 + r * stride == SKT - 1;
#ifdef FLASH_CAUSAL
                                const uint32_t mask = diag
                                    ? (tail ? cb_ctail : cb_causal)
                                    : (tail ? cb_tail : NO_MASK);
#else
                                const uint32_t mask = diag || !tail ? NO_MASK : cb_tail;
#endif
                                score(i0 + t, r, t - s0, mask);
                            }
                            cb_push_back(cb_p, SC);
                            cb_push_back(cb_ds, SC);
                            accumulate(
                                r, KQ ? i0 + s0 : s0 - lo, nt_block - s0, first_chunk[r]
                            );
                            first_chunk[r] = false;
                        }
                        if constexpr (!KQ) {
                            cb_pop_front(cb_qt, SC * DT);
                            cb_pop_front(cb_dot, SC * DV);
                        }
#endif
                    }
                    cb_pop_front(cb_q, QB * DT);
                    cb_pop_front(cb_do, QB * DV);
                    cb_pop_front(cb_wd, NST * QB);
                }
            }
#ifdef FLASH_VARLEN
            // Key rows no query sees: zero gradients.
            for (uint32_t r = 0; r < rows; ++r) {
                if (first_chunk[r]) {
                    clear_row(cb_dk, r, DT);
                    clear_row(cb_dv, r, DV);
                }
            }
            if (rows > 0) {
                cb_pop_front(cb_plan, ktt_flash::kPlanPages);
            }
#endif
            // The accumulators are read back from L1.
            pack_barrier();
            for (uint32_t r = 0; r < rows; ++r) {
                finalize(r);
            }
            cb_pop_front(cb_k, R * DT);
            cb_pop_front(cb_v, R * DV);
#ifdef FLASH_MASK_BCAST
            cb_pop_front(cb_mb, R);
#endif
        }
    }
    PACK((flash::restore_constants()));
}
