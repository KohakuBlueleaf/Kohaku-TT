// Flash attention backward compute, dQ and the row statistics (w from the log-sum-exp,
// D = rowsum(dO O)) per job of up to R query tile rows (design:
// docs/reference/kernels/flash.md, Backward). FLASH_BW_KQ: the statistics go out in
// column layout.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
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
    using namespace ktt_flash_bw::dq;

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t KB = get_compile_time_arg_val(3);
    constexpr uint32_t A_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t SQT = get_compile_time_arg_val(6);  // query tiles per head
    constexpr uint32_t SKT = get_compile_time_arg_val(7);  // key tiles per head
    constexpr uint32_t QTAIL = get_compile_time_arg_val(
        8
    );  // queries in the last query tile; 0: 32
    constexpr uint32_t TAIL = get_compile_time_arg_val(
        9
    );  // keys in the last key tile; 0: 32
    constexpr uint32_t NO_MASK = 0xffffffff;
    constexpr uint32_t SC = ktt_flash_bw::kChunk;
    constexpr uint32_t DST = ktt_flash_bw::kDstTiles;
#ifdef FLASH_BW_KQ
    constexpr uint32_t NST = 2;  // statistics tiles written per query tile row
#else
    constexpr uint32_t NST = 1;
#endif
#ifdef FLASH_VARLEN
    // Runtime args per job: also the stream's first key tile. The masks come per row
    // from the plan (cb_um), not from constant tiles.
    constexpr uint32_t JA = 5;
    static_assert(QTAIL == 0 && TAIL == 0, "the packed layout masks through its plan");
#else
    constexpr uint32_t JA = 4;
#endif

    constexpr uint32_t width(uint32_t n, uint32_t i0) {
        return n - i0 < DST ? n - i0 : DST;
    }

    bool pack_fp32 = false;
    bool pack_acc = false;

    void pack_format(bool fp32) {
        if (fp32 != pack_fp32) {
            pack_reconfig_data_format(fp32 ? cb_ds : cb_acc, fp32 ? cb_acc : cb_ds);
            pack_fp32 = fp32;
        }
    }

    void pack_accumulate(bool acc) {
        if (acc != pack_acc) {
            pack_reconfig_l1_acc(acc ? 1 : 0);
            pack_acc = acc;
        }
    }

    // Math-thread vector section: waits for the matmuls, holds the vector unit.
    void math_vector_begin() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH)));
        MATH((math::set_dst_write_addr<
              DstTileShape::Tile32x32,
              UnpackDestination::SrcRegs>(0)));
    }

    void math_vector_end() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::WAIT_SFPU)));
    }

    // The constant tiles, built by the vector unit and packed once: ones (cb_ones);
    // with FLASH_CAUSAL the diagonal mask (cb_causal); with TAIL the last key tile's
    // mask (cb_tail) and with both the two combined (cb_ctail); with QTAIL the last
    // query tile's row mask (cb_rtail).
    void constant_tiles() {
        tile_regs_acquire_math_clear();
        math_vector_begin();
        MATH((math::clear_dst_reg_addr()));
        MATH((flash::const_lanes()));
        MATH((flash::const_tile(flash::Const::Ones, 0, 0)));
#ifdef FLASH_CAUSAL
        MATH((flash::const_tile(flash::Const::Causal, 1, 0)));
#endif
        if constexpr (TAIL != 0) {
            MATH((flash::const_tile(flash::Const::Tail, 2, TAIL)));
#ifdef FLASH_CAUSAL
            MATH((flash::const_tile(flash::Const::CausalTail, 3, TAIL)));
#endif
        }
        MATH((math::clear_dst_reg_addr()));
        math_vector_end();
        tile_regs_commit();
        tile_regs_wait();
        pack_format(false);
        pack_accumulate(false);
        cb_reserve_back(cb_ones, 1);
        pack_tile(0, cb_ones);
        cb_push_back(cb_ones, 1);
#ifdef FLASH_CAUSAL
        cb_reserve_back(cb_causal, 1);
        pack_tile(1, cb_causal);
        cb_push_back(cb_causal, 1);
#endif
        if constexpr (TAIL != 0) {
            cb_reserve_back(cb_tail, 1);
            pack_tile(2, cb_tail);
            cb_push_back(cb_tail, 1);
#ifdef FLASH_CAUSAL
            cb_reserve_back(cb_ctail, 1);
            pack_tile(3, cb_ctail);
            cb_push_back(cb_ctail, 1);
#endif
        }
        tile_regs_release_math_clear();
        if constexpr (QTAIL != 0) {
            tile_regs_acquire_math_clear();
            math_vector_begin();
            MATH((math::clear_dst_reg_addr()));
            MATH((flash::const_lanes()));
            MATH((flash::const_tile(flash::Const::TailRows, 0, QTAIL)));
            MATH((math::clear_dst_reg_addr()));
            math_vector_end();
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_rtail, 1);
            pack_tile(0, cb_rtail);
            cb_push_back(cb_rtail, 1);
            tile_regs_release_math_clear();
            cb_wait_front(cb_rtail, 1);
        }
        cb_wait_front(cb_ones, 1);
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

    // Row r's statistics (fp32, flash_bw.hpp): its row tile to cb_w slot r; to cb_dout
    // the row tile, or with FLASH_BW_KQ its column tiles (w, then D). `tail`: the tile
    // holds rows past the last query, whose w is masked.
    void row_d(uint32_t r, bool tail) {
        ZONE("D");
        cb_reserve_back(cb_prod, DV);
        for (uint32_t t0 = 0; t0 < DV; t0 += DST) {
            const uint32_t w = width(DV, t0);
            tile_regs_acquire_math_clear();
            mul_init(cb_do, cb_oin, false);
            for (uint32_t t = 0; t < w; ++t) {
                mul_tiles(cb_do, cb_oin, r * DV + t0 + t, r * DV + t0 + t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_format(false);
            pack_accumulate(false);
            for (uint32_t t = 0; t < w; ++t) {
                pack_tile(t, cb_prod);
            }
            tile_regs_release_math_clear();
        }
        cb_push_back(cb_prod, DV);

        cb_wait_front(cb_prod, DV);
        tile_regs_acquire_math_clear();
        mm_no_mop_init_short(cb_prod, cb_ones, false, 1, 1, DV);
        for (uint32_t t = 0; t < DV; ++t) {
            matmul_block_no_mop(cb_prod, cb_ones, t, 0, 0, false, 1, 1, DV);
        }
        if (tail) {
            copy_init(cb_rtail);
            copy_tile(cb_rtail, 0, 1);
        }
        reconfig_data_format_srca(cb_ones, cb_lse);
        copy_init(cb_lse);
        copy_tile(cb_lse, r, 3);
        reconfig_data_format_srca(cb_lse, cb_v);
        tile_regs_commit();
        tile_regs_wait();
        vector_begin();
        if (tail) {
            PACK((flash::wd_step_tail()));
        } else {
            PACK((flash::wd_step()));
        }
#ifdef FLASH_BW_KQ
        // Column layouts of w (Dst 3, even columns) and D (Dst 0) into Dst 2 and 1.
        PACK((flash::cols_step(3, 2)));
        PACK((flash::cols_step(0, 1)));
#endif
        vector_end();
        pack_format(true);
        pack_accumulate(false);
        pack_tile<true>(3, cb_w, r);
        cb_reserve_back(cb_dout, NST);
#ifdef FLASH_BW_KQ
        pack_tile(2, cb_dout);
        pack_tile(1, cb_dout);
#else
        pack_tile(3, cb_dout);
#endif
        cb_push_back(cb_dout, NST);
        tile_regs_release_math_clear();
        cb_pop_front(cb_prod, DV);
    }

    // dS of row r against key tile k of the block into slot t of the reserved cb_ds
    // chunk; `mask`: the buffer of the score tile's mask tile (its tile `mt`), or
    // NO_MASK.
    void score(uint32_t r, uint32_t k, uint32_t t, uint32_t mask, uint32_t mt = 0) {
        ZONE("S");
        tile_regs_acquire_math_clear();
        if (mask != NO_MASK) {
            // The mask first; the score matmul accumulates onto it.
            copy_init(mask);
            copy_tile(mask, mt, 0);
        }
#ifdef FLASH_MASK_BCAST
        else {
            // The key mask's row of the key tile (a mask tile holds it).
            unary_bcast_init<BroadcastType::ROW>(cb_mb);
            unary_bcast<BroadcastType::ROW>(cb_mb, k, 0);
            unary_bcast_uninit<BroadcastType::ROW>(cb_mb);
        }
#endif
        mm_no_mop_init_short(cb_q, cb_k, true, 1, 1, DT);
        for (uint32_t d = 0; d < DT; ++d) {
            matmul_block_no_mop(cb_q, cb_k, r * DT + d, d * KB + k, 0, true, 1, 1, DT);
        }
        mm_no_mop_init_short(cb_do, cb_v, true, 1, 1, DV);
        for (uint32_t u = 0; u < DV; ++u) {
            matmul_block_no_mop(cb_do, cb_v, r * DV + u, k * DV + u, 1, true, 1, 1, DV);
        }
        reconfig_data_format_srca(cb_v, cb_w);
        copy_init(cb_w);
        copy_tile(cb_w, r, 3);
        reconfig_data_format_srca(cb_w, cb_v);
        tile_regs_commit();

        tile_regs_wait();
        vector_begin();
        PACK((flash::p_ds_step<false>()));
        vector_end();
        pack_format(false);
        pack_accumulate(false);
        pack_tile<true>(1, cb_ds, t);
        tile_regs_release_math_clear();
    }

    // dQ_r += dS K over key tiles k .. k + nt - 1 of the block (the oldest cb_ds
    // chunk).
    void accumulate(uint32_t r, uint32_t k, uint32_t nt, bool first) {
        ZONE("A");
        cb_wait_front(cb_ds, SC);
        for (uint32_t d0 = 0; d0 < DT; d0 += DST) {
            const uint32_t w = width(DT, d0);
            tile_regs_acquire_math_clear();
            mm_no_mop_init_short(cb_ds, cb_k, false, 1, 1, nt);
            for (uint32_t d = 0; d < w; ++d) {
                for (uint32_t t = 0; t < nt; ++t) {
                    matmul_block_no_mop(
                        cb_ds, cb_k, t, (d0 + d) * KB + k + t, d, false, 1, 1, nt
                    );
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_format(true);
            pack_accumulate(!first);
            for (uint32_t d = 0; d < w; ++d) {
                pack_tile<true>(d, cb_acc, r * DT + d0 + d);
            }
            tile_regs_release_math_clear();
        }
        cb_pop_front(cb_ds, SC);
    }

    // dQ_r = scale x the fp32 accumulators, to bf16.
    void finalize(uint32_t r) {
        ZONE("F");
        cb_reserve_back(cb_out, DT);
        for (uint32_t d0 = 0; d0 < DT; d0 += DST) {
            const uint32_t w = width(DT, d0);
            tile_regs_acquire_math_clear();
            reconfig_data_format_srca(cb_k, cb_acc);
            copy_init(cb_acc);
            for (uint32_t d = 0; d < w; ++d) {
                copy_tile(cb_acc, r * DT + d0 + d, d);
            }
            reconfig_data_format_srca(cb_acc, cb_k);
            tile_regs_commit();
            tile_regs_wait();
            vector_begin();
            PACK((flash::scale_step(w, SCALE_BITS)));
            vector_end();
            pack_format(false);
            pack_accumulate(false);
            for (uint32_t d = 0; d < w; ++d) {
                pack_tile(d, cb_out);
            }
            tile_regs_release_math_clear();
        }
        cb_push_back(cb_out, DT);
    }
}  // namespace

// Runtime args: iterations, jobs, then per job: query tile rows (at most R), the global
// index of its first row, key tiles, the global row step between its rows.
// FLASH_VARLEN: the first row is the job's first query tile row, and a fifth value the
// stream's first key tile.
void kernel_main() {
    const uint32_t iters = get_arg_val<uint32_t>(0);
    const uint32_t jobs = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_ds);
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
    MATH((flash::init(__builtin_bit_cast(float, A_BITS))));
    PACK((flash::addr_init()));
    constant_tiles();
    cb_reserve_back(cb_w, R);
    cb_reserve_back(cb_acc, R * DT);

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t rows = get_arg_val<uint32_t>(2 + JA * job);
            const uint32_t row0 = get_arg_val<uint32_t>(3 + JA * job);
            const uint32_t nk = get_arg_val<uint32_t>(4 + JA * job);
            const uint32_t stride = get_arg_val<uint32_t>(5 + JA * job);
#ifdef FLASH_VARLEN
            if (nk == 0) {
                continue;  // an empty job (the job list is padded to a fixed length)
            }
            // The rows' records (key tiles relative to the stream's first, key0).
            using ktt_flash::kPlanPages;
            using ktt_flash::kRecordsPerPage;
            using ktt_flash::kRecordWords;
            const uint32_t key0 = get_arg_val<uint32_t>(6 + JA * job);
            ktt_flash::RowWindow win[R];
            bool started[R];
            cb_wait_front(cb_plan, kPlanPages);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t idx = row0 % kRecordsPerPage + r;
                const uint32_t pg = idx / kRecordsPerPage,
                               w0 = idx % kRecordsPerPage * kRecordWords;
                win[r].first = read_tile_value(cb_plan, pg, w0) - key0;
                win[r].last = read_tile_value(cb_plan, pg, w0 + 1) - key0;
                win[r].a = read_tile_value(cb_plan, pg, w0 + 2);
                win[r].b = read_tile_value(cb_plan, pg, w0 + 3);
                win[r].u0 = 0;
                started[r] = false;
            }
#endif
            // Last key tile row r sees.
            auto last_key = [&](uint32_t r) {
#ifdef FLASH_CAUSAL
                const uint32_t rowg = row0 + r * stride;
                return rowg < nk - 1 ? rowg : nk - 1;
#else
                (void)row0;
                (void)stride;
                (void)r;
                return nk - 1;
#endif
            };
            {
                ZONE("WQ");
                cb_wait_front(cb_q, R * DT);
                cb_wait_front(cb_do, R * DV);
                cb_wait_front(cb_oin, R * DV);
                cb_wait_front(cb_lse, R);
            }
            for (uint32_t r = 0; r < rows; ++r) {
                row_d(r, QTAIL != 0 && row0 + r * stride == SQT - 1);
            }
            // The row statistics tiles (cb_w) are read back from L1.
            pack_barrier();
            for (uint32_t k_start = 0; k_start < nk; k_start += KB) {
                const uint32_t kb = nk - k_start < KB ? nk - k_start : KB;
                {
                    ZONE("WK");
                    cb_wait_front(cb_k, DT * KB);
                    cb_wait_front(cb_v, KB * DV);
#ifdef FLASH_MASK_BCAST
                    cb_wait_front(cb_mb, KB);
#endif
                }
                // Per chunk of up to SC key tiles, rows in pairs: the dS tiles of both
                // rows, then both rows' dQ accumulations.
                for (uint32_t k0 = 0; k0 < kb; k0 += SC) {
                    const uint32_t nt_block = kb - k0 < SC ? kb - k0 : SC;
                    const uint32_t kg = k_start + k0;
#ifdef FLASH_VARLEN
                    // Row r's tiles of this chunk: [lo(r), lo(r) + visible(r)) of the
                    // chunk.
                    (void)last_key;
                    auto lo = [&](uint32_t r) {
                        return kg > win[r].first ? 0u : win[r].first - kg;
                    };
                    auto visible = [&](uint32_t r) {
                        const uint32_t first_t = kg > win[r].first ? kg : win[r].first;
                        const uint32_t last_t = kg + nt_block - 1 < win[r].last
                            ? kg + nt_block - 1
                            : win[r].last;
                        return first_t > last_t ? 0u : last_t - first_t + 1;
                    };
                    for (uint32_t r = 0; r < rows; r += 2) {
                        const uint32_t pair = r + 1 < rows ? 2 : 1;
                        for (uint32_t p = 0; p < pair; ++p) {
                            const uint32_t nt = visible(r + p);
                            if (nt == 0) {
                                continue;
                            }
                            const ktt_flash::RowWindow& w = win[r + p];
                            const uint32_t rel0 = kg + lo(r + p) - w.first;
                            bool any = false;
                            for (uint32_t t = 0; t < nt; ++t) {
                                any = any || w.masked(rel0 + t);
                            }
                            if (any) {
                                cb_wait_front(cb_um, SC);
                            }
                            cb_reserve_back(cb_ds, SC);
                            for (uint32_t t = 0; t < nt; ++t) {
                                score(
                                    r + p,
                                    k0 + lo(r + p) + t,
                                    t,
                                    w.masked(rel0 + t) ? cb_um : NO_MASK,
                                    t
                                );
                            }
                            cb_push_back(cb_ds, SC);
                            if (any) {
                                cb_pop_front(cb_um, SC);
                            }
                        }
                        for (uint32_t p = 0; p < pair; ++p) {
                            const uint32_t nt = visible(r + p);
                            if (nt > 0) {
                                accumulate(r + p, k0 + lo(r + p), nt, !started[r + p]);
                                started[r + p] = true;
                            }
                        }
                    }
#else
                    const bool first = kg == 0;
                    auto visible = [&](uint32_t r) {
                        const uint32_t last = last_key(r);
                        return kg > last
                            ? 0u
                            : (last - kg + 1 < nt_block ? last - kg + 1 : nt_block);
                    };
                    for (uint32_t r = 0; r < rows; r += 2) {
                        const uint32_t pair = r + 1 < rows ? 2 : 1;
                        for (uint32_t p = 0; p < pair; ++p) {
                            const uint32_t nt = visible(r + p);
                            if (nt == 0) {
                                continue;
                            }
                            cb_reserve_back(cb_ds, SC);
                            for (uint32_t t = 0; t < nt; ++t) {
#ifdef FLASH_CAUSAL
                                const bool diag = kg + t == row0 + (r + p) * stride;
#else
                                const bool diag = false;
#endif
                                const bool tail = TAIL != 0 && kg + t == SKT - 1;
#ifdef FLASH_CAUSAL
                                const uint32_t mask = diag
                                    ? (tail ? cb_ctail : cb_causal)
                                    : (tail ? cb_tail : NO_MASK);
#else
                                const uint32_t mask = diag || !tail ? NO_MASK : cb_tail;
#endif
                                score(r + p, k0 + t, t, mask);
                            }
                            cb_push_back(cb_ds, SC);
                        }
                        for (uint32_t p = 0; p < pair; ++p) {
                            const uint32_t nt = visible(r + p);
                            if (nt > 0) {
                                accumulate(r + p, k0, nt, first);
                            }
                        }
                    }
#endif
                }
                cb_pop_front(cb_k, DT * KB);
                cb_pop_front(cb_v, KB * DV);
#ifdef FLASH_MASK_BCAST
                cb_pop_front(cb_mb, KB);
#endif
            }
            // The accumulators are read back from L1.
            pack_barrier();
            for (uint32_t r = 0; r < rows; ++r) {
                finalize(r);
            }
            cb_pop_front(cb_q, R * DT);
            cb_pop_front(cb_do, R * DV);
            cb_pop_front(cb_oin, R * DV);
            cb_pop_front(cb_lse, R);
#ifdef FLASH_VARLEN
            cb_pop_front(cb_plan, ktt_flash::kPlanPages);
#endif
        }
    }
    PACK((flash::restore_constants()));
}
