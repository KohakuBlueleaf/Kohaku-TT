// Flash attention compute: per job, up to R query tile rows against the job's key
// tiles, single pass. Keys arrive in blocks of KB tiles (K d-major, V key-major); each
// block is split into sub-chunks of up to 3 key tiles. Step (r, c): S = Q_r K_c^T (plus
// mask terms) into Dst 0-2 with the row's state tile in Dst 3; the pack thread updates
// the row reference, writes P = exp (bf16) and the state. Then O_r += P V and the row
// sum (P times a ones tile) accumulate in L1 (fp32); when the reference moved, O_r and
// l_r are first scaled by alpha. Steps run in pairs (two score sections, then two
// output sections). Design: docs/reference/kernels/flash.md.
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
    using namespace ktt_flash;

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t NK = get_compile_time_arg_val(3);  // key tiles of the tensors
    constexpr uint32_t KB = get_compile_time_arg_val(4);
    constexpr uint32_t A_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t THR_BITS = get_compile_time_arg_val(6);
    constexpr uint32_t BLK = get_compile_time_arg_val(
        7
    );  // causal block in key tiles; 0: token causal
    constexpr uint32_t TAIL = get_compile_time_arg_val(
        8
    );  // keys in the last key tile; 0: 32
#ifdef FLASH_STATS
    constexpr uint32_t STATS = 1;
#else
    constexpr uint32_t STATS = 0;
#endif
    // Accumulator columns per row: O, the row sum l (column DV) and with FLASH_STATS
    // the exact fp32 lane sums of P (column DV + 1).
    constexpr uint32_t W = DV + 1 + STATS;
    constexpr uint32_t LANE_COL = DV + 1;
    constexpr uint32_t SC = kSubChunk;
    constexpr uint32_t
        ANY_WORD = 257;  // fp32 state tile: row 0, column 17 (the tile's flag)

    constexpr uint32_t cb_e = cb_ones, cb_m = cb_causal;
#ifdef FLASH_VARLEN
    constexpr uint32_t
        JA = 5;  // runtime args per job: also the stream's first key tile
#else
    constexpr uint32_t JA = 4;
#endif

    // Chunk index of each row's last step (its final state slot).
    uint32_t last_c[R];

#ifdef FLASH_DUMP
    // Dumped row of the current job (R: none). Per step to cb_dump: Dst 0-3 before the
    // vector step (S, the state tile with the lane maxima), Dst 0-3 after (fp32 P, the
    // state tile).
    uint32_t dump_r = R;

    // Dst tile t to cb_dump (one page; the packer in fp32).
    void dump_tile(uint32_t t) {
        cb_reserve_back(cb_dump, 1);
        pack_tile(t, cb_dump);
        cb_push_back(cb_dump, 1);
    }
#endif

    static_assert(KB % SC == 0, "key block must be whole sub-chunks");

    struct Step {
        uint32_t r;   // query tile row
        uint32_t c;   // chunk index along the keys (0 = first)
        uint32_t k0;  // first key tile of the chunk within the block
        uint32_t nt;  // key tiles in the chunk
        bool last;    // the row's last chunk
        bool diag;    // the chunk's last tile is the row's diagonal tile (causal mask)
        bool tail;    // the chunk's last tile is the last key tile and TAIL > 0
        uint32_t mask = 0;  // FLASH_VARLEN: bit t set when tile t takes its mask tile
                            // (cb_um slot t)
    };

    uint32_t slot(uint32_t r, uint32_t c) {
        return 2 * r + (c & 1);
    }

    // Math thread: lane-wise row maxima of the section.
    void state_vector(uint32_t nt) {
#ifdef TRISC_MATH
        if (nt == 3) {
            flash::state_max<3>();
        } else if (nt == 2) {
            flash::state_max<2>();
        } else {
            flash::state_max<1>();
        }
#endif
    }

    // Pack thread: row state update and P of the section.
    void exp_vector(bool first, uint32_t nt) {
#ifdef TRISC_PACK
        flash::exp_section_begin();
        if (first) {
            flash::state_update<true>(THR_BITS);
        } else {
            flash::state_update<false>(THR_BITS);
        }
        if (nt == 3) {
            flash::exp_step<3>();
        } else if (nt == 2) {
            flash::exp_step<2>();
        } else {
            flash::exp_step<1>();
        }
        flash::exp_section_end();
#endif
    }

    // Math thread: vector work after the section's matmuls, holding the SFPU mutex.
    void math_vector_begin() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH)));
        MATH((math::set_dst_write_addr<
              DstTileShape::Tile32x32,
              UnpackDestination::SrcRegs>(0)));
        MATH((t6_mutex_acquire(mutex::SFPU)));
    }

    void math_vector_end() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::WAIT_SFPU)));
        MATH((t6_mutex_release(mutex::SFPU)));
    }

    // Packer output format (fp32: cb_o, cb_st, cb_lane, cb_stats; bf16: cb_p, cb_out)
    // and L1 accumulation, changed only on transitions.
    bool pack_fp32 = false;
    bool pack_acc = false;

    void pack_format(bool fp32) {
        if (fp32 != pack_fp32) {
            pack_reconfig_data_format(fp32 ? cb_p : cb_o, fp32 ? cb_o : cb_p);
            pack_fp32 = fp32;
        }
    }

    void pack_accumulate(bool acc) {
        if (acc != pack_acc) {
            pack_reconfig_l1_acc(acc ? 1 : 0);
            pack_acc = acc;
        }
    }

    // Pack thread: vector work waits for the math section and holds the SFPU mutex;
    // packs wait for the vector work.
    void vector_begin() {
        PACK((TTI_SEMWAIT(
            p_stall::STALL_SFPU | p_stall::STALL_SYNC,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_ZERO
        )));
        PACK((math::set_dst_write_addr<
              DstTileShape::Tile32x32,
              UnpackDestination::SrcRegs>(0)));
        PACK((t6_mutex_acquire(mutex::SFPU)));
    }

    void vector_end() {
        PACK((
            TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_PACK, p_stall::WAIT_SFPU)
        ));
        PACK((t6_mutex_release(mutex::SFPU)));
    }

    // S = Q_r K^T (raw scores) and the row state into Dst; state update and P on the
    // pack thread.
    void score(const Step& s) {
        ZONE("S");
        const bool first = s.c == 0;
        tile_regs_acquire_math_clear();
        // First chunk: any slot (defines every state row).
        reconfig_data_format_srca(cb_v, cb_st);
        copy_init(cb_st);
        copy_tile(cb_st, first ? slot(s.r, 1) : slot(s.r, s.c - 1), 3);
        reconfig_data_format_srca(cb_st, cb_v);
        // Mask terms are copied into the score tiles first; the score matmul
        // accumulates onto them. A user mask also masks the keys past the last one (its
        // padding).
#if defined(FLASH_VARLEN)
#ifdef FLASH_MASK_BCAST
        // The key mask's row for the tiles without a mask tile (a mask tile holds it).
        if (s.mask != (1u << s.nt) - 1) {
            unary_bcast_init<BroadcastType::ROW>(cb_mb);
            for (uint32_t t = 0; t < s.nt; ++t) {
                if (!((s.mask >> t) & 1)) {
                    unary_bcast<BroadcastType::ROW>(cb_mb, s.k0 + t, t);
                }
            }
            unary_bcast_uninit<BroadcastType::ROW>(cb_mb);
        }
#endif
        if (s.mask != 0) {
            cb_wait_front(cb_um, SC);
            copy_init(cb_um);
            for (uint32_t t = 0; t < s.nt; ++t) {
                if ((s.mask >> t) & 1) {
                    copy_tile(cb_um, t, t);
                }
            }
        }
#elif defined(FLASH_MASK_ROW)
        cb_wait_front(cb_um, SC);
        copy_init(cb_um);
        for (uint32_t t = 0; t < s.nt; ++t) {
            copy_tile(cb_um, t, t);
        }
#elif defined(FLASH_MASK_BCAST)
        unary_bcast_init<BroadcastType::ROW>(cb_mb);
        for (uint32_t t = 0; t < s.nt; ++t) {
            unary_bcast<BroadcastType::ROW>(cb_mb, s.k0 + t, t);
        }
        unary_bcast_uninit<BroadcastType::ROW>(cb_mb);
#else
        if (s.diag || s.tail) {
#ifdef FLASH_CAUSAL
            const uint32_t mask = s.diag ? (s.tail ? cb_ctail : cb_m) : cb_tail;
#else
            const uint32_t mask = cb_tail;
#endif
            copy_init(mask);
            copy_tile(mask, 0, s.nt - 1);
        }
#endif
        mm_no_mop_init_short(cb_q, cb_k, true, s.nt, 1, DT);
        for (uint32_t d = 0; d < DT; ++d) {
            matmul_block_no_mop(
                cb_q, cb_k, s.r * DT + d, d * KB + s.k0, 0, true, s.nt, 1, DT
            );
        }
#ifdef FLASH_MASK_ROW
        cb_pop_front(cb_um, SC);
#endif
#ifdef FLASH_VARLEN
        if (s.mask != 0) {
            cb_pop_front(cb_um, SC);
        }
#endif
        math_vector_begin();
        state_vector(s.nt);
        math_vector_end();
        tile_regs_commit();

        cb_reserve_back(cb_p, SC);
        cb_reserve_back(cb_stok, 1);
        tile_regs_wait();
#ifdef FLASH_DUMP
        const bool dump = s.r == dump_r;
        if (dump) {
            pack_format(true);
            pack_accumulate(false);
            for (uint32_t t = 0; t < 4; ++t) {
                dump_tile(t);
            }
            PACK((TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK)));
        }
#endif
        vector_begin();
        exp_vector(first, s.nt);
        vector_end();
        pack_format(false);
        pack_accumulate(false);
        for (uint32_t t = 0; t < s.nt; ++t) {
            pack_tile(t, cb_p);
        }
        pack_format(true);
        pack_tile<true>(3, cb_st, slot(s.r, s.c));
#ifdef FLASH_DUMP
        if (dump) {
            for (uint32_t t = 0; t < 4; ++t) {
                dump_tile(t);
            }
        }
#endif
#ifdef FLASH_STATS
        // Lane sums of the step's fp32 P into tile 0 once the packer has read it, to
        // cb_lane.
        cb_reserve_back(cb_lane, 1);
        PACK((TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK)));
        vector_begin();
        PACK((flash::lane_sum_step(s.nt)));
        vector_end();
        pack_format(true);
        pack_tile(0, cb_lane);
        cb_push_back(cb_lane, 1);
#endif
        tile_regs_release_math_clear();
        cb_push_back(cb_p, SC);
        cb_push_back(cb_stok, 1);
    }

    // Dst t = the step's lane-sum tile (cb_lane front, fp32).
    void lane_column(uint32_t t) {
#ifdef FLASH_STATS
        reconfig_data_format_srca(cb_v, cb_lane);
        copy_init(cb_lane);
        copy_tile(cb_lane, 0, t);
        reconfig_data_format_srca(cb_lane, cb_v);
#else
        (void)t;
#endif
    }

    // Dst 0..: P V columns [j0, j0 + wv) of the block, then the row-sum column if
    // `sum`.
    void pv_columns(const Step& s, uint32_t j0, uint32_t wv, bool sum) {
        if (wv > 0) {
            mm_no_mop_init_short(cb_p, cb_v, false, wv, 1, s.nt);
            for (uint32_t k = 0; k < s.nt; ++k) {
                matmul_block_no_mop(
                    cb_p, cb_v, k, (s.k0 + k) * DV + j0, 0, false, wv, 1, s.nt
                );
            }
        }
        if (sum) {
            mm_no_mop_init_short(cb_p, cb_e, false, 1, 1, s.nt);
            for (uint32_t k = 0; k < s.nt; ++k) {
                matmul_block_no_mop(cb_p, cb_e, k, 0, wv, false, 1, 1, s.nt);
            }
        }
    }

    // O_r (and l_r) += P V (and P 1); scaled by alpha first when the tile's reference
    // moved.
    void output(const Step& s) {
        ZONE("O");
        const bool first = s.c == 0;
        cb_wait_front(cb_stok, 1);
        cb_wait_front(cb_p, SC);
        const uint32_t moved = first ? 0
                                     : read_tile_value(cb_st, slot(s.r, s.c), ANY_WORD);
#ifdef FLASH_STATS
        cb_wait_front(cb_lane, 1);
#endif
        if (moved == 0) {
            for (uint32_t j0 = 0; j0 < W; j0 += 4) {
                const uint32_t w = W - j0 < 4 ? W - j0 : 4;
                const uint32_t wv = j0 < DV ? (DV - j0 < w ? DV - j0 : w) : 0;
                tile_regs_acquire_math_clear();
                pv_columns(s, j0, wv, j0 <= DV && DV < j0 + w);
                if (STATS && LANE_COL < j0 + w) {
                    lane_column(LANE_COL - j0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_format(true);
                pack_accumulate(!first);
                for (uint32_t t = 0; t < w; ++t) {
                    pack_tile<true>(t, cb_o, s.r * W + j0 + t);
                }
                tile_regs_release_math_clear();
            }
        } else {
            for (uint32_t j = 0; j < W; ++j) {
                tile_regs_acquire_math_clear();
                if (STATS && j == LANE_COL) {
                    lane_column(0);
                } else {
                    pv_columns(s, j, j < DV ? 1 : 0, j == DV);
                }
                reconfig_data_format_srca(cb_v, cb_st);
                copy_init(cb_st);
                copy_tile(cb_st, slot(s.r, s.c), 2);
                copy_tile(cb_st, slot(s.r, s.c - 1), 3);
                copy_init(cb_o);
                copy_tile(cb_o, s.r * W + j, 1);
                reconfig_data_format_srca(cb_o, cb_v);
                tile_regs_commit();
                tile_regs_wait();
                vector_begin();
                PACK((flash::rescale_step()));
                vector_end();
                pack_format(true);
                pack_accumulate(false);
                pack_tile<true>(0, cb_o, s.r * W + j);
                tile_regs_release_math_clear();
            }
        }
#ifdef FLASH_DUMP
        if (s.r == dump_r) {
            // The row's O, l (and lane sums) after this step, fp32.
            for (uint32_t j0 = 0; j0 < W; j0 += 4) {
                const uint32_t w = W - j0 < 4 ? W - j0 : 4;
                tile_regs_acquire_math_clear();
                reconfig_data_format_srca(cb_v, cb_o);
                copy_init(cb_o);
                for (uint32_t t = 0; t < w; ++t) {
                    copy_tile(cb_o, s.r * W + j0 + t, t);
                }
                reconfig_data_format_srca(cb_o, cb_v);
                tile_regs_commit();
                tile_regs_wait();
                pack_format(true);
                pack_accumulate(false);
                for (uint32_t t = 0; t < w; ++t) {
                    dump_tile(t);
                }
                tile_regs_release_math_clear();
            }
        }
#endif
        cb_pop_front(cb_p, SC);
        cb_pop_front(cb_stok, 1);
#ifdef FLASH_STATS
        cb_pop_front(cb_lane, 1);
#endif
        if (s.last) {
            cb_reserve_back(cb_otok, 1);
            cb_push_back(cb_otok, 1);
        }
    }

    // Output row r = O_r / l_r (bf16).
    void finalize(uint32_t r) {
        cb_wait_front(cb_otok, 1);
        cb_pop_front(cb_otok, 1);
#ifdef FLASH_STATS
        // Row log-sum-exp: the final reference into Dst 2, the row's lane sums into
        // Dst 3.
        tile_regs_acquire_math_clear();
        reconfig_data_format_srca(cb_v, cb_st);
        copy_init(cb_st);
        copy_tile(cb_st, slot(r, last_c[r]), 2);
        copy_init(cb_o);
        copy_tile(cb_o, r * W + LANE_COL, 3);
        reconfig_data_format_srca(cb_o, cb_v);
        tile_regs_commit();
        tile_regs_wait();
        vector_begin();
        {
            ZONE("LSE");
            PACK((flash::stats_step()));
        }
        vector_end();
        pack_format(true);
        pack_accumulate(false);
        cb_reserve_back(cb_stats, 1);
        pack_tile(2, cb_stats);
        cb_push_back(cb_stats, 1);
        tile_regs_release_math_clear();
#endif
        cb_reserve_back(cb_out, DV);
        for (uint32_t j0 = 0; j0 < DV; j0 += 3) {
            const uint32_t w = DV - j0 < 3 ? DV - j0 : 3;
            tile_regs_acquire_math_clear();
            reconfig_data_format_srca(cb_v, cb_o);
            copy_init(cb_o);
            for (uint32_t t = 0; t < w; ++t) {
                copy_tile(cb_o, r * W + j0 + t, t);
            }
            copy_tile(cb_o, r * W + DV, 3);
            reconfig_data_format_srca(cb_o, cb_v);
            tile_regs_commit();
            tile_regs_wait();
            vector_begin();
            PACK((flash::normalize_step(w)));
            vector_end();
            pack_format(false);
            pack_accumulate(false);
            for (uint32_t t = 0; t < w; ++t) {
                pack_tile(t, cb_out);
            }
            tile_regs_release_math_clear();
        }
        cb_push_back(cb_out, DV);
    }

    // The constant tiles, built by the vector unit and packed once: ones (cb_ones);
    // with FLASH_CAUSAL the diagonal mask (cb_causal); with TAIL the last key tile's
    // mask (cb_tail); with both the two combined (cb_ctail).
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
        cb_reserve_back(cb_e, 1);
        pack_tile(0, cb_e);
        cb_push_back(cb_e, 1);
#ifdef FLASH_CAUSAL
        cb_reserve_back(cb_m, 1);
        pack_tile(1, cb_m);
        cb_push_back(cb_m, 1);
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
        cb_wait_front(cb_e, 1);
#ifdef FLASH_CAUSAL
        cb_wait_front(cb_m, 1);
#endif
        if constexpr (TAIL != 0) {
            cb_wait_front(cb_tail, 1);
#ifdef FLASH_CAUSAL
            cb_wait_front(cb_ctail, 1);
#endif
        }
    }
}  // namespace

// Runtime args: iterations, jobs, then per job: query tile rows (at most R), the global
// index of its first row, key tiles streamed, the global row step between its rows.
void kernel_main() {
    const uint32_t iters = get_arg_val<uint32_t>(0);
    const uint32_t jobs = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_p);
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
    MATH((flash::init(__builtin_bit_cast(float, A_BITS))));
    MATH((flash::macro_init()));
    PACK((flash::addr_init()));
    constant_tiles();
    cb_reserve_back(cb_st, 2 * R);
    cb_reserve_back(cb_o, R * W);

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t rows = get_arg_val<uint32_t>(2 + JA * job);
            const uint32_t row0 = get_arg_val<uint32_t>(3 + JA * job);
            const uint32_t nkj = get_arg_val<uint32_t>(4 + JA * job);
            const uint32_t stride = get_arg_val<uint32_t>(5 + JA * job);
#ifdef FLASH_DUMP
            // Trailing args: the dumped job and its row.
            dump_r = job == get_arg_val<uint32_t>(2 + JA * jobs)
                ? get_arg_val<uint32_t>(3 + JA * jobs)
                : R;
#endif
#ifdef FLASH_VARLEN
            if (nkj == 0) {
                continue;  // an empty job (the job list is padded to a fixed length)
            }
            // The rows' records (key tiles relative to the stream's first, key0) and
            // their steps so far.
            (void)stride;
            const uint32_t key0 = get_arg_val<uint32_t>(6 + JA * job);
            RowWindow win[R];
            uint32_t steps[R];
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
                steps[r] = 0;
            }
#endif
            {
                ZONE("W_q");
                cb_wait_front(cb_q, R * DT);
            }
            uint32_t c_base = 0;
            for (uint32_t k_start = 0; k_start < nkj; k_start += KB) {
                const uint32_t kb = nkj - k_start < KB ? nkj - k_start : KB;
                const uint32_t nsub = (kb + SC - 1) / SC;
                {
                    ZONE("W_kv");
                    cb_wait_front(cb_k, DT * KB);
                    cb_wait_front(cb_v, KB * DV);
                }
#ifdef FLASH_MASK_BCAST
                cb_wait_front(cb_mb, KB);
#endif
                // Visible steps in (chunk, row) order; pairs hold two different rows.
                Step pending{};
                bool has_pending = false;
                for (uint32_t sub = 0; sub < nsub; ++sub) {
                    const uint32_t k0 = sub * SC;
                    const uint32_t kg = k_start + k0;
                    const uint32_t nt_block = kb - k0 < SC ? kb - k0 : SC;
                    for (uint32_t r = 0; r < rows; ++r) {
#ifdef FLASH_VARLEN
                        // The row's tiles of this chunk: [lo, hi] of the stream.
                        const RowWindow& w = win[r];
                        const uint32_t lo = kg > w.first ? kg : w.first;
                        const uint32_t hi = kg + nt_block - 1 < w.last
                            ? kg + nt_block - 1
                            : w.last;
                        if (lo > hi) {
                            continue;
                        }
                        uint32_t mask = 0;
                        for (uint32_t t = lo; t <= hi; ++t) {
                            mask |= (w.masked(t - w.first) ? 1u : 0u) << (t - lo);
                        }
                        const bool last = hi == w.last;
                        const Step s{
                            r,
                            steps[r],
                            k0 + (lo - kg),
                            hi - lo + 1,
                            last,
                            false,
                            false,
                            mask
                        };
                        if (last) {
                            last_c[r] = steps[r];
                        }
                        ++steps[r];
#else
#ifdef FLASH_CAUSAL
                        const uint32_t rowg = row0 + r * stride;
                        constexpr uint32_t blk = BLK ? BLK : 1;
                        const uint32_t vis = BLK ? (rowg / blk + 1) * blk - 1 : rowg;
                        const uint32_t last_key = vis < nkj - 1 ? vis : nkj - 1;
#else
                        const uint32_t last_key = nkj - 1;
#endif
                        if (kg > last_key) {
                            continue;
                        }
                        const uint32_t nt = last_key - kg + 1 < nt_block
                            ? last_key - kg + 1
                            : nt_block;
                        const bool last = kg + nt - 1 == last_key;
#ifdef FLASH_CAUSAL
                        const bool diag = BLK == 0 && last && last_key == rowg;
#else
                        const bool diag = false;
#endif
                        const bool tail = TAIL != 0 && kg + nt == NK;
                        const Step s{r, c_base + sub, k0, nt, last, diag, tail};
                        if (last) {
                            last_c[r] = c_base + sub;
                        }
#endif
                        if (has_pending && pending.r != s.r) {
                            score(pending);
                            score(s);
                            output(pending);
                            output(s);
                            has_pending = false;
                        } else {
                            if (has_pending) {
                                score(pending);
                                output(pending);
                            }
                            pending = s;
                            has_pending = true;
                        }
                    }
                }
                if (has_pending) {
                    score(pending);
                    output(pending);
                }
                cb_pop_front(cb_k, DT * KB);
                cb_pop_front(cb_v, KB * DV);
#ifdef FLASH_MASK_BCAST
                cb_pop_front(cb_mb, KB);
#endif
                c_base += nsub;
            }
            {
                ZONE("F");
                for (uint32_t r = 0; r < rows; ++r) {
                    finalize(r);
                }
            }
            cb_pop_front(cb_q, R * DT);
#ifdef FLASH_VARLEN
            cb_pop_front(cb_plan, kPlanPages);
#endif
        }
    }
    PACK((flash::restore_constants()));
}
