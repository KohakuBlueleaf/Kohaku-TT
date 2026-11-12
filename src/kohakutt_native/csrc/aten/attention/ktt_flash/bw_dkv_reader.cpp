// Flash attention backward dK / dV reader: per job its K and V rows, then for each of
// the G query heads of the K / V head its query blocks of QB tiles (Q, dO and the NST
// statistics tiles per query tile row), multicast down the lane as the forward reader
// does with K / V.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "flash.hpp"
#include "flash_bw.hpp"
#include "flash_dm.hpp"

// Compile-time args: DT, DV, R, QB, SQT (query tiles per head), G, NST, then
// TensorAccessorArgs of Q, K, V, dO, the statistics. Runtime args: the five addresses,
// iterations, multicast rectangle (sx, sy, ex, ey), receivers, sender (x, y), ready /
// valid semaphore ids, 1 on the lane sender, jobs, then per job: first key tile row
// (global), key tile rows, row stride, first query head (global), first query tile,
// query tiles; then key tiles per head x 64 + keys in the last key tile (0: 32), the
// queries in the last query tile (0: 32).
// FLASH_VARLEN (packed layouts, flash.hpp): also TensorAccessorArgs of the
// key-side mask tiles and plan; per job: first key row (tile row x Hk + head), key tile
// rows, row stride (Hk), first query head, first query tile, query tiles, first key
// tile row; then the mask and plan addresses, plan pages, query tiles x 64 + queries in
// the last tile (0: 32), the same of the keys, query heads H.
void kernel_main() {
    using namespace ktt_flash_bw::dkv;
    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t QB = get_compile_time_arg_val(3);
    [[maybe_unused]] constexpr uint32_t SQT = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t NST = get_compile_time_arg_val(6);
    constexpr auto aq = TensorAccessorArgs<7>();
    constexpr auto ak = TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    constexpr auto av = TensorAccessorArgs<ak.next_compile_time_args_offset()>();
    constexpr auto ado = TensorAccessorArgs<av.next_compile_time_args_offset()>();
    constexpr auto awd = TensorAccessorArgs<ado.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t page = get_tile_size(cb_q);
    const uint32_t page32 = get_tile_size(cb_wd);
    const auto q = TensorAccessor(aq, get_arg_val<uint32_t>(a++), page);
    const auto k = TensorAccessor(ak, get_arg_val<uint32_t>(a++), page);
    const auto v = TensorAccessor(av, get_arg_val<uint32_t>(a++), page);
    const auto dout = TensorAccessor(ado, get_arg_val<uint32_t>(a++), page);
    const auto wd = TensorAccessor(awd, get_arg_val<uint32_t>(a++), page32);
    const uint32_t iters = get_arg_val<uint32_t>(a++);
    const uint32_t sx = get_arg_val<uint32_t>(a++), sy = get_arg_val<uint32_t>(a++);
    const uint32_t ex = get_arg_val<uint32_t>(a++), ey = get_arg_val<uint32_t>(a++);
    const uint32_t receivers = get_arg_val<uint32_t>(a++);
    const uint32_t snd_x = get_arg_val<uint32_t>(a++),
                   snd_y = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t valid_addr = get_semaphore(get_arg_val<uint32_t>(a++));
    const bool sender = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t jobs = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ready_addr
    );
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        valid_addr
    );

#ifdef FLASH_VARLEN
    using ktt_flash::kPlanPages;
    using ktt_flash::kRecordsPerPage;
    using ktt_flash::kRecordWords;
    using ktt_flash::RowWindow;
    constexpr uint32_t JA = 7;
    constexpr uint32_t SC = ktt_flash_bw::kChunk;
    constexpr auto au = TensorAccessorArgs<awd.next_compile_time_args_offset()>();
    constexpr auto ap = TensorAccessorArgs<au.next_compile_time_args_offset()>();
    const uint32_t plan_page = get_tile_size(cb_plan);
    const uint32_t tail0 = a + JA * jobs;
    const auto umask = TensorAccessor(au, get_arg_val<uint32_t>(tail0), page);
    const auto plan = TensorAccessor(ap, get_arg_val<uint32_t>(tail0 + 1), plan_page);
    const uint32_t plan_pages = get_arg_val<uint32_t>(tail0 + 2);
    const uint32_t q_tiles = get_arg_val<uint32_t>(tail0 + 3) / 64;
    const uint32_t q_tail = get_arg_val<uint32_t>(tail0 + 3) % 64;
    const uint32_t k_tiles = get_arg_val<uint32_t>(tail0 + 4) / 64;
    const uint32_t k_tail = get_arg_val<uint32_t>(tail0 + 4) % 64;
    const uint32_t QH = get_arg_val<uint32_t>(tail0 + 5);
#ifdef FLASH_MASK_BCAST
    // The key mask [1, Tk]: one row tile per key tile, a further accessor and address.
    constexpr auto akm = TensorAccessorArgs<ap.next_compile_time_args_offset()>();
    const auto kmask = TensorAccessor(akm, get_arg_val<uint32_t>(tail0 + 6), page);
#endif
    auto zero_rows = [](uint32_t addr, uint32_t first) {
        ktt_flash::zero_tile_rows(
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr), first
        );
    };
#else
    constexpr uint32_t JA = 6;
    const uint32_t k_tiles = get_arg_val<uint32_t>(a + JA * jobs) / 64;
    const uint32_t q_tail = get_arg_val<uint32_t>(a + JA * jobs + 1);
#ifdef FLASH_LENGTHS
    // After the query tail: the key lengths (TILE int32 [B]: batch b at row 0, column b
    // % 32 of tile b / 32) and the K / V heads per batch.
    const auto lens = TensorAccessor(
        TensorAccessorArgs<awd.next_compile_time_args_offset()>(),
        get_arg_val<uint32_t>(a + JA * jobs + 2),
        get_tile_size(cb_lens)
    );
    const uint32_t HK = get_arg_val<uint32_t>(a + JA * jobs + 3);
    // The keys of K / V (the forward's clamp): Sk from the key tiles and tail argument.
    const uint32_t key_tail = get_arg_val<uint32_t>(a + JA * jobs) % 64;
    const uint32_t keys = (key_tail != 0 ? k_tiles - 1 : k_tiles) * 32 + key_tail;
    cb_reserve_back(cb_lens, 1);
    const auto* len_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(cb_lens)
    );
    uint32_t len_tile = 0xffffffffu;
    uint32_t k_tail = 0, KT = k_tiles, built_tail = 0;
#else
    const uint32_t k_tail = get_arg_val<uint32_t>(a + JA * jobs) % 64;
    const uint32_t KT = k_tiles;
#endif
#endif

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t krow0 = get_arg_val<uint32_t>(a + JA * job);
            const uint32_t rows = get_arg_val<uint32_t>(a + JA * job + 1);
            const uint32_t stride = get_arg_val<uint32_t>(a + JA * job + 2);
            const uint32_t head0 = get_arg_val<uint32_t>(a + JA * job + 3);
            const uint32_t q_first = get_arg_val<uint32_t>(a + JA * job + 4);
            const uint32_t nq = get_arg_val<uint32_t>(a + JA * job + 5);
#if defined(FLASH_LENGTHS) && !defined(FLASH_VARLEN)
            // The batch's key length (clamped to [1, Sk]): KT key tiles, k_tail keys
            // in the last; (k_tail, KT) to cb_len. A core without rows has its first
            // row clipped to the chunk's end (> 0): its head is that of the row before.
            {
                const uint32_t b = (rows > 0 ? krow0 : krow0 - 1) / k_tiles / HK;
                if (b / 32 != len_tile) {
                    len_tile = b / 32;
                    noc_async_read_page(len_tile, lens, get_write_ptr(cb_lens));
                    noc_async_read_barrier();
                }
                uint32_t len = len_words[b % 32 < 16 ? b % 32 : 256 + b % 32 - 16];
                len = len < 1 ? 1 : (len > keys ? keys : len);
                KT = (len + 31) / 32;
                k_tail = len % 32;
                // A new tail: its mask tiles (cb_tail, causal cb_ctail).
                const bool fresh = k_tail != 0 && k_tail != built_tail;
                if (fresh) {
#ifdef FLASH_CAUSAL
                    ktt_flash::push_mask_tiles(cb_tail, cb_ctail, k_tail, true);
#else
                    ktt_flash::push_mask_tiles(cb_tail, cb_ctail, k_tail, false);
#endif
                    built_tail = k_tail;
                }
                cb_reserve_back(cb_len, 1);
                auto* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_write_ptr(cb_len)
                );
                tok[0] = k_tail;
                tok[1] = KT;
                tok[2] = fresh ? 1 : 0;
                cb_push_back(cb_len, 1);
            }
#endif
#ifdef FLASH_VARLEN
            const uint32_t tkrow0 = get_arg_val<uint32_t>(a + JA * job + 6);
            if (rows == 0 && nq == 0) {
                continue;  // an empty job (the job list is padded to a fixed length)
            }
            // The job's key row records (query tiles relative to the stream's first).
            RowWindow win[R];
            if (rows > 0) {
                cb_reserve_back(cb_plan, kPlanPages);
                const uint32_t pb = get_write_ptr(cb_plan);
                const uint32_t p0 = tkrow0 / kRecordsPerPage;
                for (uint32_t p = 0; p < kPlanPages; ++p) {
                    const uint32_t pg = p0 + p < plan_pages ? p0 + p : plan_pages - 1;
                    noc_async_read_page(pg, plan, pb + p * plan_page);
                }
                noc_async_read_barrier();
                const auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pb);
                for (uint32_t i = 0; i < rows; ++i) {
                    const auto* rec = words +
                        (tkrow0 % kRecordsPerPage + i) * kRecordWords;
                    win[i] = RowWindow{
                        rec[0] - q_first, rec[1] - q_first, rec[2], rec[3], rec[4]
                    };
                }
                cb_push_back(cb_plan, kPlanPages);
            }
#endif

            cb_reserve_back(cb_k, R * DT);
            cb_reserve_back(cb_v, R * DV);
            const uint32_t kb = get_write_ptr(cb_k), vb = get_write_ptr(cb_v);
#ifdef FLASH_MASK_BCAST
            cb_reserve_back(cb_mb, R);
            const uint32_t mb = get_write_ptr(cb_mb);
#endif
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t row = krow0 + r * stride;
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_read_page(row * DT + d, k, kb + (r * DT + d) * page);
                }
                for (uint32_t u = 0; u < DV; ++u) {
                    noc_async_read_page(row * DV + u, v, vb + (r * DV + u) * page);
                }
#ifdef FLASH_MASK_BCAST
                noc_async_read_page(tkrow0 + r, kmask, mb + r * page);
#endif
            }
            // The rows' reads complete under the first query block's barrier.
            bool rows_pending = true;
            auto push_rows = [&] {
                if (rows_pending) {
#ifdef FLASH_VARLEN
                    // Rows past the last key hold undefined values: K and V to zero.
                    if (k_tail != 0 && rows > 0 && tkrow0 + rows == k_tiles) {
                        for (uint32_t d = 0; d < DT; ++d) {
                            zero_rows(kb + ((rows - 1) * DT + d) * page, k_tail);
                        }
                        for (uint32_t u = 0; u < DV; ++u) {
                            zero_rows(vb + ((rows - 1) * DV + u) * page, k_tail);
                        }
#ifdef FLASH_MASK_BCAST
                        ktt_flash::zero_row0_cols(
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                mb + (rows - 1) * page
                            ),
                            k_tail
                        );
#endif
                    }
#else
                    // Rows past the last key of a head hold undefined values: K and V
                    // to zero.
                    bool zeroed = false;
                    for (uint32_t r = 0; k_tail != 0 && r < rows; ++r) {
                        if ((krow0 + r * stride) % k_tiles != KT - 1) {
                            continue;
                        }
                        for (uint32_t d = 0; d < DT; ++d) {
                            ktt_flash::zero_rows_async(
                                kb + (r * DT + d) * page, k_tail
                            );
                        }
                        for (uint32_t u = 0; u < DV; ++u) {
                            ktt_flash::zero_rows_async(
                                vb + (r * DV + u) * page, k_tail
                            );
                        }
                        zeroed = true;
                    }
                    if (zeroed) {
                        noc_async_read_barrier();
                    }
#endif
                    cb_push_back(cb_k, R * DT);
                    cb_push_back(cb_v, R * DV);
#ifdef FLASH_MASK_BCAST
                    cb_push_back(cb_mb, R);
#endif
                    rows_pending = false;
                }
            };

            for (uint32_t hh = 0; hh < G; ++hh) {
#ifndef FLASH_VARLEN
                const uint32_t row_base = (head0 + hh) * SQT;
#endif
                for (
                    uint32_t i_start = q_first; i_start < q_first + nq; i_start += QB
                ) {
                    const uint32_t qb = q_first + nq - i_start < QB
                        ? q_first + nq - i_start
                        : QB;
                    cb_reserve_back(cb_q, QB * DT);
                    cb_reserve_back(cb_do, QB * DV);
                    cb_reserve_back(cb_wd, NST * QB);
                    const uint32_t qbase = get_write_ptr(cb_q),
                                   dobase = get_write_ptr(cb_do);
                    const uint32_t wbase = get_write_ptr(cb_wd);
                    if (sender) {
                        for (uint32_t i = 0; i < qb; ++i) {
#ifdef FLASH_VARLEN
                            const uint32_t row = (i_start + i) * QH + head0 + hh;
#else
                            const uint32_t row = row_base + i_start + i;
#endif
                            for (uint32_t d = 0; d < DT; ++d) {
                                noc_async_read_page(
                                    row * DT + d, q, qbase + (i * DT + d) * page
                                );
                            }
                            for (uint32_t u = 0; u < DV; ++u) {
                                noc_async_read_page(
                                    row * DV + u, dout, dobase + (i * DV + u) * page
                                );
                            }
                            for (uint32_t t = 0; t < NST; ++t) {
                                noc_async_read_page(
                                    NST * row + t, wd, wbase + (NST * i + t) * page32
                                );
                            }
                        }
                        noc_async_read_barrier();
#ifdef FLASH_VARLEN
                        // Rows past the last query hold undefined values: Q and dO to
                        // zero.
                        if (q_tail != 0 && i_start + qb == q_tiles) {
                            for (uint32_t d = 0; d < DT; ++d) {
                                zero_rows(qbase + ((qb - 1) * DT + d) * page, q_tail);
                            }
                            for (uint32_t u = 0; u < DV; ++u) {
                                zero_rows(dobase + ((qb - 1) * DV + u) * page, q_tail);
                            }
                        }
#else
                        // Rows past the last query hold undefined values: Q and dO to
                        // zero.
                        if (q_tail != 0 && i_start + qb == SQT) {
                            for (uint32_t d = 0; d < DT; ++d) {
                                ktt_flash::zero_rows_async(
                                    qbase + ((qb - 1) * DT + d) * page, q_tail
                                );
                            }
                            for (uint32_t u = 0; u < DV; ++u) {
                                ktt_flash::zero_rows_async(
                                    dobase + ((qb - 1) * DV + u) * page, q_tail
                                );
                            }
                            noc_async_read_barrier();
                        }
#endif
                        push_rows();
                        if (receivers > 0) {
                            noc_semaphore_wait(ready, receivers);
                            noc_semaphore_set(ready, 0);
                            noc_async_write_multicast(
                                qbase,
                                get_noc_multicast_addr(sx, sy, ex, ey, qbase),
                                QB * DT * page,
                                receivers
                            );
                            noc_async_write_multicast(
                                dobase,
                                get_noc_multicast_addr(sx, sy, ex, ey, dobase),
                                QB * DV * page,
                                receivers
                            );
                            noc_async_write_multicast(
                                wbase,
                                get_noc_multicast_addr(sx, sy, ex, ey, wbase),
                                NST * QB * page32,
                                receivers
                            );
                            noc_async_writes_flushed();
                            *valid = qbase;
                            noc_semaphore_set_multicast(
                                valid_addr,
                                get_noc_multicast_addr(sx, sy, ex, ey, valid_addr),
                                receivers
                            );
                            noc_async_writes_flushed();
                        }
                    } else {
                        noc_semaphore_set(valid, 0);
                        noc_semaphore_inc(get_noc_addr(snd_x, snd_y, ready_addr), 1);
                        if (rows_pending) {
                            noc_async_read_barrier();
                            push_rows();
                        }
                        noc_semaphore_wait_min(valid, 1);
                    }
                    cb_push_back(cb_q, QB * DT);
                    cb_push_back(cb_do, QB * DV);
                    cb_push_back(cb_wd, NST * QB);
#ifdef FLASH_VARLEN
                    // The mask tiles of this block's chunks in the compute's order
                    // (chunk, key row): a row's SC slots of a chunk, filled where its
                    // tile takes a mask.
                    for (uint32_t i0 = 0; i0 < qb; i0 += SC) {
                        const uint32_t nt = qb - i0 < SC ? qb - i0 : SC;
                        const uint32_t ig = i_start + i0 - q_first;
                        for (uint32_t r = 0; r < rows; ++r) {
                            const RowWindow& w = win[r];
                            if (w.empty()) {
                                continue;
                            }
                            const uint32_t lo = ig > w.first ? ig : w.first;
                            const uint32_t hi = ig + nt - 1 < w.last ? ig + nt - 1
                                                                     : w.last;
                            bool any = false;
                            for (uint32_t t = lo; t <= hi && lo <= hi; ++t) {
                                any = any || w.masked(t - w.first);
                            }
                            if (!any) {
                                continue;
                            }
                            cb_reserve_back(cb_um, SC);
                            const uint32_t ub = get_write_ptr(cb_um);
                            for (uint32_t t = lo; t <= hi; ++t) {
                                if (w.masked(t - w.first)) {
                                    noc_async_read_page(
                                        w.mask_page(t - w.first),
                                        umask,
                                        ub + (t - lo) * page
                                    );
                                }
                            }
                            noc_async_read_barrier();
#ifdef FLASH_MASK_BCAST
                            // The keys a mask tile leaves visible take the key mask's
                            // value.
                            for (uint32_t t = lo; t <= hi; ++t) {
                                if (w.masked(t - w.first)) {
                                    ktt_flash::fill_zeros_from_row0(
                                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                            ub + (t - lo) * page
                                        ),
                                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                            mb + r * page
                                        )
                                    );
                                }
                            }
#endif
                            cb_push_back(cb_um, SC);
                        }
                    }
#endif
                }
            }
            if (rows_pending) {
                noc_async_read_barrier();
                push_rows();
            }
        }
    }
    noc_async_atomic_barrier();
}
