// Flash attention backward dQ reader: per job its Q, dO, O rows and their log-sum-exp
// tiles (one per row), then the job's K / V key blocks, multicast down the lane as the
// forward reader does.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "flash.hpp"
#include "flash_bw.hpp"
#include "flash_dm.hpp"

// Compile-time args: DT, DV, R, SKT (key tiles per head), KB, then TensorAccessorArgs
// of Q, K, V, dO, O, log-sum-exp. Runtime args: the six addresses, iterations,
// multicast rectangle (sx, sy, ex, ey), receivers, sender (x, y), ready / valid
// semaphore ids, 1 on the lane sender, jobs, then per job: first query tile row
// (global), K / V head (global), query tile rows, key tiles, row stride; then the keys
// in the last key tile (0: 32). FLASH_VARLEN (packed layouts, flash.hpp): also
// TensorAccessorArgs of the mask tiles and the plan; per job: first query row (tile row
// x H + head), first K page, first V page, query tile rows, key tiles, row stride (H),
// the stream's first key tile, first query tile row; then the mask and plan addresses,
// plan pages, query tiles x 64 + queries in the last tile (0: 32), the same of the
// keys, pages per key tile in K and in V.
void kernel_main() {
    using namespace ktt_flash_bw::dq;
    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    [[maybe_unused]] constexpr uint32_t SKT = get_compile_time_arg_val(3);
    constexpr uint32_t KB = get_compile_time_arg_val(4);
    constexpr auto aq = TensorAccessorArgs<5>();
    constexpr auto ak = TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    constexpr auto av = TensorAccessorArgs<ak.next_compile_time_args_offset()>();
    constexpr auto ado = TensorAccessorArgs<av.next_compile_time_args_offset()>();
    constexpr auto ao = TensorAccessorArgs<ado.next_compile_time_args_offset()>();
    constexpr auto al = TensorAccessorArgs<ao.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t page = get_tile_size(cb_q);
    const uint32_t page32 = get_tile_size(cb_lse);
    const auto q = TensorAccessor(aq, get_arg_val<uint32_t>(a++), page);
    const auto k = TensorAccessor(ak, get_arg_val<uint32_t>(a++), page);
    const auto v = TensorAccessor(av, get_arg_val<uint32_t>(a++), page);
    const auto dout = TensorAccessor(ado, get_arg_val<uint32_t>(a++), page);
    const auto o = TensorAccessor(ao, get_arg_val<uint32_t>(a++), page);
    const auto lse = TensorAccessor(al, get_arg_val<uint32_t>(a++), page32);
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
    constexpr uint32_t JA = 8;
    constexpr uint32_t SC = ktt_flash_bw::kChunk;
    constexpr auto au = TensorAccessorArgs<al.next_compile_time_args_offset()>();
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
    const uint32_t KST = get_arg_val<uint32_t>(tail0 + 5);
    const uint32_t VST = get_arg_val<uint32_t>(tail0 + 6);
#ifdef FLASH_MASK_BCAST
    // The key mask [1, Tk]: one row tile per key tile, a further accessor and address.
    constexpr auto akm = TensorAccessorArgs<ap.next_compile_time_args_offset()>();
    const auto kmask = TensorAccessor(akm, get_arg_val<uint32_t>(tail0 + 7), page);
#endif
    auto zero_rows = [](uint32_t addr, uint32_t first) {
        ktt_flash::zero_tile_rows(
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr), first
        );
    };
#else
    constexpr uint32_t JA = 5;
    constexpr uint32_t KST = DT, VST = DV;
#ifdef FLASH_LENGTHS
    // After the key tail: the key lengths (TILE int32 [B]: batch b at row 0, column b
    // % 32 of tile b / 32) and the K / V heads per batch.
    const auto lens = TensorAccessor(
        TensorAccessorArgs<al.next_compile_time_args_offset()>(),
        get_arg_val<uint32_t>(a + JA * jobs + 1),
        get_tile_size(cb_lens)
    );
    const uint32_t HK = get_arg_val<uint32_t>(a + JA * jobs + 2);
    // The keys of K / V (the forward's clamp): the key tail argument gives Sk.
    const uint32_t key_tail = get_arg_val<uint32_t>(a + JA * jobs);
    const uint32_t keys = (key_tail != 0 ? SKT - 1 : SKT) * 32 + key_tail;
    cb_reserve_back(cb_lens, 1);
    const auto* len_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(cb_lens)
    );
    uint32_t len_tile = 0xffffffffu;
    uint32_t k_tail = 0, built_tail = 0;
#else
    const uint32_t k_tail = get_arg_val<uint32_t>(a + JA * jobs);
#endif
#endif

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
#ifdef FLASH_VARLEN
            const uint32_t row0 = get_arg_val<uint32_t>(a + JA * job);
            const uint32_t k0 = get_arg_val<uint32_t>(a + JA * job + 1);
            const uint32_t v0 = get_arg_val<uint32_t>(a + JA * job + 2);
            const uint32_t rows = get_arg_val<uint32_t>(a + JA * job + 3);
            const uint32_t nk = get_arg_val<uint32_t>(a + JA * job + 4);
            const uint32_t stride = get_arg_val<uint32_t>(a + JA * job + 5);
            const uint32_t key0 = get_arg_val<uint32_t>(a + JA * job + 6);
            const uint32_t trow0 = get_arg_val<uint32_t>(a + JA * job + 7);
            if (nk == 0) {
                continue;  // an empty job (the job list is padded to a fixed length)
            }
            // The job's row records: the plan pages from its first row's on.
            RowWindow win[R];
            {
                cb_reserve_back(cb_plan, kPlanPages);
                const uint32_t pb = get_write_ptr(cb_plan);
                const uint32_t p0 = trow0 / kRecordsPerPage;
                for (uint32_t p = 0; p < kPlanPages; ++p) {
                    const uint32_t pg = p0 + p < plan_pages ? p0 + p : plan_pages - 1;
                    noc_async_read_page(pg, plan, pb + p * plan_page);
                }
                noc_async_read_barrier();
                const auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pb);
                for (uint32_t i = 0; i < rows; ++i) {
                    const auto* rec = words +
                        (trow0 % kRecordsPerPage + i) * kRecordWords;
                    win[i] = RowWindow{
                        rec[0] - key0, rec[1] - key0, rec[2], rec[3], rec[4]
                    };
                }
                cb_push_back(cb_plan, kPlanPages);
            }
#else
            const uint32_t row0 = get_arg_val<uint32_t>(a + JA * job);
            const uint32_t kvh = get_arg_val<uint32_t>(a + JA * job + 1);
            const uint32_t rows = get_arg_val<uint32_t>(a + JA * job + 2);
            uint32_t nk = get_arg_val<uint32_t>(a + JA * job + 3);
            const uint32_t stride = get_arg_val<uint32_t>(a + JA * job + 4);
            const uint32_t k0 = kvh * SKT * DT, v0 = kvh * SKT * DV;
#ifdef FLASH_LENGTHS
            // The batch's key length (clamped to [1, Sk]): KT key tiles, the
            // job's nk of them, k_tail keys in the last; (nk, k_tail, KT) to cb_len.
            const uint32_t b = kvh / HK;
            if (b / 32 != len_tile) {
                len_tile = b / 32;
                noc_async_read_page(len_tile, lens, get_write_ptr(cb_lens));
                noc_async_read_barrier();
            }
            uint32_t len = len_words[b % 32 < 16 ? b % 32 : 256 + b % 32 - 16];
            len = len < 1 ? 1 : (len > keys ? keys : len);
            const uint32_t KT = (len + 31) / 32;
            k_tail = len % 32;
            nk = nk < KT ? nk : KT;
            // A new tail: its mask tiles (cb_tail, causal cb_ctail) for the compute.
            const bool fresh = k_tail != 0 && k_tail != built_tail;
            if (fresh) {
#ifdef FLASH_CAUSAL
                ktt_flash::push_mask_tiles(cb_tail, cb_ctail, k_tail, true);
#else
                ktt_flash::push_mask_tiles(cb_tail, cb_ctail, k_tail, false);
#endif
                built_tail = k_tail;
            }
            {
                cb_reserve_back(cb_len, 1);
                auto* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_write_ptr(cb_len)
                );
                tok[0] = nk;
                tok[1] = k_tail;
                tok[2] = KT;
                tok[3] = fresh ? 1 : 0;
                cb_push_back(cb_len, 1);
            }
#else
            constexpr uint32_t KT = SKT;
#endif
#endif

            cb_reserve_back(cb_q, R * DT);
            cb_reserve_back(cb_do, R * DV);
            cb_reserve_back(cb_oin, R * DV);
            cb_reserve_back(cb_lse, R);
            const uint32_t qb = get_write_ptr(cb_q), dob = get_write_ptr(cb_do);
            const uint32_t ob = get_write_ptr(cb_oin), lb = get_write_ptr(cb_lse);
            for (uint32_t i = 0; i < rows; ++i) {
                const uint32_t row = row0 + i * stride;
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_read_page(row * DT + d, q, qb + (i * DT + d) * page);
                }
                for (uint32_t t = 0; t < DV; ++t) {
                    noc_async_read_page(row * DV + t, dout, dob + (i * DV + t) * page);
                    noc_async_read_page(row * DV + t, o, ob + (i * DV + t) * page);
                }
                noc_async_read_page(row, lse, lb + i * page32);
            }
            // The rows' reads complete under the first key block's barrier.
            bool rows_pending = true;
            auto push_rows = [&] {
                if (rows_pending) {
#ifdef FLASH_VARLEN
                    // Rows past the last query hold undefined values: Q, dO and O to
                    // zero.
                    if (q_tail != 0 && rows > 0 && trow0 + rows == q_tiles) {
                        for (uint32_t d = 0; d < DT; ++d) {
                            zero_rows(qb + ((rows - 1) * DT + d) * page, q_tail);
                        }
                        for (uint32_t t = 0; t < DV; ++t) {
                            zero_rows(dob + ((rows - 1) * DV + t) * page, q_tail);
                            zero_rows(ob + ((rows - 1) * DV + t) * page, q_tail);
                        }
                    }
#endif
                    cb_push_back(cb_q, R * DT);
                    cb_push_back(cb_do, R * DV);
                    cb_push_back(cb_oin, R * DV);
                    cb_push_back(cb_lse, R);
                    rows_pending = false;
                }
            };

            for (uint32_t k_start = 0; k_start < nk; k_start += KB) {
                cb_reserve_back(cb_k, DT * KB);
                cb_reserve_back(cb_v, KB * DV);
                const uint32_t kbase = get_write_ptr(cb_k), vbase = get_write_ptr(cb_v);
                const uint32_t kb = nk - k_start < KB ? nk - k_start : KB;
#ifdef FLASH_MASK_BCAST
                cb_reserve_back(cb_mb, KB);
                const uint32_t mbase = get_write_ptr(cb_mb);
#endif
                if (sender) {
                    for (uint32_t j = 0; j < kb; ++j) {
                        for (uint32_t d = 0; d < DT; ++d) {
                            noc_async_read_page(
                                k0 + (k_start + j) * KST + d,
                                k,
                                kbase + (d * KB + j) * page
                            );
                        }
                        for (uint32_t u = 0; u < DV; ++u) {
                            noc_async_read_page(
                                v0 + (k_start + j) * VST + u,
                                v,
                                vbase + (j * DV + u) * page
                            );
                        }
#ifdef FLASH_MASK_BCAST
                        noc_async_read_page(
                            key0 + k_start + j, kmask, mbase + j * page
                        );
#endif
                    }
                    noc_async_read_barrier();
#ifdef FLASH_VARLEN
                    if (k_tail != 0 && key0 + k_start + kb == k_tiles) {
                        for (uint32_t d = 0; d < DT; ++d) {
                            zero_rows(kbase + (d * KB + kb - 1) * page, k_tail);
                        }
                        for (uint32_t u = 0; u < DV; ++u) {
                            zero_rows(vbase + ((kb - 1) * DV + u) * page, k_tail);
                        }
#ifdef FLASH_MASK_BCAST
                        ktt_flash::zero_row0_cols(
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                mbase + (kb - 1) * page
                            ),
                            k_tail
                        );
#endif
                    }
#else
                    // Rows past the last key hold undefined values: K and V to zero.
                    if (k_tail != 0 && k_start + kb == KT) {
                        for (uint32_t d = 0; d < DT; ++d) {
                            ktt_flash::zero_rows_async(
                                kbase + (d * KB + kb - 1) * page, k_tail
                            );
                        }
                        for (uint32_t u = 0; u < DV; ++u) {
                            ktt_flash::zero_rows_async(
                                vbase + ((kb - 1) * DV + u) * page, k_tail
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
                            kbase,
                            get_noc_multicast_addr(sx, sy, ex, ey, kbase),
                            DT * KB * page,
                            receivers
                        );
                        noc_async_write_multicast(
                            vbase,
                            get_noc_multicast_addr(sx, sy, ex, ey, vbase),
                            KB * DV * page,
                            receivers
                        );
#ifdef FLASH_MASK_BCAST
                        noc_async_write_multicast(
                            mbase,
                            get_noc_multicast_addr(sx, sy, ex, ey, mbase),
                            KB * page,
                            receivers
                        );
#endif
                        noc_async_writes_flushed();
                        *valid = kbase;
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
                cb_push_back(cb_k, DT * KB);
                cb_push_back(cb_v, KB * DV);
#ifdef FLASH_MASK_BCAST
                cb_push_back(cb_mb, KB);
#endif
#ifdef FLASH_VARLEN
                // The mask tiles of this block's chunks in the compute's order (chunk,
                // row): a row's SC slots of a chunk, filled where its tile takes a
                // mask.
                for (uint32_t k0c = 0; k0c < kb; k0c += SC) {
                    const uint32_t nt = kb - k0c < SC ? kb - k0c : SC;
                    const uint32_t kg = k_start + k0c;
                    for (uint32_t r = 0; r < rows; ++r) {
                        const RowWindow& w = win[r];
                        const uint32_t lo = kg > w.first ? kg : w.first;
                        const uint32_t hi = kg + nt - 1 < w.last ? kg + nt - 1 : w.last;
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
                                        mbase + (t - k_start) * page
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
            if (rows_pending) {
                noc_async_read_barrier();
                push_rows();
            }
        }
    }
    noc_async_atomic_barrier();
}
