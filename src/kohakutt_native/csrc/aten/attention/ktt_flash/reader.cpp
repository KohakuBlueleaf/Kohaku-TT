// Flash attention reader: each job of each iteration, its Q rows (cb_q, R x DT slots
// per job), then the job's K / V key blocks (cb_k d-major [d][KB], cb_v key-major
// [KB][DV]). Lane sender: reads each K / V block from DRAM, waits until every receiver
// has freed the slot, multicasts K and V down the lane, then multicasts the VALID flag
// (the K slot address). Receiver: clears VALID, signals the sender, waits for VALID.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "flash.hpp"
#include "flash_dm.hpp"

// Compile-time args: DT, DV, R, NK (key tiles of the tensors), KB, then
// TensorAccessorArgs of Q, K, V and the user mask. Runtime args: Q, K, V, user mask
// addresses, iterations, multicast rectangle (sx, sy, ex, ey), receivers, sender (x,
// y), ready / valid semaphore ids, 1 on the lane sender, jobs, then per job: first Q
// page, first K page, first V page, query tile rows, key tiles, row stride, first user
// mask page; then the keys in the last key tile (0: 32). FLASH_MASK_ROW: per-row user
// mask tiles in step order (cb_um). FLASH_MASK_BCAST: one user mask row per key block,
// multicast with K / V (cb_mb).
void kernel_main() {
    using namespace ktt_flash;
    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t SKT = get_compile_time_arg_val(3);
    constexpr uint32_t KB = get_compile_time_arg_val(4);
    constexpr uint32_t SC = kSubChunk;
    constexpr auto aq = TensorAccessorArgs<5>();
    constexpr auto ak = TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    constexpr auto av = TensorAccessorArgs<ak.next_compile_time_args_offset()>();
    constexpr auto au = TensorAccessorArgs<av.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t page = get_tile_size(cb_q);
    const auto q = TensorAccessor(aq, get_arg_val<uint32_t>(a++), page);
    const auto k = TensorAccessor(ak, get_arg_val<uint32_t>(a++), page);
    const auto v = TensorAccessor(av, get_arg_val<uint32_t>(a++), page);
    const auto umask = TensorAccessor(au, get_arg_val<uint32_t>(a++), page);
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

    (void)umask;
#ifdef FLASH_VARLEN
    // Per job also its first query tile row; the 7th value is the stream's first key
    // tile.
    constexpr uint32_t JA = 8;
    constexpr auto ap = TensorAccessorArgs<au.next_compile_time_args_offset()>();
    const uint32_t plan_page = get_tile_size(cb_plan);
    const auto plan = TensorAccessor(
        ap, get_arg_val<uint32_t>(a + JA * jobs), plan_page
    );
    const uint32_t plan_pages = get_arg_val<uint32_t>(a + JA * jobs + 1);
    // Tiles and tokens in the last tile (0: 32) of the queries and of the keys, as
    // tiles * 64 + tail.
    const uint32_t q_tiles = get_arg_val<uint32_t>(a + JA * jobs + 2) / 64;
    const uint32_t q_tail = get_arg_val<uint32_t>(a + JA * jobs + 2) % 64;
    const uint32_t k_tiles = get_arg_val<uint32_t>(a + JA * jobs + 3) / 64;
    const uint32_t k_tail = get_arg_val<uint32_t>(a + JA * jobs + 3) % 64;
    // Pages from one key tile to the next in K and in V (the heads of a key tile lie
    // side by side: Hk x DT and Hk x DV).
    const uint32_t KST = get_arg_val<uint32_t>(a + JA * jobs + 4);
    const uint32_t VST = get_arg_val<uint32_t>(a + JA * jobs + 5);
#ifdef FLASH_MASK_BCAST
    // The key mask [1, Tk]: one row tile per key tile, a further accessor and address.
    constexpr auto akm = TensorAccessorArgs<ap.next_compile_time_args_offset()>();
    const auto kmask = TensorAccessor(
        akm, get_arg_val<uint32_t>(a + JA * jobs + 6), page
    );
#endif
    // Zeroes rows [first, 32) of the bf16 tile at L1 address `addr` (four 16 x 16
    // faces): the rows past the last token, whose contents are undefined.
    auto zero_rows = [](uint32_t addr, uint32_t first) {
        auto* half = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t r = first; r < 32; ++r) {
            const uint32_t base = (r / 16) * 512 + (r % 16) * 16;
            for (uint32_t c = 0; c < 16; ++c) {
                half[base + c] = 0;
                half[base + 256 + c] = 0;
            }
        }
    };
#else
    constexpr uint32_t JA = 7;
    constexpr uint32_t KST = DT, VST = DV;
#ifdef FLASH_LENGTHS
    // The key lengths (TILE int32 [B], in the user mask's place; a job's u0 is its
    // batch): batch b at row 0, column b % 32 of tile b / 32, read into cb_lens.
    const auto lens = TensorAccessor(
        au, get_arg_val<uint32_t>(3), get_tile_size(cb_lens)
    );
    cb_reserve_back(cb_lens, 1);
    const auto* len_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(cb_lens)
    );
    uint32_t len_tile = 0xffffffffu;
    uint32_t k_tail = 0, built_tail = 0;
#else
    const uint32_t k_tail = get_arg_val<uint32_t>(a + JA * jobs);
#endif
#ifdef FLASH_MASK_BCAST
    const auto& kmask = umask;
#endif
#endif

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t q0 = get_arg_val<uint32_t>(a + JA * job);
            const uint32_t k0 = get_arg_val<uint32_t>(a + JA * job + 1);
            const uint32_t v0 = get_arg_val<uint32_t>(a + JA * job + 2);
            const uint32_t rows = get_arg_val<uint32_t>(a + JA * job + 3);
            [[maybe_unused]] uint32_t NK = get_arg_val<uint32_t>(a + JA * job + 4);
            const uint32_t stride = get_arg_val<uint32_t>(a + JA * job + 5);
            const uint32_t u0 = get_arg_val<uint32_t>(a + JA * job + 6);
#ifdef FLASH_VARLEN
            [[maybe_unused]] const uint32_t KT = SKT;
#elif defined(FLASH_LENGTHS)
            // The batch's key length (clamped to [1, Sk tiles x 32]): KT key tiles, the
            // job's NK of them, k_tail keys in the last; (NK, k_tail, KT) to cb_len.
            if (u0 / 32 != len_tile) {
                len_tile = u0 / 32;
                noc_async_read_page(len_tile, lens, get_write_ptr(cb_lens));
                noc_async_read_barrier();
            }
            uint32_t len = len_words[u0 % 32 < 16 ? u0 % 32 : 256 + u0 % 32 - 16];
            len = len < 1 ? 1 : (len > SKT * 32 ? SKT * 32 : len);
            const uint32_t KT = (len + 31) / 32;
            k_tail = len % 32;
            NK = NK < KT ? NK : KT;
            // A new tail: its mask tiles (cb_tail, causal cb_ctail) for the compute.
            const bool fresh = k_tail != 0 && k_tail != built_tail;
            if (fresh) {
#ifdef FLASH_CAUSAL
                push_mask_tiles(cb_tail, cb_ctail, k_tail, true);
#else
                push_mask_tiles(cb_tail, cb_ctail, k_tail, false);
#endif
                built_tail = k_tail;
            }
            {
                cb_reserve_back(cb_len, 1);
                auto* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_write_ptr(cb_len)
                );
                tok[0] = NK;
                tok[1] = k_tail;
                tok[2] = KT;
                tok[3] = fresh ? 1 : 0;
                cb_push_back(cb_len, 1);
            }
#else
            constexpr uint32_t KT = SKT;
#endif
#ifdef FLASH_VARLEN
            if (NK == 0) {
                continue;  // an empty job (the job list is padded to a fixed length)
            }
            // The job's row records: the plan pages from its first row's on.
            const uint32_t row0 = get_arg_val<uint32_t>(a + JA * job + 7);
            RowWindow win[R];
            {
                cb_reserve_back(cb_plan, kPlanPages);
                const uint32_t pb = get_write_ptr(cb_plan);
                const uint32_t p0 = row0 / kRecordsPerPage;
                for (uint32_t p = 0; p < kPlanPages; ++p) {
                    const uint32_t pg = p0 + p < plan_pages ? p0 + p : plan_pages - 1;
                    noc_async_read_page(pg, plan, pb + p * plan_page);
                }
                noc_async_read_barrier();
                const auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pb);
                for (uint32_t i = 0; i < rows; ++i) {
                    const auto* rec = words +
                        (row0 % kRecordsPerPage + i) * kRecordWords;
                    win[i] = RowWindow{
                        rec[0] - u0, rec[1] - u0, rec[2], rec[3], rec[4]
                    };
                }
                cb_push_back(cb_plan, kPlanPages);
            }
#endif

            cb_reserve_back(cb_q, R * DT);
            const uint32_t qb = get_write_ptr(cb_q);
            for (uint32_t i = 0; i < rows; ++i) {
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_read_page(
                        q0 + i * stride * DT + d, q, qb + (i * DT + d) * page
                    );
                }
            }
            noc_async_read_barrier();
#ifdef FLASH_VARLEN
            if (q_tail != 0 && rows > 0 && row0 + rows == q_tiles) {
                for (uint32_t d = 0; d < DT; ++d) {
                    zero_rows(qb + ((rows - 1) * DT + d) * page, q_tail);
                }
            }
#endif
            cb_push_back(cb_q, R * DT);

            // One key block: K, V (and the broadcast user mask row) into this core and
            // the lane. Returns the block's mask rows (L1 address; 0 without a
            // broadcast mask).
            auto kv_block = [&](uint32_t k_start) -> uint32_t {
                cb_reserve_back(cb_k, DT * KB);
                cb_reserve_back(cb_v, KB * DV);
                const uint32_t kbase = get_write_ptr(cb_k);
                const uint32_t vbase = get_write_ptr(cb_v);
#ifdef FLASH_MASK_BCAST
                cb_reserve_back(cb_mb, KB);
                const uint32_t mbase = get_write_ptr(cb_mb);
#else
                const uint32_t mbase = 0;
#endif
                if (sender) {
                    const uint32_t kb = NK - k_start < KB ? NK - k_start : KB;
                    for (uint32_t j = 0; j < kb; ++j) {
                        for (uint32_t d = 0; d < DT; ++d) {
                            noc_async_read_page(
                                k0 + (k_start + j) * KST + d,
                                k,
                                kbase + (d * KB + j) * page
                            );
                        }
                    }
                    for (uint32_t j = 0; j < kb; ++j) {
                        for (uint32_t c = 0; c < DV; ++c) {
                            noc_async_read_page(
                                v0 + (k_start + j) * VST + c,
                                v,
                                vbase + (j * DV + c) * page
                            );
                        }
                    }
#ifdef FLASH_MASK_BCAST
                    for (uint32_t j = 0; j < kb; ++j) {
                        noc_async_read_page(u0 + k_start + j, kmask, mbase + j * page);
                    }
#endif
                    noc_async_read_barrier();
#ifdef FLASH_VARLEN
                    // u0 is the stream's first key tile: the block's last tile is the
                    // keys'.
                    if (k_tail != 0 && u0 + k_start + kb == k_tiles) {
                        for (uint32_t d = 0; d < DT; ++d) {
                            zero_rows(kbase + (d * KB + kb - 1) * page, k_tail);
                        }
                        for (uint32_t c = 0; c < DV; ++c) {
                            zero_rows(vbase + ((kb - 1) * DV + c) * page, k_tail);
                        }
#ifdef FLASH_MASK_BCAST
                        zero_row0_cols(
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
                            zero_rows_async(kbase + (d * KB + kb - 1) * page, k_tail);
                        }
                        for (uint32_t c = 0; c < DV; ++c) {
                            zero_rows_async(vbase + ((kb - 1) * DV + c) * page, k_tail);
                        }
                        noc_async_read_barrier();
                    }
#endif
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
                    noc_semaphore_wait_min(valid, 1);
                }
                cb_push_back(cb_k, DT * KB);
                cb_push_back(cb_v, KB * DV);
#ifdef FLASH_MASK_BCAST
                cb_push_back(cb_mb, KB);
#endif
                return mbase;
            };

            uint32_t mb_next = kv_block(0);
            for (uint32_t k_start = 0; k_start < NK; k_start += KB) {
                [[maybe_unused]] const uint32_t mb_block = mb_next;
                if (k_start + KB < NK) {
                    mb_next = kv_block(k_start + KB);
                }
#ifdef FLASH_VARLEN
                // The mask tiles of this block's steps in the compute's order (chunk,
                // row): a step's SC slots, filled where its tile takes a mask.
                const uint32_t kbv = NK - k_start < KB ? NK - k_start : KB;
                for (uint32_t k0c = 0; k0c < kbv; k0c += SC) {
                    const uint32_t nt = kbv - k0c < SC ? kbv - k0c : SC;
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
                                fill_zeros_from_row0(
                                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                        ub + (t - lo) * page
                                    ),
                                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                        mb_block + (t - k_start) * page
                                    )
                                );
                            }
                        }
#endif
                        cb_push_back(cb_um, SC);
                    }
                }
#endif
#ifdef FLASH_MASK_ROW
                // Per-row user mask tiles of this block in the compute's step order
                // (chunk, row).
                const uint32_t kb = NK - k_start < KB ? NK - k_start : KB;
                for (uint32_t k0c = 0; k0c < kb; k0c += SC) {
                    const uint32_t nt = kb - k0c < SC ? kb - k0c : SC;
                    for (uint32_t r = 0; r < rows; ++r) {
                        cb_reserve_back(cb_um, SC);
                        const uint32_t ub = get_write_ptr(cb_um);
                        for (uint32_t t = 0; t < nt; ++t) {
                            noc_async_read_page(
                                u0 + r * stride * SKT + k_start + k0c + t,
                                umask,
                                ub + t * page
                            );
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_um, SC);
                    }
                }
#endif
            }
        }
    }
    noc_async_atomic_barrier();
}
