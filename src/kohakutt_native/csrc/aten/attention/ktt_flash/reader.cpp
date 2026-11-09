// Flash attention reader: each job of each iteration, its Q rows (cb_q, R x DT slots per job),
// then the job's K / V key blocks (cb_k d-major [d][KB], cb_v key-major [KB][DV]). Lane sender:
// reads each K / V block from DRAM, waits until every receiver has freed the slot, multicasts K
// and V down the lane, then multicasts the VALID flag (the K slot address). Receiver: clears
// VALID, signals the sender, waits for VALID.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "flash.hpp"

// Compile-time args: DT, DV, R, NK (key tiles of the tensors), KB, then TensorAccessorArgs of Q,
// K, V and the user mask. Runtime args: Q, K, V, user mask addresses, iterations, multicast
// rectangle (sx, sy, ex, ey), receivers, sender (x, y), ready / valid semaphore ids, 1 on the
// lane sender, jobs, then per job: first Q page, first K page, first V page, query tile rows,
// key tiles, row stride, first user mask page. FLASH_MASK_ROW: per-row user mask tiles in step
// order (cb_um). FLASH_MASK_BCAST: one user mask row per key block, multicast with K / V (cb_mb).
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
    const uint32_t snd_x = get_arg_val<uint32_t>(a++), snd_y = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t valid_addr = get_semaphore(get_arg_val<uint32_t>(a++));
    const bool sender = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t jobs = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);

    (void)umask;

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t job = 0; job < jobs; ++job) {
            const uint32_t q0 = get_arg_val<uint32_t>(a + 7 * job);
            const uint32_t k0 = get_arg_val<uint32_t>(a + 7 * job + 1);
            const uint32_t v0 = get_arg_val<uint32_t>(a + 7 * job + 2);
            const uint32_t rows = get_arg_val<uint32_t>(a + 7 * job + 3);
            const uint32_t NK = get_arg_val<uint32_t>(a + 7 * job + 4);
            const uint32_t stride = get_arg_val<uint32_t>(a + 7 * job + 5);
            const uint32_t u0 = get_arg_val<uint32_t>(a + 7 * job + 6);

            cb_reserve_back(cb_q, R * DT);
            const uint32_t qb = get_write_ptr(cb_q);
            for (uint32_t i = 0; i < rows; ++i) {
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_read_page(q0 + i * stride * DT + d, q, qb + (i * DT + d) * page);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_q, R * DT);

            // One key block: K, V (and the broadcast user mask row) into this core and the lane.
            auto kv_block = [&](uint32_t k_start) {
                cb_reserve_back(cb_k, DT * KB);
                cb_reserve_back(cb_v, KB * DV);
                const uint32_t kbase = get_write_ptr(cb_k);
                const uint32_t vbase = get_write_ptr(cb_v);
#ifdef FLASH_MASK_BCAST
                cb_reserve_back(cb_mb, KB);
                const uint32_t mbase = get_write_ptr(cb_mb);
#endif
                if (sender) {
                    const uint32_t kb = NK - k_start < KB ? NK - k_start : KB;
                    for (uint32_t j = 0; j < kb; ++j) {
                        for (uint32_t d = 0; d < DT; ++d) {
                            noc_async_read_page(k0 + (k_start + j) * DT + d, k, kbase + (d * KB + j) * page);
                        }
                    }
                    for (uint32_t i = 0; i < kb * DV; ++i) {
                        noc_async_read_page(v0 + k_start * DV + i, v, vbase + i * page);
                    }
#ifdef FLASH_MASK_BCAST
                    for (uint32_t j = 0; j < kb; ++j) {
                        noc_async_read_page(u0 + k_start + j, umask, mbase + j * page);
                    }
#endif
                    noc_async_read_barrier();
                    if (receivers > 0) {
                        noc_semaphore_wait(ready, receivers);
                        noc_semaphore_set(ready, 0);
                        noc_async_write_multicast(
                            kbase, get_noc_multicast_addr(sx, sy, ex, ey, kbase), DT * KB * page, receivers
                        );
                        noc_async_write_multicast(
                            vbase, get_noc_multicast_addr(sx, sy, ex, ey, vbase), KB * DV * page, receivers
                        );
#ifdef FLASH_MASK_BCAST
                        noc_async_write_multicast(
                            mbase, get_noc_multicast_addr(sx, sy, ex, ey, mbase), KB * page, receivers
                        );
#endif
                        noc_async_writes_flushed();
                        *valid = kbase;
                        noc_semaphore_set_multicast(
                            valid_addr, get_noc_multicast_addr(sx, sy, ex, ey, valid_addr), receivers
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
            };

            kv_block(0);
            for (uint32_t k_start = 0; k_start < NK; k_start += KB) {
                if (k_start + KB < NK) {
                    kv_block(k_start + KB);
                }
#ifdef FLASH_MASK_ROW
                // Per-row user mask tiles of this block in the compute's step order (chunk, row).
                const uint32_t kb = NK - k_start < KB ? NK - k_start : KB;
                for (uint32_t k0c = 0; k0c < kb; k0c += SC) {
                    const uint32_t nt = kb - k0c < SC ? kb - k0c : SC;
                    for (uint32_t r = 0; r < rows; ++r) {
                        cb_reserve_back(cb_um, SC);
                        const uint32_t ub = get_write_ptr(cb_um);
                        for (uint32_t t = 0; t < nt; ++t) {
                            noc_async_read_page(u0 + r * stride * SKT + k_start + k0c + t, umask, ub + t * page);
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
