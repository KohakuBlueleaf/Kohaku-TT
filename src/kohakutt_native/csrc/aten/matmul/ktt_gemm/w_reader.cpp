// Block matmul W stream (RISCV_1, dedicated NoC 0). Per output block (bh, bw) and K
// block: the W block (kb x bn tiles, CB 1, row stride bn). IN1_SENDER: the core reads
// the W block from DRAM and multicasts it down its column; receivers free the slot on
// the sender's semaphore, then wait for VALID. The whole matmul repeats `iters` times
// (1 in normal use).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"
#include "hostdevcommon/common_values.hpp"

namespace {

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of w.
// Common runtime args: w address, K tiles, N tiles, kb, K blocks, bn, blocks per core
// (rows, columns), iterations, semaphore ids (in1 sender, in1 receiver), w stored [N,
// K], K ranges the K blocks are taken from in turn (1: in order), sentinel. Per-core
// runtime args: first column tile, in1 multicast rectangle (start x, y, end x, y) and
// receiver count, in1 sender (x, y), sentinel.
void kernel_main() {
    if (get_common_arg_val<uint32_t>(13) != ktt_gemm::kArgSentinel ||
        get_arg_val<uint32_t>(8) != ktt_gemm::kArgSentinel) {
        return;
    }
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t kt = get_common_arg_val<uint32_t>(1);
    const uint32_t nt = get_common_arg_val<uint32_t>(2);
    const uint32_t kb = get_common_arg_val<uint32_t>(3);
    const uint32_t nk = get_common_arg_val<uint32_t>(4);
    const uint32_t bn = get_common_arg_val<uint32_t>(5);
    const uint32_t nbh = get_common_arg_val<uint32_t>(6);
    const uint32_t nbw = get_common_arg_val<uint32_t>(7);
    const uint32_t iters = get_common_arg_val<uint32_t>(8);
    const uint32_t in1_s = get_semaphore(get_common_arg_val<uint32_t>(9));
    const uint32_t in1_r = get_semaphore(get_common_arg_val<uint32_t>(10));
    const bool w_nk = get_common_arg_val<uint32_t>(11) != 0;
    const uint32_t parts = get_common_arg_val<uint32_t>(12);

    const uint32_t n0 = get_arg_val<uint32_t>(0);
    const uint32_t mc1_sx = get_arg_val<uint32_t>(1), mc1_sy = get_arg_val<uint32_t>(2);
    const uint32_t mc1_ex = get_arg_val<uint32_t>(3), mc1_ey = get_arg_val<uint32_t>(4);
    const uint32_t mc1_num = get_arg_val<uint32_t>(5);
    const uint32_t s1_x = get_arg_val<uint32_t>(6), s1_y = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_w = 1;
    constexpr auto w_args = TensorAccessorArgs<0>();
    const uint32_t w_tile = get_tile_size(cb_w);
    [[maybe_unused]] const auto w = TensorAccessor(w_args, w_addr, w_tile);
    const uint32_t w_block = kb * bn;

#ifdef IN1_SENDER
    *word(in1_r) = VALID;
#endif

    for (uint32_t blk = 0; blk < iters * nbh * nbw; ++blk) {
        const uint32_t c0 = n0 + (blk % nbw) * bn;
        for (uint32_t kblk = 0; kblk < nk; ++kblk) {
            const uint32_t k0 = ktt_gemm::k_block(kblk, nk, parts) * kb;
            const uint32_t kc = kt - k0 < kb ? kt - k0 : kb;
            cb_reserve_back(cb_w, w_block);
#ifdef IN1_SENDER
            const uint32_t base = get_write_ptr(cb_w);
            // Pages in storage order; slot (k, c) at k bn + c either way.
            if (w_nk) {
                for (uint32_t c = 0; c < bn && c0 + c < nt; ++c) {
                    const uint32_t page = (c0 + c) * kt + k0;
                    for (uint32_t k = 0; k < kc; ++k) {
                        noc_async_read_page(page + k, w, base + (k * bn + c) * w_tile);
                    }
                }
            } else {
                for (uint32_t k = 0; k < kc; ++k) {
                    const uint32_t page = (k0 + k) * nt + c0;
                    for (uint32_t c = 0; c < bn && c0 + c < nt; ++c) {
                        noc_async_read_page(page + c, w, base + (k * bn + c) * w_tile);
                    }
                }
            }
            noc_async_read_barrier();
            if (mc1_num > 0) {
                noc_semaphore_wait(word(in1_s), mc1_num);
                noc_semaphore_set(word(in1_s), 0);
                noc_async_write_multicast(
                    base,
                    get_noc_multicast_addr(mc1_sx, mc1_sy, mc1_ex, mc1_ey, base),
                    w_block * w_tile,
                    mc1_num
                );
                noc_async_writes_flushed();
                noc_semaphore_set_multicast(
                    in1_r,
                    get_noc_multicast_addr(mc1_sx, mc1_sy, mc1_ex, mc1_ey, in1_r),
                    mc1_num
                );
            }
#else
            noc_semaphore_set(word(in1_r), INVALID);
            noc_semaphore_inc(get_noc_addr(s1_x, s1_y, in1_s), 1);
            noc_semaphore_wait(word(in1_r), VALID);
#endif
            cb_push_back(cb_w, w_block);
        }
    }
    (void)mc1_sx, (void)mc1_sy, (void)mc1_ex, (void)mc1_ey, (void)mc1_num, (void)s1_x,
        (void)s1_y;
    (void)in1_s, (void)nt, (void)w_nk;
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
