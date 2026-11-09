// Block matmul reader (RISCV_1, dedicated NoC). Per output block (bh, bw) and K block: the A
// block (bm x kb tiles, CB 0, row stride kb) and the W block (kb x bn tiles, CB 1, row stride
// bn). IN0_SENDER: the core reads its A block from DRAM and, when it has receivers, multicasts
// it along its panel row. IN1_SENDER: the core reads the W block and multicasts it to up to two
// rectangles inside its panel. Receivers free the slot on the sender's semaphore, then wait for
// VALID. The whole matmul repeats `iters` times (1 in normal use).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"
#include "hostdevcommon/common_values.hpp"

namespace {

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of a, then of w.
// Common runtime args: a address, w address, K tiles, N tiles, M tiles, kb, K blocks, bm, bn,
// blocks per core (rows, columns), iterations, semaphore ids (in0 sender, in0 receiver, in1
// sender, in1 receiver), w stored [N, K], sentinel.
// Per-core runtime args: first row tile, first column tile, in0 multicast rectangle (start x,
// y, end x, y) and receiver count, in0 sender (x, y), in1 multicast rectangle and receiver
// count, in1 sender (x, y), second in1 multicast rectangle and receiver count (0: none),
// sentinel.
void kernel_main() {
    if (get_common_arg_val<uint32_t>(17) != ktt_gemm::kArgSentinel ||
        get_arg_val<uint32_t>(21) != ktt_gemm::kArgSentinel) {
        return;
    }
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t kt = get_common_arg_val<uint32_t>(2);
    const uint32_t nt = get_common_arg_val<uint32_t>(3);
    const uint32_t mt = get_common_arg_val<uint32_t>(4);
    const uint32_t kb = get_common_arg_val<uint32_t>(5);
    const uint32_t nk = get_common_arg_val<uint32_t>(6);
    const uint32_t bm = get_common_arg_val<uint32_t>(7);
    const uint32_t bn = get_common_arg_val<uint32_t>(8);
    const uint32_t nbh = get_common_arg_val<uint32_t>(9);
    const uint32_t nbw = get_common_arg_val<uint32_t>(10);
    const uint32_t iters = get_common_arg_val<uint32_t>(11);
    const uint32_t in0_s = get_semaphore(get_common_arg_val<uint32_t>(12));
    const uint32_t in0_r = get_semaphore(get_common_arg_val<uint32_t>(13));
    const uint32_t in1_s = get_semaphore(get_common_arg_val<uint32_t>(14));
    const uint32_t in1_r = get_semaphore(get_common_arg_val<uint32_t>(15));
    const bool w_nk = get_common_arg_val<uint32_t>(16) != 0;

    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const uint32_t mc0_sx = get_arg_val<uint32_t>(2), mc0_sy = get_arg_val<uint32_t>(3);
    const uint32_t mc0_ex = get_arg_val<uint32_t>(4), mc0_ey = get_arg_val<uint32_t>(5);
    const uint32_t mc0_num = get_arg_val<uint32_t>(6);
    const uint32_t s0_x = get_arg_val<uint32_t>(7), s0_y = get_arg_val<uint32_t>(8);
    const uint32_t mc1_sx = get_arg_val<uint32_t>(9), mc1_sy = get_arg_val<uint32_t>(10);
    const uint32_t mc1_ex = get_arg_val<uint32_t>(11), mc1_ey = get_arg_val<uint32_t>(12);
    const uint32_t mc1_num = get_arg_val<uint32_t>(13);
    const uint32_t s1_x = get_arg_val<uint32_t>(14), s1_y = get_arg_val<uint32_t>(15);
    const uint32_t mc2_sx = get_arg_val<uint32_t>(16), mc2_sy = get_arg_val<uint32_t>(17);
    const uint32_t mc2_ex = get_arg_val<uint32_t>(18), mc2_ey = get_arg_val<uint32_t>(19);
    const uint32_t mc2_num = get_arg_val<uint32_t>(20);

    constexpr uint32_t cb_a = 0, cb_w = 1;
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const uint32_t a_tile = get_tile_size(cb_a), w_tile = get_tile_size(cb_w);
    [[maybe_unused]] const auto a = TensorAccessor(a_args, a_addr, a_tile);
    [[maybe_unused]] const auto w = TensorAccessor(w_args, w_addr, w_tile);
    const uint32_t a_block = bm * kb, w_block = kb * bn;

#ifdef IN0_SENDER
    *word(in0_r) = VALID;
#endif
#ifdef IN1_SENDER
    *word(in1_r) = VALID;
#endif

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t bh = 0; bh < nbh; ++bh) {
            const uint32_t r0 = m0 + bh * bm;
            for (uint32_t bw = 0; bw < nbw; ++bw) {
                const uint32_t c0 = n0 + bw * bn;
                for (uint32_t kblk = 0; kblk < nk; ++kblk) {
                    const uint32_t k0 = kblk * kb;
                    const uint32_t kc = kt - k0 < kb ? kt - k0 : kb;

                    cb_reserve_back(cb_a, a_block);
#ifdef IN0_SENDER
                    {
                        const uint32_t base = get_write_ptr(cb_a);
                        for (uint32_t r = 0; r < bm; ++r) {
                            if (r0 + r >= mt) {
                                break;
                            }
                            const uint32_t page = (r0 + r) * kt + k0;
                            for (uint32_t k = 0; k < kc; ++k) {
                                noc_async_read_page(page + k, a, base + (r * kb + k) * a_tile);
                            }
                        }
                        noc_async_read_barrier();
                        if (mc0_num > 0) {
                            noc_semaphore_wait(word(in0_s), mc0_num);
                            noc_semaphore_set(word(in0_s), 0);
                            noc_async_write_multicast(
                                base, get_noc_multicast_addr(mc0_sx, mc0_sy, mc0_ex, mc0_ey, base),
                                a_block * a_tile, mc0_num
                            );
                            noc_async_writes_flushed();
                            noc_semaphore_set_multicast(
                                in0_r, get_noc_multicast_addr(mc0_sx, mc0_sy, mc0_ex, mc0_ey, in0_r), mc0_num
                            );
                        }
                    }
#else
                    noc_semaphore_set(word(in0_r), INVALID);
                    noc_semaphore_inc(get_noc_addr(s0_x, s0_y, in0_s), 1);
                    noc_semaphore_wait(word(in0_r), VALID);
#endif
                    cb_push_back(cb_a, a_block);

                    cb_reserve_back(cb_w, w_block);
#ifdef IN1_SENDER
                    {
                        const uint32_t base = get_write_ptr(cb_w);
                        // Pages in storage order (consecutive pages, consecutive DRAM banks);
                        // slot (k, c) at k bn + c either way.
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
                        if (mc1_num + mc2_num > 0) {
                            noc_semaphore_wait(word(in1_s), mc1_num + mc2_num);
                            noc_semaphore_set(word(in1_s), 0);
                            if (mc1_num > 0) {
                                noc_async_write_multicast(
                                    base, get_noc_multicast_addr(mc1_sx, mc1_sy, mc1_ex, mc1_ey, base),
                                    w_block * w_tile, mc1_num
                                );
                            }
                            if (mc2_num > 0) {
                                noc_async_write_multicast(
                                    base, get_noc_multicast_addr(mc2_sx, mc2_sy, mc2_ex, mc2_ey, base),
                                    w_block * w_tile, mc2_num
                                );
                            }
                            noc_async_writes_flushed();
                            if (mc1_num > 0) {
                                noc_semaphore_set_multicast(
                                    in1_r, get_noc_multicast_addr(mc1_sx, mc1_sy, mc1_ex, mc1_ey, in1_r), mc1_num
                                );
                            }
                            if (mc2_num > 0) {
                                noc_semaphore_set_multicast(
                                    in1_r, get_noc_multicast_addr(mc2_sx, mc2_sy, mc2_ex, mc2_ey, in1_r), mc2_num
                                );
                            }
                        }
                    }
#else
                    noc_semaphore_set(word(in1_r), INVALID);
                    noc_semaphore_inc(get_noc_addr(s1_x, s1_y, in1_s), 1);
                    noc_semaphore_wait(word(in1_r), VALID);
#endif
                    cb_push_back(cb_w, w_block);
                }
            }
        }
    }
    (void)mc0_sx, (void)mc0_sy, (void)mc0_ex, (void)mc0_ey, (void)mc0_num, (void)s0_x, (void)s0_y;
    (void)mc1_sx, (void)mc1_sy, (void)mc1_ex, (void)mc1_ey, (void)mc1_num, (void)s1_x, (void)s1_y;
    (void)mc2_sx, (void)mc2_sy, (void)mc2_ex, (void)mc2_ey, (void)mc2_num;
    (void)in0_s, (void)in1_s, (void)mt, (void)nt, (void)w_nk;
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
