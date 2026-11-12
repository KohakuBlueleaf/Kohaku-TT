// Block matmul A stream and output writer (RISCV_0, dedicated NoC 1). Per output block
// (bh, bw), with HAS_BIAS its bn bias tiles (CB 3); per K block: the A block (bm x kb
// tiles, CB 0, row stride kb). IN0_SENDER:
// the core reads its A block from DRAM and, when it has receivers, multicasts it along
// its panel row; receivers free the slot on the sender's semaphore, then wait for
// VALID. An output block (sh x sw subblocks from CB 16 in compute's order, tiles
// outside the core's range dropped) goes to DRAM after the first A block of the next
// output block. The whole matmul repeats `iters` times (1 in normal use).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"
#include "hostdevcommon/common_values.hpp"

namespace {

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of a, then of the output (and of the bias).
// Common runtime args: a address, output address, K tiles, N tiles, M tiles, kb, K
// blocks, bm, bn, blocks per core (rows, columns), iterations, semaphore ids (in0
// sender, in0 receiver), sh, sw, K ranges the K blocks are taken from in turn (1: in
// order), first semaphore id of the ranges' progress counts and increments per K block
// (0: A is not waited for), bias address, sentinel. Per-core runtime args: first row
// tile, first column tile, end row tile, end column tile, in0 multicast rectangle
// (start x, y, end x, y) and receiver count, in0 sender (x, y), sentinel.
void kernel_main() {
    if (get_common_arg_val<uint32_t>(20) != ktt_gemm::kArgSentinel ||
        get_arg_val<uint32_t>(11) != ktt_gemm::kArgSentinel) {
        return;
    }
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
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
    const uint32_t sh = get_common_arg_val<uint32_t>(14);
    const uint32_t sw = get_common_arg_val<uint32_t>(15);
    const uint32_t parts = get_common_arg_val<uint32_t>(16);
    const uint32_t progress = get_common_arg_val<uint32_t>(17);
    const uint32_t per_block = get_common_arg_val<uint32_t>(18);
    [[maybe_unused]] const uint32_t bias_addr = get_common_arg_val<uint32_t>(19);

    const uint32_t m0 = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const uint32_t m_end = get_arg_val<uint32_t>(2);
    const uint32_t n_end = get_arg_val<uint32_t>(3);
    const uint32_t mc0_sx = get_arg_val<uint32_t>(4), mc0_sy = get_arg_val<uint32_t>(5);
    const uint32_t mc0_ex = get_arg_val<uint32_t>(6), mc0_ey = get_arg_val<uint32_t>(7);
    const uint32_t mc0_num = get_arg_val<uint32_t>(8);
    const uint32_t s0_x = get_arg_val<uint32_t>(9), s0_y = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_a = 0, cb_out = 16;
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto
        out_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const uint32_t a_tile = get_tile_size(cb_a), out_tile = get_tile_size(cb_out);
    [[maybe_unused]] const auto a = TensorAccessor(a_args, a_addr, a_tile);
    const auto out = TensorAccessor(out_args, out_addr, out_tile);
#ifdef HAS_BIAS
    constexpr uint32_t cb_bias = 3;
    constexpr auto
        bias_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto bias = TensorAccessor(bias_args, bias_addr, get_tile_size(cb_bias));
#endif
    const uint32_t a_block = bm * kb, sub = sh * sw;

    // Output block (r0, c0) from CB 16 to DRAM.
    const auto write_block = [&](uint32_t r0, uint32_t c0) {
        for (uint32_t i = 0; i < bm; i += sh) {
            for (uint32_t j = 0; j < bn; j += sw) {
                cb_wait_front(cb_out, sub);
                uint32_t src = get_read_ptr(cb_out);
                for (uint32_t r = 0; r < sh; ++r) {
                    const uint32_t row = r0 + i + r;
                    for (uint32_t c = 0; c < sw; ++c) {
                        const uint32_t col = c0 + j + c;
                        if (row < m_end && col < n_end) {
                            noc_async_write_page(row * nt + col, out, src);
                        }
                        src += out_tile;
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out, sub);
            }
        }
    };

#ifdef IN0_SENDER
    *word(in0_r) = VALID;
#endif

    bool pending = false;
    uint32_t pending_r0 = 0, pending_c0 = 0;
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t bh = 0; bh < nbh; ++bh) {
            const uint32_t r0 = m0 + bh * bm;
            for (uint32_t bw = 0; bw < nbw; ++bw) {
#ifdef HAS_BIAS
                cb_reserve_back(cb_bias, bn);
                for (uint32_t c = 0; c < bn; ++c) {
                    const uint32_t col = n0 + bw * bn + c;
                    if (col < n_end) {
                        noc_async_read_page(
                            col, bias, get_write_ptr(cb_bias) + c * out_tile
                        );
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_bias, bn);
#endif
                for (uint32_t kblk = 0; kblk < nk; ++kblk) {
                    const uint32_t k0 = ktt_gemm::k_block(kblk, nk, parts) * kb;
                    const uint32_t kc = kt - k0 < kb ? kt - k0 : kb;
                    cb_reserve_back(cb_a, a_block);
#ifdef IN0_SENDER
                    if (per_block > 0) {
                        // A's tiles of this K block are written by another kernel of
                        // the program.
                        noc_semaphore_wait_min(
                            word(get_semaphore(progress + kblk % parts)),
                            per_block * (kblk / parts + 1)
                        );
                    }
                    const uint32_t base = get_write_ptr(cb_a);
                    for (uint32_t r = 0; r < bm; ++r) {
                        if (r0 + r >= mt) {
                            break;
                        }
                        const uint32_t page = (r0 + r) * kt + k0;
                        for (uint32_t k = 0; k < kc; ++k) {
                            noc_async_read_page(
                                page + k, a, base + (r * kb + k) * a_tile
                            );
                        }
                    }
                    noc_async_read_barrier();
                    if (mc0_num > 0) {
                        noc_semaphore_wait(word(in0_s), mc0_num);
                        noc_semaphore_set(word(in0_s), 0);
                        noc_async_write_multicast(
                            base,
                            get_noc_multicast_addr(
                                mc0_sx, mc0_sy, mc0_ex, mc0_ey, base
                            ),
                            a_block * a_tile,
                            mc0_num
                        );
                        noc_async_writes_flushed();
                        noc_semaphore_set_multicast(
                            in0_r,
                            get_noc_multicast_addr(
                                mc0_sx, mc0_sy, mc0_ex, mc0_ey, in0_r
                            ),
                            mc0_num
                        );
                    }
#else
                    noc_semaphore_set(word(in0_r), INVALID);
                    noc_semaphore_inc(get_noc_addr(s0_x, s0_y, in0_s), 1);
                    noc_semaphore_wait(word(in0_r), VALID);
#endif
                    cb_push_back(cb_a, a_block);
                    if (kblk == 0 && pending) {
                        write_block(pending_r0, pending_c0);
                        pending = false;
                    }
                }
                pending = true;
                pending_r0 = r0;
                pending_c0 = n0 + bw * bn;
            }
        }
    }
    if (pending) {
        write_block(pending_r0, pending_c0);
    }
    (void)mc0_sx, (void)mc0_sy, (void)mc0_ex, (void)mc0_ey, (void)mc0_num, (void)s0_x,
        (void)s0_y;
    (void)in0_s, (void)mt, (void)progress, (void)per_block;
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
