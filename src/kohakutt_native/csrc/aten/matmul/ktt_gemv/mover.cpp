// ktt_gemv data mover, one per RISC (dm0: NoC 0, CB 1; dm1: NoC 1, CB 2): its share of
// the W blocks; with HAS_X x into CB 0, HAS_BIAS a group's bias tiles into CB 3,
// HAS_OUT the output from CB 16 to DRAM.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "args.hpp"

namespace {

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Args (built and listed in program.cpp): compile-time the side, then x, w, output
// (bias) accessors; common runtime addresses and sizes; per core its share.
void kernel_main() {
    constexpr uint32_t kNone = 0xffff, kFixed = 9;
    const uint32_t senders = get_common_arg_val<uint32_t>(10);
    if (get_common_arg_val<uint32_t>(12) != ktt_gemv::kArgSentinel ||
        get_arg_val<uint32_t>(kFixed + 2 * senders) != ktt_gemv::kArgSentinel) {
        return;
    }
    constexpr uint32_t side = get_compile_time_arg_val(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t mt = get_common_arg_val<uint32_t>(3);
    const uint32_t kt = get_common_arg_val<uint32_t>(4);
    const uint32_t nt = get_common_arg_val<uint32_t>(5);
    const uint32_t sw = get_common_arg_val<uint32_t>(6);
    const uint32_t kb = get_common_arg_val<uint32_t>(7);
    const bool w_nk = get_common_arg_val<uint32_t>(8) != 0;
    const uint32_t xc = get_common_arg_val<uint32_t>(9);
    [[maybe_unused]] const uint32_t bias_addr = get_common_arg_val<uint32_t>(11);

    const uint32_t c0 = get_arg_val<uint32_t>(0);
    const uint32_t nc = get_arg_val<uint32_t>(1);
    const uint32_t mine = get_arg_val<uint32_t>(2);
    const uint32_t sx = get_arg_val<uint32_t>(3), sy = get_arg_val<uint32_t>(4);
    const uint32_t ex = get_arg_val<uint32_t>(5), ey = get_arg_val<uint32_t>(6);
    const uint32_t receivers = get_arg_val<uint32_t>(7);
    const uint32_t me = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_x = 0, cb_w = 1 + side, cb_bias = 3, cb_out = 16;
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto
        w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto
        out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t x_tile = get_tile_size(cb_x), w_tile = get_tile_size(cb_w);
    const uint32_t out_tile = get_tile_size(cb_out);
    [[maybe_unused]] const auto x = TensorAccessor(x_args, x_addr, x_tile);
    const auto w = TensorAccessor(w_args, w_addr, w_tile);
    [[maybe_unused]] const auto out = TensorAccessor(out_args, out_addr, out_tile);
#ifdef HAS_BIAS
    constexpr auto
        bias_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto bias = TensorAccessor(bias_args, bias_addr, get_tile_size(cb_bias));
#endif
    const uint32_t groups = (nc + sw - 1) / sw, nkb = (kt + kb - 1) / kb;

#ifdef HAS_X
    // Semaphore 0: a sender's count of ready receivers; 1 + s: chunks sender s sent.
    const uint32_t ready = get_semaphore(0);
    const auto valid = [](uint32_t s) {
        return get_semaphore(1 + s);
    };
    const uint32_t chunks = (kt + xc - 1) / xc;
    const auto chunk_tiles = [&](uint32_t j) {
        return (kt - j * xc < xc ? kt - j * xc : xc) * mt;
    };
    cb_reserve_back(cb_x, mt * kt);
    const uint32_t x_base = get_write_ptr(cb_x);
    uint32_t pushed = 0;  // x chunks pushed to compute, in order
    // Pushes the x chunks that have arrived (all with `wait`); chunk j is sender
    // (j % S)'s (j / S + 1)-th.
    const auto take_x = [&](bool wait) {
        while (pushed < chunks) {
            const uint32_t s = pushed % senders, n = pushed / senders + 1;
            invalidate_l1_cache();
            if (*word(valid(s)) < n) {
                if (!wait) {
                    return;
                }
                noc_semaphore_wait_min(word(valid(s)), n);
            }
            cb_push_back(cb_x, chunk_tiles(pushed));
            ++pushed;
        }
    };
    // Each core of the rectangle tells every other sender its x buffer is reserved.
    for (uint32_t s = 0; s < senders && receivers > 0; ++s) {
        if (s != me) {
            const uint32_t px = get_arg_val<uint32_t>(kFixed + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(kFixed + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, ready), 1);
        }
    }
    if (me != kNone) {
        // Chunks me, me + S, ...: read (the next issued before one is multicast),
        // multicast, counted in valid(me) on every core of the rectangle.
        const auto read_chunk = [&](uint32_t j) {
            const uint32_t k0 = j * xc, kc = chunk_tiles(j) / mt;
            for (uint32_t k = 0; k < kc; ++k) {
                for (uint32_t r = 0; r < mt; ++r) {
                    noc_async_read_page(
                        r * kt + k0 + k, x, x_base + ((k0 + k) * mt + r) * x_tile
                    );
                }
            }
        };
        if (receivers > 0) {
            noc_semaphore_wait(word(ready), receivers);
            noc_semaphore_set(word(ready), 0);
        }
        if (me < chunks) {
            read_chunk(me);
        }
        uint32_t sent = 0;
        for (uint32_t j = me; j < chunks; j += senders) {
            noc_async_read_barrier();
            if (j + senders < chunks) {
                read_chunk(j + senders);
            }
            *word(valid(me)) = ++sent;
            if (receivers > 0) {
                const uint32_t at = x_base + j * xc * mt * x_tile;
                noc_async_write_multicast(
                    at,
                    get_noc_multicast_addr(sx, sy, ex, ey, at),
                    chunk_tiles(j) * x_tile,
                    receivers
                );
                noc_async_writes_flushed();
                noc_semaphore_set_multicast(
                    valid(me),
                    get_noc_multicast_addr(sx, sy, ex, ey, valid(me)),
                    receivers
                );
            }
            take_x(false);
        }
    }
#else
    const auto take_x = [](bool) {};
#endif

    // This mover's W blocks: per group of sw columns, its K blocks in order; x chunks
    // are passed on between them (compute needs both).
    uint32_t b = 0;
    for (uint32_t g = 0; g < groups && mine != 2; ++g) {
        const uint32_t col0 = c0 + g * sw;
        const uint32_t gw = nc - g * sw < sw ? nc - g * sw : sw;
#ifdef HAS_BIAS
        // The group's bias tiles (row 0 is broadcast to every row by compute).
        cb_reserve_back(cb_bias, sw);
        for (uint32_t c = 0; c < gw; ++c) {
            noc_async_read_page(col0 + c, bias, get_write_ptr(cb_bias) + c * out_tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_bias, sw);
#endif
        for (uint32_t j = 0; j < nkb; ++j, ++b) {
            if (mine == 0 && ktt_gemv::on_dm1(b) != (side == 1)) {
                continue;
            }
            const uint32_t k0 = j * kb, kc = kt - k0 < kb ? kt - k0 : kb;
            while (!cb_pages_reservable_at_back(cb_w, sw * kb)) {
                take_x(false);
            }
            cb_reserve_back(cb_w, sw * kb);
            const uint32_t base = get_write_ptr(cb_w);
            for (uint32_t c = 0; c < gw; ++c) {
                for (uint32_t k = 0; k < kc; ++k) {
                    const uint32_t page = w_nk ? (col0 + c) * kt + k0 + k
                                               : (k0 + k) * nt + col0 + c;
                    noc_async_read_page(page, w, base + (k * sw + c) * w_tile);
                }
            }
            take_x(false);
            noc_async_read_barrier();
            cb_push_back(cb_w, sw * kb);
            take_x(false);
        }
    }
    take_x(true);

#ifdef HAS_OUT
    // Output: per group, mt x sw tiles (row-major); columns past the core's end
    // dropped.
    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t col0 = c0 + g * sw;
        const uint32_t gw = nc - g * sw < sw ? nc - g * sw : sw;
        cb_wait_front(cb_out, mt * sw);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t r = 0; r < mt; ++r) {
            for (uint32_t c = 0; c < gw; ++c) {
                noc_async_write_page(
                    r * nt + col0 + c, out, src + (r * sw + c) * out_tile
                );
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, mt * sw);
    }
#endif
    (void)sx, (void)sy, (void)ex, (void)ey, (void)receivers, (void)me, (void)xc,
        (void)cb_bias;
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
