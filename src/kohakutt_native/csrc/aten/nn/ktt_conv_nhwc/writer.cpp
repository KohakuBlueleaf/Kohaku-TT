// Nhwc conv writer (RISCV_0, NoC1 on every compute core). Steps as the reader's:
// s = (it npb + pb) Ct + k, Cin block cb = (k + rot) % Ct. Planes transport runs ahead
// of the builds (4-slot ring). Step s:
// - Lead (no feeder): wait for step s's input reads, issue step s + 2's (transaction id
// 1 +
//   parity). Receivers: wait for flag >= s + 1, then count in at the feeder for step s
//   + 1 once its slot is free. A core's count therefore never exceeds the steps sent +
//   1, and the feeder's test (count >= receivers x (h + 1)) means every receiver
//   counted in for step h.
// - `ready` = s + 1 for the reader; the odd A tiles of step s (transaction id 3) with
// pixel
//   block pb's copy table (2-slot ring, transaction id 4, read one pixel block ahead);
//   with k0 < n_even, after `a0 go` >= s + 1, even tiles [k0, n_even) into A0 block s,
//   then `a0 done`.
// After the last Cin step of a pixel block: its bm_b x (nnb bn_b) output tiles (page m
// Nt + n), in compute's (nb, m, j) order. With `side`: compute's transposed input tiles
// of step s - 2 (CB 17, bm_b per step) to page cb Mt + m of X^T before step s's builds,
// the last two steps' after the loop.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTile = 2048, kFace = 512, kRow = 32, kZero = 0xffffffffu,
                       kBuildTrid = 3, kTableTrid = 4, kSlots = 4;

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of X, table, OUT, X^T (OUT's again without a
// side output). Common runtime args: X, table, OUT addresses, Ct, Nt, bn_b, nnb, npb,
// iters, semaphore ids (planes count, planes flag, planes ready, planes done, a0 go, a0
// done), table page bytes, planes slot bytes, subblock width, Mt, bm_b, lowest /
// highest tap offset in pixels (int32), X^T address. Per core: first output tile m0,
// n0, first table page, odd tiles per step, even tiles per step, k0, lead (1 / 0),
// feeder (x, y), rot, side (1 / 0).
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        t_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto
        o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto
        s_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t ct = get_common_arg_val<uint32_t>(3);
    const uint32_t nt = get_common_arg_val<uint32_t>(4);
    const uint32_t bn_b = get_common_arg_val<uint32_t>(5);
    const uint32_t nnb = get_common_arg_val<uint32_t>(6);
    const uint32_t npb = get_common_arg_val<uint32_t>(7);
    const uint32_t iters = get_common_arg_val<uint32_t>(8);
    const uint32_t p_count = get_semaphore(get_common_arg_val<uint32_t>(9));
    const uint32_t p_flag = get_semaphore(get_common_arg_val<uint32_t>(10));
    const uint32_t ready = get_semaphore(get_common_arg_val<uint32_t>(11));
    const uint32_t done = get_semaphore(get_common_arg_val<uint32_t>(12));
    const uint32_t a0_go = get_semaphore(get_common_arg_val<uint32_t>(13));
    const uint32_t a0_done = get_semaphore(get_common_arg_val<uint32_t>(14));
    const uint32_t page_bytes = get_common_arg_val<uint32_t>(15);
    const uint32_t slot_bytes = get_common_arg_val<uint32_t>(16);
    const uint32_t sub = get_common_arg_val<uint32_t>(17);
    const int32_t mt = static_cast<int32_t>(get_common_arg_val<uint32_t>(18));
    const int32_t bm_b = static_cast<int32_t>(get_common_arg_val<uint32_t>(19));
    const int32_t min_off = static_cast<int32_t>(get_common_arg_val<uint32_t>(20));
    const int32_t max_off = static_cast<int32_t>(get_common_arg_val<uint32_t>(21));
    const uint32_t s_addr = get_common_arg_val<uint32_t>(22);

    const int32_t m0 = static_cast<int32_t>(get_arg_val<uint32_t>(0));
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const uint32_t tab_page0 = get_arg_val<uint32_t>(2);
    const uint32_t n_odd = get_arg_val<uint32_t>(3);
    const uint32_t n_even = get_arg_val<uint32_t>(4);
    const uint32_t k0 = get_arg_val<uint32_t>(5);
    const bool lead = get_arg_val<uint32_t>(6) != 0;
    const uint32_t l_x = get_arg_val<uint32_t>(7), l_y = get_arg_val<uint32_t>(8);
    const uint32_t rot = get_arg_val<uint32_t>(9);
    const bool side = get_arg_val<uint32_t>(10) != 0;

    constexpr uint32_t cb_a0 = 2, cb_a1 = 3, cb_planes = 5, cb_tab = 9, cb_out = 16,
                       cb_side = 17;
    const auto x = TensorAccessor(x_args, x_addr, kTile);
    const auto tab = TensorAccessor(t_args, t_addr, page_bytes);
    const auto out = TensorAccessor(o_args, o_addr, kTile);
    const auto xt = TensorAccessor(s_args, s_addr, kTile);
    const uint32_t steps = iters * npb * ct;
    const uint32_t planes = get_write_ptr(cb_planes);
    const uint32_t tables = get_write_ptr(cb_tab);
    const uint32_t a0_base = get_write_ptr(cb_a0);

    // Input tiles [lo, lo + n) of pixel block pb (a block past Mt: one tile).
    auto range = [&](uint32_t pb, uint32_t& lo, uint32_t& n) {
        const int32_t mp = m0 + static_cast<int32_t>(pb) * bm_b;
        if (mp >= mt) {
            lo = 0, n = 1;
            return;
        }
        int32_t g_lo = 32 * mp + min_off, g_hi = 32 * (mp + bm_b) - 1 + max_off;
        g_lo = g_lo < 0 ? 0 : g_lo;
        g_hi = g_hi > 32 * mt - 1 ? 32 * mt - 1 : g_hi;
        lo = static_cast<uint32_t>(g_lo / 32);
        n = static_cast<uint32_t>(g_hi / 32) + 1 - lo;
    };
    // Lead: step h's input tiles from DRAM into planes slot h % 4 (plane L channels
    // 0-15, plane R 16-31, pixel-contiguous), transaction id 1 + (h & 1).
    auto planes_read = [&](uint32_t h) {
        const uint32_t cb = (h % ct + rot) % ct;
        uint32_t lo, n;
        range((h / ct) % npb, lo, n);
        const uint32_t slot = planes + (h % kSlots) * slot_bytes;
        noc_async_read_set_trid(1 + (h & 1));
        for (uint32_t t = 0; t < n; ++t) {
            const uint32_t page = (lo + t) * ct + cb;
            for (uint32_t f = 0; f < 4; ++f) {
                const uint32_t plane = slot + (f & 1) * (slot_bytes / 2);
                noc_async_read(
                    x.get_noc_addr(page, f * kFace),
                    plane + (32 * t + 16 * (f >> 1)) * kRow,
                    kFace
                );
            }
        }
    };
    // Receiver: the next step's slot is free; count in at the feeder.
    auto planes_ready = [&]() {
        noc_semaphore_inc(get_noc_addr(l_x, l_y, p_count), 1);
    };
    // Copy table of pixel block g (over all iterations) into table slot g & 1.
    auto table_read = [&](uint32_t g) {
        noc_async_read_set_trid(kTableTrid);
        noc_async_read_page(
            tab_page0 + 2 * (g % npb) + 1, tab, tables + (g & 1) * page_bytes
        );
    };
    // Step h's bm_b transposed input tiles from CB 17 to X^T pages cb Mt + m (none past
    // Mt).
    auto side_write = [&](uint32_t h) {
        const uint32_t cb = (h % ct + rot) % ct;
        const int32_t mp = m0 + static_cast<int32_t>((h / ct) % npb) * bm_b;
        for (int32_t m = mp; m < mp + bm_b; ++m) {
            cb_wait_front(cb_side, 1);
            if (m < mt) {
                noc_async_write_page(
                    cb * static_cast<uint32_t>(mt) + m, xt, get_read_ptr(cb_side)
                );
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_side, 1);
        }
    };

    table_read(0);
    noc_async_read_barrier_with_trid(kTableTrid);
    if (lead) {
        planes_read(0);
        if (steps > 1) {
            planes_read(1);
        }
    } else {
        planes_ready();
    }
    for (uint32_t s = 0; s < steps; ++s) {
        const uint32_t k = s % ct;
        const uint32_t pbg = s / ct;
        if (lead) {
            // Step s's reads (issued two steps ahead) before step s + 2's, which share
            // its id.
            noc_async_read_barrier_with_trid(1 + (s & 1));
            if (s + 2 < steps) {
                // Slot s + 2 last held step s - 2: wait for the reader's builds of it.
                if (s >= 2) {
                    noc_semaphore_wait_min(word(done), s - 1);
                }
                planes_read(s + 2);
            }
        } else {
            noc_semaphore_wait_min(word(p_flag), s + 1);
            if (s + 1 < steps) {
                // Counted in only after step s arrived (a core's count never exceeds
                // the steps sent + 1). Slot s + 1 last held step s - 3.
                if (s >= 3) {
                    noc_semaphore_wait_min(word(done), s - 2);
                }
                planes_ready();
            }
        }
        *word(ready) = s + 1;
        if (k == 0) {
            // This block's table was read one block ahead; read the next block's now.
            noc_async_read_barrier_with_trid(kTableTrid);
            if (pbg + 1 < iters * npb) {
                table_read(pbg + 1);
            }
        }
        const uint32_t table = tables + (pbg & 1) * page_bytes;
        const uint32_t slot = planes + (s % kSlots) * slot_bytes;

        // Copies tiles [i_lo, i_hi) of the build list (table entries from word p on)
        // into dst + i kTile.
        uint32_t p = 0;
        auto build = [&](uint32_t dst, uint32_t i_lo, uint32_t i_hi) {
            noc_async_read_set_trid(kBuildTrid);
            noc_async_read_set_state(get_noc_addr(0));
            for (uint32_t i = i_lo; i < i_hi; ++i) {
                const uint32_t runs = *word(table + 4 * p);
                const uint32_t dst_tile = dst + i * kTile;
                for (uint32_t r = 0; r < runs; ++r) {
                    const uint32_t e = table + 4 * (p + 1 + 3 * r);
                    const uint32_t src = *word(e + 4);
                    noc_async_read_with_state(
                        src == kZero ? MEM_ZEROS_BASE : slot + src,
                        dst_tile + *word(e),
                        *word(e + 8)
                    );
                }
                p += 1 + 3 * runs;
            }
            noc_async_read_barrier_with_trid(kBuildTrid);
        };
        if (n_odd > 0) {
            cb_reserve_back(cb_a1, n_odd);
            if (side && s >= 2) {
                // The A1 ring has room for step s: compute finished step s - 2.
                side_write(s - 2);
            }
            build(get_write_ptr(cb_a1), 0, n_odd);
            cb_push_back(cb_a1, n_odd);
        }
        if (k0 < n_even) {
            // A0 block s sits at ring position s % 2 (capacity 2 n_even, n_even per
            // push).
            noc_semaphore_wait_min(word(a0_go), s + 1);
            build(a0_base + (s & 1) * n_even * kTile, k0, n_even);
            *word(a0_done) = s + 1;
        }

        if (k == ct - 1) {
            const int32_t mp = m0 + static_cast<int32_t>(pbg % npb) * bm_b;
            for (uint32_t nb = 0; nb < nnb; ++nb) {
                for (int32_t m = mp; m < mp + bm_b; ++m) {
                    for (uint32_t j0 = 0; j0 < bn_b; j0 += sub) {
                        for (uint32_t j = j0; j < j0 + sub; ++j) {
                            cb_wait_front(cb_out, 1);
                            if (m < mt) {
                                noc_async_write_page(
                                    static_cast<uint32_t>(m) * nt + n0 + nb * bn_b + j,
                                    out,
                                    get_read_ptr(cb_out)
                                );
                                noc_async_writes_flushed();
                            }
                            cb_pop_front(cb_out, 1);
                        }
                    }
                }
            }
        }
    }
    if (side) {
        for (uint32_t h = steps > 2 ? steps - 2 : 0; h < steps; ++h) {
            side_write(h);
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
