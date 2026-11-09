// Nhwc conv writer (RISCV_0, NoC1 on every compute core). Planes transport runs one block ahead
// of the builds. Step g (Cin block cb = (g % Ct + rot) % Ct):
// - Lead (no feeder): issue block g + 2's input reads as planes (transaction id 1 + parity,
//   4-slot ring); multicast block g + 1 to its receivers, if any. Receivers: count in at the
//   feeder for block g + 1 once its slot is free, then wait for flag >= g + 1.
// - `ready` = g + 1 for the reader; the odd A tiles of block g (transaction id 3); with k0 <
//   n_even, after `a0 go` >= g + 1, even tiles [k0, n_even) into A0 block g, then `a0 done`.
// After the last block of an iteration: the core's output tiles (page m Nt + n).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTile = 2048, kFace = 512, kRow = 32, kZero = 0xffffffffu, kBuildTrid = 3, kSlots = 4;

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of X, table, OUT. Common runtime args: X, table, OUT
// addresses, Ct, Nt, bn, iters, semaphore ids (planes count, planes flag, planes ready, planes
// done, a0 go, a0 done), table page bytes, subblock width, Mt. Per core: m0, m1, n0, first input
// tile, input tiles, table page, odd tile count, lead (1 / 0), multicast rectangle (start x, y,
// end x, y, given for NoC1) and receiver count, lead (x, y), rot, k0, even tile count.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto t_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t ct = get_common_arg_val<uint32_t>(3);
    const uint32_t nt = get_common_arg_val<uint32_t>(4);
    const uint32_t bn = get_common_arg_val<uint32_t>(5);
    const uint32_t iters = get_common_arg_val<uint32_t>(6);
    const uint32_t p_count = get_semaphore(get_common_arg_val<uint32_t>(7));
    const uint32_t p_flag = get_semaphore(get_common_arg_val<uint32_t>(8));
    const uint32_t ready = get_semaphore(get_common_arg_val<uint32_t>(9));
    const uint32_t done = get_semaphore(get_common_arg_val<uint32_t>(10));
    const uint32_t a0_go = get_semaphore(get_common_arg_val<uint32_t>(11));
    const uint32_t a0_done = get_semaphore(get_common_arg_val<uint32_t>(12));
    const uint32_t page_bytes = get_common_arg_val<uint32_t>(13);
    const uint32_t sub = get_common_arg_val<uint32_t>(14);
    const uint32_t mt = get_common_arg_val<uint32_t>(15);

    const uint32_t m0 = get_arg_val<uint32_t>(0), m1 = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);
    const uint32_t tile_lo = get_arg_val<uint32_t>(3);
    const uint32_t t_in = get_arg_val<uint32_t>(4);
    const uint32_t tab_page = get_arg_val<uint32_t>(5);
    const uint32_t n_odd = get_arg_val<uint32_t>(6);
    const bool lead = get_arg_val<uint32_t>(7) != 0;
    const uint32_t mc_sx = get_arg_val<uint32_t>(8), mc_sy = get_arg_val<uint32_t>(9);
    const uint32_t mc_ex = get_arg_val<uint32_t>(10), mc_ey = get_arg_val<uint32_t>(11);
    const uint32_t mc_num = get_arg_val<uint32_t>(12);
    const uint32_t l_x = get_arg_val<uint32_t>(13), l_y = get_arg_val<uint32_t>(14);
    const uint32_t rot = get_arg_val<uint32_t>(15);
    const uint32_t k0 = get_arg_val<uint32_t>(16);
    const uint32_t n_even = get_arg_val<uint32_t>(17);

    constexpr uint32_t cb_a0 = 2, cb_a1 = 3, cb_planes = 5, cb_tab = 9, cb_out = 16;
    const uint32_t a0_base = get_write_ptr(cb_a0);
    const auto x = TensorAccessor(x_args, x_addr, kTile);
    const auto tab = TensorAccessor(t_args, t_addr, page_bytes);
    const auto out = TensorAccessor(o_args, o_addr, kTile);
    const uint32_t total = iters * ct;

    cb_reserve_back(cb_tab, 1);
    const uint32_t table = get_write_ptr(cb_tab);
    noc_async_read_set_trid(kBuildTrid);
    noc_async_read_page(tab_page, tab, table);
    noc_async_read_barrier_with_trid(kBuildTrid);
    const uint32_t planes = get_write_ptr(cb_planes);
    const uint32_t slot_bytes = t_in * kTile;

    // Lead: block h's input tiles from DRAM into planes slot h % 4 (plane L channels 0-15, plane R
    // 16-31, pixel-contiguous), transaction id 1 + (h & 1).
    auto planes_read = [&](uint32_t h) {
        const uint32_t cb = (h % ct + rot) % ct;
        const uint32_t slot = planes + (h % kSlots) * slot_bytes;
        noc_async_read_set_trid(1 + (h & 1));
        for (uint32_t t = 0; t < t_in; ++t) {
            const uint32_t page = (tile_lo + t) * ct + cb;
            for (uint32_t f = 0; f < 4; ++f) {
                const uint32_t plane = slot + (f & 1) * (slot_bytes / 2);
                noc_async_read(x.get_noc_addr(page, f * kFace), plane + (32 * t + 16 * (f >> 1)) * kRow, kFace);
            }
        }
    };
    // Lead: block h read; multicast its slot once every receiver counted in for it; flag = h + 1.
    auto planes_send = [&](uint32_t h) {
        noc_async_read_barrier_with_trid(1 + (h & 1));
        if (mc_num > 0) {
            const uint32_t slot = planes + (h % kSlots) * slot_bytes;
            noc_semaphore_wait_min(word(p_count), mc_num * (h + 1));
            noc_async_write_multicast(slot, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, slot), slot_bytes, mc_num);
            noc_async_writes_flushed();
            *word(p_flag) = h + 1;
            noc_semaphore_set_multicast(p_flag, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, p_flag), mc_num);
        }
    };
    // Receiver: the next block's slot is free; count in at the lead.
    auto planes_ready = [&]() {
        noc_semaphore_inc(get_noc_addr(l_x, l_y, p_count), 1);
    };

    if (lead) {
        planes_read(0);
        if (total > 1) {
            planes_read(1);
        }
        planes_send(0);
    } else {
        planes_ready();
    }
    for (uint32_t g = 0; g < total; ++g) {
        if (lead) {
            if (g + 2 < total) {
                // Slot g + 2 last held block g - 2: wait for the reader's builds of it.
                if (g >= 2) {
                    noc_semaphore_wait_min(word(done), g - 1);
                }
                planes_read(g + 2);
            }
            if (g + 1 < total) {
                planes_send(g + 1);
            }
        } else {
            if (g + 1 < total) {
                // Slot g + 1 last held block g - 3.
                if (g >= 3) {
                    noc_semaphore_wait_min(word(done), g - 2);
                }
                planes_ready();
            }
            noc_semaphore_wait_min(word(p_flag), g + 1);
        }
        *word(ready) = g + 1;
        const uint32_t slot = planes + (g % kSlots) * slot_bytes;

        // Copies tiles [k_lo, k_hi) of a build list (table entries from word p on) into dst + k kTile.
        uint32_t p = 0;
        auto build = [&](uint32_t dst, uint32_t k_lo, uint32_t k_hi) {
            noc_async_read_set_trid(kBuildTrid);
            noc_async_read_set_state(get_noc_addr(0));
            for (uint32_t k = k_lo; k < k_hi; ++k) {
                const uint32_t runs = *word(table + 4 * p);
                const uint32_t dst_tile = dst + k * kTile;
                for (uint32_t r = 0; r < runs; ++r) {
                    const uint32_t e = table + 4 * (p + 1 + 3 * r);
                    const uint32_t src = *word(e + 4);
                    noc_async_read_with_state(
                        src == kZero ? MEM_ZEROS_BASE : slot + src, dst_tile + *word(e), *word(e + 8)
                    );
                }
                p += 1 + 3 * runs;
            }
            noc_async_read_barrier_with_trid(kBuildTrid);
        };
        if (n_odd > 0) {
            cb_reserve_back(cb_a1, n_odd);
            build(get_write_ptr(cb_a1), 0, n_odd);
            cb_push_back(cb_a1, n_odd);
        }
        if (k0 < n_even) {
            // A0 block g sits at ring position g % 2 (capacity 2 n_even, n_even per push).
            noc_semaphore_wait_min(word(a0_go), g + 1);
            build(a0_base + (g & 1) * n_even * kTile, k0, n_even);
            *word(a0_done) = g + 1;
        }

        if (g % ct == ct - 1) {
            for (uint32_t m = m0; m < m1; ++m) {
                for (uint32_t j0 = 0; j0 < bn; j0 += sub) {
                    for (uint32_t j = j0; j < j0 + sub; ++j) {
                        cb_wait_front(cb_out, 1);
                        if (m < mt) {
                            noc_async_write_page(m * nt + n0 + j, out, get_read_ptr(cb_out));
                            noc_async_writes_flushed();
                        }
                        cb_pop_front(cb_out, 1);
                    }
                }
            }
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
