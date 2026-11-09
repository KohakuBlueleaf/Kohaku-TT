// Nhwc conv reader (RISCV_1, NoC0 on every compute core). W' transport runs one block ahead
// of the builds. Step g (Cin block cb = (g % Ct + rot) % Ct):
// - Column top: issue the DRAM reads of W' block g + 2 (transaction id 1 + parity, 4-slot
//   ring), multicast block g + 1 down the column, set its flag to g + 2. Other cores: count
//   in at the top for block g + 1 once its slot is free, then wait for flag >= g + 1.
// - Push W' block g. Reserve A0 block g, `a0 go` = g + 1; build even A tiles [0, k0)
//   (transaction id 3) from planes slot g; the writer builds [k0, n_even) into the same
//   block and sets `a0 done`; push; `done` = g + 1.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTile = 2048, kZero = 0xffffffffu, kBuildTrid = 3, kSlots = 4;

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of W', table. Common runtime args: W', table addresses,
// Ct, Nt, taps, bn, iters, semaphore ids (W count, W flag, planes ready, planes done, a0 go, a0
// done), table page bytes. Per core: n0, table page, even tile count, top (1 / 0), multicast
// rectangle (start x, y, end x, y) and receiver count, top (x, y), rot, input tiles, k0.
void kernel_main() {
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto t_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ct = get_common_arg_val<uint32_t>(2);
    const uint32_t nt = get_common_arg_val<uint32_t>(3);
    const uint32_t taps = get_common_arg_val<uint32_t>(4);
    const uint32_t bn = get_common_arg_val<uint32_t>(5);
    const uint32_t iters = get_common_arg_val<uint32_t>(6);
    const uint32_t w_count = get_semaphore(get_common_arg_val<uint32_t>(7));
    const uint32_t w_flag = get_semaphore(get_common_arg_val<uint32_t>(8));
    const uint32_t ready = get_semaphore(get_common_arg_val<uint32_t>(9));
    const uint32_t done = get_semaphore(get_common_arg_val<uint32_t>(10));
    const uint32_t a0_go = get_semaphore(get_common_arg_val<uint32_t>(11));
    const uint32_t a0_done = get_semaphore(get_common_arg_val<uint32_t>(12));
    const uint32_t page_bytes = get_common_arg_val<uint32_t>(13);

    const uint32_t n0 = get_arg_val<uint32_t>(0);
    const uint32_t tab_page = get_arg_val<uint32_t>(1);
    const uint32_t n_even = get_arg_val<uint32_t>(2);
    const bool top = get_arg_val<uint32_t>(3) != 0;
    const uint32_t mc_sx = get_arg_val<uint32_t>(4), mc_sy = get_arg_val<uint32_t>(5);
    const uint32_t mc_ex = get_arg_val<uint32_t>(6), mc_ey = get_arg_val<uint32_t>(7);
    const uint32_t mc_num = get_arg_val<uint32_t>(8);
    const uint32_t s_x = get_arg_val<uint32_t>(9), s_y = get_arg_val<uint32_t>(10);
    const uint32_t rot = get_arg_val<uint32_t>(11);
    const uint32_t t_in = get_arg_val<uint32_t>(12);
    const uint32_t k0 = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_w = 1, cb_a0 = 2, cb_planes = 5, cb_tab = 8;
    const auto w = TensorAccessor(w_args, w_addr, kTile);
    const auto tab = TensorAccessor(t_args, t_addr, page_bytes);
    const uint32_t w_block = taps * bn;
    const uint32_t w_bytes = w_block * kTile;
    const uint32_t total = iters * ct;

    cb_reserve_back(cb_tab, 1);
    const uint32_t table = get_write_ptr(cb_tab);
    noc_async_read_set_trid(kBuildTrid);
    noc_async_read_page(tab_page, tab, table);
    noc_async_read_barrier_with_trid(kBuildTrid);
    const uint32_t w_base = get_write_ptr(cb_w);
    const uint32_t planes = get_write_ptr(cb_planes);
    const uint32_t slot_bytes = t_in * kTile;

    // Top: W' block h from DRAM into ring slot h % 4, transaction id 1 + (h & 1).
    auto w_read = [&](uint32_t h) {
        const uint32_t cb = (h % ct + rot) % ct;
        const uint32_t dst = w_base + (h % kSlots) * w_bytes;
        noc_async_read_set_trid(1 + (h & 1));
        for (uint32_t t = 0; t < taps; ++t) {
            const uint32_t page = (cb * taps + t) * nt + n0;
            for (uint32_t j = 0; j < bn; ++j) {
                noc_async_read_page(page + j, w, dst + (t * bn + j) * kTile);
            }
        }
    };
    // Top: block h read; multicast it once every receiver counted in for it; flag = h + 1.
    auto w_send = [&](uint32_t h) {
        noc_async_read_barrier_with_trid(1 + (h & 1));
        if (mc_num > 0) {
            const uint32_t slot = w_base + (h % kSlots) * w_bytes;
            noc_semaphore_wait_min(word(w_count), mc_num * (h + 1));
            noc_async_write_multicast(slot, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, slot), w_bytes, mc_num);
            noc_async_writes_flushed();
            *word(w_flag) = h + 1;
            noc_semaphore_set_multicast(w_flag, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, w_flag), mc_num);
        }
    };
    // Receiver: the next block's slot is free; count in at the top.
    auto w_ready = [&]() {
        noc_semaphore_inc(get_noc_addr(s_x, s_y, w_count), 1);
    };

    cb_reserve_back(cb_w, w_block);
    if (top) {
        w_read(0);
        if (total > 1) {
            cb_reserve_back(cb_w, 2 * w_block);
            w_read(1);
        }
        w_send(0);
    } else {
        w_ready();
    }
    for (uint32_t g = 0; g < total; ++g) {
        if (top) {
            if (g + 2 < total) {
                // Frees slot g + 2 (compute is done with g - 2).
                cb_reserve_back(cb_w, 3 * w_block);
                w_read(g + 2);
            }
            if (g + 1 < total) {
                w_send(g + 1);
            }
        } else {
            if (g + 1 < total) {
                cb_reserve_back(cb_w, 2 * w_block);
                w_ready();
            }
            noc_semaphore_wait_min(word(w_flag), g + 1);
        }
        cb_push_back(cb_w, w_block);

        // The writer has planes slot g.
        noc_semaphore_wait_min(word(ready), g + 1);
        if (n_even > 0) {
            cb_reserve_back(cb_a0, n_even);
            if (k0 < n_even) {
                *word(a0_go) = g + 1;
            }
            const uint32_t ab = get_write_ptr(cb_a0);
            const uint32_t slot = planes + (g % kSlots) * slot_bytes;
            noc_async_read_set_trid(kBuildTrid);
            noc_async_read_set_state(get_noc_addr(0));
            uint32_t p = 0;
            for (uint32_t k = 0; k < k0; ++k) {
                const uint32_t runs = *word(table + 4 * p);
                const uint32_t dst_tile = ab + k * kTile;
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
            if (k0 < n_even) {
                noc_semaphore_wait_min(word(a0_done), g + 1);
            }
            cb_push_back(cb_a0, n_even);
        }
        *word(done) = g + 1;
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
