// Nhwc conv weight-gradient reader (RISCV_1, NoC0 on every compute core). The core
// walks bo Cout tiles x nq Cin blocks of bc tiles, each over npb pixel blocks of kb
// tiles: step s = ((it bo + i) nq + q) npb + pb. X^T transport runs one step ahead:
// - Column top: issue the DRAM reads of X^T block s + 2 (bc x kb tiles, page cb Mt + m,
// into
//   ring slot (s + 2) % 4 at (j kb + m); transaction id 1 + parity), multicast block s
//   + 1 down the column, set its flag to s + 2. Other cores: wait for flag >= s + 1.
// - Push X^T block s; other cores then count in at the top for block s + 1 once its
// slot is
//   free. Reserve G block s (even tiles), `a0 go` = s + 1; build even tiles [0, k0)
//   (transaction id 3) from planes slot s with pixel block pb's copy table (transaction
//   id 4, read one step ahead when npb > 1); the writer builds [k0, n_even) into the
//   same block and sets `a0 done`; push; `done` = s + 1.
// Count-in invariant: a core counts in for block s + 1 only after block s arrived, so
// its count never exceeds the blocks sent + 1, and the top's test (count >= receivers x
// (h + 1)) means every receiver counted in for block h.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTile = 2048, kZero = 0xffffffffu, kBuildTrid = 3,
                       kTableTrid = 4, kSlots = 4;

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of X^T, table. Common runtime args: X^T, table
// addresses, Ct, Mt, bc, kb, nq, npb, bo, iters, semaphore ids (X count, X flag, planes
// ready, planes done, a0 go, a0 done), table page bytes, planes slot bytes. Per core:
// first Cin tile cb0, first pixel tile, pixel tiles, first table page, even tiles per
// step, top (1 / 0), multicast rectangle (start x, y, end x, y) and receiver count, top
// (x, y), k0.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto
        t_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ct = get_common_arg_val<uint32_t>(2);
    const uint32_t mt = get_common_arg_val<uint32_t>(3);
    const uint32_t bc = get_common_arg_val<uint32_t>(4);
    const uint32_t kb = get_common_arg_val<uint32_t>(5);
    const uint32_t nq = get_common_arg_val<uint32_t>(6);
    const uint32_t npb = get_common_arg_val<uint32_t>(7);
    const uint32_t bo = get_common_arg_val<uint32_t>(8);
    const uint32_t iters = get_common_arg_val<uint32_t>(9);
    const uint32_t x_count = get_semaphore(get_common_arg_val<uint32_t>(10));
    const uint32_t x_flag = get_semaphore(get_common_arg_val<uint32_t>(11));
    const uint32_t ready = get_semaphore(get_common_arg_val<uint32_t>(12));
    const uint32_t done = get_semaphore(get_common_arg_val<uint32_t>(13));
    const uint32_t a0_go = get_semaphore(get_common_arg_val<uint32_t>(14));
    const uint32_t a0_done = get_semaphore(get_common_arg_val<uint32_t>(15));
    const uint32_t page_bytes = get_common_arg_val<uint32_t>(16);
    const uint32_t slot_bytes = get_common_arg_val<uint32_t>(17);

    const uint32_t cb0 = get_arg_val<uint32_t>(0);
    const uint32_t m_lo = get_arg_val<uint32_t>(1);
    const uint32_t m_cnt = get_arg_val<uint32_t>(2);
    const uint32_t tab_page0 = get_arg_val<uint32_t>(3);
    const uint32_t n_even = get_arg_val<uint32_t>(4);
    const bool top = get_arg_val<uint32_t>(5) != 0;
    const uint32_t mc_sx = get_arg_val<uint32_t>(6), mc_sy = get_arg_val<uint32_t>(7);
    const uint32_t mc_ex = get_arg_val<uint32_t>(8), mc_ey = get_arg_val<uint32_t>(9);
    const uint32_t mc_num = get_arg_val<uint32_t>(10);
    const uint32_t s_x = get_arg_val<uint32_t>(11), s_y = get_arg_val<uint32_t>(12);
    const uint32_t k0 = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_x = 1, cb_a0 = 2, cb_planes = 5, cb_tab = 8;
    const auto xt = TensorAccessor(x_args, x_addr, kTile);
    const auto tab = TensorAccessor(t_args, t_addr, page_bytes);
    const uint32_t x_block = bc * kb;
    const uint32_t x_bytes = x_block * kTile;
    const uint32_t steps = iters * bo * nq * npb;
    const uint32_t x_base = get_write_ptr(cb_x);
    const uint32_t planes = get_write_ptr(cb_planes);
    const uint32_t tables = get_write_ptr(cb_tab);

    // Copy table of step h's pixel block (page tab_page0 + 2 (h % npb)) into table slot
    // h & 1 (slot 0 with one pixel block).
    auto table_read = [&](uint32_t h) {
        noc_async_read_set_trid(kTableTrid);
        noc_async_read_page(
            tab_page0 + 2 * (h % npb), tab, tables + (npb > 1 ? h & 1 : 0) * page_bytes
        );
    };
    // Top: X^T block of step h (Cin tiles cb0 + q bc + j, pixel tiles of block h % npb;
    // tiles past Ct or the core's pixel range are not read) from DRAM into ring slot h
    // % 4, transaction id 1 + (h & 1).
    auto x_read = [&](uint32_t h) {
        const uint32_t q = (h / npb) % nq, pb = h % npb;
        const uint32_t dst = x_base + (h % kSlots) * x_bytes;
        noc_async_read_set_trid(1 + (h & 1));
        for (uint32_t j = 0; j < bc; ++j) {
            const uint32_t cb = cb0 + q * bc + j;
            if (cb >= ct) {
                break;
            }
            for (uint32_t m = 0; m < kb && pb * kb + m < m_cnt; ++m) {
                noc_async_read_page(
                    cb * mt + m_lo + pb * kb + m, xt, dst + (j * kb + m) * kTile
                );
            }
        }
    };
    // Top: block h read; multicast it once every receiver counted in for it; flag = h
    // + 1.
    auto x_send = [&](uint32_t h) {
        noc_async_read_barrier_with_trid(1 + (h & 1));
        if (mc_num > 0) {
            const uint32_t slot = x_base + (h % kSlots) * x_bytes;
            noc_semaphore_wait_min(word(x_count), mc_num * (h + 1));
            noc_async_write_multicast(
                slot,
                get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, slot),
                x_bytes,
                mc_num
            );
            noc_async_writes_flushed();
            *word(x_flag) = h + 1;
            noc_semaphore_set_multicast(
                x_flag,
                get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, x_flag),
                mc_num
            );
        }
    };
    // Receiver: the next block's slot is free; count in at the top.
    auto x_ready = [&]() {
        noc_semaphore_inc(get_noc_addr(s_x, s_y, x_count), 1);
    };

    table_read(0);
    noc_async_read_barrier_with_trid(kTableTrid);
    cb_reserve_back(cb_x, x_block);
    if (top) {
        x_read(0);
        if (steps > 1) {
            cb_reserve_back(cb_x, 2 * x_block);
            x_read(1);
        }
        x_send(0);
    } else {
        x_ready();
    }
    for (uint32_t s = 0; s < steps; ++s) {
        if (top) {
            if (s + 2 < steps) {
                // Frees slot s + 2 (compute is done with s - 2).
                cb_reserve_back(cb_x, 3 * x_block);
                x_read(s + 2);
            }
            if (s + 1 < steps) {
                x_send(s + 1);
            }
        } else {
            noc_semaphore_wait_min(word(x_flag), s + 1);
        }
        cb_push_back(cb_x, x_block);
        if (!top && s + 1 < steps) {
            // Counted in only after block s arrived (a core's count never exceeds the
            // blocks sent + 1). Block s + 1's slot last held block s - 3.
            cb_reserve_back(cb_x, x_block);
            x_ready();
        }

        uint32_t table = tables;
        if (npb > 1) {
            // This step's table was read one step ahead; read the next step's now.
            noc_async_read_barrier_with_trid(kTableTrid);
            if (s + 1 < steps) {
                table_read(s + 1);
            }
            table = tables + (s & 1) * page_bytes;
        }
        // The writer has planes slot s.
        noc_semaphore_wait_min(word(ready), s + 1);
        if (n_even > 0) {
            cb_reserve_back(cb_a0, n_even);
            if (k0 < n_even) {
                *word(a0_go) = s + 1;
            }
            const uint32_t ab = get_write_ptr(cb_a0);
            const uint32_t slot = planes + (s % kSlots) * slot_bytes;
            noc_async_read_set_trid(kBuildTrid);
            noc_async_read_set_state(get_noc_addr(0));
            uint32_t p = 0;
            for (uint32_t i = 0; i < k0; ++i) {
                const uint32_t runs = *word(table + 4 * p);
                const uint32_t dst_tile = ab + i * kTile;
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
            if (k0 < n_even) {
                noc_semaphore_wait_min(word(a0_done), s + 1);
            }
            cb_push_back(cb_a0, n_even);
        }
        *word(done) = s + 1;
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
