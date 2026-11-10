// Nhwc conv weight-gradient writer (RISCV_0, NoC1 on every compute core). Steps as the
// reader's: s = ((it bo + i) nq + q) npb + pb. Planes of dY (Cout tile n0 + i, pixel
// block pb with its halo) run ahead of the builds (4-slot ring):
// - Lead (no feeder): wait for step s's reads, issue step s + 2's (transaction id 1 +
// parity).
//   Receivers: wait for flag >= s + 1, then count in at the feeder for step s + 1 once
//   its slot is free (so a core's count never exceeds the steps sent + 1, and the
//   feeder's sum test holds for every receiver).
// - `ready` = s + 1 for the reader; the odd G tiles of step s (transaction id 3) with
// pixel
//   block pb's copy table (transaction id 4, read one step ahead when npb > 1); with
//   k0 < n_even, after `a0 go` >= s + 1, even tiles [k0, n_even) into block s, then `a0
//   done`.
// After the last pixel block of (i, q): the bc x taps output tiles in compute's (tap,
// j) order, tile (cb0 + q bc + j, taps - 1 - tap, n0 + i) to page ((row0 + cb) taps +
// t) Nt + n; tiles past Ct or Nt are dropped.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTile = 2048, kFace = 512, kRow = 32, kZero = 0xffffffffu,
                       kBuildTrid = 3, kTableTrid = 4, kSlots = 4;

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of dY, table, OUT. Common runtime args: dY,
// table, OUT addresses, Nt, taps, bc, kb, nq, npb, bo, iters, semaphore ids (planes
// count, planes flag, planes ready, planes done, a0 go, a0 done), table page bytes,
// planes slot bytes, Mt, Ct, lowest / highest tap offset in pixels (int32), output page
// bytes. Per core: first Cin tile cb0, first Cout tile n0, first pixel tile, pixel
// tiles, first table page, odd tiles per step, even tiles per step, k0, lead (1 / 0),
// feeder (x, y), first output row tile row0.
void kernel_main() {
    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto
        t_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto
        o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t nt = get_common_arg_val<uint32_t>(3);
    const uint32_t taps = get_common_arg_val<uint32_t>(4);
    const uint32_t bc = get_common_arg_val<uint32_t>(5);
    const int32_t kb = static_cast<int32_t>(get_common_arg_val<uint32_t>(6));
    const uint32_t nq = get_common_arg_val<uint32_t>(7);
    const uint32_t npb = get_common_arg_val<uint32_t>(8);
    const uint32_t bo = get_common_arg_val<uint32_t>(9);
    const uint32_t iters = get_common_arg_val<uint32_t>(10);
    const uint32_t p_count = get_semaphore(get_common_arg_val<uint32_t>(11));
    const uint32_t p_flag = get_semaphore(get_common_arg_val<uint32_t>(12));
    const uint32_t ready = get_semaphore(get_common_arg_val<uint32_t>(13));
    const uint32_t done = get_semaphore(get_common_arg_val<uint32_t>(14));
    const uint32_t a0_go = get_semaphore(get_common_arg_val<uint32_t>(15));
    const uint32_t a0_done = get_semaphore(get_common_arg_val<uint32_t>(16));
    const uint32_t page_bytes = get_common_arg_val<uint32_t>(17);
    const uint32_t slot_bytes = get_common_arg_val<uint32_t>(18);
    const int32_t mt = static_cast<int32_t>(get_common_arg_val<uint32_t>(19));
    const uint32_t ct = get_common_arg_val<uint32_t>(20);
    const int32_t min_off = static_cast<int32_t>(get_common_arg_val<uint32_t>(21));
    const int32_t max_off = static_cast<int32_t>(get_common_arg_val<uint32_t>(22));
    const uint32_t out_page = get_common_arg_val<uint32_t>(23);

    const uint32_t cb0 = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const int32_t m_lo = static_cast<int32_t>(get_arg_val<uint32_t>(2));
    const int32_t m_cnt = static_cast<int32_t>(get_arg_val<uint32_t>(3));
    const uint32_t tab_page0 = get_arg_val<uint32_t>(4);
    const uint32_t n_odd = get_arg_val<uint32_t>(5);
    const uint32_t n_even = get_arg_val<uint32_t>(6);
    const uint32_t k0 = get_arg_val<uint32_t>(7);
    const bool lead = get_arg_val<uint32_t>(8) != 0;
    const uint32_t l_x = get_arg_val<uint32_t>(9), l_y = get_arg_val<uint32_t>(10);
    const uint32_t row0 = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_a0 = 2, cb_a1 = 3, cb_planes = 5, cb_tab = 9, cb_out = 16;
    const auto dy = TensorAccessor(y_args, y_addr, kTile);
    const auto tab = TensorAccessor(t_args, t_addr, page_bytes);
    const auto out = TensorAccessor(o_args, o_addr, out_page);
    const uint32_t per_n = nq * npb;
    const uint32_t steps = iters * bo * per_n;
    const uint32_t planes = get_write_ptr(cb_planes);
    const uint32_t tables = get_write_ptr(cb_tab);
    const uint32_t a0_base = get_write_ptr(cb_a0);

    // Input tiles [lo, lo + n) of pixel block pb (a block past the core's pixel tiles:
    // one tile).
    auto range = [&](uint32_t pb, uint32_t& lo, uint32_t& n) {
        const int32_t first = static_cast<int32_t>(pb) * kb;
        if (first >= m_cnt) {
            lo = 0, n = 1;
            return;
        }
        const int32_t mp = m_lo + first;
        int32_t g_lo = 32 * mp + min_off, g_hi = 32 * (mp + kb) - 1 + max_off;
        g_lo = g_lo < 0 ? 0 : g_lo;
        g_hi = g_hi > 32 * mt - 1 ? 32 * mt - 1 : g_hi;
        lo = static_cast<uint32_t>(g_lo / 32);
        n = static_cast<uint32_t>(g_hi / 32) + 1 - lo;
    };
    // Lead: step h's dY tiles (Cout tile n0 + i; none past Nt) from DRAM into planes
    // slot h % 4 (plane L channels 0-15, plane R 16-31, pixel-contiguous), transaction
    // id 1 + (h & 1).
    auto planes_read = [&](uint32_t h) {
        const uint32_t ch = n0 + (h / per_n) % bo;
        if (ch >= nt) {
            return;
        }
        uint32_t lo, n;
        range(h % npb, lo, n);
        const uint32_t slot = planes + (h % kSlots) * slot_bytes;
        noc_async_read_set_trid(1 + (h & 1));
        for (uint32_t t = 0; t < n; ++t) {
            const uint32_t page = (lo + t) * nt + ch;
            for (uint32_t f = 0; f < 4; ++f) {
                const uint32_t plane = slot + (f & 1) * (slot_bytes / 2);
                noc_async_read(
                    dy.get_noc_addr(page, f * kFace),
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
    // Copy table of step h's pixel block into table slot h & 1 (slot 0 with one pixel
    // block).
    auto table_read = [&](uint32_t h) {
        noc_async_read_set_trid(kTableTrid);
        noc_async_read_page(
            tab_page0 + 2 * (h % npb) + 1,
            tab,
            tables + (npb > 1 ? h & 1 : 0) * page_bytes
        );
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
        const uint32_t pb = s % npb;
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
        uint32_t table = tables;
        if (npb > 1) {
            // This step's table was read one step ahead; read the next step's now.
            noc_async_read_barrier_with_trid(kTableTrid);
            if (s + 1 < steps) {
                table_read(s + 1);
            }
            table = tables + (s & 1) * page_bytes;
        }
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
            build(get_write_ptr(cb_a1), 0, n_odd);
            cb_push_back(cb_a1, n_odd);
        }
        if (k0 < n_even) {
            // G block s (even tiles) sits at ring position s % 2 (capacity 2 n_even,
            // n_even per push).
            noc_semaphore_wait_min(word(a0_go), s + 1);
            build(a0_base + (s & 1) * n_even * kTile, k0, n_even);
            *word(a0_done) = s + 1;
        }

        if (pb == npb - 1) {
            const uint32_t n = n0 + (s / per_n) % bo;
            const uint32_t q = (s / npb) % nq;
            for (uint32_t tap = 0; tap < taps; ++tap) {
                for (uint32_t j = 0; j < bc; ++j) {
                    const uint32_t cb = cb0 + q * bc + j;
                    cb_wait_front(cb_out, 1);
                    if (cb < ct && n < nt) {
                        noc_async_write_page(
                            ((row0 + cb) * taps + taps - 1 - tap) * nt + n,
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
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
