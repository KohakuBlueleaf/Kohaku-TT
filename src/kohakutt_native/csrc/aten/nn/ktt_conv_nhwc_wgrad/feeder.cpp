// Nhwc conv weight-gradient planes feeder (RISCV_0, NoC1; a core right of its slot in
// the same panel, one per (slot, row) group). Steps s = ((it bo + i) nq + q) npb + pb:
// issue step s + 2's dY reads (Cout tile n0 + i, pixel block pb with its halo) as
// planes (transaction id 1 + parity, 4-slot ring); multicast step s + 1's slot to the
// group once its cores counted in; set their flag to s + 2.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTile = 2048, kFace = 512, kRow = 32, kSlots = 4;

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of dY. Common runtime args: dY address, Nt, kb,
// nq, npb, bo, iters, semaphore ids (planes count, planes flag), planes slot bytes, Mt,
// lowest / highest tap offset in pixels (int32). Per core: the group's first Cout tile
// n0, first pixel tile, pixel tiles, multicast rectangle (start x, y, end x, y, given
// for NoC1) and receiver count.
void kernel_main() {
    constexpr auto y_args = TensorAccessorArgs<0>();
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t nt = get_common_arg_val<uint32_t>(1);
    const int32_t kb = static_cast<int32_t>(get_common_arg_val<uint32_t>(2));
    const uint32_t nq = get_common_arg_val<uint32_t>(3);
    const uint32_t npb = get_common_arg_val<uint32_t>(4);
    const uint32_t bo = get_common_arg_val<uint32_t>(5);
    const uint32_t iters = get_common_arg_val<uint32_t>(6);
    const uint32_t p_count = get_semaphore(get_common_arg_val<uint32_t>(7));
    const uint32_t p_flag = get_semaphore(get_common_arg_val<uint32_t>(8));
    const uint32_t slot_bytes = get_common_arg_val<uint32_t>(9);
    const int32_t mt = static_cast<int32_t>(get_common_arg_val<uint32_t>(10));
    const int32_t min_off = static_cast<int32_t>(get_common_arg_val<uint32_t>(11));
    const int32_t max_off = static_cast<int32_t>(get_common_arg_val<uint32_t>(12));

    const uint32_t n0 = get_arg_val<uint32_t>(0);
    const int32_t m_lo = static_cast<int32_t>(get_arg_val<uint32_t>(1));
    const int32_t m_cnt = static_cast<int32_t>(get_arg_val<uint32_t>(2));
    const uint32_t mc_sx = get_arg_val<uint32_t>(3), mc_sy = get_arg_val<uint32_t>(4);
    const uint32_t mc_ex = get_arg_val<uint32_t>(5), mc_ey = get_arg_val<uint32_t>(6);
    const uint32_t mc_num = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_planes = 5;
    const auto dy = TensorAccessor(y_args, y_addr, kTile);
    const uint32_t planes = get_write_ptr(cb_planes);
    const uint32_t per_n = nq * npb;
    const uint32_t steps = iters * bo * per_n;

    // Input tiles [lo, lo + n) of pixel block pb (a block past the group's pixel tiles:
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
    // Step h's dY tiles (none past Nt) from DRAM into planes slot h % 4, transaction id
    // 1 + (h & 1).
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
    // Step h read; multicast its slot once every receiver counted in for it; flag = h
    // + 1.
    auto planes_send = [&](uint32_t h) {
        noc_async_read_barrier_with_trid(1 + (h & 1));
        const uint32_t slot = planes + (h % kSlots) * slot_bytes;
        noc_semaphore_wait_min(word(p_count), mc_num * (h + 1));
        noc_async_write_multicast(
            slot,
            get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, slot),
            slot_bytes,
            mc_num
        );
        noc_async_writes_flushed();
        *word(p_flag) = h + 1;
        noc_semaphore_set_multicast(
            p_flag, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, p_flag), mc_num
        );
    };

    planes_read(0);
    if (steps > 1) {
        planes_read(1);
    }
    planes_send(0);
    for (uint32_t s = 0; s < steps; ++s) {
        if (s + 2 < steps) {
            planes_read(s + 2);
        }
        if (s + 1 < steps) {
            planes_send(s + 1);
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
}
