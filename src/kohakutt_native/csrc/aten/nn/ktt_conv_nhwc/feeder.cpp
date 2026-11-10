// Nhwc conv planes feeder (RISCV_0, NoC1; a core right of its pixel block in the same
// panel, one per pixel group). Steps s = (it npb + pb) Ct + k, Cin block cb = (k + rot)
// % Ct: issue step s + 2's input reads as planes (transaction id 1 + parity, 4-slot
// ring); multicast step s + 1's slot to the group once its cores counted in; set their
// flag to s + 2.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTile = 2048, kFace = 512, kRow = 32, kSlots = 4;

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of X. Common runtime args: X address, Ct, npb,
// iters, semaphore ids (planes count, planes flag), planes slot bytes, Mt, bm_b, lowest
// / highest tap offset in pixels (int32). Per core: the group's first output tile m0,
// multicast rectangle (start x, y, end x, y, given for NoC1) and receiver count, rot.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ct = get_common_arg_val<uint32_t>(1);
    const uint32_t npb = get_common_arg_val<uint32_t>(2);
    const uint32_t iters = get_common_arg_val<uint32_t>(3);
    const uint32_t p_count = get_semaphore(get_common_arg_val<uint32_t>(4));
    const uint32_t p_flag = get_semaphore(get_common_arg_val<uint32_t>(5));
    const uint32_t slot_bytes = get_common_arg_val<uint32_t>(6);
    const int32_t mt = static_cast<int32_t>(get_common_arg_val<uint32_t>(7));
    const int32_t bm_b = static_cast<int32_t>(get_common_arg_val<uint32_t>(8));
    const int32_t min_off = static_cast<int32_t>(get_common_arg_val<uint32_t>(9));
    const int32_t max_off = static_cast<int32_t>(get_common_arg_val<uint32_t>(10));

    const int32_t m0 = static_cast<int32_t>(get_arg_val<uint32_t>(0));
    const uint32_t mc_sx = get_arg_val<uint32_t>(1), mc_sy = get_arg_val<uint32_t>(2);
    const uint32_t mc_ex = get_arg_val<uint32_t>(3), mc_ey = get_arg_val<uint32_t>(4);
    const uint32_t mc_num = get_arg_val<uint32_t>(5);
    const uint32_t rot = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_planes = 5;
    const auto x = TensorAccessor(x_args, x_addr, kTile);
    const uint32_t planes = get_write_ptr(cb_planes);
    const uint32_t steps = iters * npb * ct;

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
    // Step h's input tiles from DRAM into planes slot h % 4, transaction id 1 + (h &
    // 1).
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
