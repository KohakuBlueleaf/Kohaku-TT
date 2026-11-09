// Nhwc conv planes feeder (RISCV_0, NoC1; a core right of its pixel block in the same panel, one
// per pixel group). Step g (Cin block cb = (g % Ct + rot) % Ct): issue block g + 2's input reads
// as planes (transaction id 1 + parity, 4-slot ring); multicast block g + 1's slot to the group
// once its cores counted in; set their flag to g + 2.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kTile = 2048, kFace = 512, kRow = 32, kSlots = 4;

    volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

}  // namespace

// Compile-time args: TensorAccessorArgs of X. Common runtime args: X address, Ct, iters,
// semaphore ids (planes count, planes flag). Per core: first input tile, input tiles, multicast
// rectangle (start x, y, end x, y, given for NoC1) and receiver count, rot.
void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ct = get_common_arg_val<uint32_t>(1);
    const uint32_t iters = get_common_arg_val<uint32_t>(2);
    const uint32_t p_count = get_semaphore(get_common_arg_val<uint32_t>(3));
    const uint32_t p_flag = get_semaphore(get_common_arg_val<uint32_t>(4));

    const uint32_t tile_lo = get_arg_val<uint32_t>(0);
    const uint32_t t_in = get_arg_val<uint32_t>(1);
    const uint32_t mc_sx = get_arg_val<uint32_t>(2), mc_sy = get_arg_val<uint32_t>(3);
    const uint32_t mc_ex = get_arg_val<uint32_t>(4), mc_ey = get_arg_val<uint32_t>(5);
    const uint32_t mc_num = get_arg_val<uint32_t>(6);
    const uint32_t rot = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_planes = 5;
    const auto x = TensorAccessor(x_args, x_addr, kTile);
    const uint32_t planes = get_write_ptr(cb_planes);
    const uint32_t slot_bytes = t_in * kTile;
    const uint32_t total = iters * ct;

    // Block h's input tiles from DRAM into planes slot h % 4, transaction id 1 + (h & 1).
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
    // Block h read; multicast its slot once every receiver counted in for it; flag = h + 1.
    auto planes_send = [&](uint32_t h) {
        noc_async_read_barrier_with_trid(1 + (h & 1));
        const uint32_t slot = planes + (h % kSlots) * slot_bytes;
        noc_semaphore_wait_min(word(p_count), mc_num * (h + 1));
        noc_async_write_multicast(slot, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, slot), slot_bytes, mc_num);
        noc_async_writes_flushed();
        *word(p_flag) = h + 1;
        noc_semaphore_set_multicast(p_flag, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, p_flag), mc_num);
    };

    planes_read(0);
    if (total > 1) {
        planes_read(1);
    }
    planes_send(0);
    for (uint32_t g = 0; g < total; ++g) {
        if (g + 2 < total) {
            planes_read(g + 2);
        }
        if (g + 1 < total) {
            planes_send(g + 1);
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
}
