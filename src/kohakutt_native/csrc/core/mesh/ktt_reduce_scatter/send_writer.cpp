// Sender writer of the 1x2 collectives: sends the tiles CB 0 holds, tile range
// [first, first + count) in chunks of `chunk`, over one fabric link into the peer's
// scratch (pages of `pack` tiles, the same address on both chips), `pack` tiles per
// packet. The last packet of each chunk also increments the semaphore of the peer drain
// owning the chunk (chunk c to drain c % drains). A ready handshake with the peer
// sender of the same link comes first.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric;

// Packet headers in flight before the writes are flushed and the first is reused.
constexpr uint32_t kHeaders = 4;
constexpr uint32_t kMaxDrains = 16;

// Compile-time args: scratch accessor, then the runtime-arg count before the drain
// list. Runtime args: scratch address, first tile, tile count, tile bytes, chunk tiles,
// drains, drain semaphore address, peer sender x / y, ready semaphore address, tiles
// per packet, then x / y of each peer drain, then the fabric connection.
void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr uint32_t kArgs = get_compile_time_arg_val(
        dst_args.next_compile_time_args_offset()
    );
    constexpr uint32_t cb = 0;

    size_t i = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(i++);
    const uint32_t first = get_arg_val<uint32_t>(i++);
    const uint32_t count = get_arg_val<uint32_t>(i++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(i++);
    const uint32_t chunk = get_arg_val<uint32_t>(i++);
    const uint32_t drains = get_arg_val<uint32_t>(i++);
    const uint32_t drain_sem = get_arg_val<uint32_t>(i++);
    const uint32_t peer_x = get_arg_val<uint32_t>(i++);
    const uint32_t peer_y = get_arg_val<uint32_t>(i++);
    const uint32_t ready_sem = get_arg_val<uint32_t>(i++);
    const uint32_t pack = get_arg_val<uint32_t>(i++);
    if (i != kArgs || drains == 0 || drains > kMaxDrains || pack == 0 ||
        chunk % pack != 0 || first % pack != 0) {
        return;  // runtime-arg layout does not match this source
    }
    uint64_t drain_sem_noc[kMaxDrains];
    for (uint32_t d = 0; d < drains; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(i++);
        const uint32_t y = get_arg_val<uint32_t>(i++);
        drain_sem_noc[d] = safe_get_noc_addr(x, y, drain_sem, 0);
    }
    auto
        sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(
            i
        );

    const auto dst = TensorAccessor(dst_args, dst_addr, pack * tile_bytes);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[kHeaders];
    for (uint32_t h = 0; h < kHeaders; ++h) {
        headers[h] = PacketHeaderPool::allocate_header();
        headers[h]->to_chip_unicast(1);
    }
    sender.open<true>();

    // Ready handshake with the peer sender of this link.
    headers[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(
        safe_get_noc_addr(peer_x, peer_y, ready_sem, 0), 1
    ));
    sender.wait_for_empty_write_slot();
    sender.send_payload_flush_blocking_from_address(
        (uint32_t)headers[0], sizeof(PACKET_HEADER_TYPE)
    );
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem);
    noc_semaphore_wait_min(ready, 1);
    noc_semaphore_set(ready, 0);

    uint32_t packet = 0;
    for (uint32_t c = 0, t = 0; t < count; ++c, t += chunk) {
        const uint32_t n = count - t < chunk ? count - t : chunk;
        const uint64_t sem_noc = drain_sem_noc[c % drains];
        cb_wait_front(cb, chunk);
        uint32_t l1 = get_read_ptr(cb);
        for (uint32_t k = 0; k < n; k += pack, ++packet) {
            const uint32_t h = packet % kHeaders;
            if (h == 0) {
                // Headers are rewritten only after their previous packets left L1.
                noc_async_writes_flushed();
            }
            // The receiving router writes on its NoC (edm_to_local_chip_noc). The
            // chunk's last packet also increments the drain semaphore by the chunk's
            // tile count, after the router flushes the chunk's writes.
            const uint32_t bytes = (n - k < pack ? n - k : pack) * tile_bytes;
            const uint64_t page_noc = dst.get_noc_addr(
                (first + t + k) / pack, 0, edm_to_local_chip_noc
            );
            if (k + pack < n) {
                headers[h]->to_noc_unicast_write(
                    NocUnicastCommandHeader{page_noc}, bytes
                );
            } else {
                headers[h]->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader(page_noc, sem_noc, n, true),
                    bytes
                );
            }
            sender.wait_for_empty_write_slot();
            sender.send_payload_without_header_non_blocking_from_address(l1, bytes);
            sender.send_payload_flush_non_blocking_from_address(
                (uint32_t)headers[h], sizeof(PACKET_HEADER_TYPE)
            );
            l1 += bytes;
        }
        // The CB slots are refilled after the pop: their payload reads must be done.
        noc_async_writes_flushed();
        cb_pop_front(cb, chunk);
    }
    noc_async_write_barrier();
    sender.close();
}
