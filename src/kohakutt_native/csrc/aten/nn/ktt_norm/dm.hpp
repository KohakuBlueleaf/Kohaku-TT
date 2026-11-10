// Data-movement helpers of ktt_norm: L1 zero fills by NoC reads and the per-row
// statistics exchange of a group of cores sharing a tile row (row 0 of fp32 tiles sent
// as row `index` of the peers' gather tiles).
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace ktt_norm {

    constexpr uint32_t kF32Tile = 4096;

    // Element offset of (r, c) in a 32 x 32 tile of four 16 x 16 row-major faces.
    inline uint32_t at(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // Byte offset of row r of an fp32 tile (columns 0-15; columns 16-31 are 1024 bytes
    // on).
    inline uint32_t row_offset(uint32_t r) {
        return ((r >> 4) * 512 + (r & 15) * 16) * 4;
    }

    // Zeroes `bytes` (a multiple of 16) at L1 address `addr` with NoC reads of the zero
    // region.
    inline void zero_l1(uint32_t addr, uint32_t bytes) {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
            const uint32_t n = bytes - off < MEM_ZEROS_SIZE ? bytes - off
                                                            : MEM_ZEROS_SIZE;
            noc_async_read(zeros, addr + off, n);
        }
        noc_async_read_barrier();
    }

    // The counter at NoC address `counter` incremented by one; returns its previous
    // value (the atomic's return lands at MEM_NOC_ATOMIC_RET_VAL_ADDR; no other RISC of
    // the core may issue an atomic meanwhile).
    inline uint32_t fetch_add(uint64_t counter) {
        noc_semaphore_inc(counter, 1);
        noc_async_atomic_barrier();
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            MEM_NOC_ATOMIC_RET_VAL_ADDR
        );
    }

    // One control word (a tile row index) into CB `cb`.
    inline void put_word(uint32_t cb, uint32_t v) {
        cb_reserve_back(cb, 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb)) = v;
        cb_push_back(cb, 1);
    }

    // The next control word of CB `cb`.
    inline uint32_t take_word(uint32_t cb) {
        cb_wait_front(cb, 1);
        invalidate_l1_cache();
        const uint32_t v = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_read_ptr(cb)
        );
        cb_pop_front(cb, 1);
        return v;
    }

    // A group of `size` cores sharing a tile row; this core is `index`. Peer p's NoC
    // coordinates are words 2 p, 2 p + 1 at `peers`.
    struct Exchange {
        uint32_t size, index;
        uint32_t peers;
        uint32_t data_sem, ready_sem;

        uint64_t peer(uint32_t p, uint32_t addr) const {
            const auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(peers);
            return get_noc_addr(w[2 * p], w[2 * p + 1], addr);
        }

        // Tells every peer this core's receive tiles are zeroed, then waits for all
        // peers.
        void ready() const {
            for (uint32_t p = 0; p < size; ++p) {
                noc_semaphore_inc(peer(p, ready_sem), 1);
            }
            noc_async_atomic_barrier();
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem), size
            );
        }

        // Row 0 of the `count` fp32 tiles at `src` to row `index` of the tiles at `dst`
        // in every peer, then one increment of every peer's data semaphore.
        void send(uint32_t src, uint32_t dst, uint32_t count) const {
            const uint32_t row = row_offset(index);
            for (uint32_t p = 0; p < size; ++p) {
                for (uint32_t t = 0; t < count; ++t) {
                    const uint32_t s = src + t * kF32Tile, d = dst + t * kF32Tile + row;
                    noc_async_write(s, peer(p, d), 64);
                    noc_async_write(s + 1024, peer(p, d + 1024), 64);
                }
            }
            noc_async_write_barrier();
            for (uint32_t p = 0; p < size; ++p) {
                noc_semaphore_inc(peer(p, data_sem), 1);
            }
            noc_async_atomic_barrier();
        }

        // Waits until every peer has sent `rounds` times.
        void wait(uint32_t rounds) const {
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_sem), size * rounds
            );
        }
    };

}  // namespace ktt_norm
