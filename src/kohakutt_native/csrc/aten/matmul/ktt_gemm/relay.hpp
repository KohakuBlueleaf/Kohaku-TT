// Block relay between neighbouring cores: the receiver frees its CB slot and waits for
// the upstream core's data; the sender waits for its downstream core's free slot and
// writes the block to the same L1 address, then sets the downstream core's flag.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

namespace ktt_gemm {

    struct Relay {
        uint32_t up_x, up_y, down_x, down_y;
        bool injector, sink;
        uint32_t ready_addr,
            flag_addr;  // semaphores: downstream slots free, data valid

        volatile tt_l1_ptr uint32_t* ready() const {
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        }

        volatile tt_l1_ptr uint32_t* flag() const {
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
        }

        // Waits for the upstream core's block in the slot this core has reserved.
        void receive() const {
            noc_semaphore_set(flag(), INVALID);
            noc_semaphore_inc(get_noc_addr(up_x, up_y, ready_addr), 1);
            noc_semaphore_wait(flag(), VALID);
        }

        // Sends `bytes` at addr to the downstream core's same address, then its flag.
        void forward(uint32_t addr, uint32_t bytes) const {
            if (sink) {
                return;
            }
            noc_semaphore_wait(ready(), 1);
            noc_semaphore_set(ready(), 0);
            noc_async_write(addr, get_noc_addr(down_x, down_y, addr), bytes);
            noc_async_writes_flushed();
            noc_semaphore_set_remote(
                flag_addr, get_noc_addr(down_x, down_y, flag_addr)
            );
            noc_async_writes_flushed();
        }

        // forward() of two buffers under one handshake.
        void forward2(
            uint32_t addr0, uint32_t bytes0, uint32_t addr1, uint32_t bytes1
        ) const {
            if (sink) {
                return;
            }
            noc_semaphore_wait(ready(), 1);
            noc_semaphore_set(ready(), 0);
            noc_async_write(addr0, get_noc_addr(down_x, down_y, addr0), bytes0);
            noc_async_write(addr1, get_noc_addr(down_x, down_y, addr1), bytes1);
            noc_async_writes_flushed();
            noc_semaphore_set_remote(
                flag_addr, get_noc_addr(down_x, down_y, flag_addr)
            );
            noc_async_writes_flushed();
        }
    };

}  // namespace ktt_gemm
