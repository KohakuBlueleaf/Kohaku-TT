// Runtime-argument sentinel: the host appends it after the last common and per-core
// argument of each ktt_gemm kernel; a kernel that does not find it returns at once.
#pragma once

#include <cstdint>

namespace ktt_gemm {

    constexpr uint32_t kArgSentinel = 0x6b747467u;

    // The K block processed as number `kblk` of `nk`: with `parts` > 1 the K blocks of
    // `parts` equal ranges in turn (nk a multiple of parts).
    inline uint32_t k_block(uint32_t kblk, uint32_t nk, uint32_t parts) {
        return parts > 1 ? (kblk % parts) * (nk / parts) + kblk / parts : kblk;
    }

}  // namespace ktt_gemm
