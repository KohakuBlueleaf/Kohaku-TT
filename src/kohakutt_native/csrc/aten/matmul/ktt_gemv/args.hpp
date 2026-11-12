// Shared by ktt_gemv's kernels: the runtime-argument sentinel and which mover reads W
// block b when the two share a core's blocks (NoC0 : NoC1 = 5 : 3, spread evenly).
#pragma once

#include <cstdint>

namespace ktt_gemv {

    constexpr uint32_t kArgSentinel = 0x6b746776u;

    // True when shared W block b is read by dm1 (BRISC, NoC 1), else by dm0.
    inline bool on_dm1(uint32_t b) {
        return ((b + 1) * 3 + 4) / 8 > (b * 3 + 4) / 8;
    }

}  // namespace ktt_gemv
