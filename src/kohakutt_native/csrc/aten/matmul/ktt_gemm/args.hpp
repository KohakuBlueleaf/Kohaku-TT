// Runtime-argument sentinel: the host appends it after the last common and per-core
// argument of each ktt_gemm kernel; a kernel that does not find it returns at once.
#pragma once

#include <cstdint>

namespace ktt_gemm {

    constexpr uint32_t kArgSentinel = 0x6b747467u;

}  // namespace ktt_gemm
