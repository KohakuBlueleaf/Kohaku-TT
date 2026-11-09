// The argument-layout version of the conv kernels (aten/nn/ktt_conv*/): the host passes
// it as the last compile-time arg and as common runtime arg 0; a kernel whose source
// expects another layout returns before reading any other argument. Changed with every
// change to a conv kernel's arguments.
#pragma once

#include <cstdint>

namespace ktt_conv {

    constexpr uint32_t kArgsVersion = 0x6B630005u;

}  // namespace ktt_conv
