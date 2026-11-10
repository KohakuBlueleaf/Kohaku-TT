// Runtime-argument layout check of the ktt_softmax kernels: the host appends kSentinel
// as the last common runtime arg and passes the common arg count as a compile-time arg.
#pragma once

#include <stdint.h>

namespace ktt_softmax {

    constexpr uint32_t kSentinel = 0x4b534d58u;

    // True when the host passed `expected` common runtime args (count: compile-time arg
    // `cta`) ending in the sentinel.
    template <uint32_t cta>
    inline bool args_ok(uint32_t expected) {
        constexpr uint32_t count = get_compile_time_arg_val(cta);
        return (
            count >= 1 && count == expected &&
            get_common_arg_val<uint32_t>(count - 1) == kSentinel
        );
    }

}  // namespace ktt_softmax
