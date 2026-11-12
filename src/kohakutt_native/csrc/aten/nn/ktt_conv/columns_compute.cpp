// Columns compute: per unit, the 32 row-major input rows of CB 0 (Ct tiles wide)
// tilized into CB 16.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"

// Runtime args (per core): units; common: Ct.
void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    const uint32_t ct = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_rows = 0, cb_out = 16;

    compute_kernel_hw_startup(cb_rows, cb_out);
    tilize_init(cb_rows, ct, cb_out);
    for (uint32_t u = 0; u < units; ++u) {
        cb_wait_front(cb_rows, ct);
        cb_reserve_back(cb_out, ct);
        tilize_block(cb_rows, ct, cb_out);
        cb_push_back(cb_out, ct);
        cb_pop_front(cb_rows, ct);
    }
    tilize_uninit(cb_rows, cb_out);
}
