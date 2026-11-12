// ktt_gemv compute: per group of sw output columns, x (CB 0, k-major) times the W
// blocks of CB 1 / 2 summed in fp32 Dst (HAS_BIAS: onto CB 3's row 0), packed to CB 16.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "args.hpp"

// Args: common M, K tiles, sw, kb, w transposed in its tiles; per core columns, W
// blocks' mover (0 both, 1 dm0, 2 dm1); each list ends with the sentinel.
void kernel_main() {
    if (get_common_arg_val<uint32_t>(5) != ktt_gemv::kArgSentinel ||
        get_arg_val<uint32_t>(2) != ktt_gemv::kArgSentinel) {
        return;
    }
    const uint32_t mt = get_common_arg_val<uint32_t>(0);
    const uint32_t kt = get_common_arg_val<uint32_t>(1);
    const uint32_t sw = get_common_arg_val<uint32_t>(2);
    const uint32_t kb = get_common_arg_val<uint32_t>(3);
    const uint32_t transpose = get_common_arg_val<uint32_t>(4);
    const uint32_t nc = get_arg_val<uint32_t>(0);
    const uint32_t from = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_x = 0, cb_w0 = 1, cb_w1 = 2, cb_bias = 3, cb_out = 16;

    const uint32_t groups = (nc + sw - 1) / sw, nkb = (kt + kb - 1) / kb;
    const uint32_t block = sw * kb, sub = mt * sw;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w0, cb_out);
    matmul_block_init(cb_x, cb_w0, transpose, sw, mt, 1);

    uint32_t b = 0;
    for (uint32_t g = 0; g < groups; ++g) {
        tile_regs_acquire();
#ifdef HAS_BIAS
        cb_wait_front(cb_bias, sw);
        reconfig_data_format(cb_w0, cb_bias, cb_x, cb_bias);
        unary_bcast_init<BroadcastType::ROW>(cb_bias);
        for (uint32_t t = 0; t < sub; ++t) {
            unary_bcast<BroadcastType::ROW>(cb_bias, t % sw, t);
        }
        unary_bcast_uninit<BroadcastType::ROW>(cb_bias);
        reconfig_data_format(cb_bias, cb_w0, cb_bias, cb_x);
        matmul_block_init(cb_x, cb_w0, transpose, sw, mt, 1);
        cb_pop_front(cb_bias, sw);
#endif
        for (uint32_t j = 0; j < nkb; ++j, ++b) {
            const uint32_t k0 = j * kb, kc = kt - k0 < kb ? kt - k0 : kb;
            if (g == 0) {
                cb_wait_front(cb_x, (k0 + kc) * mt);
            }
            const uint32_t cb_w = (from == 0 ? ktt_gemv::on_dm1(b) : from == 2) ? cb_w1
                                                                                : cb_w0;
            cb_wait_front(cb_w, block);
            for (uint32_t k = 0; k < kc; ++k) {
                matmul_block(
                    cb_x, cb_w, (k0 + k) * mt, k * sw, 0, transpose, sw, mt, 1
                );
            }
            cb_pop_front(cb_w, block);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, sub);
        tile_regs_wait();
        for (uint32_t t = 0; t < sub; ++t) {
            pack_tile(t, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, sub);
    }
}
