// Fused-chain compute template: this core's output tiles column by column (the reader's
// order). Per tile, input i into DST slot i (fp32; row / column / scalar inputs
// broadcast by the unpacker), FUSED_BODY (generated SFPU calls on the slots), slot
// OUT_SLOT packed into CB 16. kohakutt.fusion writes one source per chain with
// NUM_INPUTS, LAYOUTS, BATCHED, OUT_SLOT and FUSED_BODY defined.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"

enum Layout : uint32_t { FULL = 0, MATRIX = 1, ROW = 2, COLUMN = 3, SCALAR = 4 };

constexpr uint32_t kInputs = NUM_INPUTS;
constexpr uint32_t kLayout[4] = LAYOUTS;
constexpr uint32_t kBatched[4] = BATCHED;

constexpr bool persistent(uint32_t i) {
    return (kLayout[i] == ROW || kLayout[i] == SCALAR) && !kBatched[i];
}

// The front tile of CB i into DST slot i, broadcast per its layout.
template <uint32_t i>
inline void load() {
    reconfig_data_format_srca(i);
    if constexpr (kLayout[i] == ROW) {
        unary_bcast_init<BroadcastType::ROW>(i);
        unary_bcast<BroadcastType::ROW>(i, 0, i);
    } else if constexpr (kLayout[i] == COLUMN) {
        unary_bcast_init<BroadcastType::COL>(i);
        unary_bcast<BroadcastType::COL>(i, 0, i);
    } else if constexpr (kLayout[i] == SCALAR) {
        unary_bcast_init<BroadcastType::SCALAR>(i);
        unary_bcast<BroadcastType::SCALAR>(i, 0, i);
    } else {
        copy_init(i);
        copy_tile(i, 0, i);
    }
}

template <uint32_t i>
inline void begin_column() {
    if constexpr (i < kInputs && persistent(i)) {
        cb_wait_front(i, 1);
    }
}

template <uint32_t i>
inline void per_tile() {
    if constexpr (i < kInputs) {
        if constexpr (!persistent(i)) {
            cb_wait_front(i, 1);
        }
        load<i>();
    }
}

template <uint32_t i>
inline void end_tile() {
    if constexpr (i < kInputs && !persistent(i)) {
        cb_pop_front(i, 1);
    }
}

template <uint32_t i>
inline void end_column() {
    if constexpr (i < kInputs && persistent(i)) {
        cb_pop_front(i, 1);
    }
}

// Runtime args: first tile, tile count, ct (tile columns).
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t ct = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = 16;

    init_sfpu(0, cb_out);
    const uint32_t end = start + count;
    const uint32_t first_row = start / ct, last_row = (end - 1) / ct;
    for (uint32_t c = 0; c < ct; ++c) {
        const uint32_t gr0 = first_row * ct + c >= start ? first_row : first_row + 1;
        const uint32_t gr1 = last_row * ct + c < end ? last_row : last_row - 1;
        if (gr1 < gr0 || gr1 == UINT32_MAX) {
            continue;
        }
        begin_column<0>();
        begin_column<1>();
        begin_column<2>();
        begin_column<3>();
        for (uint32_t gr = gr0; gr <= gr1; ++gr) {
            tile_regs_acquire();
            per_tile<0>();
            per_tile<1>();
            per_tile<2>();
            per_tile<3>();
            FUSED_BODY
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(OUT_SLOT, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
            end_tile<0>();
            end_tile<1>();
            end_tile<2>();
            end_tile<3>();
        }
        end_column<0>();
        end_column<1>();
        end_column<2>();
        end_column<3>();
    }
}
