"""Flash attention forward at the metal level: the backend's own kernel (ktt_flash)
as one model, with its lanes, blocks, step pairs, pack-thread exp and row state;
torch op ``kohakutt_example::flash_km``.
Usage: python examples/kernels/dsl_metal/flash_km.py [--heads H] [--seq S]"""

import argparse
import contextlib
import math
import struct
import textwrap

import torch

import kohakutt
import kohakutt.dsl.metal as km

# Score tiles per step, and the most query tile rows per job, key tiles per block and
# head-dim tiles; a block has at most NSUB steps per row.
SC = 3
R_MOST, KB_MOST, DT_MOST = 13, 12, 8
NSUB = KB_MOST // SC
# Bytes of a bf16 tile, an fp32 tile and a token page as the backend's plan counts
# them.
TILE, TILE32, TOKEN = 2048, 4096, 64
# Pages of the constant buffer: ones, the causal mask, the tail mask, both masks.
ONES, CAUSAL, TAIL, BOTH = 0, 1, 2, 3
LOG2E = 1.4426950408889634
# ktt_flash's register scopes (the math thread clears the registers, not the packer)
# and its matmul (from the replay buffer), as raw renames of the typed calls.
MATH_CLEAR = True
NO_MOP = True
# Device-profiler zones around each step's score and output (profiler builds).
ZONES = False

# ktt_flash's vector-unit code (its accurate exp only): the math thread's lane maxima
# and constant tiles, the pack thread's row state, exp, rescale and normalization.
VECTOR = km.SfpuFn(
    "flash_km_vector",
    textwrap.dedent("""
    }  // namespace
    #if defined(TRISC_MATH) || defined(TRISC_PACK)
    #include "cmath_common.h"
    #include "llk_math_eltwise_unary_sfpu_init.h"
    #include "sfpi.h"
    #endif
    namespace {
    #if defined(TRISC_MATH) || defined(TRISC_PACK)
    namespace fk {
    using namespace ckernel;
    constexpr uint32_t L0 = p_sfpu::LREG0, L1 = p_sfpu::LREG1, L2 = p_sfpu::LREG2;
    constexpr uint32_t L3 = p_sfpu::LREG3, L4 = p_sfpu::LREG4, L5 = p_sfpu::LREG5;
    constexpr uint32_t L6 = p_sfpu::LREG6, L7 = p_sfpu::LREG7;
    constexpr uint32_t C512 = 11, CA = 12, CB2 = 13, CB1 = 14;
    constexpr uint32_t ONE = p_sfpu::LCONST_1, ZERO = p_sfpu::LCONST_0;
    // Dst rows of the state tile (slot 3); a band is 4 tile rows.
    constexpr uint32_t ST = 192;
    constexpr float kB2 = 0.33718854f, kB1 = -0.01674134f, kB0 = 0.68127801f;
    constexpr float kU0 = 639.0f;

    inline void load_bits(uint32_t lreg, uint32_t bits) {
        TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
        TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xffff);
    }

    inline void load_const(uint32_t lreg, float x) {
        load_bits(lreg, __builtin_bit_cast(uint32_t, x));
    }

    inline void set_config(uint32_t c, uint32_t bits) {
        load_bits(L0, bits);
        TT_SFPCONFIG(0, c, 0);
        TTI_SFPNOP;
    }

    inline void next_band(uint32_t j) {
        math::inc_dst_addr<4>();
        if (j == 4) {
            math::inc_dst_addr<8>();
            math::inc_dst_addr<8>();
        }
    }

    // Math thread: LReg 11-14 = 512, a, B2, B1; the lane-max macro (sequence 2).
    inline void math_init(uint32_t a_bits) {
        llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>();
        set_config(C512, 0x44000000u);
        set_config(CA, a_bits);
        set_config(CB2, __builtin_bit_cast(uint32_t, kB2));
        set_config(CB1, __builtin_bit_cast(uint32_t, kB1));
        TTI_SFPSWAP(0, L0, 12, 1);
        TTI_SFPNOP;
        set_config(6, 0x84u);
        TTI_SFPCONFIG(0, 8, 1);
        TTI_SFPNOP;
    }

    inline void math_begin() {
        TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH);
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(
            0);
        t6_mutex_acquire(mutex::SFPU);
    }

    inline void math_end() {
        TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::WAIT_SFPU);
        t6_mutex_release(mutex::SFPU);
    }

    template <uint32_t nt>
    inline void band_lane_max() {
        constexpr uint32_t offs[4] = {0, 2, 16, 18};
        TTI_SFPLOAD(L0, 0, ADDR_MOD_7, 0);
    #pragma GCC unroll 12
        for (uint32_t i = 1; i < 4 * nt; ++i) {
            TT_SFPLOADMACRO(
                (2 << 2) | (1 + (i & 1)), 0, ADDR_MOD_7, 64 * (i / 4) + offs[i % 4]);
            TTI_SFPNOP;
        }
        TTI_SFPNOP;
    }

    template <uint32_t nt>
    inline void state_max() {
        math::clear_dst_reg_addr();
    #pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            band_lane_max<nt>();
            TTI_SFPSTORE(L0, 0, ADDR_MOD_7, ST + 2);
        }
        math::clear_dst_reg_addr();
    }

    // Math thread: each band's lane maxima of score slots 0 .. nt - 1 into the state
    // tile's left odd columns.
    [[gnu::noinline]] void max_section(uint32_t nt) {
        math_begin();
        if (nt == 3) {
            state_max<3>();
        } else if (nt == 2) {
            state_max<2>();
        } else {
            state_max<1>();
        }
        math_end();
    }

    inline void const_lanes() {
        TTI_SFPMOV(0, 15, L4, 0);
        TTI_SFPLOADI(L3, sfpi::SFPLOADI_MOD0_USHORT, 14);
        TTI_SFPAND(0, L3, L4, 0);
        TTI_SFPMOV(0, 15, L5, 0);
        TTI_SFPSHFT(0xffc, 0, L5, 1);
        TTI_SFPCAST(L4, L4, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
        TTI_SFPCAST(L5, L5, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
        TTI_SFPNOP;
        TTI_SFPMAD(L5, ONE, L4, L6, 1);
        TTI_SFPNOP;
    }

    inline uint32_t bf16_bits(int32_t x) {
        return __builtin_bit_cast(uint32_t, static_cast<float>(x)) >> 16;
    }

    inline void clamp01() {
        TTI_SFPSWAP(0, L0, ZERO, 1);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L0, ONE, 9);
        TTI_SFPNOP;
    }

    // Constant tile `kind` (0 ones, 1 causal, 2 tail, 3 both) into slot t; `tail` is
    // the first masked key column.
    inline void const_tile(uint32_t kind, uint32_t t, int32_t tail) {
    #pragma GCC unroll 1
        for (uint32_t f = 0; f < 4; ++f) {
    #pragma GCC unroll 1
            for (uint32_t j = 0; j < 4; ++j) {
    #pragma GCC unroll 1
                for (uint32_t p = 0; p < 2; ++p) {
                    const int32_t row = 16 * static_cast<int32_t>(f >> 1) +
                        4 * static_cast<int32_t>(j);
                    const int32_t col = 16 * static_cast<int32_t>(f & 1) +
                        static_cast<int32_t>(p);
                    if (kind == 0) {
                        TTI_SFPMOV(0, ONE, L0, 0);
                    } else if (kind == 1) {
                        TTI_SFPMOV(0, L6, L0, 0);
                        TT_SFPADDI(bf16_bits(col - row), L0, 0);
                        TTI_SFPNOP;
                        clamp01();
                        TTI_SFPMULI(0xF14A, L0, 0);
                    } else if (kind == 2) {
                        TTI_SFPMOV(0, L4, L0, 0);
                        TT_SFPADDI(bf16_bits(col - tail + 1), L0, 0);
                        TTI_SFPNOP;
                        clamp01();
                        TTI_SFPMULI(0xF14A, L0, 0);
                    } else {
                        TTI_SFPMOV(0, L6, L0, 0);
                        TT_SFPADDI(bf16_bits(col - row), L0, 0);
                        TTI_SFPNOP;
                        clamp01();
                        TTI_SFPMOV(0, L0, L2, 0);
                        TTI_SFPMOV(0, L4, L0, 0);
                        TT_SFPADDI(bf16_bits(col - tail + 1), L0, 0);
                        TTI_SFPNOP;
                        clamp01();
                        TTI_SFPMAD(L0, ONE, L2, L0, 0);
                        TTI_SFPNOP;
                        clamp01();
                        TTI_SFPMULI(0xF14A, L0, 0);
                    }
                    TTI_SFPNOP;
                    TT_SFPSTORE(L0, 0, ADDR_MOD_7, 64 * t + 16 * f + 4 * j + 2 * p);
                }
            }
        }
    }

    // Math thread: ones, the causal mask and, with a ragged last key tile, the tail
    // masks into slots 0-3.
    [[gnu::noinline]] void const_section(uint32_t tail) {
        math_begin();
        math::clear_dst_reg_addr();
        const_lanes();
        const_tile(0, 0, 0);
        const_tile(1, 1, 0);
        if (tail != 0) {
            const_tile(2, 2, static_cast<int32_t>(tail));
            const_tile(3, 3, static_cast<int32_t>(tail));
        }
        math::clear_dst_reg_addr();
        math_end();
    }

    // Pack thread: address modes 7 (no step) and 6 (4 rows).
    inline void addr_init() {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(
            ADDR_MOD_7);
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(
            ADDR_MOD_6);
    }

    inline void pack_begin() {
        TTI_SEMWAIT(
            p_stall::STALL_SFPU | p_stall::STALL_SYNC,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_ZERO);
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(
            0);
        t6_mutex_acquire(mutex::SFPU);
    }

    inline void pack_end() {
        TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_PACK, p_stall::WAIT_SFPU);
        t6_mutex_release(mutex::SFPU);
    }

    inline void row_segment_max() {
        TTI_SFPSHFT2(0, L0, L1, 3);
        TTI_SFPSWAP(0, L0, L1, 1);
        TTI_SFPSHFT2(0, L0, L1, 3);
        TTI_SFPSHFT2(0, L1, L1, 3);
        TTI_SFPSWAP(0, L0, L1, 1);
        TTI_SFPSHFT2(0, L0, L1, 3);
        TTI_SFPSHFT2(0, L1, L1, 3);
        TTI_SFPSHFT2(0, L1, L1, 3);
        TTI_SFPSHFT2(0, L1, L1, 3);
        TTI_SFPSWAP(0, L0, L1, 1);
    }

    inline void band_state_first() {
        TTI_SFPMAD(L0, CA, L7, L1, 1);
        TTI_SFPNOP;
        TTI_SFPSTORE(L1, 0, ADDR_MOD_7, ST + 0);
        TTI_SFPSTORE(L1, 0, ADDR_MOD_7, ST + 16);
    }

    // Where a c + b_old > thr (L5), b becomes 639 - a c; L6 |= the band's mask.
    inline void band_state_next() {
        TTI_SFPLOAD(L1, 0, ADDR_MOD_7, ST + 0);
        TTI_SFPMAD(L0, CA, L1, L2, 0);
        TTI_SFPMAD(L2, ONE, L7, L3, 2);
        TTI_SFPMOV(0, L2, L4, 0);
        TTI_SFPGT(0, L5, L4, 8);
        TTI_SFPOR(0, L4, L6, 0);
        TTI_SFPAND(0, L4, L3, 0);
        TTI_SFPMAD(L3, ONE, L1, L1, 1);
        TTI_SFPNOP;
        TTI_SFPSTORE(L1, 0, ADDR_MOD_7, ST + 0);
        TTI_SFPSTORE(L1, 0, ADDR_MOD_7, ST + 16);
    }

    // The tile's flag (1.0 when a row moved) to right odd of rows 0-3 (word 257).
    inline void store_any_flag() {
        TTI_SFPMOV(0, L6, L0, 0);
        TTI_SFPMOV(0, ZERO, L1, 0);
        TTI_SFPMOV(0, ZERO, L2, 0);
        TTI_SFPMOV(0, ZERO, L3, 0);
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPOR(0, L1, L0, 0);
        TTI_SFPOR(0, L2, L0, 0);
        TTI_SFPOR(0, L3, L0, 0);
        TTI_SFPLOADI(L1, sfpi::SFPLOADI_MOD0_FLOATB, 0x3F80);
        TTI_SFPAND(0, L1, L0, 0);
        TTI_SFPSTORE(L0, 0, ADDR_MOD_7, ST + 18 - 44);
    }

    template <bool first>
    inline void state_update(uint32_t thr_bits) {
        math::clear_dst_reg_addr();
        load_const(L7, kU0);
        if constexpr (!first) {
            TTI_SFPMOV(0, ZERO, L6, 0);
            load_bits(L5, thr_bits);
        }
    #pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            TTI_SFPLOAD(L0, 0, ADDR_MOD_7, ST + 2);
            row_segment_max();
            if constexpr (first) {
                band_state_first();
            } else {
                band_state_next();
            }
        }
        if constexpr (!first) {
            store_any_flag();
        }
        math::clear_dst_reg_addr();
    }

    inline void exp_pair_body() {
        TTI_SFPLOAD(L0, 0, ADDR_MOD_7, 0);
        TTI_SFPLOAD(L3, 0, ADDR_MOD_7, 2);
        TTI_SFPMAD(L0, CA, L7, L0, 0);
        TTI_SFPMAD(L3, CA, L7, L3, 0);
        TTI_SFPSWAP(0, L0, C512, 1);
        TTI_SFPSWAP(0, L3, C512, 1);
        TTI_SFPSHFT(9, L0, L0, 5);
        TTI_SFPSHFT(9, L3, L3, 5);
        TTI_SFPSETEXP(127, L0, L1, 1);
        TTI_SFPSETEXP(127, L3, L4, 1);
        TTI_SFPMAD(L1, CB2, CB1, L2, 0);
        TTI_SFPMAD(L4, CB2, CB1, L5, 0);
        TTI_SFPMAD(L2, L1, L6, L2, 0);
        TTI_SFPMAD(L5, L4, L6, L5, 0);
        TTI_SFPSETEXP(0, L2, L0, 2);
        TTI_SFPSETEXP(0, L5, L3, 2);
        TTI_SFPSTORE(L0, 0, ADDR_MOD_7, 0);
        TTI_SFPSTORE(L3, 0, ADDR_MOD_6, 2);
    }

    template <uint32_t t>
    inline void exp_tile() {
        constexpr uint32_t body = 19;
        load_replay_buf<Exec>(0, body, [] {
            TTI_SFPLOAD(L7, 0, ADDR_MOD_7, 64 * (3 - t));
            exp_pair_body();
        });
    #pragma GCC unroll 1
        for (uint32_t i = 1; i < 16; ++i) {
            TTI_REPLAY(0, body, 0, 0);
        }
    }

    template <uint32_t nt>
    inline void exp_step() {
        load_const(L6, kB0);
        math::clear_dst_reg_addr();
        exp_tile<0>();
        if constexpr (nt > 1) {
            exp_tile<1>();
        }
        if constexpr (nt > 2) {
            exp_tile<2>();
        }
        math::clear_dst_reg_addr();
    }

    // Pack thread: the row state of the step from its lane maxima, then P = exp of
    // score slots 0 .. nt - 1 in place.
    [[gnu::noinline]] void exp_section(uint32_t first, uint32_t nt, uint32_t thr) {
        pack_begin();
        if (first != 0) {
            state_update<true>(thr);
        } else {
            state_update<false>(thr);
        }
        if (nt == 3) {
            exp_step<3>();
        } else if (nt == 2) {
            exp_step<2>();
        } else {
            exp_step<1>();
        }
        pack_end();
    }

    // L7 = alpha = 2^(b_new - b_old) of the band (1 where b did not move); b_new from
    // slot 2, b_old from slot 3.
    inline void band_alpha() {
        TTI_SFPLOAD(L1, 0, ADDR_MOD_7, 128);
        TTI_SFPLOAD(L2, 0, ADDR_MOD_7, ST);
        TTI_SFPMAD(L1, ONE, L2, L4, 2);
        TTI_SFPMOV(0, ZERO, L3, 0);
        TTI_SFPGT(0, L4, L3, 8);
        load_const(L5, kU0);
        TTI_SFPMAD(L4, ONE, L5, L5, 0);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L5, C512, 1);
        TTI_SFPSHFT(9, L5, L5, 5);
        TTI_SFPSETEXP(127, L5, L2, 1);
        TTI_SFPMAD(L2, CB2, CB1, L4, 0);
        load_const(L0, kB0);
        TTI_SFPMAD(L4, L2, L0, L4, 0);
        TTI_SFPNOP;
        TTI_SFPSETEXP(0, L4, L5, 2);
        TTI_SFPAND(0, L3, L5, 0);
        TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_FLOATB, 0x3F80);
        TTI_SFPNOT(0, L3, L4, 0);
        TTI_SFPAND(0, L4, L7, 0);
        TTI_SFPOR(0, L5, L7, 0);
    }

    // Pack thread: slot 0 += alpha * slot 1 (alpha from the states in slots 2, 3).
    [[gnu::noinline]] void rescale_section() {
        pack_begin();
    #pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            band_alpha();
            TTI_SFPLOAD(L0, 0, ADDR_MOD_7, 0);
            TTI_SFPLOAD(L1, 0, ADDR_MOD_7, 64);
            TTI_SFPMAD(L1, L7, L0, L0, 0);
            TTI_SFPNOP;
            TTI_SFPSTORE(L0, 0, ADDR_MOD_7, 0);
            TTI_SFPLOAD(L0, 0, ADDR_MOD_7, 2);
            TTI_SFPLOAD(L1, 0, ADDR_MOD_7, 66);
            TTI_SFPMAD(L1, L7, L0, L0, 0);
            TTI_SFPNOP;
            TTI_SFPSTORE(L0, 0, ADDR_MOD_7, 2);
            TTI_SFPLOAD(L0, 0, ADDR_MOD_7, 16);
            TTI_SFPLOAD(L1, 0, ADDR_MOD_7, 80);
            TTI_SFPMAD(L1, L7, L0, L0, 0);
            TTI_SFPNOP;
            TTI_SFPSTORE(L0, 0, ADDR_MOD_7, 16);
            TTI_SFPLOAD(L0, 0, ADDR_MOD_7, 18);
            TTI_SFPLOAD(L1, 0, ADDR_MOD_7, 82);
            TTI_SFPMAD(L1, L7, L0, L0, 0);
            TTI_SFPNOP;
            TTI_SFPSTORE(L0, 0, ADDR_MOD_7, 18);
        }
        math::clear_dst_reg_addr();
        pack_end();
    }

    // Pack thread: slots 0 .. w - 1 *= 1 / slot 3 (the row sums in every column).
    [[gnu::noinline]] void normalize_section(uint32_t w) {
        pack_begin();
    #pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            TTI_SFPLOAD(L7, 0, ADDR_MOD_7, ST + 0);
            TTI_SFPARECIP(0, L7, L1, 0);
            TTI_SFPMAD(L7, L1, ONE, L2, 1);
            TTI_SFPMAD(L1, L2, L1, L1, 0);
            TTI_SFPMAD(L7, L1, ONE, L2, 1);
            TTI_SFPMAD(L1, L2, L1, L7, 0);
    #pragma GCC unroll 1
            for (uint32_t t = 0; t < w; ++t) {
                const uint32_t base = 64 * t;
                TT_SFPLOAD(L0, 0, ADDR_MOD_7, base);
                TTI_SFPMUL(L0, L7, ZERO, L0, 0);
                TT_SFPSTORE(L0, 0, ADDR_MOD_7, base);
                TT_SFPLOAD(L0, 0, ADDR_MOD_7, base + 2);
                TTI_SFPMUL(L0, L7, ZERO, L0, 0);
                TT_SFPSTORE(L0, 0, ADDR_MOD_7, base + 2);
                TT_SFPLOAD(L0, 0, ADDR_MOD_7, base + 16);
                TTI_SFPMUL(L0, L7, ZERO, L0, 0);
                TT_SFPSTORE(L0, 0, ADDR_MOD_7, base + 16);
                TT_SFPLOAD(L0, 0, ADDR_MOD_7, base + 18);
                TTI_SFPMUL(L0, L7, ZERO, L0, 0);
                TT_SFPSTORE(L0, 0, ADDR_MOD_7, base + 18);
            }
        }
        math::clear_dst_reg_addr();
        pack_end();
    }
    }  // namespace fk
    #endif

    // Unpack thread: the step's tile flag (word 257 of held state page `at`; 0 on the
    // row's first chunk) into the front page of token buffer `tok`.
    inline void flag_to_token(uint32_t st, uint32_t at, uint32_t tok, uint32_t first) {
    #ifdef TRISC_UNPACK
        const auto& s = get_local_cb_interface(st);
        const uint32_t from = (s.fifo_rd_ptr + s.fifo_page_size * at) << 4;
        const uint32_t to = get_local_cb_interface(tok).fifo_rd_ptr << 4;
        reinterpret_cast<volatile uint32_t*>(to)[0] =
            first != 0 ? 0u : reinterpret_cast<volatile uint32_t*>(from)[257];
    #endif
    }

    // Pack thread: packer accumulation into L1 on or off, switched only on a change.
    uint32_t acc_on = 0;
    inline void accumulate(uint32_t on) {
    #ifdef TRISC_PACK
        if (on != acc_on) {
            pack_reconfig_l1_acc(on);
            acc_on = on;
        }
    #endif
    }

    // Held fp32 page `index` of `cb` into slot `dst`; the first source then back to
    // the format of `restore`.
    [[gnu::noinline]] void copy_held(
        uint32_t restore, uint32_t cb, uint32_t index, uint32_t dst) {
        reconfig_data_format_srca(restore, cb);
        copy_init(cb);
        copy_tile(cb, index, dst);
        reconfig_data_format_srca(cb, restore);
    }

    // The rescale's operands: held states `now` and `old` into slots 2 and 3, held
    // O page `at` into slot 1.
    [[gnu::noinline]] void copy_rescale(
        uint32_t restore, uint32_t st, uint32_t now, uint32_t old, uint32_t o,
        uint32_t at) {
        reconfig_data_format_srca(restore, st);
        copy_init(st);
        copy_tile(st, now, 2);
        copy_tile(st, old, 3);
        copy_init(o);
        copy_tile(o, at, 1);
        reconfig_data_format_srca(o, restore);
    }
    """),
    clobbers=["lreg11"],
)

# Rows `first` .. 31 of the bf16 tile at L1 `addr` to zero: NoC reads of the core's
# zero region (ktt_flash's reader), complete after a read barrier.
ZERO_ROWS = km.CppFn(
    "flash_km_zero_rows",
    textwrap.dedent("""
    inline void zero_rows(uint32_t addr, uint32_t first) {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t face = 0; face < 4; ++face) {
            const uint32_t r0 = face < 2 ? first : (first > 16 ? first - 16 : 0);
            if (r0 < 16) {
                noc_async_read(zeros, addr + face * 512 + r0 * 32, (16 - r0) * 32);
            }
        }
    }
    """),
)


def raw(k, code, uses=(), sfpu=None, clobbers=(), **args):
    """Inline C++ `code` in kernel `k`, touching buffers `uses`; `args` are the
    kernel values it reads, named in C++."""
    uses = [km.CBUse(cb) for cb in uses]
    k.inline(code, uses=uses, sfpu=sfpu, clobbers=clobbers, args=args)


class Values:
    """Named values of the program (runtime values, sized values, definitions)."""

    def __init__(self, **values):
        self.__dict__.update(values)


def job(k, jj, v):
    """The running core's job `jj` in kernel `k` (every kernel derives it alike): query
    tile rows `rows` of global row `qrow` on, `stride` apart, against key tiles 0 ..
    nk - 1 of K / V row `krow` in `nb` blocks; `real` is false on a padding job."""
    length = v.length
    if v.causal:
        # Chunk `chunk` of head slot * lanes + lane: rows [c Sq / nc, (c + 1) Sq / nc).
        chunk = k.let("chunk", v.nc - 1 - jj // v.slots)
        head = k.let("head", jj % v.slots * v.lanes + v.lane)
        real = head < v.bh
        r1 = k.let("r1", (chunk + 1) * v.sqt // v.nc)
        first = k.let("first", chunk * v.sqt // v.nc + v.yy)
        count = (r1 - km.min(first, r1) + length - 1) // length
        rows = km.select(real, count, 0)
        start, stride, nk = km.min(first, r1), length, r1
        nb = k.let("nb", (r1 + v.kb - 1) // v.kb)
    else:
        # Chunk jj of the lane's rows: the first head's part, whole heads, the last
        # head's part, each cut in chunks of `span` rows.
        one, two = jj < v.c1, jj < v.c2
        later = km.select(one, 0, jj - v.c1)
        whole = km.select(two, v.h_lo + 1 + later // v.cph, v.h_hi)
        head = k.let("head", km.select(one, v.h_lo, whole))
        real = jj < v.chunks
        seg0 = km.select(one, v.lo, head * v.sqt)
        seg1 = km.select(one, v.lo + v.f, km.select(two, head * v.sqt + v.sqt, v.hi))
        nth = km.select(one, jj, km.select(two, later % v.cph, jj - v.c2))
        pos = k.let("pos", seg0 + nth * v.span)
        n = k.let("n", km.select(real, km.min(v.span, seg1 - pos), 0))
        r0 = k.let("r0", pos - head * v.sqt)
        per = k.let("per", (n + length - 1) // length)
        first = k.let("first", km.min(r0 + v.yy * per, r0 + n))
        rows = km.select(real, km.min(per, r0 + n - first), 0)
        start, stride, nk, nb = first, 1, v.skt, v.nb
    head = k.let("hd", km.select(real, head, 0))
    start = k.let("start", start)
    kv_row = (head // v.heads * v.hk + head % v.heads // v.group) * v.skt
    return Values(
        real=real,
        rows=k.let("rows", rows),
        qrow=k.let("qrow", head * v.sqt + start),
        krow=k.let("krow", kv_row),
        start=start,
        stride=stride,
        nk=nk,
        nb=nb,
    )


def block(k, j, blk, v):
    """Block `blk` of job `j`: key tiles kstart .. kstart + kbn - 1; per sub-chunk of
    3 tiles its first visible row (`lows`) and the steps up to its end (`ends`); the
    steps of sub-chunks with two or more rows go in pairs, then the rest singly."""
    kstart = k.let("kstart", blk * v.kb)
    kbn = k.let("kbn", km.min(v.kb, j.nk - kstart))
    nsub = (kbn + SC - 1) // SC
    lows, counts = [], []
    for sub in range(NSUB):
        if v.causal:
            kg = kstart + SC * sub
            behind = km.select(kg > j.start, kg - j.start, 0)
            low = k.var(f"low{sub}", (behind + j.stride - 1) // j.stride)
            seen = km.select(low < j.rows, j.rows - low, 0)
        else:
            low, seen = 0, j.rows
        lows.append(low)
        counts.append(k.var(f"seen{sub}", km.select(nsub > sub, seen, 0)))
    ends = [counts[0]]
    for sub in range(1, NSUB):
        ends.append(k.var(f"end{sub}", ends[-1] + counts[sub]))
    many = km.select(counts[0] > 1, counts[0], 0)
    ones = km.select(counts[0].eq(1), 1, 0)
    for count in counts[1:]:
        many = many + km.select(count > 1, count, 0)
        ones = ones + km.select(count.eq(1), 1, 0)
    many = k.var("many", many)
    return Values(
        kstart=kstart,
        kbn=kbn,
        lows=lows,
        ends=ends,
        pairs=k.var("pairs", many // 2),
        singles=k.var("singles", many % 2 + ones),
        rest=k.var("rest", many // 2 * 2),
    )


def step(k, j, b, s, v):
    """Step `s` (in sub-chunk, row order) of block `b` of job `j`: its row `r`, chunk
    `c` along the keys, first key tile `k0` in the block, score tiles `nt`, state
    pages `slot` and `prev`, mask page `mask` (0: none), `first` chunk of the row."""
    sub = s >= b.ends[0]
    for end in b.ends[1 : NSUB - 1]:
        sub = sub + (s >= end)
    sub = k.var("sub", sub)
    low, before = b.lows[NSUB - 1], b.ends[NSUB - 2]
    for at in range(NSUB - 2, -1, -1):
        low = km.select(sub.eq(at), b.lows[at], low)
    for at in range(NSUB - 2, 0, -1):
        before = km.select(sub.eq(at), b.ends[at - 1], before)
    before = km.select(sub.eq(0), 0, before)
    r = k.var("r", low + s - before)
    c = k.var("kc", b.kstart // SC + sub)
    k0 = k.var("k0", SC * sub)
    kg = b.kstart + k0
    rowg = j.start + r * j.stride
    nt = km.min(SC, b.kbn - k0)
    if v.causal:
        nt = km.min(nt, km.min(rowg, j.nk - 1) - kg + 1)
    nt = k.var("nt", nt).bounded(1, SC + 1)
    tail = km.select(v.tail.eq(0), 0, km.select((kg + nt).eq(v.skt), TAIL, 0))
    mask = tail
    if v.causal:
        diag = (kg + nt - 1).eq(rowg)
        mask = km.select(diag, km.select(tail.eq(0), CAUSAL, BOTH), tail)
    return Values(
        row=r,
        c=c,
        k0=k0,
        nt=nt,
        first=c.eq(0),
        mask=k.var("mask", mask),
        slot=k.var("slot", 2 * r + c % 2),
        prev=k.var("prev", 2 * r + (c + 1) % 2),
    )


def declare(p, causal, length):
    """The runtime values, sized values and per-core definitions of the program."""
    cols, rows = p.machine.cols, p.machine.rows
    if rows % length:
        raise ValueError("flash_km: the lane length divides the grid's rows")
    per_col = rows // length
    names = ["dt", "dvt", "r", "kb", "sqt", "skt", "bh", "heads", "group", "hk"]
    names += ["jobs", "tail", "a", "thr"]
    names += ["nc", "slots"] if causal else ["total", "span", "cph", "nb"]
    v = Values(**{name: p.runtime(name) for name in names})
    v.causal, v.length, v.lanes = causal, length, cols * per_col
    x, y = p.core % cols, p.core // cols
    v.lane = p.define("lane", x * per_col + y // length)
    v.yy = p.define("yy", y % length)
    if not causal:
        # The lane's rows [lo, hi) of the (batch, head, row) order: c1 chunks of the
        # first head's part, up to c2 with the whole heads, then the last head's part.
        v.lo = p.define("lane_lo", v.lane * v.total // v.lanes)
        v.hi = p.define("lane_hi", (v.lane + 1) * v.total // v.lanes)
        v.h_lo = p.define("head_lo", v.lo // v.sqt)
        v.f = p.define("head_rows", km.min(v.hi, (v.h_lo + 1) * v.sqt) - v.lo)
        last_head = km.select(v.hi > v.lo, (v.hi + v.sqt - 1) // v.sqt - 1, v.h_lo)
        v.h_hi = p.define("head_hi", last_head)
        more = v.h_hi > v.h_lo
        v.c1 = p.define("chunks_lo", (v.f + v.span - 1) // v.span)
        whole = km.select(more, v.h_hi - v.h_lo - 1, 0) * v.cph
        v.c2 = p.define("chunks_mid", v.c1 + whole)
        last = (v.hi - v.h_hi * v.sqt + v.span - 1) // v.span
        v.chunks = p.define("chunks", v.c2 + km.select(more, last, 0))
    v.q_pages = p.sized("q_pages", of=v.r * v.dt, exact=True, limit=R_MOST * DT_MOST)
    # Key block pages: at least 3 key tiles of K and V (a step's tiles fit).
    blocks = range(2 * SC, 2 * KB_MOST * DT_MOST + 1)
    v.kv_pages = p.sized("kv_pages", of=v.kb * (v.dt + v.dvt), classes=blocks)
    v.st_pages = p.sized("st_pages", of=2 * v.r, exact=True, limit=2 * R_MOST)
    v.o_pages = p.sized(
        "o_pages", of=v.r * (v.dvt + 1), exact=True, limit=R_MOST * (DT_MOST + 1)
    )
    v.dvs = p.sized("dvs", of=v.dvt, exact=True, limit=DT_MOST)
    return v


def reader(k, role, q, kt, vt, cb, v, lane=None):
    """Reader kernel `k`: per job its query rows, then its key blocks, read ("read"),
    read and multicast down the lane ("send") or taken from the lane ("receive");
    K tile (key t, d) at page d kb + t of a block, V tile (t, e) at kb dt + t dvt + e.
    """
    if role == "send":
        lane.open(k)
    if role != "receive":
        k.helper(ZERO_ROWS)
    with k.loop(v.jobs, name="jj") as jj:
        j = job(k, jj, v)
        slot = k.reserve(cb.q, v.q_pages)
        with k.loop(j.rows, name="i") as i, k.loop(v.dt, name="d") as d:
            at = (i * v.dt + d).bounded(0, v.q_pages)
            k.read(q.page((j.qrow + i * j.stride) * v.dt + d), slot[at])
        k.reads_done()
        k.push(cb.q, v.q_pages)
        with k.loop(j.nb, name="blk") as blk:
            kstart = k.let("kstart", blk * v.kb)
            kbn = k.let("kbn", km.min(v.kb, j.nk - kstart))
            kv = k.reserve(cb.kv, v.kv_pages)
            if role == "receive":
                lane.receive(k)
            else:
                with k.when(j.real):
                    with k.loop(kbn, name="t") as t, k.loop(v.dt, name="d") as d:
                        at = (d * v.kb + t).bounded(0, v.kv_pages)
                        k.read(kt.page((j.krow + kstart + t) * v.dt + d), kv[at])
                    with k.loop(kbn, name="t") as t, k.loop(v.dvt, name="e") as e:
                        at = (v.kb * v.dt + t * v.dvt + e).bounded(0, v.kv_pages)
                        k.read(vt.page((j.krow + kstart + t) * v.dvt + e), kv[at])
                k.reads_done()
                # Rows past the last key of a ragged last key tile: K and V to zero.
                last = (kstart + kbn).eq(v.skt)
                with k.when(j.real), k.when(v.tail.ne(0)), k.when(last):
                    raw(
                        k,
                        ZERO_CODE,
                        uses=[cb.kv],
                        km_kv=kv,
                        km_last=kbn - 1,
                        km_kb=v.kb,
                        km_dt=v.dt,
                        km_dv=v.dvt,
                        km_tail=v.tail,
                    )
                    k.reads_done()
                if role == "send":
                    lane.send(k, kv)
            k.push(cb.kv, v.kv_pages)
    if role == "receive":
        k.atomics_done()
    else:
        k.writes_done()


ZERO_CODE = """
for (uint32_t d = 0; d < km_dt; ++d) {
    zero_rows(km_kv + (d * km_kb + km_last) * 2048, km_tail);
}
for (uint32_t e = 0; e < km_dv; ++e) {
    zero_rows(km_kv + (km_kb * km_dt + km_last * km_dv + e) * 2048, km_tail);
}
"""


def score(c, cb, v, s):
    """Step `s`'s scores: the row's state and mask page into the registers, q k^T
    added onto them, the lane maxima; on the pack thread the row state and P; P (bf16)
    and the state packed, a token pushed."""
    c.acquire()
    raw(c, "copy_held(cb_kv, cb_state, km_at, 3);", uses=[cb.kv, cb.st], km_at=s.prev)
    with c.when(s.mask.ne(0)):
        c.init("copy", cb.const)
        c.copy(cb.const, s.mask.bounded(0, 4), s.nt - 1)
    m = c.matmul_init(cb.q, cb.kv, transpose=1, ct=s.nt, rt=1, kt=v.dt)
    with c.loop(v.dt, name="d") as d:
        ia = (s.row * v.dt + d).bounded(0, v.q_pages)
        ib = (d * v.kb + s.k0).bounded(0, v.kv_pages - s.nt + 1)
        c.matmul(m, ia, ib, 0)
    raw(c, "MATH((fk::max_section(km_nt)));", sfpu="math", km_nt=s.nt)
    c.commit()
    c.reserve(cb.p, SC)
    c.reserve(cb.tok, 1)
    code = "PACK((fk::exp_section(km_first, km_nt, km_thr)));"
    raw(c, code, sfpu="pack", km_first=s.first, km_nt=s.nt, km_thr=v.thr)
    raw(c, "accumulate(0);")
    c.pack_format(cb.p)
    with c.loop(SC, name="t") as t:
        c.pack(t, cb.p, at=t)
    c.pack_format(cb.st)
    c.pack(3, cb.st, at=s.slot.bounded(0, v.st_pages))
    c.release()
    c.push(cb.p, SC)
    c.push(cb.tok, 1)


def output(c, cb, v, s):
    """Step `s`'s output: O and l of its row += P V and P 1 by packer accumulation
    (written on the row's first chunk), or, when a row of the tile moved its reference,
    column by column alpha O + P V on the pack thread."""
    width = v.dvt + 1
    c.wait(cb.tok, 1)
    c.wait(cb.p, SC)
    code = "flag_to_token(cb_state, km_slot, cb_steps, km_first);"
    raw(c, code, uses=[cb.st, cb.tok], km_slot=s.slot, km_first=s.first)
    moved = c.read_word(cb.tok, 0, name="moved")
    with c.when(moved.eq(0)):
        raw(c, "accumulate(km_add);", km_add=km.select(s.first, 0, 1))
        with c.block_loop(width, 4, names=("j0", "w")) as (j0, w):
            wv = c.let("wv", km.select(j0 < v.dvt, km.min(w, v.dvt - j0), 0))
            c.acquire()
            with c.when(wv.ne(0)):
                m = c.matmul_init(cb.p, cb.kv, ct=wv, rt=1, kt=s.nt)
                with c.loop(s.nt, name="t") as t:
                    ib = v.kb * v.dt + (s.k0 + t) * v.dvt + j0
                    c.matmul(m, t, ib.bounded(0, v.kv_pages - wv + 1), 0)
            with c.when(v.dvt < j0 + w):
                m = c.matmul_init(cb.p, cb.const, ct=1, rt=1, kt=s.nt)
                with c.loop(s.nt, name="t") as t:
                    c.matmul(m, t, ONES, wv.bounded(0, 4))
            c.commit()
            with c.loop(w, name="t") as t:
                c.pack(t, cb.o, at=(s.row * width + j0 + t).bounded(0, v.o_pages))
            c.release()
    with c.otherwise(), c.loop(width, name="col") as col:
        rescaled(c, cb, v, s, col)
    c.pop(cb.p, SC)
    c.pop(cb.tok, 1)


def rescaled(c, cb, v, s, col):
    """Column `col` of step `s`'s row (an O column, or l at dvt) = alpha times its old
    value + its column of P V (or P 1), alpha from the step's old and new state."""
    at = s.row * (v.dvt + 1) + col
    c.acquire()
    with c.when(col < v.dvt):
        m = c.matmul_init(cb.p, cb.kv, ct=1, rt=1, kt=s.nt)
        with c.loop(s.nt, name="t") as t:
            ib = v.kb * v.dt + (s.k0 + t) * v.dvt + col
            c.matmul(m, t, ib.bounded(0, v.kv_pages), 0)
    with c.otherwise():
        m = c.matmul_init(cb.p, cb.const, ct=1, rt=1, kt=s.nt)
        with c.loop(s.nt, name="t") as t:
            c.matmul(m, t, ONES, 0)
    code = "copy_rescale(cb_kv, cb_state, km_now, km_old, cb_acc, km_at);"
    uses = [cb.kv, cb.st, cb.o]
    raw(c, code, uses=uses, km_now=s.slot, km_old=s.prev, km_at=at)
    c.commit()
    raw(c, "PACK((fk::rescale_section()));", sfpu="pack")
    raw(c, "accumulate(0);")
    c.pack(0, cb.o, at=at.bounded(0, v.o_pages))
    c.release()


def finalize(c, cb, v, row):
    """Output row `row` = O / l (bf16), three columns per register scope."""
    width = v.dvt + 1
    c.reserve(cb.out, v.dvs)
    with c.block_loop(v.dvt, 3, names=("f0", "fw")) as (f0, fw):
        c.acquire()
        c.format(srca=cb.o)
        c.init("copy", cb.o)
        with c.loop(fw, name="t") as t:
            c.copy(cb.o, (row * width + f0 + t).bounded(0, v.o_pages), t)
        c.copy(cb.o, (row * width + v.dvt).bounded(0, v.o_pages), 3)
        c.format(srca=cb.kv)
        c.commit()
        raw(c, "PACK((fk::normalize_section(km_w)));", sfpu="pack", km_w=fw)
        raw(c, "accumulate(0);")
        c.pack_format(cb.out)
        with c.loop(fw, name="t") as t:
            c.pack(t, cb.out, at=(f0 + t).bounded(0, v.dvs))
        c.release()
    c.push(cb.out, v.dvs)


def compute(p, cb, v):
    """The compute kernel: the constant tiles, then per job and key block the steps of
    its rows, two at a time (both scores, then both outputs), then the rows' output."""
    c = p.kernel("compute", fp32_dest=True, approx=False, unpack_fp32=[cb.st, cb.o])
    c.helper(VECTOR)
    renames = []
    if MATH_CLEAR:
        c.include("api/compute/experimental/reg_api.h")
        renames += ["tile_regs_acquire tile_regs_acquire_math_clear"]
        renames += ["tile_regs_release tile_regs_release_math_clear"]
    if NO_MOP:
        c.include("api/compute/experimental/matmul_custom.h")
        renames += ["matmul_block_init mm_no_mop_init_short"]
        renames += ["matmul_block matmul_block_no_mop"]
    if renames:
        code = "\n".join(f"#define {rename}" for rename in renames)
        c.helper(km.CppFn("flash_km_renames", code))
    c.startup(cb.q, cb.kv, cb.p, reverse=True)
    raw(c, "MATH((fk::math_init(km_a)));", sfpu="math", clobbers=["lreg11"], km_a=v.a)
    raw(c, "PACK((fk::addr_init()));", sfpu="pack")
    c.acquire()
    raw(c, "MATH((fk::const_section(km_tail)));", sfpu="math", km_tail=v.tail)
    c.commit()
    c.reserve(cb.const, 4)
    for t in range(4):
        c.pack(t, cb.const)
    c.release()
    c.push(cb.const, 4)
    c.wait(cb.const, 4)
    c.hold(cb.st)
    c.hold(cb.o)
    with c.loop(v.jobs, name="jj") as jj:
        j = job(c, jj, v)
        c.wait(cb.q, v.q_pages)
        with c.loop(j.nb, name="blk") as blk:
            b = block(c, j, blk, v)
            c.wait(cb.kv, v.kv_pages)
            # Groups of steps: the pairs, then the single steps; a group's scores,
            # then its outputs.
            zone = c.zone if ZONES else (lambda name: contextlib.nullcontext())
            c.pack_format(cb.o)
            with c.loop(b.pairs + b.singles, name="group") as g:
                paired = g < b.pairs
                size = c.var("size", km.select(paired, 2, 1))
                first = c.var("s0", km.select(paired, 2 * g, b.rest + g - b.pairs))
                for name, part in (("S", score), ("O", output)):
                    with c.loop(size, name="half") as half:
                        s = step(c, j, b, first + half, v)
                        with zone(name):
                            part(c, cb, v, s)
            c.pop(cb.kv, v.kv_pages)
        c.fence(cb.done)
        with c.loop(j.rows, name="row") as row:
            finalize(c, cb, v, row)
        c.pop(cb.q, v.q_pages)
    raw(c, "accumulate(0);")
    c.pop(cb.const, 4)


def writer(p, out, cb, v):
    """The writer kernel: each job's output rows."""
    w = p.kernel("dm1", noc=1)
    with w.loop(v.jobs, name="jj") as jj:
        j = job(w, jj, v)
        with w.loop(j.rows, name="i") as i:
            w.wait(cb.out, v.dvs)
            with w.loop(v.dvt, name="e") as e:
                w.write(cb.out, out.page((j.qrow + i * j.stride) * v.dvt + e), e)
            w.flushed()
            w.pop(cb.out, v.dvs)
    w.writes_done()


@km.program("flash")
def flash_program(
    p,
    q: km.TensorRef,
    k: km.TensorRef,
    v: km.TensorRef,
    out: km.Out,
    causal=False,
    length=5,
):
    """softmax(q k^T scale [+ causal mask]) v for bf16 q [B, H, Sq, D] and k, v [B, Hk,
    Sk, D] on the whole grid in lanes of `length` cores down each column (a divisor of
    the grid's rows); runtime values as plan() gives them."""
    vals = declare(p, causal, length)
    cb = Values(
        q=p.cb(0, like=q, pages=vals.q_pages),
        kv=p.cb(1, like=k, pages=2 * vals.kv_pages, name="kv"),
        const=p.cb(3, dtype="bfloat16", pages=4, name="consts"),
        p=p.cb(4, dtype="bfloat16", pages=2 * SC, name="probs"),
        st=p.cb(5, dtype="float32", pages=vals.st_pages, name="state"),
        o=p.cb(6, dtype="float32", pages=vals.o_pages, name="acc"),
        tok=p.words(7, pages=4, name="steps"),
        done=p.words(8, pages=2, name="done"),
        out=p.cb(16, like=out, pages=2 * vals.dvs),
    )
    rows, length = p.machine.rows, vals.length
    if length == 1:
        reader(p.kernel("dm0", noc=0), "read", q, k, v, cb, vals)
    for g in range(rows // length if length > 1 else 0):
        top = p.cores.rows(g * length)
        below = p.cores.rows(g * length + 1, (g + 1) * length - 1)
        lane = p.lane(top, below, name=f"kv{g}")
        send = p.kernel("dm0", cores=top, noc=0, name=f"kv_send{g}")
        reader(send, "send", q, k, v, cb, vals, lane)
        recv = p.kernel("dm0", cores=below, noc=0, name=f"kv_recv{g}")
        reader(recv, "receive", q, k, v, cb, vals, lane)
    compute(p, cb, vals)
    writer(p, out, cb, vals)


def blocks(dt, dvt, budget):
    """ktt_flash's query tile rows per job and key tiles per block for head-dim tiles
    dt, dvt: the largest under `budget` bytes of buffers per core; (0, 0): none."""

    def size(r, kb):
        bf16 = r * dt + 2 * dt * kb + 2 * kb * dvt + 1 + 2 * SC + 2 * dvt + 4
        return bf16 * TILE + (r * (dvt + 1) + 2 * r) * TILE32 + (r + 8) * TOKEN

    r, kb = R_MOST, KB_MOST
    while size(r, kb) > budget:
        if kb > 6:
            kb -= SC
        elif r > 4:
            r -= 1
        elif kb > SC:
            kb -= SC
        elif r > 1:
            r -= 1
        else:
            return 0, 0
    return r, kb


def lane_chunks(bh, sqt, span, lanes):
    """Per lane its chunks (head, first row, rows): even ranges of the (batch, head,
    row) order cut at heads and every `span` rows (ktt_flash's non-causal plan)."""
    total, out = bh * sqt, []
    for i in range(lanes):
        lo, hi = i * total // lanes, (i + 1) * total // lanes
        chunks, pos = [], lo
        while pos < hi:
            head, r0 = divmod(pos, sqt)
            n = min(hi - pos, sqt - r0, span)
            chunks.append((head, r0, n))
            pos += n
        out.append(chunks)
    return out


def causal_chunks(bh, sqt, span, lanes):
    """Per lane its chunks (head, first row, rows) in the causal plan: chunk c of
    head s lanes + lane at job (nc - 1 - c) slots + s, None past the heads."""
    nc, slots = -(-sqt // span), -(-bh // lanes)
    out = []
    for lane in range(lanes):
        chunks = []
        for jj in range(nc * slots):
            c, head = nc - 1 - jj // slots, jj % slots * lanes + lane
            r0, r1 = c * sqt // nc, (c + 1) * sqt // nc
            chunks.append((head, r0, r1 - r0) if head < bh else None)
        out.append(chunks)
    return out


def busiest(causal, bh, sqt, skt, r, length, rows, cols):
    """(score tiles of the busiest core, jobs per lane) of the plan in lanes of
    `length` cores on a grid of `cols` x `rows`."""
    lanes, span = cols * (rows // length), length * r
    chunks = (causal_chunks if causal else lane_chunks)(bh, sqt, span, lanes)
    worst = 0
    for lane in chunks:
        for yy in range(length):
            tiles = 0
            for chunk in lane:
                if chunk is None:
                    continue
                _, r0, n = chunk
                if causal:
                    mine = range(r0 + yy, r0 + n, length)
                    tiles += sum(min(row, skt - 1) + 1 for row in mine)
                else:
                    per = -(-n // length)
                    first = min(r0 + yy * per, r0 + n)
                    tiles += min(per, r0 + n - first) * skt
            worst = max(worst, tiles)
    return worst, max(len(lane) for lane in chunks)


def bits(value):
    """The fp32 bits of `value`."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32(value):
    """`value` rounded to fp32."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def plan(q_shape, k_shape, v_shape, causal, scale, budget, cols, rows):
    """(runtime values, lane length) of flash_program for these shapes on a grid of
    `cols` x `rows` cores: ktt_flash's blocks, lane length and (non-causal) jobs."""
    b, h, sq, d = q_shape
    hk, sk, dv = k_shape[1], k_shape[2], v_shape[3]
    dt, dvt, sqt, skt = d // 32, dv // 32, -(-sq // 32), -(-sk // 32)
    r, kb = blocks(dt, dvt, budget)
    if r == 0:
        raise ValueError("flash_km: the buffers do not fit L1")
    best, length, jobs = None, 1, 0
    for size in (n for n in range(1, rows + 1) if rows % n == 0):
        tiles, count = busiest(causal, b * h, sqt, skt, r, size, rows, cols)
        if (causal or count <= 46) and (best is None or tiles <= best):
            best, length, jobs = tiles, size, count
    span, lanes = length * r, cols * (rows // length)
    sc = f32(scale if scale is not None else 1.0 / math.sqrt(d))
    values = {
        "dt": dt,
        "dvt": dvt,
        "r": r,
        "kb": kb,
        "sqt": sqt,
        "skt": skt,
        "bh": b * h,
        "heads": h,
        "group": h // hk,
        "hk": hk,
        "jobs": jobs,
        "tail": sk % 32,
        "a": bits(f32(sc * LOG2E)),
        "thr": bits(655.0),
    }
    if causal:
        values.update(nc=-(-sqt // span), slots=-(-(b * h) // lanes))
    else:
        values.update(total=b * h * sqt, span=span, cph=-(-sqt // span))
        values.update(nb=-(-skt // kb))
    return values, length


SCHEMA = (
    "kohakutt_example::flash_km(Tensor q, Tensor k, Tensor v, bool causal=False, "
    "float? scale=None) -> Tensor"
)


@km.op(SCHEMA, flash_program)
def flash(q, k, v, causal=False, scale=None):
    """The launch of softmax(q k^T scale [+ causal mask]) v for bf16 tt tensors q [B,
    H, Sq, D] and k, v [B, Hk, Sk, D] (D a multiple of 32 up to 256, Hk dividing H;
    causal: Sq == Sk); `scale` defaults to 1 / sqrt(D)."""
    shapes = [tuple(t.shape) for t in (q, k, v)]
    if any(t.dtype != torch.bfloat16 or t.dim() != 4 for t in (q, k, v)):
        raise TypeError("flash_km: q, k and v are 4-D bfloat16")
    (b, h, sq, d), (bk, hk, sk, dk), (bv, hv, sv, dv) = shapes
    fits = d % 32 == 0 and dv % 32 == 0 and d <= 32 * DT_MOST and dv <= 32 * DT_MOST
    if not (
        fits and b == bk == bv and hk == hv and h % hk == 0 and sk == sv and dk == d
    ):
        raise ValueError(f"flash_km: shapes {shapes} do not match")
    if causal and sq != sk:
        raise ValueError("flash_km: causal needs Sq == Sk")
    cols, rows = torch.ops.ktt.compute_grid()
    budget = int(torch.ops.ktt.cb_budget())
    values, length = plan(*shapes, causal, scale, budget, cols, rows)
    return km.Call(
        inputs={"q": q, "k": k, "v": v},
        outputs={"out": ([b, h, sq, dv], q.dtype)},
        runtime=values,
        static={"causal": bool(causal), "length": length},
    )


def reference(q, k, v, causal):
    """softmax(q k^T / sqrt(D) [+ causal mask]) v in float64 on the host."""
    q, k, v = (t.double() for t in (q, k, v))
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), -math.inf)
    return torch.softmax(s, -1) @ v


def main():
    """Runs flash_km and the backend's attention on tt tensors, non-causal and causal at
    head dims 64 and 128; prints whether they agree bit for bit and their errors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=300)
    args = parser.parse_args()
    dev = kohakutt.device(0)
    try:
        for dim in (64, 128):
            for causal in (False, True):
                torch.manual_seed(dim + causal)
                shape = (args.batch, args.heads, args.seq, dim)
                q, k, v = (torch.randn(shape).bfloat16() for _ in range(3))
                qt, kt, vt = (t.to(dev) for t in (q, k, v))
                got = torch.ops.kohakutt_example.flash_km(qt, kt, vt, causal).cpu()
                want = torch.nn.functional.scaled_dot_product_attention(
                    qt, kt, vt, is_causal=causal
                ).cpu()
                exact = reference(q, k, v, causal)
                ours = (got.double() - exact).abs().max().item()
                theirs = (want.double() - exact).abs().max().item()
                print(
                    f"flash_km {shape} causal={causal}: equal to the backend: "
                    f"{torch.equal(got, want)}; max error vs float64 {ours:.3g} "
                    f"(backend {theirs:.3g})"
                )
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
