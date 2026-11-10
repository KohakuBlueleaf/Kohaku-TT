// Complex multiply for the broadcast binary compute kernel on fp32 tiles holding
// (re, im) at columns (2 i, 2 i + 1): an SFPU register loads the even or the odd
// columns.
#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "ckernel.h"
#include "ckernel_ops.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

namespace ktt_binary {

    // One face (16 Dst rows) of tile `out` = tile `in0` * tile `in1` as complex
    // numbers; CONJ_A / CONJ_B conjugate that operand. Every multiply and add rounds to
    // fp32.
    template <bool CONJ_A, bool CONJ_B>
    inline void complex_mul_face(uint32_t in0, uint32_t in1, uint32_t out) {
        using namespace ckernel;
        constexpr uint32_t L0 = p_sfpu::LREG0, L1 = p_sfpu::LREG1, L2 = p_sfpu::LREG2;
        constexpr uint32_t L3 = p_sfpu::LREG3, L4 = p_sfpu::LREG4, L5 = p_sfpu::LREG5;
        constexpr uint32_t L6 = p_sfpu::LREG6, L7 = p_sfpu::LREG7;
        constexpr uint32_t zero = p_sfpu::LCONST_0, one = p_sfpu::LCONST_1;
        // A tile is 64 Dst rows; +2 selects the odd columns.
        const uint32_t a = in0 * 64, b = in1 * 64, o = out * 64;
        // SFPMUL's modifier 1 negates the product. With s_a, s_b the signs of the
        // operands' imaginary parts: re = ar br - s_a s_b ai bi, im = s_b ar bi + s_a
        // ai br.
        constexpr uint32_t neg_ii = CONJ_A == CONJ_B ? 1 : 0;
        constexpr uint32_t neg_ri = CONJ_B ? 1 : 0;
        constexpr uint32_t neg_ir = CONJ_A ? 1 : 0;
        for (uint32_t group = 0; group < 4; ++group) {
            TT_SFPLOAD(L0, 0, ADDR_MOD_7, a);
            TT_SFPLOAD(L1, 0, ADDR_MOD_7, a + 2);
            TT_SFPLOAD(L2, 0, ADDR_MOD_7, b);
            TT_SFPLOAD(L3, 0, ADDR_MOD_7, b + 2);
            TTI_SFPMUL(L0, L2, zero, L4, 0);
            TTI_SFPMUL(L1, L3, zero, L5, neg_ii);
            TTI_SFPMUL(L0, L3, zero, L6, neg_ri);
            TTI_SFPMUL(L1, L2, zero, L7, neg_ir);
            TTI_SFPADD(one, L4, L5, L4, 0);
            TTI_SFPADD(one, L6, L7, L6, 0);
            TT_SFPSTORE(L4, 0, ADDR_MOD_7, o);
            TT_SFPSTORE(L6, 0, ADDR_MOD_7, o + 2);
            TTI_INCRWC(0, 4, 0, 0);
        }
    }

    // Tile `odst` = tile `idst0` * tile `idst1` of Dst as interleaved complex pairs.
    template <bool CONJ_A, bool CONJ_B>
    ALWI void complex_mul_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
        MATH((_llk_math_eltwise_binary_sfpu_params_(
            complex_mul_face<CONJ_A, CONJ_B>, idst0, idst1, odst, VectorMode::RC
        )));
    }

}  // namespace ktt_binary
