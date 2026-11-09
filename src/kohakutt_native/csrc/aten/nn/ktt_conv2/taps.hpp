// Device-side helpers shared by the gather conv's reader, compute and writer: the
// shape from the common runtime args, channel-tile ranges and the taps of a phase.
#pragma once

#include <stdint.h>

#include "../ktt_conv/args.hpp"

namespace ktt_conv2 {

    // True when compile-time arg `ct` and common runtime arg 0 both hold this source's
    // argument-layout version.
    template <uint32_t ct>
    inline bool args_match() {
        return get_compile_time_arg_val(ct) == ktt_conv::kArgsVersion &&
            get_common_arg_val<uint32_t>(0) == ktt_conv::kArgsVersion;
    }

    constexpr uint32_t kMaxTaps = 16;

    // Element offset of (r, c) in a 32x32 tile stored as four 16x16 faces.
    inline uint32_t face_offset(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    // Common runtime args from `at` on: Ot, Kmax, Cg, Og, O, KH, KW, K (channel tile
    // pages per tap), nt max, phase tiles, Sw, the most taps of an h phase, then axis h
    // and axis w (each: phases S, S tap counts, each phase's (offset, kernel index)
    // pairs).
    struct Shape {
        uint32_t ot, kmax, cg, og, o, kh, kw, kpages, nt_max, phase_tiles, phases_w;
        uint32_t rows_h, axis_h, axis_w;
    };

    // Args of the axis at `base`: its length.
    inline uint32_t axis_args(uint32_t base) {
        const uint32_t phases = get_common_arg_val<uint32_t>(base);
        uint32_t total = 1 + phases;
        for (uint32_t f = 0; f < phases; ++f) {
            total += 2 * get_common_arg_val<uint32_t>(base + 1 + f);
        }
        return total;
    }

    inline Shape read_shape(uint32_t at) {
        Shape s;
        s.ot = get_common_arg_val<uint32_t>(at++);
        s.kmax = get_common_arg_val<uint32_t>(at++);
        s.cg = get_common_arg_val<uint32_t>(at++);
        s.og = get_common_arg_val<uint32_t>(at++);
        s.o = get_common_arg_val<uint32_t>(at++);
        s.kh = get_common_arg_val<uint32_t>(at++);
        s.kw = get_common_arg_val<uint32_t>(at++);
        s.kpages = get_common_arg_val<uint32_t>(at++);
        s.nt_max = get_common_arg_val<uint32_t>(at++);
        s.phase_tiles = get_common_arg_val<uint32_t>(at++);
        s.phases_w = get_common_arg_val<uint32_t>(at++);
        s.rows_h = get_common_arg_val<uint32_t>(at++);
        s.axis_h = at;
        s.axis_w = at + axis_args(at);
        return s;
    }

    // Input channel tiles [k0, k1) of output column tile j.
    inline void channel_tiles(uint32_t j, const Shape& s, uint32_t& k0, uint32_t& k1) {
        const uint32_t lo = 32 * j, hi = 32 * j + 31 < s.o ? 32 * j + 31 : s.o - 1;
        k0 = (lo / s.og) * s.cg / 32;
        k1 = ((hi / s.og + 1) * s.cg - 1) / 32 + 1;
    }

    // Phase f's taps of the axis at `base`: offsets and kernel indices; the count.
    inline uint32_t axis_taps(uint32_t base, uint32_t f, uint32_t* off, uint32_t* idx) {
        const uint32_t phases = get_common_arg_val<uint32_t>(base);
        uint32_t at = base + 1 + phases;
        for (uint32_t g = 0; g < f; ++g) {
            at += 2 * get_common_arg_val<uint32_t>(base + 1 + g);
        }
        const uint32_t count = get_common_arg_val<uint32_t>(base + 1 + f);
        for (uint32_t i = 0; i < count; ++i) {
            off[i] = get_common_arg_val<uint32_t>(at + 2 * i);
            idx[i] = get_common_arg_val<uint32_t>(at + 2 * i + 1);
        }
        return count;
    }

    // The taps of an output row tile's phase: nh rows x nw columns; tap (a, b) reads
    // offsets (roff[a], coff[b]) and kernel tap ridx[a] KW + cidx[b]; clo / chi: the
    // column offsets' range.
    struct Phase {
        uint32_t nh = 0, nw = 0, clo = 0, chi = 0;
        uint32_t roff[kMaxTaps], ridx[kMaxTaps], coff[kMaxTaps], cidx[kMaxTaps];
    };

    // A gather work item (per-core args after the item count, kItemArgs each): row tiles
    // [m0, m1), column tiles [n0, n0 + nt), tap rows [a0, a1) of each phase, and the
    // output row-tile base mb (row tile m is written as mb + m).
    struct Item {
        uint32_t m0, m1, n0, nt, a0, a1, mb;
    };
    constexpr uint32_t kItemArgs = 7;

    inline Item read_item(uint32_t it) {
        const uint32_t at = 1 + kItemArgs * it;
        return Item{
            get_arg_val<uint32_t>(at),
            get_arg_val<uint32_t>(at + 1),
            get_arg_val<uint32_t>(at + 2),
            get_arg_val<uint32_t>(at + 3),
            get_arg_val<uint32_t>(at + 4),
            get_arg_val<uint32_t>(at + 5),
            get_arg_val<uint32_t>(at + 6)
        };
    }

    // The taps of row tile m's phase, tap rows [a0, a1) only.
    inline void load_phase(const Shape& s, uint32_t m, Phase& p, uint32_t a0, uint32_t a1) {
        const uint32_t f = m / s.phase_tiles;
        uint32_t nh = axis_taps(s.axis_h, f / s.phases_w, p.roff, p.ridx);
        const uint32_t end = a1 < nh ? a1 : nh;
        nh = a0 < end ? end - a0 : 0;
        for (uint32_t a = 0; a < nh; ++a) {
            p.roff[a] = p.roff[a0 + a];
            p.ridx[a] = p.ridx[a0 + a];
        }
        p.nh = nh;
        p.nw = axis_taps(s.axis_w, f % s.phases_w, p.coff, p.cidx);
        p.clo = 0xFFFFFFFFu;
        p.chi = 0;
        for (uint32_t b = 0; b < p.nw; ++b) {
            p.clo = p.coff[b] < p.clo ? p.coff[b] : p.clo;
            p.chi = p.coff[b] > p.chi ? p.coff[b] : p.chi;
        }
    }

    // Taps of row tile m's phase in tap rows [a0, a1) (no offsets).
    inline uint32_t phase_taps(const Shape& s, uint32_t m, uint32_t a0, uint32_t a1) {
        const uint32_t f = m / s.phase_tiles;
        const uint32_t fh = f / s.phases_w, fw = f % s.phases_w;
        const uint32_t nh = get_common_arg_val<uint32_t>(s.axis_h + 1 + fh);
        const uint32_t nw = get_common_arg_val<uint32_t>(s.axis_w + 1 + fw);
        const uint32_t end = a1 < nh ? a1 : nh;
        return (a0 < end ? end - a0 : 0) * nw;
    }

}  // namespace ktt_conv2
