// Shared by the row-form group norm kernels (rows_*.cpp) and their host: x as TILE
// [.., images x rows, C] (a tile is 32 rows x 32 channels), tile (m, cb) at page m Ct +
// cb. Image n owns the row tiles whose first row is its own; segment (n, cb) is those
// tiles of channel tile cb. The tiles in segment order are cut into one range [q0, q1)
// per core, walked as pieces (the part of one segment inside the range).
#pragma once

#include <stdint.h>

namespace ktt_group_norm {

    constexpr uint32_t kF32Tile = 4096;

    // Element offset of (r, c) in a 32 x 32 tile of four 16 x 16 row-major faces.
    inline uint32_t at(uint32_t r, uint32_t c) {
        return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
    }

    struct Geometry {
        uint32_t ct, rows, images;

        // First row tile of image n (n = images: the row tiles in all).
        uint32_t first(uint32_t n) const {
            return (n * rows + 31) / 32;
        }

        uint32_t len(uint32_t n) const {
            return first(n + 1) - first(n);
        }

        // Rows of image n in its last row tile (1 .. 32).
        uint32_t tail_rows(uint32_t n) const {
            return (n + 1) * rows - 32 * (first(n + 1) - 1);
        }

        // True when image n's last row tile also holds rows of image n + 1.
        bool shared(uint32_t n) const {
            return n + 1 < images && tail_rows(n) < 32;
        }

        // Position of segment (n, cb)'s first tile in the tile sequence.
        uint32_t start(uint32_t n, uint32_t cb) const {
            return ct * first(n) + cb * len(n);
        }
    };

    // The piece of the range [q, q1) inside q's segment (n, cb): from its row tile i,
    // `len` row tiles; `last`: it ends the segment.
    struct Piece {
        uint32_t n, cb, i, len;
        bool last;

        Piece(const Geometry& g, uint32_t q, uint32_t q1) {
            n = static_cast<uint32_t>(static_cast<uint64_t>(q / g.ct) * 32 / g.rows);
            n = n < g.images ? n : g.images - 1;
            while (n + 1 < g.images && g.ct * g.first(n + 1) <= q) {
                ++n;
            }
            while (n > 0 && g.ct * g.first(n) > q) {
                --n;
            }
            const uint32_t ms = g.len(n), in = q - g.ct * g.first(n);
            cb = in / ms;
            i = in % ms;
            const uint32_t left = ms - i;
            len = q1 - q < left ? q1 - q : left;
            last = len == left;
        }

        // Segment index n Ct + cb.
        uint32_t segment(const Geometry& g) const {
            return n * g.ct + cb;
        }

        // Page of the piece's tile t.
        uint32_t page(const Geometry& g, uint32_t t) const {
            return (g.first(n) + i + t) * g.ct + cb;
        }

        // True when the piece's tile t is its image's last row tile.
        bool ends_image(const Geometry& g, uint32_t t) const {
            return i + t + 1 == g.len(n);
        }
    };

    // The groups of cg channels that have a channel in channel tile cb of C channels,
    // and the channel tiles their channels lie in.
    struct Span {
        uint32_t g_lo, g_hi, t_lo, t_hi;

        Span(uint32_t cb, uint32_t cg, uint32_t c) {
            const uint32_t last = (cb * 32 + 32 < c ? cb * 32 + 32 : c) - 1;
            g_lo = cb * 32 / cg;
            g_hi = last / cg;
            t_lo = g_lo * cg / 32;
            t_hi = ((g_hi + 1) * cg - 1) / 32;
        }

        uint32_t tiles() const {
            return t_hi - t_lo + 1;
        }
    };

}  // namespace ktt_group_norm
