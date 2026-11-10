// Order of the exchanged tiles of the 1x2 collectives.
#pragma once

#include <cstdint>

namespace ktt_rs {

    // Page of exchanged tile u in its tensor: u, or with `rows` > 0 the tile of a
    // [rows, cols]-tile tensor taken column by column.
    inline uint32_t page_of(uint32_t u, uint32_t rows, uint32_t cols) {
        return rows == 0 ? u : (u % rows) * cols + u / rows;
    }

}  // namespace ktt_rs
