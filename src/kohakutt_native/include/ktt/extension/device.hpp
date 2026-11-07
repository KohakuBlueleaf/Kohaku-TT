// Kohaku-TT extension API, device side (TT-NN / Metalium headers only): named program
// builders that ktt/extension/torch.hpp runs on tt tensors.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

#define KTT_EXTENSION_API __attribute__((visibility("default")))

namespace ktt::extension {

    // The numerics a program runs with: the caller's kohakutt.numerics scope when the
    // op was queued, with the program's Requirements applied.
    struct Numerics {
        tt::tt_metal::MathFidelity fidelity;
        bool approx;
    };

    // What a program needs or can use: fidelity raised to `floor` and lowered to
    // `ceiling`; `exact` turns SFPU approximations off. Ignored in a strict scope.
    struct Requirements {
        std::optional<tt::tt_metal::MathFidelity> floor;
        std::optional<tt::tt_metal::MathFidelity> ceiling;
        bool exact = false;
    };

    // Builds the program over `tensors` (the inputs, then the allocated outputs: TILE,
    // DRAM interleaved) with `numerics`. Runs on the stream worker at every call.
    using Builder = std::function<tt::tt_metal::ProgramDescriptor(
        const std::vector<ttnn::Tensor>& tensors, const Numerics& numerics
    )>;

    // Registers `build` as program `name` (e.g. "myns::swiglu") with `requirements`;
    // replaces an earlier registration of the same name.
    KTT_EXTENSION_API void register_program(
        const std::string& name, Builder build, Requirements requirements = {}
    );

    // A compute kernel config of `numerics`: fp32 destination accumulation, fp32
    // unpack-to-DST for the circular buffers in `unpack_fp32`.
    KTT_EXTENSION_API tt::tt_metal::ComputeConfigDescriptor compute_config(
        const Numerics& numerics, const std::vector<uint8_t>& unpack_fp32 = {}
    );

    // The compute grid (columns, rows) of the open device.
    KTT_EXTENSION_API tt::tt_metal::CoreCoord grid();

    // Number of 32x32 tiles of a TILE tensor.
    KTT_EXTENSION_API uint32_t tile_count(const ttnn::Tensor& t);

    // `tiles` split row-major over the grid, at least `min_per_core` per core:
    // cores[i] takes tiles [start(i), start(i) + count(i)).
    struct Split {
        std::vector<tt::tt_metal::CoreCoord> cores;
        tt::tt_metal::CoreRangeSet range;
        uint32_t base = 0, extra = 0;

        uint32_t count(size_t i) const {
            return base + (i < extra ? 1 : 0);
        }

        uint32_t start(size_t i) const {
            return static_cast<uint32_t>(i) * base + std::min<uint32_t>(i, extra);
        }
    };

    KTT_EXTENSION_API Split split_tiles(uint32_t tiles, uint32_t min_per_core = 1);

}  // namespace ktt::extension

// Registers program `name` with builder `fn` (and optional Requirements) when the
// extension library loads.
#define KTT_REGISTER_PROGRAM(name, fn, ...)                          \
    static const bool ktt_registered_##fn = [] {                     \
        ::ktt::extension::register_program(name, fn, ##__VA_ARGS__); \
        return true;                                                 \
    }()
