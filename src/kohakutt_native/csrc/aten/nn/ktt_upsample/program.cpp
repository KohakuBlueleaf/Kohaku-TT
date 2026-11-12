// Host side of ktt_upsample: output tiles split over the grid, scratch buffers, runtime
// args; nearest upsampling (row copies) and pixel shuffle.
#include "aten/nn/ktt_upsample/program.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_upsample {
    namespace {

        // Per-core runtime arguments of every kernel: the first unit.
        constexpr uint32_t kArgs = 1;

        // Runs data-movement kernel `source` over `units` output tiles of `out` from
        // `xs` (one memory layout): scratch CB 0 of `in_tiles` tiles, CB 1 of one tile;
        // common args (x addresses, out address, cores, tail..., element bytes).
        ttnn::Tensor gather(
            const std::string& source,
            const std::vector<ttnn::Tensor>& xs,
            const ttnn::Tensor& out,
            uint32_t units,
            uint32_t in_tiles,
            const std::vector<uint32_t>& tail
        ) {
            using namespace tt::tt_metal;
            // Kernels with several inputs read their count at compile time, last.
            const auto parts = static_cast<uint32_t>(xs.size());
            program::mark(program::Phase::Alloc);
            program::Key key("ktt_upsample");
            key.add(source).add(out).add(units).add(in_tiles);
            for (const auto& x : xs) {
                key.add(x);
            }
            const auto& x = xs.front();
            auto io = xs;
            io.push_back(out);
            for (const int64_t v : tail) {
                key.add(v);
            }
            return program::launch(key, io, [&] {
                const bool wide =
                    (x.dtype() == DataType::FLOAT32 || x.dtype() == DataType::INT32 ||
                     x.dtype() == DataType::UINT32);
                const uint32_t elem = wide ? 4 : 2;
                const auto grid = detail::device()->compute_with_storage_grid_size();
                const uint32_t cores = (std::max<uint32_t>(
                    1, std::min<uint32_t>(grid.x * grid.y, units)
                ));
                std::vector<CoreRange> ranges;
                for (uint32_t c = 0; c < cores; ++c) {
                    ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
                }
                const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();
                ProgramDescriptor desc;
                const auto df =
                    (elem == 4 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b);
                for (
                    auto [index, tiles] :
                    {std::pair<uint8_t, uint32_t>{0, in_tiles}, {1, 1}}
                ) {
                    desc.cbs.push_back(
                        CBDescriptor{
                            .total_size = tiles * 1024 * elem,
                            .core_ranges = range,
                            .format_descriptors = {CBFormatDescriptor{
                                .buffer_index = index,
                                .data_format = df,
                                .page_size = tiles * 1024 * elem
                            }}
                        }
                    );
                }
                const auto args = program::core_index_args(cores);
                std::vector<uint32_t> common;
                for (const auto& t : io) {
                    common.push_back(t.buffer()->address());
                }
                common.push_back(cores);
                common.insert(common.end(), tail.begin(), tail.end());
                common.push_back(elem);
                program::mark(program::Phase::Args);
                auto ct = TensorAccessorArgs(*x.buffer()).get_compile_time_args();
                const auto more = (TensorAccessorArgs(*out.buffer())
                                       .get_compile_time_args());
                ct.insert(ct.end(), more.begin(), more.end());
                if (parts > 1) {
                    ct.push_back(parts);
                }
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(source),
                        .core_ranges = range,
                        .compile_time_args = ct,
                        .runtime_args = args,
                        .common_runtime_args = common,
                        .config = ReaderConfigDescriptor{}
                    }
                );
                constexpr auto all = program::Slot::kCommon;
                std::vector<program::Slot> slots;
                for (uint32_t i = 0; i < io.size(); ++i) {
                    slots.push_back(
                        {.kernel = 0, .entry = all, .position = i, .tensor = i}
                    );
                }
                const auto n_common = static_cast<uint32_t>(common.size());
                return program::Built{
                    std::move(desc),
                    std::move(slots),
                    {{.kernel = 0, .per_core = kArgs, .common = n_common}},
                    // Reserved once and written in place: the input tiles of one output
                    // tile, and the output tile.
                    {{0, {}, uint64_t{in_tiles} * 1024 * elem}, {1, {}, 1024 * elem}}
                };
            });
        }

        // Common runtime args of rows.cpp (addresses, workers, units, then mode, elem,
        // kh, kw, H, W, tile counts, planes / images), and its per-core arg (the core
        // index).
        constexpr uint32_t kRowsCommon = 15, kRowsArgs = 1;

        // rows.cpp over `units` work units of `out` from `x` on both data-movement
        // RISCs: per RISC an input tile cache of 4 tiles (2 when L1 is short) and two
        // output tiles, within program::cb_budget().
        ttnn::Tensor row_copy(
            const ttnn::Tensor& x,
            const ttnn::Tensor& out,
            uint32_t units,
            const std::vector<uint32_t>& tail
        ) {
            using namespace tt::tt_metal;
            program::mark(program::Phase::Alloc);
            const uint32_t tile = (x.dtype() == DataType::FLOAT32 ? 4 : 2) * 1024;
            const uint32_t budget = program::cb_budget();
            const uint32_t slots = 2 * (4 + 2) * tile <= budget ? 4 : 2;
            if (2 * (slots + 2) * tile > budget) {
                throw std::runtime_error(
                    "ktt_upsample: the row copy's circular buffers (" +
                    std::to_string(2 * (slots + 2) * tile) +
                    " B per core) do not fit L1 (" + std::to_string(budget) + " B free)"
                );
            }
            program::Key key("ktt_upsample_rows");
            key.add(x).add(out).add(units).add(static_cast<int64_t>(slots));
            for (const int64_t v : tail) {
                key.add(v);
            }
            return program::launch(key, {x, out}, [&] {
                const uint32_t elem = x.dtype() == DataType::FLOAT32 ? 4 : 2;
                const auto grid = detail::device()->compute_with_storage_grid_size();
                const uint32_t cores = (std::max<uint32_t>(
                    1, std::min<uint32_t>(grid.x * grid.y, (units + 1) / 2)
                ));
                std::vector<CoreRange> ranges;
                for (uint32_t c = 0; c < cores; ++c) {
                    ranges.emplace_back(CoreCoord(c % grid.x, c / grid.x));
                }
                const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();
                ProgramDescriptor desc;
                const auto df =
                    (elem == 4 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b);
                std::vector<program::Ring> rings;
                for (uint8_t role = 0; role < 2; ++role) {
                    for (uint32_t k = 0; k < 2; ++k) {
                        const uint8_t index = role * 2 + k;
                        const uint32_t bytes = (k == 0 ? slots : 2) * 1024 * elem;
                        desc.cbs.push_back(
                            CBDescriptor{
                                .total_size = bytes,
                                .core_ranges = range,
                                .format_descriptors = {CBFormatDescriptor{
                                    .buffer_index = index,
                                    .data_format = df,
                                    .page_size = bytes
                                }}
                            }
                        );
                        rings.push_back({index, {}, bytes});
                    }
                }
                const auto args = program::core_index_args(cores);
                std::vector<uint32_t> common{
                    x.buffer()->address(), out.buffer()->address(), 2 * cores, units
                };
                common.insert(common.end(), tail.begin(), tail.end());
                program::mark(program::Phase::Args);
                auto ct = TensorAccessorArgs(*x.buffer()).get_compile_time_args();
                const auto more = (TensorAccessorArgs(*out.buffer())
                                       .get_compile_time_args());
                ct.insert(ct.end(), more.begin(), more.end());
                for (uint32_t role = 0; role < 2; ++role) {
                    KernelDescriptor kernel{
                        .kernel_source = program::source("nn/ktt_upsample/rows.cpp"),
                        .core_ranges = range,
                        .compile_time_args = ct,
                        .defines =
                            {{"ROLE", std::to_string(role)},
                             {"SLOTS", std::to_string(slots)}},
                        .runtime_args = args,
                        .common_runtime_args = common
                    };
                    if (role == 0) {
                        kernel.config = ReaderConfigDescriptor{};
                    } else {
                        kernel.config = WriterConfigDescriptor{};
                    }
                    desc.kernels.push_back(std::move(kernel));
                }
                constexpr auto all = program::Slot::kCommon;
                std::vector<program::Slot> slots;
                for (uint32_t k = 0; k < 2; ++k) {
                    slots.push_back(
                        {.kernel = k, .entry = all, .position = 0, .tensor = 0}
                    );
                    slots.push_back(
                        {.kernel = k, .entry = all, .position = 1, .tensor = 1}
                    );
                }
                return program::Built{
                    std::move(desc),
                    slots,
                    {{.kernel = 0, .per_core = kRowsArgs, .common = kRowsCommon},
                     {.kernel = 1, .per_core = kRowsArgs, .common = kRowsCommon}},
                    rings
                };
            });
        }

    }  // namespace

    ttnn::Tensor rows(
        const ttnn::Tensor& x,
        uint32_t kh,
        uint32_t kw,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& shape = x.logical_shape();
        const uint32_t rank = shape.rank();
        const uint32_t h = shape[rank - 2], w = shape[rank - 1];
        uint32_t planes = 1;
        ttsl::SmallVector<uint32_t> out_dims;
        for (uint32_t d = 0; d + 2 < rank; ++d) {
            planes *= shape[d];
            out_dims.push_back(shape[d]);
        }
        out_dims.push_back(h * kh);
        out_dims.push_back(w * kw);
        auto out = (ttnn::empty(
            ttnn::Shape(out_dims), x.dtype(), Layout::TILE, detail::device(), memory
        ));
        const uint32_t iht = (h + 31) / 32, iwt = (w + 31) / 32;
        const uint32_t oht = (h * kh + 31) / 32, owt = (w * kw + 31) / 32;
        const uint32_t elem = x.dtype() == DataType::FLOAT32 ? 4 : 2;
        // Factors dividing 32: one unit per input tile (mode 2).
        const bool tiled = 32 % kh == 0 && 32 % kw == 0;
        return row_copy(
            x,
            out,
            planes * (tiled ? iht * iwt : oht * owt),
            {tiled ? 2u : 0u, elem, kh, kw, h, w, iht, iwt, oht, owt, planes}
        );
    }

    ttnn::Tensor rows_nhwc(
        const ttnn::Tensor& x,
        uint32_t images,
        uint32_t h,
        uint32_t w,
        uint32_t kh,
        uint32_t kw,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const uint32_t c = x.logical_shape()[-1];
        const uint32_t pixels = images * h * kh * w * kw;
        auto out = (ttnn::empty(
            ttnn::Shape({1, 1, pixels, c}),
            x.dtype(),
            Layout::TILE,
            detail::device(),
            memory
        ));
        const uint32_t iht = (images * h * w + 31) / 32, ct = (c + 31) / 32;
        const uint32_t oht = (pixels + 31) / 32;
        const uint32_t elem = x.dtype() == DataType::FLOAT32 ? 4 : 2;
        // Output image rows of whole tiles: one unit per (input image row, channel
        // tile) (mode 3).
        const bool whole = w * kw % 32 == 0;
        return row_copy(
            x,
            out,
            whole ? images * h * ct : oht * ct,
            {whole ? 3u : 1u, elem, kh, kw, h, w, iht, ct, oht, ct, images}
        );
    }

    ttnn::Tensor shuffle(
        const ttnn::Tensor& x, uint32_t r, const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& shape = x.logical_shape();
        const uint32_t rank = shape.rank();
        const uint32_t c = shape[rank - 3] / (r * r);
        const uint32_t h = shape[rank - 2], w = shape[rank - 1];
        uint32_t prefix = 1;
        ttsl::SmallVector<uint32_t> out_dims;
        for (uint32_t d = 0; d + 3 < rank; ++d) {
            prefix *= shape[d];
            out_dims.push_back(shape[d]);
        }
        out_dims.push_back(c);
        out_dims.push_back(h * r);
        out_dims.push_back(w * r);
        auto out = (ttnn::empty(
            ttnn::Shape(out_dims), x.dtype(), Layout::TILE, detail::device(), memory
        ));
        const uint32_t iht = (h + 31) / 32, iwt = (w + 31) / 32;
        const uint32_t oht = (h * r + 31) / 32, owt = (w * r + 31) / 32;
        return gather(
            "nn/ktt_upsample/shuffle.cpp",
            {x},
            out,
            prefix * c * oht * owt,
            r * r,
            {prefix * c, c, iht, iwt, oht, owt, r}
        );
    }

    ttnn::Tensor columns(
        const ttnn::Tensor& x,
        uint32_t first,
        uint32_t step,
        uint32_t count,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& shape = x.logical_shape();
        const uint32_t rank = shape.rank();
        const uint32_t h = shape[rank - 2], w = shape[rank - 1];
        uint32_t planes = 1;
        ttsl::SmallVector<uint32_t> out_dims;
        for (uint32_t d = 0; d + 2 < rank; ++d) {
            planes *= shape[d];
            out_dims.push_back(shape[d]);
        }
        out_dims.push_back(h);
        out_dims.push_back(count);
        auto out = (ttnn::empty(
            ttnn::Shape(out_dims), x.dtype(), Layout::TILE, detail::device(), memory
        ));
        const uint32_t ht = (h + 31) / 32, iwt = (w + 31) / 32;
        const uint32_t owt = (count + 31) / 32;
        return gather(
            "nn/ktt_upsample/columns.cpp",
            {x},
            out,
            planes * ht * owt,
            step + 1,
            {planes, ht, iwt, owt, first, step, count}
        );
    }

    ttnn::Tensor select_last(
        const ttnn::Tensor& x, uint32_t j, const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& shape = x.logical_shape();
        const uint32_t rank = shape.rank();
        const uint32_t n = shape[rank - 2];
        const uint32_t r = rank >= 3 ? shape[rank - 3] : 1;
        uint32_t planes = 1;
        ttsl::SmallVector<uint32_t> out_dims;
        for (uint32_t d = 0; d + 3 < rank; ++d) {
            planes *= shape[d];
            out_dims.push_back(shape[d]);
        }
        out_dims.push_back(r);
        out_dims.push_back(n);
        auto out = (ttnn::empty(
            ttnn::Shape(out_dims), x.dtype(), Layout::TILE, detail::device(), memory
        ));
        const uint32_t nt = (n + 31) / 32;
        const uint32_t oht = (r + 31) / 32, owt = nt;
        return gather(
            "nn/ktt_upsample/select.cpp",
            {x},
            out,
            planes * oht * owt,
            32,
            {planes, r, nt, oht, owt, j}
        );
    }

    ttnn::Tensor interleave(
        const std::vector<ttnn::Tensor>& xs, const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        const auto& shape = xs.front().logical_shape();
        const uint32_t rank = shape.rank();
        const uint32_t k = static_cast<uint32_t>(xs.size());
        const uint32_t h = shape[rank - 2], n = shape[rank - 1];
        uint32_t planes = 1;
        ttsl::SmallVector<uint32_t> out_dims;
        for (uint32_t d = 0; d + 2 < rank; ++d) {
            planes *= shape[d];
            out_dims.push_back(shape[d]);
        }
        out_dims.push_back(h);
        out_dims.push_back(n * k);
        auto out = (ttnn::empty(
            ttnn::Shape(out_dims),
            xs.front().dtype(),
            Layout::TILE,
            detail::device(),
            memory
        ));
        const uint32_t ht = (h + 31) / 32, iwt = (n + 31) / 32;
        const uint32_t owt = (n * k + 31) / 32;
        return gather(
            "nn/ktt_upsample/interleave.cpp",
            xs,
            out,
            planes * ht * owt,
            2 * k,
            {planes, ht, iwt, owt, n}
        );
    }

}  // namespace ktt::bridge::ktt_upsample
