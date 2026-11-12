// Host side of ktt_conv: the 2-D split (column tile groups x output row tile ranges),
// circular buffers and runtime args of reader / compute / writer.
#include "aten/nn/ktt_conv/program.hpp"

#include "aten/nn/ktt_conv/args.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::ktt_conv {
    namespace {

        constexpr uint32_t kTile = 2048;       // bf16 tile bytes
        constexpr uint32_t kRowsPerPage = 32;  // pixel rows per page of to_rows()
        // L1 bytes per core for the conv's circular buffers.
        constexpr uint32_t kBudget = 900 * 1024;
        // Runtime args the kernel sources read, per core and common: reader, writer,
        // compute.
        constexpr uint32_t kReaderArgs = 4, kWriterArgs = 4, kComputeArgs = 2;
        constexpr uint32_t kReaderCommon = 16, kWriterCommon = 2, kComputeCommon = 2;
        // The layout kernels': one per core; common: version, two addresses, unit step,
        // shape.
        constexpr uint32_t kLayoutArgs = 1;
        constexpr uint32_t kRowsInCommon = 15, kRowsOutCommon = 14,
                           kRowsIn1dCommon = 11, kRowsOut1dCommon = 10;

        // CB bytes of a core with nt column tiles.
        uint32_t cb_bytes(const Geometry& g, uint32_t nt) {
            const uint32_t ct = g.c / 32, k = g.kh * g.kw * ct;
            return (k * nt + 2 * ct + k + 2 * nt) * kTile;
        }

        // Bytes of the reader's two window sets (kh windows of 32 + (kw - 1) dw rows).
        uint32_t window_bytes(const Geometry& g) {
            return 2 * g.kh * (32 + (g.kw - 1) * g.dw) * g.c * 2;
        }

        // Tile rows per layout unit (8, 4 or 2): the fewest that give the grid's cores
        // a unit each, for `tiles` units of whole tiles.
        uint32_t rows_per_unit(uint32_t tiles) {
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = grid.x * grid.y;
            uint32_t rows = 8;
            while (rows > 2 && tiles * (32 / rows) < cores) {
                rows /= 2;
            }
            return rows;
        }

        // True unless KTT_CONV_WINDOW=0 (one read per tap row instead of shared
        // windows).
        bool window_enabled() {
            static const bool on = [] {
                const char* v = std::getenv("KTT_CONV_WINDOW");
                return !(v && std::string(v) == "0");
            }();
            return on;
        }

    }  // namespace

    std::optional<uint32_t> columns_per_core(const Geometry& g) {
        const uint32_t ot = (g.o + 31) / 32;
        for (uint32_t nt : {4u, 2u, 1u}) {
            if (nt <= ot && cb_bytes(g, nt) <= kBudget) {
                return nt;
            }
        }
        return std::nullopt;
    }

    ttnn::Tensor run(
        const ttnn::Tensor& x_rows,
        const ttnn::Tensor& w,
        const Geometry& g,
        const ttnn::ComputeKernelConfig& config
    ) {
        using namespace tt::tt_metal;
        const uint32_t ct = g.c / 32, ot = (g.o + 31) / 32;
        const uint32_t nt = *columns_per_core(g);
        const uint32_t out_rows = g.n * g.oh * g.wp;
        const uint32_t mt = (out_rows + 31) / 32;
        const uint32_t in_rows = g.n * g.hp * g.wp;

        auto out = ttnn::empty(
            ttnn::Shape({out_rows, g.o}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_conv");
        key.add(x_rows).add(w).add(out).add(static_cast<int64_t>(config.math_fidelity));
        for (
            const int64_t v : {g.n, g.c, g.hp, g.wp, g.o, g.kh, g.kw, g.dh, g.dw, g.oh}
        ) {
            key.add(v);
        }
        return program::launch(key, {x_rows, w, out}, [&] {
            // Column groups of nt tiles; the cores of the grid split evenly among them,
            // each core a contiguous range of output row tiles.
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t cores = grid.x * grid.y;
            const uint32_t groups = (ot + nt - 1) / nt;
            const uint32_t per_group = (std::max<uint32_t>(
                1, std::min<uint32_t>(cores / std::max<uint32_t>(groups, 1), mt)
            ));

            struct Work {
                CoreCoord core;
                uint32_t m0, m1, n0, n;
            };

            std::vector<Work> work;
            for (uint32_t gi = 0; gi < groups; ++gi) {
                const uint32_t n0 = gi * nt, n = std::min(nt, ot - n0);
                for (uint32_t p = 0; p < per_group; ++p) {
                    const uint32_t m0 = mt * p / per_group,
                                   m1 = mt * (p + 1) / per_group;
                    if (m1 == m0) {
                        continue;
                    }
                    const uint32_t index = static_cast<uint32_t>(work.size());
                    if (index >= cores) {
                        break;
                    }
                    work.push_back(
                        {CoreCoord(index % grid.x, index / grid.x), m0, m1, n0, n}
                    );
                }
            }
            std::vector<CoreRange> ranges;
            for (const auto& wk : work) {
                ranges.emplace_back(wk.core, wk.core);
            }
            const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            const uint32_t k = g.kh * g.kw * ct;
            const auto tiles_cb = [&](uint8_t index,
                                      uint32_t tiles,
                                      const CoreRangeSet& on) {
                return CBDescriptor{
                    .total_size = tiles * kTile,
                    .core_ranges = on,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index, .data_format = bf16, .page_size = kTile
                    }}
                };
            };
            desc.cbs.push_back(tiles_cb(0, 2 * ct, range));
            desc.cbs.push_back(tiles_cb(1, k * nt, range));
            desc.cbs.push_back(tiles_cb(2, k, range));
            // The weight block is one fill per launch.
            std::vector<program::Ring> rings = {
                {.cb = 0, .pushes = {ct}},
                {.cb = 1, .written = static_cast<uint64_t>(k) * nt * kTile},
                {.cb = 2, .pushes = {ct}},
            };
            // The output ring: two pushes of the core's column tiles (the last group
            // may have fewer than nt).
            std::map<uint32_t, std::vector<CoreRange>> by_width;
            for (const auto& wk : work) {
                by_width[wk.n].emplace_back(wk.core, wk.core);
            }
            for (const auto& [width, on] : by_width) {
                rings.push_back(
                    {.cb = 16,
                     .pushes = {width},
                     .at = static_cast<int32_t>(desc.cbs.size())}
                );
                desc.cbs.push_back(
                    tiles_cb(16, 2 * width, CoreRangeSet(on).merge_ranges())
                );
            }
            const uint32_t window =
                (window_enabled() && g.kw > 1 &&
                 cb_bytes(g, nt) + window_bytes(g) <= kBudget);
            if (window) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = window_bytes(g),
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = 3,
                            .data_format = bf16,
                            .page_size = window_bytes(g)
                        }}
                    }
                );
                // Two window sets addressed from the base.
                rings.push_back({.cb = 3, .written = window_bytes(g)});
            }

            const uint32_t row_bytes = g.c * 2;
            const uint32_t per_page = x_rows.buffer()->page_size() / row_bytes;
            // Per core: the block (m0, m1, n0, n) to the reader and writer, its row and
            // column tiles to compute.
            KernelDescriptor::RuntimeArgs block_args, compute_args;
            for (const auto& wk : work) {
                block_args.push_back({wk.core, {wk.m0, wk.m1, wk.n0, wk.n}});
                compute_args.push_back({wk.core, {wk.m1 - wk.m0, wk.n}});
            }
            program::mark(program::Phase::Args);

            auto reader_ct = (TensorAccessorArgs(*x_rows.buffer())
                                  .get_compile_time_args());
            const auto w_ct = TensorAccessorArgs(*w.buffer()).get_compile_time_args();
            reader_ct.insert(reader_ct.end(), w_ct.begin(), w_ct.end());
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv/reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = reader_ct,
                    .runtime_args = block_args,
                    .common_runtime_args =
                        {x_rows.buffer()->address(),
                         w.buffer()->address(),
                         ot,
                         ct,
                         g.kh,
                         g.kw,
                         g.dh,
                         g.dw,
                         g.wp,
                         g.hp * g.wp,
                         g.oh * g.wp,
                         in_rows,
                         out_rows,
                         row_bytes,
                         window,
                         per_page},
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv/writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*out.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = block_args,
                    .common_runtime_args = {out.buffer()->address(), ot},
                    .config = WriterConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv/compute.cpp"),
                    .core_ranges = range,
                    .runtime_args = compute_args,
                    .common_runtime_args = {g.kh * g.kw, ct},
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = config.math_fidelity,
                        .fp32_dest_acc_en = true,
                        .math_approx_mode = false
                    }
                }
            );
            constexpr auto all = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {{.kernel = 0, .entry = all},
                 {.kernel = 0, .entry = all, .position = 1, .tensor = 1},
                 {.kernel = 1, .entry = all, .tensor = 2}},
                {
                    {.kernel = 0, .per_core = kReaderArgs, .common = kReaderCommon},
                    {.kernel = 1, .per_core = kWriterArgs, .common = kWriterCommon},
                    {.kernel = 2, .per_core = kComputeArgs, .common = kComputeCommon},
                },
                std::move(rings)
            };
        });
    }

    bool columns_fit(const Columns& g) {
        // CB 0 and CB 16: two units of Ct tiles each.
        return g.cp % 32 == 0 && g.cp > 0 && 4 * (g.cp / 32) * kTile <= kBudget;
    }

    ttnn::Tensor to_columns(
        const ttnn::Tensor& x_rows, const Columns& g, uint32_t row0, uint32_t out_rows
    ) {
        using namespace tt::tt_metal;
        const uint32_t ct = g.cp / 32, taps = g.kh * g.kw;
        const uint32_t mt = (out_rows + 31) / 32;
        auto cols = ttnn::empty(
            ttnn::Shape({out_rows, taps * g.cp}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            MemoryConfig{}
        );
        program::mark(program::Phase::Alloc);
        program::Key key("ktt_conv_columns");
        key.add(x_rows).add(cols).add(row0);
        for (
            const int64_t v :
            {g.n, g.cp, g.hp, g.wp, g.kh, g.kw, g.sh, g.sw, g.dh, g.dw, g.oh, g.ow}
        ) {
            key.add(v);
        }
        return program::launch(key, {x_rows, cols}, [&] {
            // Units (m, tap), m-major, in contiguous ranges over the grid.
            const auto grid = detail::device()->compute_with_storage_grid_size();
            const uint32_t units = mt * taps;
            const uint32_t cores = std::min<uint32_t>(grid.x * grid.y, units);
            std::vector<CoreRange> ranges;
            KernelDescriptor::RuntimeArgs range_args, count_args;
            for (uint32_t i = 0; i < cores; ++i) {
                const CoreCoord core(i % grid.x, i / grid.x);
                const uint32_t u0 = static_cast<uint32_t>(uint64_t{units} * i / cores);
                const uint32_t u1 = static_cast<uint32_t>(
                    uint64_t{units} * (i + 1) / cores
                );
                ranges.emplace_back(core, core);
                range_args.push_back({core, {u0, u1}});
                count_args.push_back({core, {u1 - u0}});
            }
            const CoreRangeSet range = CoreRangeSet(ranges).merge_ranges();

            ProgramDescriptor desc;
            const auto bf16 = tt::DataFormat::Float16_b;
            for (const uint8_t index : {uint8_t{0}, uint8_t{16}}) {
                desc.cbs.push_back(
                    CBDescriptor{
                        .total_size = 2 * ct * kTile,
                        .core_ranges = range,
                        .format_descriptors = {CBFormatDescriptor{
                            .buffer_index = index,
                            .data_format = bf16,
                            .page_size = kTile
                        }}
                    }
                );
            }
            std::vector<program::Ring> rings = {
                {.cb = 0, .pushes = {ct}}, {.cb = 16, .pushes = {ct}}
            };
            const uint32_t row_bytes = g.cp * 2;
            const uint32_t per_page = x_rows.buffer()->page_size() / row_bytes;
            program::mark(program::Phase::Args);
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv/columns_reader.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*x_rows.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = range_args,
                    .common_runtime_args =
                        {x_rows.buffer()->address(),
                         per_page,
                         row_bytes,
                         ct,
                         g.kh,
                         g.kw,
                         g.sh,
                         g.sw,
                         g.dh,
                         g.dw,
                         g.hp,
                         g.wp,
                         g.oh,
                         g.ow,
                         row0,
                         row0 + out_rows},
                    .config = ReaderConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv/columns_writer.cpp"),
                    .core_ranges = range,
                    .compile_time_args = TensorAccessorArgs(*cols.buffer())
                                             .get_compile_time_args(),
                    .runtime_args = range_args,
                    .common_runtime_args = {cols.buffer()->address(), ct, taps},
                    .config = WriterConfigDescriptor{}
                }
            );
            desc.kernels.push_back(
                KernelDescriptor{
                    .kernel_source = program::source("nn/ktt_conv/columns_compute.cpp"),
                    .core_ranges = range,
                    .runtime_args = count_args,
                    .common_runtime_args = {ct},
                    .config = ComputeConfigDescriptor{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false
                    }
                }
            );
            constexpr auto all = program::Slot::kCommon;
            return program::Built{
                std::move(desc),
                {{.kernel = 0, .entry = all}, {.kernel = 1, .entry = all, .tensor = 1}},
                {
                    {.kernel = 0, .per_core = 2, .common = 16},
                    {.kernel = 1, .per_core = 2, .common = 3},
                    {.kernel = 2, .per_core = 1, .common = 1},
                },
                std::move(rings)
            };
        });
    }

    namespace {

        // A one-page scratch circular buffer: its bytes, and the most the kernel writes
        // from its base.
        struct Scratch {
            uint8_t index;
            uint32_t bytes, written;
        };

        // A single data-movement kernel `source` on the first `units` cores of the grid
        // with scratch buffers `cbs`; per-core arg: the core index; common args
        // (`common_args` read by the source): the argument-layout version, the
        // addresses of io's two tensors, the core count, then `tail`; compile-time
        // args: io's accessors, then the version.
        ttnn::Tensor layout_op(
            const std::vector<ttnn::Tensor>& io,
            const std::string& source,
            const std::vector<Scratch>& cbs,
            const std::vector<uint32_t>& tail,
            uint32_t units,
            uint32_t common_args
        ) {
            using namespace tt::tt_metal;
            program::mark(program::Phase::Alloc);
            program::Key key("ktt_conv_layout");
            key.add(source).add(units);
            for (const auto& t : io) {
                key.add(t);
            }
            for (const auto& [index, bytes, written] : cbs) {
                key.add(index).add(bytes);
            }
            for (const int64_t v : tail) {
                key.add(v);
            }
            return program::launch(key, io, [&] {
                const std::vector<uint32_t> head = {
                    ::ktt_conv::kArgsVersion,
                    io[0].buffer()->address(),
                    io[1].buffer()->address()
                };
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
                std::vector<program::Ring> rings;
                for (const auto& [index, bytes, written] : cbs) {
                    desc.cbs.push_back(
                        CBDescriptor{
                            .total_size = bytes,
                            .core_ranges = range,
                            .format_descriptors = {CBFormatDescriptor{
                                .buffer_index = index,
                                .data_format = tt::DataFormat::Float16_b,
                                .page_size = bytes
                            }}
                        }
                    );
                    rings.push_back({.cb = index, .written = written});
                }
                const auto args = program::core_index_args(cores);
                std::vector<uint32_t> common = head;
                common.push_back(cores);
                common.insert(common.end(), tail.begin(), tail.end());
                program::mark(program::Phase::Args);
                std::vector<uint32_t> ct;
                for (size_t i = 0; i + 1 < io.size(); ++i) {
                    const auto more = (TensorAccessorArgs(*io[i].buffer())
                                           .get_compile_time_args());
                    ct.insert(ct.end(), more.begin(), more.end());
                }
                const auto last = (TensorAccessorArgs(*io.back().buffer())
                                       .get_compile_time_args());
                ct.insert(ct.end(), last.begin(), last.end());
                ct.push_back(::ktt_conv::kArgsVersion);
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
                return program::Built{
                    std::move(desc),
                    {{.kernel = 0, .entry = all, .position = 1},
                     {.kernel = 0, .entry = all, .position = 2, .tensor = 1}},
                    {{.kernel = 0, .per_core = kLayoutArgs, .common = common_args}},
                    std::move(rings)
                };
            });
        }

    }  // namespace

    ttnn::Tensor to_rows(
        const ttnn::Tensor& x, const Geometry& g, uint32_t ph, uint32_t pw
    ) {
        using namespace tt::tt_metal;
        const auto& shape = x.logical_shape();
        const uint32_t h = shape[shape.rank() - 2], w = shape[shape.rank() - 1];
        const uint32_t pixels = g.n * g.hp * g.wp;
        const uint32_t cp = (g.c + 31) / 32 * 32;
        auto rows = ttnn::empty(
            ttnn::Shape(
                {(pixels + kRowsPerPage - 1) / kRowsPerPage, kRowsPerPage * cp}
            ),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR,
            detail::device(),
            MemoryConfig{}
        );
        const uint32_t tiles = g.n * ((h + 31) / 32) * ((w + 31) / 32) * (cp / 32);
        const uint32_t rpu = rows_per_unit(tiles);
        const uint32_t units = std::max(tiles * (32 / rpu), g.n * g.hp);
        layout_op(
            {x, rows},
            "nn/ktt_conv/rows_in.cpp",
            {{0, 32 * kTile, 32 * kTile},
             {1, 32 * kTile, 32 * kTile},
             {2, cp * 2, cp * 2}},
            {g.n, g.c, cp, h, w, g.hp, g.wp, ph, pw, kRowsPerPage, rpu},
            units,
            kRowsInCommon
        );
        return rows;
    }

    ttnn::Tensor to_nchw(
        const ttnn::Tensor& out,
        const Geometry& g,
        uint32_t ow,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        return to_nchw(out, OutRows{g.n, g.o, g.oh, ow, g.wp, g.oh}, memory);
    }

    ttnn::Tensor to_nchw(
        const ttnn::Tensor& out,
        const OutRows& r,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        auto nchw = ttnn::empty(
            ttnn::Shape({r.n, r.o, r.oh, r.ow}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            memory
        );
        const uint32_t
            tiles = (r.n * ((r.oh + 31) / 32) * ((r.ow + 31) / 32) * ((r.o + 31) / 32));
        const uint32_t rpu = rows_per_unit(tiles);
        layout_op(
            {out, nchw},
            "nn/ktt_conv/rows_out.cpp",
            // Source tiles: two per (unit row, column phase).
            {{0, 16 * r.sw * kTile, 2 * rpu * r.sw * kTile},
             {1, 32 * kTile, 32 * kTile}},
            {r.n, r.o, r.oh, r.ow, r.pitch, r.ohq, r.sh, r.sw, r.phase_rows, rpu},
            tiles * (32 / rpu),
            kRowsOutCommon
        );
        return nchw;
    }

    ttnn::Tensor to_rows_1d(
        const ttnn::Tensor& x, uint32_t n, uint32_t c, uint32_t wp, uint32_t pw
    ) {
        using namespace tt::tt_metal;
        const auto& shape = x.logical_shape();
        const uint32_t len = shape[shape.rank() - 1];
        const uint32_t cp = (c + 31) / 32 * 32;
        const uint32_t pixels = n * wp;
        auto rows = ttnn::empty(
            ttnn::Shape(
                {(pixels + kRowsPerPage - 1) / kRowsPerPage, kRowsPerPage * cp}
            ),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR,
            detail::device(),
            MemoryConfig{}
        );
        const uint32_t units = std::max(n * (cp / 32) * ((len + 31) / 32), n);
        layout_op(
            {x, rows},
            "nn/ktt_conv/rows_in_1d.cpp",
            {{0, kTile, kTile}, {1, kTile, kTile}, {2, cp * 2, cp * 2}},
            {n, c, cp, len, wp, pw, kRowsPerPage},
            units,
            kRowsIn1dCommon
        );
        return rows;
    }

    ttnn::Tensor to_ncl(
        const ttnn::Tensor& out,
        const OutRows& r,
        const tt::tt_metal::MemoryConfig& memory
    ) {
        using namespace tt::tt_metal;
        auto ncl = ttnn::empty(
            ttnn::Shape({r.n, r.o, r.ow}),
            DataType::BFLOAT16,
            Layout::TILE,
            detail::device(),
            memory
        );
        const uint32_t units = r.n * ((r.o + 31) / 32) * ((r.ow + 31) / 32);
        layout_op(
            {out, ncl},
            "nn/ktt_conv/rows_out_1d.cpp",
            {{0, 2 * r.sw * kTile, 2 * r.sw * kTile}, {1, kTile, kTile}},
            {r.n, r.o, r.ow, r.pitch, r.sw, r.phase_rows},
            units,
            kRowsOut1dCommon
        );
        return ncl;
    }

}  // namespace ktt::bridge::ktt_conv
