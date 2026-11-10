// Kohaku-TT's own collectives on a 1x2 mesh, one program each: per fabric link, a
// sender core in the Tensix row under the link's Ethernet tile reads this chip's tiles
// (reader RISC) and sends them into the peer's scratch (writer RISC); drain cores add
// each received tile to the kept tile (and to an addend tile) on the SFPU in fp32 and
// write the output. Reduce-scatter exchanges the peer's half of the rows, all-reduce
// every tile.
#include "core/mesh/reduce_scatter.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "aten/matmul/ktt_gemm/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/numerics.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/generic/device/generic_op_device_operation.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::bridge::mesh_rs {
    namespace {

        using tt::tt_metal::CoreCoord;
        using tt::tt_metal::CoreRange;
        using tt::tt_metal::CoreRangeSet;

        constexpr uint32_t kMaxLinks = 4;
        // Drain cores per link (reduce-scatter; all-reduce, at most 16); tiles per
        // drain chunk (drain CBs hold two); tiles per sender read batch (the sender CB
        // holds two).
        constexpr uint32_t kDrainsPerLink = 4;
        constexpr uint32_t kReduceDrainsPerLink = 12;
        static_assert(
            kDrainsPerLink <= kReduceDrainsPerLink && kReduceDrainsPerLink <= 16
        );
        constexpr uint32_t kChunk = 4;
        constexpr uint32_t kSendBatch = 32;
        static_assert(kSendBatch % kChunk == 0);

        // NoCs: readers (DRAM reads) on NoC0, writers (fabric sends, DRAM writes) on
        // the other one (dedicated NoC mode: one RISC per NoC).
        constexpr uint32_t kReaderNoc = 0;
        constexpr uint32_t kWriterNoc = 1 - kReaderNoc;
        static_assert(kWriterNoc != kReaderNoc);

        // Cores and global semaphores of the open mesh, built on first use.
        struct Plan {
            uint32_t links = 0;
            // Per chip (mesh column), per link: the sender's logical core.
            std::vector<std::vector<CoreCoord>> senders;
            // Per link, per drain: the drain's logical core (the same on both chips).
            std::vector<std::vector<CoreCoord>> drains;
            std::optional<tt::tt_metal::GlobalSemaphore> drain_sem, ready_sem;
        };

        std::unique_ptr<Plan>& g_plan = *new std::unique_ptr<Plan>();

        tt::tt_metal::NOC noc_of(uint32_t noc) {
            return noc == 0 ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1;
        }

        // The free row-0 Tensix column whose virtual x is the nearest at or above the
        // virtual x of Ethernet core `eth` (NoC1 reaches the Ethernet row in one -y
        // hop, then -x); marks it used. Coordinates from the device's SoC descriptor.
        CoreCoord under(
            tt::tt_metal::IDevice* device,
            const CoreCoord& eth,
            uint32_t columns,
            std::vector<bool>& used
        ) {
            const auto eth_x = static_cast<int64_t>(
                device->ethernet_core_from_logical_core(eth).x
            );
            int64_t best = -1, best_cost = 0;
            for (uint32_t x = 0; x < columns; ++x) {
                if (used[x]) {
                    continue;
                }
                const auto wx = static_cast<int64_t>(
                    device->worker_core_from_logical_core({x, 0}).x
                );
                // Hops of the -x leg; columns left of the Ethernet core wrap around.
                const int64_t cost = wx >= eth_x ? wx - eth_x : 1000 + eth_x - wx;
                if (best < 0 || cost < best_cost) {
                    best = x;
                    best_cost = cost;
                }
            }
            if (best < 0) {
                throw std::runtime_error("kohakutt_native: no free sender column");
            }
            used[best] = true;
            return {static_cast<uint32_t>(best), 0};
        }

        Plan* plan() {
            if (g_plan) {
                return g_plan.get();
            }
            auto* mesh = detail::device();
            auto p = std::make_unique<Plan>();
            const ttnn::MeshCoordinate c0(0, 0), c1(0, 1);
            const auto n0 = mesh->get_fabric_node_id(c0),
                       n1 = mesh->get_fabric_node_id(c1);
            const auto l01 = tt::tt_fabric::get_forwarding_link_indices(n0, n1);
            const auto l10 = tt::tt_fabric::get_forwarding_link_indices(n1, n0);
            p->links = static_cast<uint32_t>(
                std::min<size_t>({l01.size(), l10.size(), kMaxLinks})
            );
            const auto grid = mesh->compute_with_storage_grid_size();
            if (p->links == 0 || grid.y < 2 ||
                (grid.y - 1) * grid.x < p->links * kReduceDrainsPerLink) {
                return nullptr;
            }
            p->senders.resize(2);
            for (int c = 0; c < 2; ++c) {
                const auto src = c == 0 ? n0 : n1, dst = c == 0 ? n1 : n0;
                const auto& links = c == 0 ? l01 : l10;
                auto* device = mesh->get_device(c == 0 ? c0 : c1);
                std::vector<bool> used(grid.x, false);
                for (uint32_t l = 0; l < p->links; ++l) {
                    const auto eth = (tt::tt_fabric::get_forwarding_eth_core(
                        *mesh, src, dst, links[l]
                    ));
                    p->senders[c].push_back(under(device, eth, grid.x, used));
                }
            }
            // Drains: rows 1 and on, row-major, dealt to the links in turn.
            p->drains.resize(p->links);
            for (uint32_t k = 0; k < p->links * kReduceDrainsPerLink; ++k) {
                const CoreCoord core(k % grid.x, 1 + k / grid.x);
                p->drains[k % p->links].push_back(core);
            }
            // The ready semaphore lives on the senders of both chips (one set).
            std::set<std::pair<uint32_t, uint32_t>> sender_cores;
            for (const auto& s : p->senders) {
                for (const auto& core : s) {
                    sender_cores.insert({core.x, core.y});
                }
            }
            std::vector<CoreRange> sender_ranges;
            for (const auto& [x, y] : sender_cores) {
                sender_ranges.emplace_back(CoreCoord(x, y));
            }
            p->ready_sem = tt::tt_metal::CreateGlobalSemaphore(
                *mesh, CoreRangeSet(sender_ranges).merge_ranges(), 0
            );
            // The drain semaphore lives on every core: a fused exchange takes its
            // drains from rows 0 and 1.
            p->drain_sem = tt::tt_metal::CreateGlobalSemaphore(
                *mesh, CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})), 0
            );
            g_plan = std::move(p);
            return g_plan.get();
        }

        tt::tt_metal::DataMovementConfigDescriptor dm(
            tt::tt_metal::DataMovementProcessor processor, uint32_t noc
        ) {
            return {
                .processor = processor,
                .noc = noc_of(noc),
                .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC
            };
        }

    }  // namespace

    void clear() {
        g_plan.reset();
    }

    namespace {

        // True when `t` is a bf16 / fp32 TILE tensor in interleaved DRAM.
        bool plain(const ttnn::Tensor& t) {
            using namespace tt::tt_metal;
            return (t.dtype() == DataType::FLOAT32 ||
                    t.dtype() == DataType::BFLOAT16) &&
                t.layout() == Layout::TILE &&
                t.memory_config().buffer_type() == BufferType::DRAM &&
                !t.memory_config().is_sharded();
        }

        // The sum over the two chips of `t` (+ `addend`'s tile of the same index on
        // each chip): `scatter` exchanges the peer's half of the rows and gives chip k
        // its half, else every tile is exchanged and both chips get the whole sum. The
        // matmul that consumes an all-reduce's output in its program: weight, its
        // orientation, compute config; exchange() sets `product`.
        struct Fuse {
            const ttnn::Tensor& w;
            bool w_nk;
            const ttnn::ComputeKernelConfig& config;
            std::optional<ttnn::Tensor> product;
        };

        // First grid row of a fused matmul (rows 0 and 1: senders and drains) and the
        // first id of its progress semaphores.
        constexpr uint32_t kFuseRow = 2, kFuseProgress = 4;

        std::optional<ttnn::Tensor> exchange(
            const ttnn::Tensor& t,
            const ttnn::Tensor* addend,
            bool scatter,
            Fuse* fuse = nullptr,
            bool columns = false
        );

    }  // namespace

    std::optional<ttnn::Tensor> reduce_scatter_rows(
        const ttnn::Tensor& t, uint32_t axis
    ) {
        const auto& s = t.padded_shape();
        const int rank = static_cast<int>(s.rank());
        for (int k = 0; k + 2 < rank; ++k) {
            if (s[k] != 1) {
                return std::nullopt;
            }
        }
        if (axis != 1 || rank < 2 || s[rank - 2] % 64 != 0 ||
            t.dtype() != tt::tt_metal::DataType::FLOAT32) {
            return std::nullopt;
        }
        return exchange(t, nullptr, true);
    }

    std::optional<ttnn::Tensor> reduce_scatter_columns(
        const ttnn::Tensor& t, uint32_t axis
    ) {
        const auto& s = t.padded_shape();
        const int rank = static_cast<int>(s.rank());
        for (int k = 0; k + 2 < rank; ++k) {
            if (s[k] != 1) {
                return std::nullopt;
            }
        }
        if (axis != 1 || rank < 2 || s[rank - 1] % 64 != 0 ||
            t.dtype() != tt::tt_metal::DataType::FLOAT32) {
            return std::nullopt;
        }
        return exchange(t, nullptr, true, nullptr, true);
    }

    std::optional<ttnn::Tensor> all_reduce(
        const ttnn::Tensor& t, const ttnn::Tensor* addend, uint32_t axis
    ) {
        if (axis != 1 ||
            (addend &&
             (!plain(*addend) || addend->dtype() != t.dtype() ||
              addend->padded_shape() != t.padded_shape()))) {
            return std::nullopt;
        }
        return exchange(t, addend, false);
    }

    std::optional<std::pair<ttnn::Tensor, ttnn::Tensor>> all_reduce_matmul(
        const ttnn::Tensor& t,
        const ttnn::Tensor* addend,
        const ttnn::Tensor& w,
        bool w_nk,
        const ttnn::ComputeKernelConfig& config
    ) {
        using tt::tt_metal::DataType;
        if (t.dtype() != DataType::BFLOAT16 || t.logical_shape().rank() != 2 ||
            t.logical_shape() != t.padded_shape() ||
            (addend &&
             (!plain(*addend) || addend->dtype() != t.dtype() ||
              addend->padded_shape() != t.padded_shape())) ||
            !ktt_gemm::supports(t, w, w_nk)) {
            return std::nullopt;
        }
        Fuse fuse{w, w_nk, config, std::nullopt};
        auto x = exchange(t, addend, false, &fuse);
        if (!x) {
            return std::nullopt;
        }
        return std::pair{*x, *fuse.product};
    }

    namespace {

        std::optional<ttnn::Tensor> exchange(
            const ttnn::Tensor& t,
            const ttnn::Tensor* addend,
            bool scatter,
            Fuse* fuse,
            bool columns
        ) {
            using namespace tt::tt_metal;
            auto* mesh = detail::device();
            const auto& shape = mesh->shape();
            if (shape.dims() != 2 || shape[0] != 1 || shape[1] != 2 || !plain(t)) {
                return std::nullopt;
            }
            const auto& s = t.padded_shape();
            const int rank = static_cast<int>(s.rank());
            auto* p = plan();
            if (!p || rank < 2) {
                return std::nullopt;
            }
            const uint32_t tiles = static_cast<uint32_t>(t.physical_volume() / 1024);
            // Tiles each chip keeps and receives.
            const uint32_t half = scatter ? tiles / 2 : tiles;
            const uint32_t per_link = scatter || fuse ? kDrainsPerLink
                                                      : kReduceDrainsPerLink;
            const auto math = numerics::effective();
            const auto format = datatype_to_dataformat_converter(t.dtype());
            const uint32_t tile_bytes = tt::tile_size(format);

            auto out_shape = t.logical_shape();
            if (scatter) {
                out_shape[columns ? rank - 1 : rank - 2] /= 2;
            }
            // Tiles per fabric packet (1, 2 or 4); the scratch has one page per packet.
            const auto payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
            const uint32_t pack = payload >= 4 * tile_bytes ? 4
                : payload >= 2 * tile_bytes                 ? 2
                                                            : 1;
            static_assert(kChunk % 4 == 0);
            auto scratch = ttnn::empty(
                ttnn::Shape({(half + pack - 1) / pack, pack * 1024}),
                t.dtype(),
                Layout::ROW_MAJOR,
                mesh,
                MemoryConfig{}
            );
            auto out = ttnn::empty(
                out_shape, t.dtype(), Layout::TILE, mesh, MemoryConfig{}
            );
            const ttnn::Tensor& extra = addend ? *addend : t;

            // Fused matmul: the tiles go column by column (rows x cols tiles), each
            // link a whole number of K blocks; the drains sit in rows 0 and 1.
            uint32_t order_rows = 0, order_cols = 0, out_cols = 0, block = 0;
            if (columns) {
                // Tiles go column by column; the output has half the columns.
                order_rows = s[rank - 2] / 32;
                order_cols = s[rank - 1] / 32;
                out_cols = order_cols / 2;
            }
            ktt_gemm::Stream stream;
            std::vector<std::vector<CoreCoord>> fused_drains;
            if (fuse) {
                const auto grid = mesh->compute_with_storage_grid_size();
                order_rows = s[rank - 2] / 32;
                order_cols = out_cols = s[rank - 1] / 32;
                stream.y0 = kFuseRow;
                stream.parts = p->links;
                stream.progress = kFuseProgress;
                stream.per_block = per_link;
                if (grid.y <= kFuseRow || order_cols % p->links != 0) {
                    return std::nullopt;
                }
                auto product = ktt_gemm::run(
                    out, fuse->w, fuse->w_nk, MemoryConfig{}, fuse->config, &stream
                );
                if (!product) {
                    return std::nullopt;
                }
                fuse->product = *product;
                block = stream.kb * order_rows;
                std::vector<CoreCoord> pool;
                for (uint32_t y = 0; y < kFuseRow; ++y) {
                    for (uint32_t x = 0; x < grid.x; ++x) {
                        bool sender = false;
                        for (const auto& chip : p->senders) {
                            for (const auto& core : chip) {
                                sender = sender || (core.x == x && core.y == y);
                            }
                        }
                        if (!sender) {
                            pool.emplace_back(x, y);
                        }
                    }
                }
                if (pool.size() < p->links * per_link || block < per_link * kChunk ||
                    block % kChunk != 0 || (tiles / p->links) % block != 0 ||
                    stream.a_senders.size() > 16) {
                    return std::nullopt;
                }
                fused_drains.resize(p->links);
                for (uint32_t k = 0; k < p->links * per_link; ++k) {
                    fused_drains[k % p->links].push_back(pool[k]);
                }
            }

            // Per link: tiles [first[l], first[l] + count[l]) of the exchanged tiles,
            // whole chunks but for the last tiles.
            std::vector<uint32_t> first(p->links), count(p->links);
            const uint32_t chunks = (half + kChunk - 1) / kChunk;
            for (uint32_t l = 0, at = 0; l < p->links; ++l) {
                const uint32_t n = chunks / p->links + (l < chunks % p->links ? 1 : 0);
                count[l] = std::min(n * kChunk, half - at);
                first[l] = at;
                at += count[l];
            }
            const auto drains_of = [&](uint32_t l) {
                if (fuse) {
                    return fused_drains[l];
                }
                return std::vector<CoreCoord>(
                    p->drains[l].begin(), p->drains[l].begin() + per_link
                );
            };

            const ttnn::MeshCoordinate coords[2] = {{0, 0}, {0, 1}};
            tt::tt_metal::experimental::MeshProgramDescriptor mesh_desc;
            for (int c = 0; c < 2; ++c) {
                const int peer = 1 - c;
                const auto src_node = mesh->get_fabric_node_id(coords[c]);
                const auto dst_node = mesh->get_fabric_node_id(coords[peer]);
                const auto links = tt::tt_fabric::get_forwarding_link_indices(
                    src_node, dst_node
                );

                ProgramDescriptor desc;
                std::vector<CoreRange> sender_ranges, drain_ranges;
                for (uint32_t l = 0; l < p->links; ++l) {
                    sender_ranges.emplace_back(p->senders[c][l]);
                    for (const auto& core : drains_of(l)) {
                        drain_ranges.emplace_back(core);
                    }
                }
                const CoreRangeSet senders(sender_ranges);
                const auto drains = CoreRangeSet(drain_ranges).merge_ranges();
                auto cb =
                    [&](const CoreRangeSet& cores, uint8_t index, uint32_t tiles) {
                        desc.cbs.push_back(
                            CBDescriptor{
                                .total_size = 2 * tiles * tile_bytes,
                                .core_ranges = cores,
                                .format_descriptors = {CBFormatDescriptor{
                                    .buffer_index = index,
                                    .data_format = format,
                                    .page_size = tile_bytes
                                }}
                            }
                        );
                    };
                cb(senders, 0, kSendBatch);
                for (const uint8_t index : {0, 1, 2, 16}) {
                    if (index != 2 || addend) {
                        cb(drains, index, kChunk);
                    }
                }

                // Senders: reader (DRAM -> CB 0) and writer (CB 0 -> fabric).
                auto send_reader_ct = TensorAccessorArgs(*t.buffer())
                                          .get_compile_time_args();
                send_reader_ct.push_back(8);
                auto send_writer_ct = TensorAccessorArgs(*scratch.buffer())
                                          .get_compile_time_args();
                send_writer_ct.push_back(11);
                KernelDescriptor::RuntimeArgs send_reader_args, send_writer_args;
                for (uint32_t l = 0; l < p->links; ++l) {
                    const auto core = p->senders[c][l];
                    send_reader_args.push_back(
                        {core,
                         {t.buffer()->address(),
                          scatter ? static_cast<uint32_t>(peer) * half : 0,
                          first[l],
                          count[l],
                          tile_bytes,
                          kSendBatch,
                          order_rows,
                          order_cols}}
                    );
                    const auto peer_sender = (mesh->worker_core_from_logical_core(
                        p->senders[peer][l]
                    ));
                    std::vector<uint32_t> args = {
                        scratch.buffer()->address(),
                        first[l],
                        count[l],
                        tile_bytes,
                        kChunk,
                        per_link,
                        static_cast<uint32_t>(p->drain_sem->address()),
                        static_cast<uint32_t>(peer_sender.x),
                        static_cast<uint32_t>(peer_sender.y),
                        static_cast<uint32_t>(p->ready_sem->address()),
                        pack,
                    };
                    for (const auto& drain : drains_of(l)) {
                        const auto v = mesh->worker_core_from_logical_core(drain);
                        args.push_back(static_cast<uint32_t>(v.x));
                        args.push_back(static_cast<uint32_t>(v.y));
                    }
                    tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                        src_node, dst_node, links[l], desc, core, args
                    );
                    send_writer_args.push_back({core, std::move(args)});
                }
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "../core/mesh/ktt_reduce_scatter/send_reader.cpp"
                        ),
                        .core_ranges = senders,
                        .compile_time_args = send_reader_ct,
                        .runtime_args = send_reader_args,
                        .config = dm(DataMovementProcessor::RISCV_1, kReaderNoc)
                    }
                );
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "../core/mesh/ktt_reduce_scatter/send_writer.cpp"
                        ),
                        .core_ranges = senders,
                        .compile_time_args = send_writer_ct,
                        .runtime_args = send_writer_args,
                        .config = dm(DataMovementProcessor::RISCV_0, kWriterNoc)
                    }
                );

                // Drains: reader (kept, scratch, addend -> CB 1, 0, 2), compute (fp32
                // adds -> CB 16), writer (CB 16 -> output).
                auto drain_reader_ct = TensorAccessorArgs(*scratch.buffer())
                                           .get_compile_time_args();
                for (const auto* source : {&t, &extra}) {
                    const auto ct = TensorAccessorArgs(*source->buffer())
                                        .get_compile_time_args();
                    drain_reader_ct.insert(drain_reader_ct.end(), ct.begin(), ct.end());
                }
                drain_reader_ct.push_back(15);
                auto drain_writer_ct = TensorAccessorArgs(*out.buffer())
                                           .get_compile_time_args();
                drain_writer_ct.push_back(12);
                KernelDescriptor::RuntimeArgs reader_args, compute_args, writer_args;
                for (uint32_t l = 0; l < p->links; ++l) {
                    const auto link_drains = drains_of(l);
                    for (uint32_t d = 0; d < per_link; ++d) {
                        const auto core = link_drains[d];
                        reader_args.push_back(
                            {core,
                             {scratch.buffer()->address(),
                              t.buffer()->address(),
                              scatter ? static_cast<uint32_t>(c) * half : 0,
                              first[l],
                              count[l],
                              tile_bytes,
                              kChunk,
                              per_link,
                              d,
                              static_cast<uint32_t>(p->drain_sem->address()),
                              extra.buffer()->address(),
                              addend ? 1u : 0u,
                              pack,
                              order_rows,
                              order_cols}}
                        );
                        compute_args.push_back({core, {count[l], kChunk, per_link, d}});
                        std::vector<uint32_t> wargs = {
                            out.buffer()->address(),
                            first[l],
                            count[l],
                            tile_bytes,
                            kChunk,
                            per_link,
                            d,
                            order_rows,
                            out_cols,
                            block,
                            kFuseProgress + l,
                            static_cast<uint32_t>(stream.a_senders.size())
                        };
                        for (const auto& watcher : stream.a_senders) {
                            const auto v = mesh->worker_core_from_logical_core(watcher);
                            wargs.push_back(static_cast<uint32_t>(v.x));
                            wargs.push_back(static_cast<uint32_t>(v.y));
                        }
                        writer_args.push_back({core, std::move(wargs)});
                    }
                }
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "../core/mesh/ktt_reduce_scatter/drain_reader.cpp"
                        ),
                        .core_ranges = drains,
                        .compile_time_args = drain_reader_ct,
                        .runtime_args = reader_args,
                        .config = dm(DataMovementProcessor::RISCV_1, kReaderNoc)
                    }
                );
                std::vector<UnpackToDestMode> unpack(
                    NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default
                );
                if (t.dtype() == DataType::FLOAT32) {
                    unpack[0] = unpack[1] = unpack[2] = UnpackToDestMode::
                        UnpackToDestFp32;
                }
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "../core/mesh/ktt_reduce_scatter/drain_compute.cpp"
                        ),
                        .core_ranges = drains,
                        .compile_time_args = {addend ? 1u : 0u},
                        .runtime_args = compute_args,
                        .config = ComputeConfigDescriptor{
                            .math_fidelity = math.fidelity,
                            .fp32_dest_acc_en = true,
                            .unpack_to_dest_mode = unpack,
                            .math_approx_mode = math.approx
                        }
                    }
                );
                desc.kernels.push_back(
                    KernelDescriptor{
                        .kernel_source = program::source(
                            "../core/mesh/ktt_reduce_scatter/drain_writer.cpp"
                        ),
                        .core_ranges = drains,
                        .compile_time_args = drain_writer_ct,
                        .runtime_args = writer_args,
                        .config = dm(DataMovementProcessor::RISCV_0, kWriterNoc)
                    }
                );
                if (fuse) {
                    desc.kernels.insert(
                        desc.kernels.end(),
                        stream.desc.kernels.begin(),
                        stream.desc.kernels.end()
                    );
                    desc.cbs.insert(
                        desc.cbs.end(), stream.desc.cbs.begin(), stream.desc.cbs.end()
                    );
                    desc.semaphores.insert(
                        desc.semaphores.end(),
                        stream.desc.semaphores.begin(),
                        stream.desc.semaphores.end()
                    );
                }
                mesh_desc.mesh_programs.emplace_back(
                    ttnn::MeshCoordinateRange(coords[c], coords[c]), std::move(desc)
                );
            }
            // Program hash: the structural hash (kernels, cores, CBs, argument counts)
            // with the tile geometry and plan folded in (tt-metal#60938).
            std::vector<ttnn::Tensor> io = {t, scratch};
            if (addend) {
                io.push_back(*addend);
            }
            if (fuse) {
                io.push_back(fuse->w);
                io.push_back(*fuse->product);
            }
            io.push_back(out);
            const ttnn::operations::generic::tensor_args_t args{io, io.back()};
            const size_t
                structural = (ttnn::operations::generic::GenericOpDeviceOperation::
                                  compute_program_hash(mesh_desc, args));
            const std::string key =
                ("ktt_reduce_scatter|" + std::to_string(tiles) + "|" +
                 std::to_string(scatter) + std::to_string(columns) + "|" +
                 std::to_string(addend != nullptr) + "|" + std::to_string(tile_bytes) +
                 "|" + std::to_string(p->links) + "|" + std::to_string(per_link) + "|" +
                 std::to_string(kChunk) + "|" + std::to_string(kSendBatch) + "|" +
                 std::to_string(pack) + "|" + stream.plan);
            for (auto& [range, desc] : mesh_desc.mesh_programs) {
                desc.custom_program_hash =
                    (structural ^
                     (std::hash<std::string>{}(key) + 0x9e3779b97f4a7c15ull +
                      (structural << 6) + (structural >> 2)));
            }
            return ttnn::generic_op(io, mesh_desc);
        }

    }  // namespace

}  // namespace ktt::bridge::mesh_rs
