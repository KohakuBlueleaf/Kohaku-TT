// Kohaku-TT's own fp32 reduce-scatter on a 1x2 mesh (rows split): per fabric link, a
// sender core in the Tensix row under the link's Ethernet tile reads this chip's tiles of
// the peer's half (reader RISC) and sends them into the peer's scratch (writer RISC);
// drain cores add each received tile to the kept tile on the SFPU in fp32 and write the
// output shard.
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
        // Drain cores per link; tiles per drain chunk (drain CBs hold two); tiles per
        // sender read batch (the sender CB holds two).
        constexpr uint32_t kDrainsPerLink = 4;
        constexpr uint32_t kChunk = 4;
        constexpr uint32_t kSendBatch = 32;
        static_assert(kSendBatch % kChunk == 0);

        // NoCs: readers (DRAM reads) on NoC0, writers (fabric sends, DRAM writes) on the
        // other one (dedicated NoC mode: one RISC per NoC).
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
        // virtual x of Ethernet core `eth` (NoC1 reaches the Ethernet row in one -y hop,
        // then -x); marks it used. Coordinates from the device's SoC descriptor.
        CoreCoord under(
            tt::tt_metal::IDevice* device,
            const CoreCoord& eth,
            uint32_t columns,
            std::vector<bool>& used
        ) {
            const auto eth_x = static_cast<int64_t>(device->ethernet_core_from_logical_core(eth).x);
            int64_t best = -1, best_cost = 0;
            for (uint32_t x = 0; x < columns; ++x) {
                if (used[x]) {
                    continue;
                }
                const auto wx = static_cast<int64_t>(device->worker_core_from_logical_core({x, 0}).x);
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
            const auto n0 = mesh->get_fabric_node_id(c0), n1 = mesh->get_fabric_node_id(c1);
            const auto l01 = tt::tt_fabric::get_forwarding_link_indices(n0, n1);
            const auto l10 = tt::tt_fabric::get_forwarding_link_indices(n1, n0);
            p->links = static_cast<uint32_t>(
                std::min<size_t>({l01.size(), l10.size(), kMaxLinks})
            );
            const auto grid = mesh->compute_with_storage_grid_size();
            if (p->links == 0 || grid.y < 2 ||
                (grid.y - 1) * grid.x < p->links * kDrainsPerLink) {
                return nullptr;
            }
            p->senders.resize(2);
            for (int c = 0; c < 2; ++c) {
                const auto src = c == 0 ? n0 : n1, dst = c == 0 ? n1 : n0;
                const auto& links = c == 0 ? l01 : l10;
                auto* device = mesh->get_device(c == 0 ? c0 : c1);
                std::vector<bool> used(grid.x, false);
                for (uint32_t l = 0; l < p->links; ++l) {
                    const auto eth = (
                        tt::tt_fabric::get_forwarding_eth_core(*mesh, src, dst, links[l])
                    );
                    p->senders[c].push_back(under(device, eth, grid.x, used));
                }
            }
            // Drains: rows 1 and on, row-major.
            p->drains.resize(p->links);
            for (uint32_t k = 0; k < p->links * kDrainsPerLink; ++k) {
                const CoreCoord core(k % grid.x, 1 + k / grid.x);
                p->drains[k / kDrainsPerLink].push_back(core);
            }
            // The ready semaphore lives on the senders of both chips (one set).
            std::set<std::pair<uint32_t, uint32_t>> sender_cores;
            for (const auto& s : p->senders) {
                for (const auto& core : s) {
                    sender_cores.insert({core.x, core.y});
                }
            }
            std::vector<CoreRange> sender_ranges, drain_ranges;
            for (const auto& [x, y] : sender_cores) {
                sender_ranges.emplace_back(CoreCoord(x, y));
            }
            for (const auto& link : p->drains) {
                for (const auto& core : link) {
                    drain_ranges.emplace_back(core);
                }
            }
            p->ready_sem = tt::tt_metal::CreateGlobalSemaphore(
                *mesh, CoreRangeSet(sender_ranges).merge_ranges(), 0
            );
            p->drain_sem = tt::tt_metal::CreateGlobalSemaphore(
                *mesh, CoreRangeSet(drain_ranges).merge_ranges(), 0
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

    std::optional<ttnn::Tensor> reduce_scatter_rows(const ttnn::Tensor& t, uint32_t axis) {
        using namespace tt::tt_metal;
        auto* mesh = detail::device();
        const auto& shape = mesh->shape();
        if (shape.dims() != 2 || shape[0] != 1 || shape[1] != 2 ||
            axis != 1 || t.dtype() != DataType::FLOAT32 || t.layout() != Layout::TILE ||
            t.memory_config().buffer_type() != BufferType::DRAM ||
            t.memory_config().is_sharded()) {
            return std::nullopt;
        }
        const auto& s = t.padded_shape();
        const int rank = static_cast<int>(s.rank());
        for (int k = 0; k + 2 < rank; ++k) {
            if (s[k] != 1) {
                return std::nullopt;
            }
        }
        if (s[rank - 2] % 64 != 0) {
            return std::nullopt;
        }
        auto* p = plan();
        if (!p) {
            return std::nullopt;
        }
        const uint32_t ht = s[rank - 2] / 32, wt = s[rank - 1] / 32;
        const uint32_t half = ht / 2 * wt;
        const auto math = numerics::effective();
        const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float32);

        auto out_shape = t.logical_shape();
        out_shape[rank - 2] /= 2;
        auto scratch = ttnn::empty(out_shape, DataType::FLOAT32, Layout::TILE, mesh, MemoryConfig{});
        auto out = ttnn::empty(out_shape, DataType::FLOAT32, Layout::TILE, mesh, MemoryConfig{});

        // Per link: tiles [first[l], first[l] + count[l]) of the half.
        std::vector<uint32_t> first(p->links), count(p->links);
        for (uint32_t l = 0, at = 0; l < p->links; ++l) {
            count[l] = half / p->links + (l < half % p->links ? 1 : 0);
            first[l] = at;
            at += count[l];
        }

        const ttnn::MeshCoordinate coords[2] = {{0, 0}, {0, 1}};
        tt::tt_metal::experimental::MeshProgramDescriptor mesh_desc;
        for (int c = 0; c < 2; ++c) {
            const int peer = 1 - c;
            const auto src_node = mesh->get_fabric_node_id(coords[c]);
            const auto dst_node = mesh->get_fabric_node_id(coords[peer]);
            const auto links = tt::tt_fabric::get_forwarding_link_indices(src_node, dst_node);

            ProgramDescriptor desc;
            std::vector<CoreRange> sender_ranges, drain_ranges;
            for (uint32_t l = 0; l < p->links; ++l) {
                sender_ranges.emplace_back(p->senders[c][l]);
                for (const auto& core : p->drains[l]) {
                    drain_ranges.emplace_back(core);
                }
            }
            const CoreRangeSet senders(sender_ranges), drains(drain_ranges);
            auto cb = [&](const CoreRangeSet& cores, uint8_t index, uint32_t tiles) {
                desc.cbs.push_back(CBDescriptor{
                    .total_size = 2 * tiles * tile_bytes,
                    .core_ranges = cores,
                    .format_descriptors = {CBFormatDescriptor{
                        .buffer_index = index,
                        .data_format = tt::DataFormat::Float32,
                        .page_size = tile_bytes
                    }}
                });
            };
            cb(senders, 0, kSendBatch);
            for (const uint8_t index : {0, 1, 16}) {
                cb(drains, index, kChunk);
            }

            // Senders: reader (DRAM -> CB 0) and writer (CB 0 -> fabric).
            auto send_reader_ct = TensorAccessorArgs(*t.buffer()).get_compile_time_args();
            send_reader_ct.push_back(6);
            auto send_writer_ct = TensorAccessorArgs(*scratch.buffer()).get_compile_time_args();
            send_writer_ct.push_back(10);
            KernelDescriptor::RuntimeArgs send_reader_args, send_writer_args;
            for (uint32_t l = 0; l < p->links; ++l) {
                const auto core = p->senders[c][l];
                send_reader_args.push_back({core, {
                    t.buffer()->address(),
                    static_cast<uint32_t>(peer) * half,
                    first[l],
                    count[l],
                    tile_bytes,
                    kSendBatch
                }});
                const auto peer_sender = (
                    mesh->worker_core_from_logical_core(p->senders[peer][l])
                );
                std::vector<uint32_t> args = {
                    scratch.buffer()->address(),
                    first[l],
                    count[l],
                    tile_bytes,
                    kChunk,
                    kDrainsPerLink,
                    static_cast<uint32_t>(p->drain_sem->address()),
                    static_cast<uint32_t>(peer_sender.x),
                    static_cast<uint32_t>(peer_sender.y),
                    static_cast<uint32_t>(p->ready_sem->address()),
                };
                for (const auto& drain : p->drains[l]) {
                    const auto v = mesh->worker_core_from_logical_core(drain);
                    args.push_back(static_cast<uint32_t>(v.x));
                    args.push_back(static_cast<uint32_t>(v.y));
                }
                tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                    src_node, dst_node, links[l], desc, core, args
                );
                send_writer_args.push_back({core, std::move(args)});
            }
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("../core/mesh/ktt_reduce_scatter/send_reader.cpp"),
                .core_ranges = senders,
                .compile_time_args = send_reader_ct,
                .runtime_args = send_reader_args,
                .config = dm(DataMovementProcessor::RISCV_1, kReaderNoc)
            });
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("../core/mesh/ktt_reduce_scatter/send_writer.cpp"),
                .core_ranges = senders,
                .compile_time_args = send_writer_ct,
                .runtime_args = send_writer_args,
                .config = dm(DataMovementProcessor::RISCV_0, kWriterNoc)
            });

            // Drains: reader (kept + scratch -> CB 1, 0), compute (fp32 add -> CB 16),
            // writer (CB 16 -> output).
            auto drain_reader_ct = TensorAccessorArgs(*scratch.buffer()).get_compile_time_args();
            const auto src_ct = TensorAccessorArgs(*t.buffer()).get_compile_time_args();
            drain_reader_ct.insert(drain_reader_ct.end(), src_ct.begin(), src_ct.end());
            drain_reader_ct.push_back(10);
            auto drain_writer_ct = TensorAccessorArgs(*out.buffer()).get_compile_time_args();
            drain_writer_ct.push_back(7);
            KernelDescriptor::RuntimeArgs reader_args, compute_args, writer_args;
            for (uint32_t l = 0; l < p->links; ++l) {
                for (uint32_t d = 0; d < kDrainsPerLink; ++d) {
                    const auto core = p->drains[l][d];
                    reader_args.push_back({core, {
                        scratch.buffer()->address(),
                        t.buffer()->address(),
                        static_cast<uint32_t>(c) * half,
                        first[l],
                        count[l],
                        tile_bytes,
                        kChunk,
                        kDrainsPerLink,
                        d,
                        static_cast<uint32_t>(p->drain_sem->address())
                    }});
                    compute_args.push_back({core, {count[l], kChunk, kDrainsPerLink, d}});
                    writer_args.push_back({core, {
                        out.buffer()->address(),
                        first[l],
                        count[l],
                        tile_bytes,
                        kChunk,
                        kDrainsPerLink,
                        d
                    }});
                }
            }
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("../core/mesh/ktt_reduce_scatter/drain_reader.cpp"),
                .core_ranges = drains,
                .compile_time_args = drain_reader_ct,
                .runtime_args = reader_args,
                .config = dm(DataMovementProcessor::RISCV_1, kReaderNoc)
            });
            std::vector<UnpackToDestMode> unpack(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
            unpack[0] = unpack[1] = UnpackToDestMode::UnpackToDestFp32;
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("../core/mesh/ktt_reduce_scatter/drain_compute.cpp"),
                .core_ranges = drains,
                .runtime_args = compute_args,
                .config = ComputeConfigDescriptor{
                    .math_fidelity = math.fidelity,
                    .fp32_dest_acc_en = true,
                    .unpack_to_dest_mode = unpack,
                    .math_approx_mode = math.approx
                }
            });
            desc.kernels.push_back(KernelDescriptor{
                .kernel_source = program::source("../core/mesh/ktt_reduce_scatter/drain_writer.cpp"),
                .core_ranges = drains,
                .compile_time_args = drain_writer_ct,
                .runtime_args = writer_args,
                .config = dm(DataMovementProcessor::RISCV_0, kWriterNoc)
            });
            mesh_desc.mesh_programs.emplace_back(
                ttnn::MeshCoordinateRange(coords[c], coords[c]), std::move(desc)
            );
        }
        // Program hash: the structural hash (kernels, cores, CBs, argument counts) with
        // the tile geometry and plan folded in (tt-metal#60938).
        const std::vector<ttnn::Tensor> io = {t, scratch, out};
        const ttnn::operations::generic::tensor_args_t args{io, io.back()};
        const size_t structural = (
            ttnn::operations::generic::GenericOpDeviceOperation::compute_program_hash(mesh_desc, args)
        );
        const std::string key = (
            "ktt_reduce_scatter|" + std::to_string(ht) + "|" + std::to_string(wt) + "|" +
            std::to_string(p->links) + "|" + std::to_string(kDrainsPerLink) + "|" +
            std::to_string(kChunk) + "|" + std::to_string(kSendBatch)
        );
        for (auto& [range, desc] : mesh_desc.mesh_programs) {
            desc.custom_program_hash = (
                structural ^ (std::hash<std::string>{}(key) + 0x9e3779b97f4a7c15ull +
                              (structural << 6) + (structural >> 2))
            );
        }
        return ttnn::generic_op(io, mesh_desc);
    }

}  // namespace ktt::bridge::mesh_rs
