// Bridge device ownership: the process's mesh device (bound as tt:0) and its stream,
// sync, counters and memory statistics.
#include <algorithm>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "core/device/handle.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"

namespace ktt::bridge {
    namespace {
        using DevicePtr = std::shared_ptr<ttnn::MeshDevice>;

        // Never destroyed at exit (after TT-Metal's teardown); close() resets it.
        DevicePtr& g_device = *new DevicePtr();
        std::atomic<bool> g_exiting{false};
        detail::AtomicCounters& g_counters = *new detail::AtomicCounters();
        std::atomic<int64_t> g_live_bytes{0};
        std::atomic<int64_t> g_peak_bytes{0};
    }  // namespace

    namespace detail {
        bool exiting() {
            return g_exiting;
        }

        ttnn::MeshDevice* device() {
            if (!g_device) {
                throw std::runtime_error("kohakutt_native: device is not open");
            }
            return g_device.get();
        }

        AtomicCounters& counters() {
            return g_counters;
        }

        std::optional<tt::tt_metal::CoreRangeSet> worker_cores(
            const std::vector<int64_t>& shape, int64_t tiles_per_core
        ) {
            using tt::tt_metal::CoreCoord;
            using tt::tt_metal::CoreRange;
            using tt::tt_metal::CoreRangeSet;

            // Core sets of the first n cores (row by row), built once per n.
            static std::vector<std::optional<CoreRangeSet>> sets;
            const auto grid = device()->compute_with_storage_grid_size();
            const int64_t total = static_cast<int64_t>(grid.x * grid.y);
            if (static_cast<int64_t>(sets.size()) != total + 1) {
                sets.assign(total + 1, std::nullopt);
                for (int64_t n = 1; n < total; ++n) {
                    std::vector<CoreRange> ranges;
                    const int64_t rows = n / grid.x, rest = n % grid.x;
                    if (rows > 0) {
                        ranges.emplace_back(
                            CoreCoord(0, 0), CoreCoord(grid.x - 1, rows - 1)
                        );
                    }
                    if (rest > 0) {
                        ranges.emplace_back(
                            CoreCoord(0, rows), CoreCoord(rest - 1, rows)
                        );
                    }
                    sets[n] = CoreRangeSet(ranges);
                }
            }

            // Tiles of the device shape: leading dims times tile rows times tile
            // columns.
            int64_t tiles = 1;
            const size_t r = shape.size();
            for (size_t i = 0; i + 2 < r; ++i) {
                tiles *= shape[i];
            }
            tiles *= ((shape[r - 2] + 31) / 32) * ((shape[r - 1] + 31) / 32);
            const int64_t n = (
                std::clamp<int64_t>(
                    (tiles + tiles_per_core - 1) / tiles_per_core, 1, total
                )
            );
            return sets[n];
        }

        void track_bytes(int64_t delta) {
            const int64_t live = g_live_bytes += delta;
            int64_t peak = g_peak_bytes;
            while (live > peak && !g_peak_bytes.compare_exchange_weak(peak, live)) {
                ;
            }
        }

        void check_shape(const Handle& h) {
            const auto& actual = h.tensor.logical_shape();
            std::vector<int64_t> got(actual.cbegin(), actual.cend());
            if (got != h.shape) {
                std::ostringstream msg;
                msg << "kohakutt_native: op produced device shape [";
                for (auto d : got) {
                    msg << d << ",";
                }
                msg << "], predicted [";
                for (auto d : h.shape) {
                    msg << d << ",";
                }
                msg << "]";
                throw std::logic_error(msg.str());
            }
        }
    }  // namespace detail

    void open(const std::vector<int>& device_ids, int rows, int cols) {
        if (g_device) {
            throw std::runtime_error("kohakutt_native: device already open");
        }
        if (static_cast<int>(device_ids.size()) != rows * cols) {
            throw std::invalid_argument(
                "kohakutt_native: device_ids must fill the mesh"
            );
        }
        // Opened on the caller's thread: TT-Metal's chip lock belongs to the opening
        // thread and is released by process teardown on the main thread.
        // KTT_L1_SMALL: bytes per core reserved for TT-NN's small L1 buffers (conv
        // sliding-window metadata), default 96 KiB.
        const char* small = std::getenv("KTT_L1_SMALL");
        const size_t l1_small = small ? std::strtoull(small, nullptr, 10) : 98304;
        g_device = ttnn::distributed::open_mesh_device(
            ttnn::MeshShape(rows, cols),
            l1_small,   // l1_small_size
            256 << 20,  // trace_region_size
            1,          // num_command_queues
            tt::tt_metal::DispatchCoreConfig{},
            std::nullopt,
            device_ids
        );
        detail::start_stream();
    }

    void mark_exiting() {
        try {
            detail::drain();
        } catch (...) {
            // Errors of queued work are moot at exit.
        }
        detail::stop_stream();
        g_exiting = true;
    }

    void close() {
        if (!g_device) {
            return;
        }
        detail::drain();
        detail::release_traces();
        detail::clear_constants();
        detail::clear_ranges();
        detail::clear_row_major();
        detail::stop_stream();  // drains the releases above
        ttnn::distributed::close_mesh_device(g_device);
        g_device.reset();
    }

    bool is_open() {
        return static_cast<bool>(g_device);
    }

    std::vector<int64_t> compute_grid() {
        std::vector<int64_t> grid;
        detail::run_sync([&] {
            auto g = detail::device()->compute_with_storage_grid_size();
            grid = {static_cast<int64_t>(g.x), static_cast<int64_t>(g.y)};
        });
        return grid;
    }

    int mesh_size() {
        return g_device ? static_cast<int>(g_device->num_devices()) : 0;
    }

    void synchronize() {
        detail::run_sync([] {
            tt::tt_metal::distributed::Synchronize(*detail::device(), std::nullopt);
        });
    }

    Counters counters() {
        return Counters{
            g_counters.uploads.load(),
            g_counters.downloads.load(),
            g_counters.ops.load(),
            g_counters.op_ns.load()
        };
    }

    void clear_program_cache() {
        detail::run_sync([] {
            detail::device()->clear_program_cache();
        });
    }

    void read_device_profiler() {
        detail::run_sync([] {
            tt::tt_metal::ReadMeshDeviceProfilerResults(*detail::device());
        });
    }

    namespace detail {

        int64_t l1_free_per_bank() {
            const auto& allocator = device()->allocator();
            const auto type = tt::tt_metal::BufferType::L1;
            const auto stats = allocator->get_statistics(type);
            return (
                static_cast<int64_t>(allocator->get_bank_size(type)) -
                static_cast<int64_t>(stats.total_allocated_bytes)
            );
        }

        namespace {
            // Every handle created (never destroyed: outlives device close at exit).
            struct Registry {
                std::mutex mutex;
                std::vector<std::weak_ptr<Handle>> handles;
                size_t pruned_at = 1024;
            };

            Registry& registry() {
                static auto& r = *new Registry;
                return r;
            }
        }  // namespace

        void register_handle(const HandlePtr& handle) {
            auto& r = registry();
            std::lock_guard lock(r.mutex);
            if (r.handles.size() >= 2 * r.pruned_at) {
                std::erase_if(r.handles, [](const auto& w) {
                    return w.expired();
                });
                r.pruned_at = std::max<size_t>(1024, r.handles.size());
            }
            r.handles.push_back(handle);
        }

        namespace {
            // Scale of KTT_L1_BUDGET, halved by every spill (stream worker only).
            double g_budget_scale = 1.0;
        }  // namespace

        bool spill_l1() {
            if (capturing()) {
                return false;
            }
            g_budget_scale /= 2;
            std::vector<HandlePtr> live;
            {
                auto& r = registry();
                std::lock_guard lock(r.mutex);
                for (const auto& weak : r.handles) {
                    if (auto h = weak.lock()) {
                        live.push_back(std::move(h));
                    }
                }
            }
            bool moved = false;
            for (const auto& h : live) {
                // Handles of queued ops hold no tensor yet.
                if (h->tensor.tensor_attributes != nullptr &&
                    h->tensor.is_allocated() &&
                    h->tensor.memory_config().buffer_type() ==
                        tt::tt_metal::BufferType::L1) {
                    const auto before = h->tensor.logical_shape();
                    const auto padded = h->tensor.padded_shape();
                    h->tensor = ttnn::to_memory_config(
                        h->tensor, tt::tt_metal::MemoryConfig{}
                    );
                    if (h->tensor.logical_shape() != before ||
                        h->tensor.padded_shape() != padded) {
                        std::ostringstream msg;
                        msg << "kohakutt_native: spill changed shape " << before << "/"
                            << padded << " -> " << h->tensor.logical_shape() << "/"
                            << h->tensor.padded_shape();
                        throw std::logic_error(msg.str());
                    }
                    moved = true;
                }
            }
            return moved;
        }

        namespace {
            thread_local int g_dram_only = 0;
            // Trace preparation / capture active (stream worker only): activations in
            // DRAM, the same placement in both passes.
            bool g_trace_dram = false;
        }  // namespace

        void trace_placement(bool dram) {
            post([dram] {
                g_trace_dram = dram;
            });
        }

        DramOnly::DramOnly() {
            ++g_dram_only;
        }

        DramOnly::~DramOnly() {
            --g_dram_only;
        }

        bool is_l1_small_full(const std::exception& e) {
            const std::string what = e.what();
            return (
                what.find("Out of Memory") != std::string::npos &&
                what.find("L1_SMALL buffer") != std::string::npos
            );
        }

        bool is_l1_clash(const std::exception& e) {
            const std::string what = e.what();
            if (is_l1_small_full(e)) {
                return false;
            }
            return (
                what.find("clash with L1 buffers") != std::string::npos ||
                what.find("beyond max L1 size") != std::string::npos ||
                (what.find("Out of Memory") != std::string::npos &&
                 what.find("L1 buffer") != std::string::npos)
            );
        }

        tt::tt_metal::MemoryConfig activation_config(int64_t bytes) {
            // KTT_L1_BUDGET: the fraction of each L1 bank activations may fill (default
            // 0.4; 0 keeps every activation in DRAM), halved by every spill.
            static const double budget = [] {
                const char* v = std::getenv("KTT_L1_BUDGET");
                return v ? std::atof(v) : 0.4;
            }();
            const auto& allocator = device()->allocator();
            const auto type = tt::tt_metal::BufferType::L1;
            const auto banks = static_cast<int64_t>(allocator->get_num_banks(type));
            const auto bank = static_cast<int64_t>(allocator->get_bank_size(type));
            const int64_t per_bank = (bytes + banks - 1) / banks + 4096;
            const auto allocated = (
                static_cast<int64_t>(
                    allocator->get_statistics(type).total_allocated_bytes
                )
            );
            if (budget > 0 && g_dram_only == 0 && !g_trace_dram &&
                allocated + per_bank <=
                    static_cast<int64_t>(budget * g_budget_scale * bank)) {
                return (
                    tt::tt_metal::MemoryConfig(
                        tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                        tt::tt_metal::BufferType::L1
                    )
                );
            }
            return tt::tt_metal::MemoryConfig{};
        }

    }  // namespace detail

    MemoryStats memory_stats() {
        // Queued work holds its operands: drained so only live tensors count.
        detail::drain();
        return MemoryStats{g_live_bytes.load(), g_peak_bytes.load()};
    }

    void reset_peak_memory() {
        detail::drain();
        g_peak_bytes = g_live_bytes.load();
    }

}  // namespace ktt::bridge
