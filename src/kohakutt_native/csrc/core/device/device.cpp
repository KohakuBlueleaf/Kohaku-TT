// Bridge device ownership: the process's mesh device (bound as tt:0) and its stream,
// sync, counters and memory statistics.
#include <algorithm>
#include <any>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/host_api.hpp>

#include "core/device/device.hpp"
#include "core/device/handle.hpp"
#include "core/mesh/reduce_scatter.hpp"
#include "core/device/program.hpp"
#include "jit_build/build_cache_telemetry.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"

namespace ktt::bridge {
    namespace {
        using DevicePtr = std::shared_ptr<ttnn::MeshDevice>;

        // Never destroyed at exit (after TT-Metal's teardown); close() resets it.
        DevicePtr& g_device = *new DevicePtr();
        // True while the open mesh runs with fabric enabled by open().
        bool g_fabric = false;
        std::atomic<bool> g_exiting{false};
        detail::AtomicCounters& g_counters = *new detail::AtomicCounters();
        std::atomic<int64_t> g_live_bytes{0};
        std::atomic<int64_t> g_peak_bytes{0};

        // Host time of each TT-NN device-operation launch by phase (launch_phases()),
        // from GraphTracker events on the launching thread.
        class LaunchPhases final : public tt::tt_metal::IGraphProcessor {
        public:
            using Clock = std::chrono::steady_clock;

            struct Totals {
                int64_t count = 0;
                int64_t alloc_ns = 0;
                int64_t args_ns = 0;
                int64_t enqueue_ns = 0;
            };

            bool is_capture_processor() const override {
                return false;
            }

            void track_allocate(const tt::tt_metal::Buffer*) override {
                if (!frames().empty()) {
                    frames().back().allocated = Clock::now();
                }
            }

            void track_program(
                tt::tt_metal::Program*, const tt::tt_metal::IDevice*
            ) override {
                if (!frames().empty() && !frames().back().program) {
                    frames().back().program = Clock::now();
                }
            }

            void track_function_start(
                std::string_view name, std::span<tt::tt_metal::TrackedArgument>
            ) override {
                const auto now = Clock::now();
                frames().push_back({std::string(name), now, now, std::nullopt});
            }

            void track_function_end() override {
                finish();
            }

            void track_function_end(const std::any&) override {
                finish();
            }

            std::map<std::string, Totals> totals() {
                std::lock_guard<std::mutex> lock(mutex_);
                return totals_;
            }

            void reset() {
                std::lock_guard<std::mutex> lock(mutex_);
                totals_.clear();
            }

        private:
            struct Frame {
                std::string name;
                Clock::time_point start;
                Clock::time_point allocated;
                std::optional<Clock::time_point> program;
            };

            static std::vector<Frame>& frames() {
                thread_local std::vector<Frame> stack;
                return stack;
            }

            void finish() {
                auto& stack = frames();
                if (stack.empty()) {
                    return;
                }
                const auto now = Clock::now();
                Frame f = std::move(stack.back());
                stack.pop_back();
                if (!f.program) {
                    // A nested function without programs: its allocations count as
                    // the enclosing launch's.
                    if (!stack.empty() && f.allocated > f.start) {
                        stack.back().allocated = f.allocated;
                    }
                    return;
                }
                using std::chrono::duration_cast, std::chrono::nanoseconds;
                const auto ns = [](auto d) {
                    return duration_cast<nanoseconds>(d).count();
                };
                const auto allocated = std::min(f.allocated, *f.program);
                std::lock_guard<std::mutex> lock(mutex_);
                auto& t = totals_[f.name];
                t.count++;
                t.alloc_ns += ns(allocated - f.start);
                t.args_ns += ns(*f.program - allocated);
                t.enqueue_ns += ns(now - *f.program);
            }

            std::mutex mutex_;
            std::map<std::string, Totals> totals_;
        };

        // Registered while a device is open with KTT_LAUNCH_PHASES=1.
        std::shared_ptr<LaunchPhases>& g_phases = *new std::shared_ptr<LaunchPhases>();
    }  // namespace

    std::vector<std::string> launch_phases() {
        std::vector<std::string> out;
        if (!g_phases) {
            return out;
        }
        detail::drain();
        for (const auto& [name, t] : g_phases->totals()) {
            out.push_back(
                name + " " + std::to_string(t.count) + " " +
                std::to_string(t.alloc_ns) + " " + std::to_string(t.args_ns) + " " +
                std::to_string(t.enqueue_ns)
            );
        }
        return out;
    }

    void reset_launch_phases() {
        if (g_phases) {
            detail::drain();
            g_phases->reset();
        }
    }

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
            const int64_t n = (std::clamp<int64_t>(
                (tiles + tiles_per_core - 1) / tiles_per_core, 1, total
            ));
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
            const auto& actual = h.stored().logical_shape();
            std::vector<int64_t> got(actual.cbegin(), actual.cend());
            const auto want =
                (h.placement.is_replicated() ? stored_dims(h.shape, h.format)
                                             : h.local);
            if (got != want) {
                std::ostringstream msg;
                msg << "kohakutt_native: op produced device shape [";
                for (auto d : got) {
                    msg << d << ",";
                }
                msg << "], predicted [";
                for (auto d : want) {
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
        // Multi-chip meshes: fabric (1-D) for TT-NN collectives; KTT_FABRIC=0 turns it
        // off.
        const char* fabric = std::getenv("KTT_FABRIC");
        g_fabric = device_ids.size() > 1 && !(fabric && std::string(fabric) == "0");
        if (g_fabric) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
        }
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
        detail::run_sync([] {
            detail::track_program_regions(true);
        });
        const char* phases = std::getenv("KTT_LAUNCH_PHASES");
        if (phases && std::string(phases) == "1") {
            g_phases = std::make_shared<LaunchPhases>();
            detail::run_sync([] {
                // GraphTracker processors are per thread: TT-NN runs on the stream.
                tt::tt_metal::GraphTracker::instance().push_processor(g_phases);
            });
        }
    }

    void mark_exiting(bool close_device) {
        // A fabric mesh (or any device when asked) is closed now, tensors alive: only
        // a close stops the fabric routers.
        if (g_device && (g_fabric || close_device)) {
            try {
                close();
            } catch (...) {
                // Errors of queued work are moot at exit.
            }
            g_exiting = true;
            return;
        }
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
        // A pending error of queued work does not stop the close; it is raised after.
        std::exception_ptr pending;
        try {
            detail::drain();
        } catch (...) {
            pending = std::current_exception();
        }
        try {
            detail::release_traces();
        } catch (...) {
            pending = pending ? pending : std::current_exception();
        }
        detail::clear_constants();
        detail::clear_ranges();
        detail::clear_row_major();
        detail::clear_conv_nhwc();
        detail::run_cleanup([] {
            mesh_rs::clear();
        });
        if (g_phases) {
            detail::run_cleanup([] {
                tt::tt_metal::GraphTracker::instance().pop_processor();
            });
            g_phases.reset();
        }
        detail::run_cleanup([] {
            detail::track_program_regions(false);
        });
        detail::stop_stream();  // drains the releases above
        program::clear_desc_cache();
        ttnn::distributed::close_mesh_device(g_device);
        g_device.reset();
        if (g_fabric) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            g_fabric = false;
        }
        if (pending) {
            std::rethrow_exception(pending);
        }
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
            g_counters.op_ns.load(),
            g_counters.programs.load(),
            g_counters.desc_hits.load(),
            g_counters.alloc_ns.load(),
            g_counters.args_ns.load(),
            g_counters.desc_ns.load(),
            g_counters.enqueue_ns.load(),
            g_counters.program_entries.load(),
            tt::tt_metal::BuildCacheTelemetry::inst().get_compile_count()
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

        // KTT_L1_BUDGET: the fraction of each L1 bank activations may fill (default
        // 0: every activation in DRAM).
        double l1_budget() {
            static const char* set = std::getenv("KTT_L1_BUDGET");
            static const double budget = set ? std::atof(set) : 0.0;
            return budget;
        }

        int64_t l1_free_per_bank(int64_t reserve) {
            const auto& allocator = device()->allocator();
            const auto type = tt::tt_metal::BufferType::L1;
            const auto stats = allocator->get_statistics(type);
            const auto bank = static_cast<int64_t>(allocator->get_bank_size(type));
            // The part of a bank outside the activation budget: the same on every call.
            const int64_t fixed = bank - static_cast<int64_t>(l1_budget() * bank);
            const int64_t free = std::min(
                fixed,
                bank - static_cast<int64_t>(stats.total_allocated_bytes) - reserve
            );
            // A program's static circular buffers grow up from the L1 base and must end
            // below the lowest live L1 buffer: the gap, not the free total, bounds
            // them.
            const auto lowest = device()->lowest_occupied_compute_l1_address();
            if (!lowest) {
                return free;
            }
            const int64_t base = (static_cast<int64_t>(
                allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)
            ));
            // Margin for buffers and alignment that callers' L1 estimates leave out.
            constexpr int64_t kMargin = 32 * 1024;
            const int64_t gap = static_cast<int64_t>(*lowest) - base - kMargin -
                reserve;
            return std::min(free, std::max<int64_t>(0, gap));
        }

        namespace {
            // Trace preparation / capture active (stream worker only): activations in
            // DRAM, the same placement in both passes.
            bool g_trace_dram = false;
        }  // namespace

        void trace_placement(bool dram) {
            post([dram] {
                g_trace_dram = dram;
            });
        }

        bool is_l1_small_full(const std::exception& e) {
            const std::string what = e.what();
            return (
                what.find("Out of Memory") != std::string::npos &&
                what.find("L1_SMALL buffer") != std::string::npos
            );
        }

        tt::tt_metal::MemoryConfig activation_config(int64_t bytes) {
            const double budget = l1_budget();
            const auto& allocator = device()->allocator();
            const auto type = tt::tt_metal::BufferType::L1;
            const auto banks = static_cast<int64_t>(allocator->get_num_banks(type));
            const auto bank = static_cast<int64_t>(allocator->get_bank_size(type));
            const int64_t per_bank = (bytes + banks - 1) / banks + 4096;
            const auto allocated = (static_cast<int64_t>(
                allocator->get_statistics(type).total_allocated_bytes
            ));
            // While traces are alive, L1 activations also stay above their programs'
            // circular buffers.
            int64_t limit = static_cast<int64_t>(budget * bank);
            if (const uint64_t cb_end = traced_cb_end()) {
                constexpr int64_t kMargin = 32 * 1024;
                const auto base = static_cast<int64_t>(
                    allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)
                );
                limit = std::min(
                    limit, bank - (static_cast<int64_t>(cb_end) - base) - kMargin
                );
            }
            if (budget > 0 && !g_trace_dram && allocated + per_bank <= limit) {
                return (tt::tt_metal::MemoryConfig(
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                    tt::tt_metal::BufferType::L1
                ));
            }
            return tt::tt_metal::MemoryConfig{};
        }

    }  // namespace detail

    void keep_in_dram(const HandlePtr& a) {
        if (detail::l1_budget() <= 0) {
            return;
        }
        detail::submit([a] {
            auto& t = a->placement.is_replicated() ? a->tensor : a->placed;
            if (t.memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {
                t = ttnn::to_memory_config(t, tt::tt_metal::MemoryConfig{});
            }
        });
    }

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
