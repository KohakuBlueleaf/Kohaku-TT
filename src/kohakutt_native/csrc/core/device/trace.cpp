// Bridge trace: TT-NN trace capture/replay of everything enqueued between begin and
// end. A trace keeps every handle created while capturing, and fences the device
// memory of op-internal temporaries freed during capture, so later allocations cannot
// land where a replay writes.
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/trace_allocation_tracker.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/mesh_buffer.hpp>

#include "core/device/handle.hpp"
#include "core/device/trace.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/trace.hpp"

namespace ktt::bridge {
    namespace {

        using tt::tt_metal::Buffer;
        using tt::tt_metal::BufferType;
        using tt::tt_metal::DeviceAddr;

        // One device buffer allocation seen while capturing.
        struct Allocation {
            const Buffer* buffer;
            DeviceAddr address;
            DeviceAddr size;
            DeviceAddr page_size;
            BufferType type;
            bool freed;
        };

        // Records buffer allocations / deallocations (background processor: no behavior
        // change).
        class AllocationLog final : public tt::tt_metal::IGraphProcessor {
        public:
            bool is_capture_processor() const override {
                return false;
            }

            void track_allocate(const Buffer* b) override {
                log.push_back({
                    b,
                    b->address(),
                    b->size(),
                    b->page_size(),
                    b->buffer_type(),
                    false
                });
            }

            void track_deallocate(Buffer* b) override {
                for (auto it = log.rbegin(); it != log.rend(); ++it) {
                    if (it->buffer == b && !it->freed) {
                        it->freed = true;
                        return;
                    }
                }
            }

            std::vector<Allocation> log;
        };

        // What a trace keeps alive: its handles and the fences over its temporaries;
        // `written`: the handles its capture copied into (each replay writes them).
        struct TraceState {
            std::vector<HandlePtr> held;
            std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>> fences;
            int64_t misplaced = 0;
            std::vector<HandlePtr> written;
        };

        // The capture in progress; at most one at a time. `held` and `written` are filled
        // on the host thread as handles are created / copied into; `log` is written on
        // the stream.
        struct Capture {
            ttnn::MeshTraceId id;
            std::vector<HandlePtr> held;
            std::shared_ptr<AllocationLog> log;
            int64_t failures;  // detail::failures() at begin
            std::vector<HandlePtr> written;
        };

        // Destroys `state` on the stream (its fences free device buffers).
        void drop_on_stream(TraceState state) {
            auto boxed = std::make_shared<TraceState>(std::move(state));
            detail::post([boxed]() mutable {
                boxed.reset();
            });
        }

        // Never destroyed at exit (after TT-Metal's teardown); close() releases them.
        std::optional<Capture>& g_capture = *new std::optional<Capture>();
        std::map<int64_t, TraceState>& g_traces = *new std::map<int64_t, TraceState>();
        std::map<std::string, HandlePtr>& g_traced_uploads = (
            *new std::map<std::string, HandlePtr>()
        );

        // Fences: the temporaries' allocation requests re-issued in order (first-fit
        // lands them on the same addresses); misplaced counts fences at another
        // address.
        TraceState fence(const std::vector<Allocation>& log) {
            TraceState state;
            for (const auto& a : log) {
                if (!a.freed ||
                    (a.type != BufferType::DRAM && a.type != BufferType::L1)) {
                    continue;
                }
                auto f = tt::tt_metal::distributed::MeshBuffer::create(
                    tt::tt_metal::distributed::ReplicatedBufferConfig{.size = a.size},
                    tt::tt_metal::distributed::DeviceLocalBufferConfig{
                        .page_size = a.page_size, .buffer_type = a.type
                    },
                    detail::device()
                );
                state.misplaced += f->address() != a.address;
                state.fences.push_back(std::move(f));
            }
            return state;
        }

    }  // namespace

    namespace detail {

        void on_wrap(const HandlePtr& handle) {
            if (g_capture) {
                g_capture->held.push_back(handle);
            }
        }

        void release_traces() {
            auto traces = (std::make_shared<std::map<int64_t, TraceState>>(
                std::move(g_traces)
            ));
            g_traces.clear();
            g_traced_uploads.clear();
            const bool capturing = g_capture.has_value();
            auto capture = std::move(g_capture);
            g_capture.reset();
            // Runs after a failed op too (a capture it interrupted is ended first).
            auto pending = run_cleanup([traces, capturing, &capture] {
                if (capturing) {
                    ttnn::operations::trace::end_trace_capture(
                        device(), capture->id, std::nullopt
                    );
                    ttnn::operations::trace::release_trace(device(), capture->id);
                    tt::tt_metal::GraphTracker::instance().pop_processor();
                }
                for (auto& [id, state] : *traces) {
                    ttnn::operations::trace::release_trace(
                        device(), ttnn::MeshTraceId(static_cast<uint32_t>(id))
                    );
                }
                traces->clear();
            });
            if (pending) {
                std::rethrow_exception(pending);
            }
        }

    }  // namespace detail

    bool capturing() {
        return g_capture.has_value();
    }

    void trace_placement(bool dram) {
        detail::trace_placement(dram);
    }

    bool traces_alive() {
        return g_capture.has_value() || !g_traces.empty() || tracing();
    }

    namespace {

        std::atomic<int64_t> g_write_epoch{0};

    }  // namespace

    int64_t write_epoch() {
        return g_write_epoch.load();
    }

    namespace detail {

        void bump_write_epoch() {
            ++g_write_epoch;
        }

    }  // namespace detail

    int64_t trace_begin() {
        if (g_capture) {
            throw std::runtime_error(
                "kohakutt_native: a trace capture is already active"
            );
        }
        auto log = std::make_shared<AllocationLog>();
        std::optional<ttnn::MeshTraceId> id;
        detail::run_sync([&] {
            tt::tt_metal::GraphTracker::instance().push_processor(log);
            id = (
                ttnn::operations::trace::begin_trace_capture(
                    detail::device(), std::nullopt
                )
            );
        });
        g_capture = Capture{*id, {}, log, detail::failures()};
        return static_cast<int64_t>(**id);
    }

    namespace {

        // Ends the active capture on the stream, also after a failed op; returns the
        // capture and, when an op of it failed, the error to raise (else null).
        std::pair<Capture, std::exception_ptr> end_capture() {
            auto capture = std::move(*g_capture);
            g_capture.reset();
            auto failed = detail::run_cleanup([&] {
                ttnn::operations::trace::end_trace_capture(
                    detail::device(), capture.id, std::nullopt
                );
                tt::tt_metal::GraphTracker::instance().pop_processor();
            });
            if (!failed && detail::failures() != capture.failures) {
                // The op's error was raised to an earlier caller.
                failed = std::make_exception_ptr(
                    std::runtime_error(
                        "kohakutt_native: an op of the trace capture failed; the trace "
                        "is released"
                    )
                );
            }
            return {std::move(capture), failed};
        }

        // Releases the ended trace of `capture` and the handles it holds.
        void discard(Capture capture) {
            const auto id = capture.id;
            auto pending = detail::run_cleanup([id] {
                ttnn::operations::trace::release_trace(detail::device(), id);
            });
            TraceState state;
            state.held = std::move(capture.held);
            drop_on_stream(std::move(state));
            if (pending) {
                std::rethrow_exception(pending);
            }
        }

    }  // namespace

    void trace_abort() {
        if (!g_capture) {
            return;
        }
        auto [capture, failed] = end_capture();
        discard(std::move(capture));
        if (failed) {
            std::rethrow_exception(failed);
        }
    }

    int64_t trace_end() {
        if (!g_capture) {
            throw std::runtime_error("kohakutt_native: no trace capture is active");
        }
        auto [capture, failed] = end_capture();
        if (failed) {
            // An op of the capture failed: the trace is released, the op's error
            // raised.
            discard(std::move(capture));
            std::rethrow_exception(failed);
        }
        TraceState state;
        detail::run_sync([&] {
            state = fence(capture.log->log);
        });
        if (take_capture_host_conversion()) {
            drop_on_stream(std::move(state));
            discard(std::move(capture));
            throw std::runtime_error(
                "kohakutt_native: a mesh conversion in the capture needs the host path "
                "(shards not on tile boundaries); the trace is released"
            );
        }
        state.held = std::move(capture.held);
        state.written = std::move(capture.written);
        auto key = static_cast<int64_t>(*capture.id);
        auto held = static_cast<int64_t>(state.held.size());
        g_traces[key] = std::move(state);
        return held;
    }

    int64_t trace_misplaced_fences(int64_t id) {
        auto it = g_traces.find(id);
        return it == g_traces.end() ? -1 : it->second.misplaced;
    }

    void trace_execute(int64_t id) {
        detail::refresh_nhwc_conv_weights();
        const int64_t epoch = write_epoch();
        detail::bump_write_epoch();
        detail::submit([id, epoch] {
            detail::refresh_traced_conv_weights(epoch);
            ttnn::operations::trace::execute_trace(
                detail::device(),
                ttnn::MeshTraceId(static_cast<uint32_t>(id)),
                std::nullopt,
                false
            );
        });
        // The replay writes the handles its capture copied into.
        if (const auto it = g_traces.find(id); it != g_traces.end()) {
            for (const auto& h : it->second.written) {
                h->writes++;
            }
        }
    }

    void trace_release(int64_t id) {
        detail::submit([id] {
            ttnn::operations::trace::release_trace(
                detail::device(), ttnn::MeshTraceId(static_cast<uint32_t>(id))
            );
        });
        auto it = g_traces.find(id);
        if (it != g_traces.end()) {
            drop_on_stream(std::move(it->second));
            g_traces.erase(it);
        }
    }

    int64_t unsafe_buffers(int64_t id) {
        namespace tracker = tt::tt_metal::distributed::trace_allocation_tracker;
        int64_t count = 0;
        detail::run_sync([&] {
            count = (
                static_cast<int64_t>(tracker::get_unsafe_tracked_ids(
                                         detail::device(),
                                         ttnn::MeshTraceId(static_cast<uint32_t>(id))
                )
                                         .size())
            );
        });
        return count;
    }

    HandlePtr clone(const HandlePtr& a) {
        return detail::timed_op(
            a->dtype,
            a->shape,
            [=] {
                return ttnn::clone(a->tensor, std::nullopt, std::nullopt, std::nullopt);
            },
            a->format
        );
    }

    HandlePtr traced_upload(const std::string& key) {
        const auto it = g_traced_uploads.find(key);
        return it == g_traced_uploads.end() ? nullptr : it->second;
    }

    void keep_traced_upload(const std::string& key, const HandlePtr& h) {
        g_traced_uploads[key] = h;
    }

    HandlePtr empty_like(const HandlePtr& a) {
        if (a->format == Format::Nhwc) {
            auto out = detail::make_handle(a->shape, a->dtype, Format::Nhwc);
            const auto dims = detail::stored_dims(a->shape, Format::Nhwc);
            const auto dtype = a->dtype;
            detail::submit([out, dims, dtype] {
                out->tensor = ttnn::empty(
                    detail::device_shape(dims),
                    detail::to_ttnn(dtype),
                    tt::tt_metal::Layout::TILE,
                    detail::device(),
                    tt::tt_metal::MemoryConfig{}
                );
            });
            return out;
        }
        return (
            a->placement.is_replicated() ? empty(a->shape, a->dtype)
                                         : empty_placed(a->shape, a->dtype, a->placement)
        );
    }

    namespace {

        // Handles copy_into() wrote in the current prepare pass or capture (host thread).
        std::set<const Handle*>& pass_writes() {
            static auto* set = new std::set<const Handle*>();
            return *set;
        }

    }  // namespace

    bool written_in_pass(const HandlePtr& h) {
        return pass_writes().count(h.get()) > 0;
    }

    void clear_pass_writes() {
        pass_writes().clear();
    }

    void copy_into(const HandlePtr& dst, const HandlePtr& src) {
        dst->writes++;
        if (g_capture) {
            g_capture->written.push_back(dst);
        }
        if (tracing()) {
            pass_writes().insert(dst.get());
        }
        // Other formats (or Nhwc of another shape): src as Nchw of dst's shape, then
        // in dst's format.
        if (dst->format != src->format ||
            (dst->format == Format::Nhwc && dst->shape != src->shape)) {
            auto from = to_nchw(src);
            if (from->shape != dst->shape) {
                from = reshape(from, dst->shape);
            }
            copy_into(dst, dst->format == Format::Nhwc ? to_nhwc(from) : from);
            return;
        }
        // Mesh-placed handles hold their shards in `placed`: the copy runs between
        // the per-chip local views.
        if (!dst->placement.is_replicated() || !src->placement.is_replicated()) {
            if (!(dst->placement == src->placement)) {
                throw std::runtime_error(
                    "copy_into: placements differ (" + dst->placement.str() + " vs " +
                    src->placement.str() + ")"
                );
            }
            copy_into(local_view(dst), local_view(src));
            return;
        }
        detail::bump_write_epoch();
        detail::counters().ops++;
        detail::submit([dst, src] {
            // Same volume, other shape (e.g. keepdim statistics): src viewed as dst
            // first.
            auto from = src->tensor;
            if (src->shape != dst->shape) {
                from = ttnn::reshape(from, dst->tensor.logical_shape());
            }
            ttnn::copy(from, dst->tensor);
        });
    }

}  // namespace ktt::bridge
