// The bridge's default stream: a FIFO drained by one worker thread. The first error is
// rethrown to the next caller; ops queued before that are skipped.
#include "core/device/stream.hpp"

#include "core/device/trace.hpp"
#include "core/device/tt_bridge.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tracy/Tracy.hpp>

// Provided by a TT-Metal that can hold back a thread's eager launches' fetch-queue
// entries (opt-in per thread); null otherwise.
extern "C" __attribute__((weak)) void tt_metal_fetch_queue_batching(int n);
extern "C" __attribute__((weak)) void tt_metal_flush_fetch_queues();

namespace ktt::bridge::detail {
    namespace {

        // A queued work item; ops (submit) are skipped while an error is pending,
        // releases (post) and sync points always run. Ops are numbered in queue order.
        struct Item {
            std::function<void()> work;
            bool skip_on_error;
            const char* label = "";
            int64_t seq = -1;
        };

        // `error` with the bridge op that queued the failing work (and its op number,
        // when known) named in its message, keeping the exception's kind.
        std::exception_ptr labelled(
            std::exception_ptr error, const char* label, int64_t seq = -1
        ) {
            if (!label || !*label) {
                return error;
            }
            const std::string head = std::string("[bridge op '") + label + "'" +
                (seq >= 0 ? " #" + std::to_string(seq) : std::string()) + "] ";
            try {
                std::rethrow_exception(error);
            } catch (const std::invalid_argument& e) {
                return std::make_exception_ptr(std::invalid_argument(head + e.what()));
            } catch (const std::out_of_range& e) {
                return std::make_exception_ptr(std::out_of_range(head + e.what()));
            } catch (const std::exception& e) {
                return std::make_exception_ptr(std::runtime_error(head + e.what()));
            } catch (...) {
                return error;
            }
        }

        struct Stream {
            std::mutex mutex;
            std::condition_variable ready;  // work queued or stopping
            std::condition_variable idle;   // queue empty and nothing running
            std::deque<Item> queue;
            // Work held back by defer() and its key.
            std::function<void()> deferred;
            const void* deferred_key = nullptr;
            std::exception_ptr error;
            int64_t failures = 0;  // work items that raised, ever
            std::atomic<int64_t> next_seq{0};
            int64_t deferred_seq = -1;
            int64_t running_seq = -1;  // the worker's current op
            int64_t popped_seq = -1;   // the last op taken from the queue
            int64_t failed_seq = -1;   // the first op whose work `error` lost
            bool reported = false;     // `error` raised while held
            // Ops below skip_until are skipped; `lost` holds the ranges [first, end)
            // skipped after a failure, whose outputs were never computed.
            int64_t skip_until = 0;
            std::vector<std::pair<int64_t, int64_t>> lost;
            std::atomic<int64_t> lost_end{0};
            bool running = false;
            bool busy = false;
            bool stopping = false;
            std::thread worker;
            std::thread::id worker_id;
        };

        // Never destroyed at exit; stop_stream() joins the worker.
        Stream& stream() {
            static auto* s = new Stream();
            return *s;
        }

        // KTT_FETCH_Q_BATCH: eager launches TT-Metal may publish to the device together
        // (default 8; 0 or 1: each one at once).
        int fetch_queue_batch() {
            const char* v = std::getenv("KTT_FETCH_Q_BATCH");
            return v ? std::atoi(v) : 8;
        }

        // Runs `hook` (a call into TT-Metal); an error it raises is the stream's
        // pending error, raised to the next op (caller holds the lock).
        void run_hook(Stream& s, std::unique_lock<std::mutex>& lock, void (*hook)()) {
            lock.unlock();
            std::exception_ptr error;
            try {
                hook();
            } catch (...) {
                error = std::current_exception();
            }
            lock.lock();
            if (error) {
                ++s.failures;
                if (!s.error) {
                    s.error = labelled(error, "launch publish");
                }
            }
        }

        // Opts the worker in to TT-Metal's launch batching; false if that failed (not
        // an op's error: what it could not publish stays held for the next publish).
        bool opt_in(std::unique_lock<std::mutex>& lock) {
            lock.unlock();
            bool done = true;
            try {
                tt_metal_fetch_queue_batching(fetch_queue_batch());
            } catch (...) {
                done = false;
            }
            lock.lock();
            return done;
        }

        void worker_loop(Stream& s) {
            bool opted_in = !tt_metal_fetch_queue_batching;
            if (!opted_in) {
                std::unique_lock<std::mutex> lock(s.mutex);
                opted_in = opt_in(lock);
            }
            for (;;) {
                std::function<void()> work;
                const char* label = "";
                {
                    std::unique_lock<std::mutex> lock(s.mutex);
                    if (s.queue.empty() && tt_metal_flush_fetch_queues) {
                        // Idle: the launches TT-Metal holds back go to the device now.
                        run_hook(s, lock, tt_metal_flush_fetch_queues);
                    }
                    if (s.queue.empty() && !opted_in) {
                        opted_in = opt_in(lock);  // retried at each idle point
                    }
                    s.ready.wait(lock, [&] {
                        return s.stopping || !s.queue.empty();
                    });
                    if (s.queue.empty()) {
                        return;
                    }
                    auto& item = s.queue.front();
                    const bool skip =
                        (item.skip_on_error && (s.error || item.seq < s.skip_until));
                    work = std::move(item.work);
                    label = item.label;
                    s.running_seq = item.seq;
                    if (item.seq >= 0) {
                        s.popped_seq = item.seq;
                    }
                    s.queue.pop_front();
                    s.busy = true;
                    if (skip) {
                        work = nullptr;
                    }
                }
                if (!work) {
                    std::lock_guard<std::mutex> lock(s.mutex);
                    s.busy = false;
                    if (s.queue.empty()) {
                        s.idle.notify_all();
                    }
                    continue;
                }
                try {
                    ZoneScopedN("ktt task");
                    work();
                } catch (...) {
                    std::lock_guard<std::mutex> lock(s.mutex);
                    ++s.failures;
                    if (!s.error) {
                        s.failed_seq = s.running_seq >= 0 ? s.running_seq
                                                          : s.popped_seq + 1;
                        s.error = labelled(
                            std::current_exception(), label, s.failed_seq
                        );
                    }
                }
                {
                    ZoneScopedN("ktt task release");
                    work = nullptr;  // captured handles are released on the worker
                }
                std::lock_guard<std::mutex> lock(s.mutex);
                s.busy = false;
                if (s.queue.empty()) {
                    s.idle.notify_all();
                }
            }
        }

        // Queues the work held back by defer() (caller holds the lock).
        void queue_deferred(Stream& s) {
            if (s.deferred) {
                s.queue.push_back({std::move(s.deferred), true, "", s.deferred_seq});
                s.deferred = nullptr;
                s.deferred_key = nullptr;
            }
        }

        // Clears and returns the pending error (caller holds the lock): every op
        // queued so far is skipped, and the ops it lost are recorded.
        std::exception_ptr take_error(Stream& s) {
            if (s.error) {
                const int64_t end = s.next_seq;
                s.lost.emplace_back(s.failed_seq, end);
                s.lost_end = end;
                s.skip_until = end;
                s.reported = false;
            }
            return std::exchange(s.error, nullptr);
        }

        // Rethrows and clears the pending error (caller holds the lock). In a capture
        // or its prepare pass it stays pending (later ops are skipped) until the pass
        // ends, and is not raised again after it.
        void rethrow_pending(Stream& s) {
            if (!s.error) {
                return;
            }
            if (capturing() || tracing()) {
                s.reported = true;
                std::rethrow_exception(s.error);
            }
            const bool reported = s.reported;
            auto error = take_error(s);
            if (!reported) {
                std::rethrow_exception(error);
            }
        }

    }  // namespace

    bool on_worker() {
        auto& s = stream();
        return s.running && std::this_thread::get_id() == s.worker_id;
    }

    void start_stream() {
        auto& s = stream();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.running) {
            return;
        }
        s.stopping = false;
        s.running = true;
        s.worker = std::thread(worker_loop, std::ref(s));
        s.worker_id = s.worker.get_id();
    }

    void stop_stream() {
        auto& s = stream();
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            if (!s.running) {
                return;
            }
            queue_deferred(s);
            s.stopping = true;
        }
        s.ready.notify_all();
        s.worker.join();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.running = false;
        s.error = nullptr;
    }

    void submit(std::function<void()> work, const char* label) {
        auto& s = stream();
        if (!s.running || on_worker()) {
            work();  // no stream (or already on it): run in place
            return;
        }
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            rethrow_pending(s);
            queue_deferred(s);
            s.queue.push_back({std::move(work), true, label, s.next_seq++});
        }
        s.ready.notify_all();
        // KTT_STREAM_SYNC=1: each op completes before its caller returns, so an op's
        // error is raised to the op that failed.
        static const bool sync = [] {
            const char* v = std::getenv("KTT_STREAM_SYNC");
            return v && std::string(v) == "1";
        }();
        if (sync) {
            drain();
        }
    }

    void defer(const void* key, std::function<void()> work) {
        auto& s = stream();
        if (!s.running || on_worker()) {
            work();
            return;
        }
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            rethrow_pending(s);
            queue_deferred(s);
            s.deferred = std::move(work);
            s.deferred_key = key;
            s.deferred_seq = s.next_seq++;
        }
        s.ready.notify_all();
    }

    std::function<void()> take_deferred(const void* key) {
        auto& s = stream();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.deferred || s.deferred_key != key) {
            return nullptr;
        }
        s.deferred_key = nullptr;
        return std::exchange(s.deferred, nullptr);
    }

    void post(std::function<void()> work) noexcept {
        auto& s = stream();
        try {
            if (!s.running || on_worker()) {
                work();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(s.mutex);
                s.queue.push_back({std::move(work), false});
            }
            s.ready.notify_one();
        } catch (...) {
            // A failed release is dropped: destructors must not throw.
        }
    }

    void run_sync(const std::function<void()>& work) {
        auto& s = stream();
        if (!s.running || on_worker()) {
            work();
            return;
        }
        std::promise<void> done;
        auto finished = done.get_future();
        {
            // Always runs (the caller waits on it); skips `work` when an earlier op
            // failed, whose error is rethrown below.
            std::lock_guard<std::mutex> lock(s.mutex);
            rethrow_pending(s);
            queue_deferred(s);
            s.queue.push_back(
                {[&] {
                     bool failed = false;
                     {
                         std::lock_guard<std::mutex> inner(s.mutex);
                         failed = static_cast<bool>(s.error);
                     }
                     try {
                         if (!failed) {
                             work();
                         }
                         done.set_value();
                     } catch (...) {
                         done.set_exception(std::current_exception());
                     }
                 },
                 false}
            );
        }
        s.ready.notify_one();
        finished.get();
        std::lock_guard<std::mutex> lock(s.mutex);
        rethrow_pending(s);
    }

    int64_t op_seq() {
        auto& s = stream();
        if (!s.running) {
            return -1;
        }
        return on_worker() ? s.running_seq : s.next_seq.load();
    }

    bool lost(int64_t seq) {
        auto& s = stream();
        if (seq < 0 || seq >= s.lost_end) {
            return false;
        }
        std::lock_guard<std::mutex> lock(s.mutex);
        for (const auto& [first, end] : s.lost) {
            if (seq >= first && seq < end) {
                return true;
            }
        }
        return false;
    }

    int64_t failures() {
        auto& s = stream();
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.failures;
    }

    std::exception_ptr run_cleanup(const std::function<void()>& work) {
        auto& s = stream();
        if (!s.running || on_worker()) {
            work();
            std::lock_guard<std::mutex> lock(s.mutex);
            return take_error(s);
        }
        std::promise<void> done;
        auto finished = done.get_future();
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            queue_deferred(s);
            s.queue.push_back(
                {[&] {
                     try {
                         work();
                         done.set_value();
                     } catch (...) {
                         done.set_exception(std::current_exception());
                     }
                 },
                 false}
            );
        }
        s.ready.notify_one();
        finished.get();
        std::lock_guard<std::mutex> lock(s.mutex);
        return take_error(s);
    }

    void drain() {
        auto& s = stream();
        if (!s.running || on_worker()) {
            return;
        }
        std::unique_lock<std::mutex> lock(s.mutex);
        if (s.deferred) {
            queue_deferred(s);
            s.ready.notify_all();
        }
        s.idle.wait(lock, [&] {
            return s.queue.empty() && !s.busy;
        });
        rethrow_pending(s);
    }

}  // namespace ktt::bridge::detail
