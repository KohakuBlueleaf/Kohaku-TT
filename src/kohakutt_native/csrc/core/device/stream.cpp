// The bridge's default stream: a FIFO drained by one worker thread. The first error is
// rethrown to the next caller; ops queued behind it are skipped.
#include "core/device/stream.hpp"

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <tracy/Tracy.hpp>

namespace ktt::bridge::detail {
    namespace {

        // A queued work item; ops (submit) are skipped while an error is pending,
        // releases (post) and sync points always run.
        struct Item {
            std::function<void()> work;
            bool skip_on_error;
        };

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

        void worker_loop(Stream& s) {
            for (;;) {
                std::function<void()> work;
                {
                    std::unique_lock<std::mutex> lock(s.mutex);
                    s.ready.wait(lock, [&] {
                        return s.stopping || !s.queue.empty();
                    });
                    if (s.queue.empty()) {
                        return;
                    }
                    const bool skip = s.queue.front().skip_on_error && s.error;
                    work = std::move(s.queue.front().work);
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
                        s.error = std::current_exception();
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
                s.queue.push_back({std::move(s.deferred), true});
                s.deferred = nullptr;
                s.deferred_key = nullptr;
            }
        }

        // Rethrows and clears the pending error (caller holds the lock).
        void rethrow_pending(Stream& s) {
            if (s.error) {
                auto error = s.error;
                s.error = nullptr;
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

    void submit(std::function<void()> work) {
        auto& s = stream();
        if (!s.running || on_worker()) {
            work();  // no stream (or already on it): run in place
            return;
        }
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            rethrow_pending(s);
            queue_deferred(s);
            s.queue.push_back({std::move(work), true});
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
            return std::exchange(s.error, nullptr);
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
        return std::exchange(s.error, nullptr);
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
