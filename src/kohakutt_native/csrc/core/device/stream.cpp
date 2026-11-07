// The bridge's default stream: a FIFO drained by one worker thread. The first error is
// rethrown to the next caller; ops queued behind it are skipped.
#include "core/device/stream.hpp"

#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <thread>

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
            std::exception_ptr error;
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
                    work();
                } catch (...) {
                    std::lock_guard<std::mutex> lock(s.mutex);
                    if (!s.error) {
                        s.error = std::current_exception();
                    }
                }
                work = nullptr;  // captured handles are released on the worker
                std::lock_guard<std::mutex> lock(s.mutex);
                s.busy = false;
                if (s.queue.empty()) {
                    s.idle.notify_all();
                }
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
            s.queue.push_back({std::move(work), true});
        }
        s.ready.notify_one();
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

    void drain() {
        auto& s = stream();
        if (!s.running || on_worker()) {
            return;
        }
        std::unique_lock<std::mutex> lock(s.mutex);
        s.idle.wait(lock, [&] {
            return s.queue.empty() && !s.busy;
        });
        rethrow_pending(s);
    }

}  // namespace ktt::bridge::detail
