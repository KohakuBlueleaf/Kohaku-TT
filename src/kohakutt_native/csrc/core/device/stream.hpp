// The bridge's default stream: every TT-NN call runs on one worker thread, in issue
// order; callers return as soon as their work is queued.
#pragma once

#include <cstdint>
#include <exception>
#include <functional>

namespace ktt::bridge::detail {

    // Queues `work` on the stream; rethrows a pending error of earlier work.
    void submit(std::function<void()> work);

    // Holds `work` back under `key`: it is queued just before the next work that is
    // submitted, run synchronously or drained for, unless take_deferred(key) removes it
    // first. Work held back earlier is queued first.
    void defer(const void* key, std::function<void()> work);

    // The work held back under `key`, removed without being queued; null when there is
    // none.
    std::function<void()> take_deferred(const void* key);

    // Queues `work` without checking errors (destructors); runs it in place when there
    // is no stream or the caller is the worker.
    void post(std::function<void()> work) noexcept;

    // Runs `work` on the stream after everything queued before it and waits for it;
    // rethrows its error or a pending one.
    void run_sync(const std::function<void()>& work);

    // Runs `work` on the stream after everything queued before it, also when earlier
    // work failed, and waits for it; rethrows `work`'s own error, else returns (and
    // clears) the pending error of earlier work, null when none.
    std::exception_ptr run_cleanup(const std::function<void()>& work);

    // Number of queued work items that raised since the process started.
    int64_t failures();

    // Waits until every queued work item has run; rethrows the first pending error.
    void drain();

    // True on the stream's worker thread.
    bool on_worker();

    // Starts / stops the worker thread (stop drains first).
    void start_stream();
    void stop_stream();

}  // namespace ktt::bridge::detail
