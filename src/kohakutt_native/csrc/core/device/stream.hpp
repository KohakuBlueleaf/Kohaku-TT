// The bridge's default stream: every TT-NN call runs on one worker thread, in issue
// order; callers return as soon as their work is queued.
#pragma once

#include <functional>

namespace ktt::bridge::detail {

    // Queues `work` on the stream; rethrows a pending error of earlier work.
    void submit(std::function<void()> work);

    // Queues `work` without checking errors (destructors); runs it in place when there
    // is no stream or the caller is the worker.
    void post(std::function<void()> work) noexcept;

    // Runs `work` on the stream after everything queued before it and waits for it;
    // rethrows its error or a pending one.
    void run_sync(const std::function<void()>& work);

    // Waits until every queued work item has run; rethrows the first pending error.
    void drain();

    // True on the stream's worker thread.
    bool on_worker();

    // Starts / stops the worker thread (stop drains first).
    void start_stream();
    void stop_stream();

}  // namespace ktt::bridge::detail
