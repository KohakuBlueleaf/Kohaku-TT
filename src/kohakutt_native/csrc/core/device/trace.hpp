// Bridge trace capture, the part not declared in tt_bridge.hpp (standard types only).
#pragma once

#include <string>

#include "core/device/tt_bridge.hpp"

namespace ktt::bridge {

    // Ends the active capture, if any, and releases its trace; also after a failed op
    // of the capture. Rethrows the pending error of queued work.
    void trace_abort();

    // A new, unwritten handle of a's device shape, dtype, mesh placement and format.
    HandlePtr empty_like(const HandlePtr& a);

    // True during a capture's prepare pass and the capture (set_tracing).
    bool tracing();

    // True when copy_into() wrote `h` earlier in the current prepare pass or capture.
    bool written_in_pass(const HandlePtr& h);
    // Forgets the writes of the previous pass (set_tracing(true) at a pass start).
    void clear_pass_writes();

    // The device copy kept for host bytes `key` uploaded while tracing, or null; keeps
    // `h` for `key`. Released with the traces (close).
    HandlePtr traced_upload(const std::string& key);
    void keep_traced_upload(const std::string& key, const HandlePtr& h);

}  // namespace ktt::bridge
