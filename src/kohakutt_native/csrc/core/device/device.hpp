// Host time of TT-NN device-operation launches by phase (standard types only: included
// by both the device and the torch side).
#pragma once

#include <string>
#include <vector>

namespace ktt::bridge {

    // One "name count alloc_ns args_ns enqueue_ns" line per device operation launched
    // since open or the last reset, with KTT_LAUNCH_PHASES=1 at open (else empty):
    // start to the last output allocation, to the first program (hash, cache lookup,
    // runtime arguments), to the end (mesh workload enqueue).
    std::vector<std::string> launch_phases();

    // Clears the launch_phases() totals.
    void reset_launch_phases();

}  // namespace ktt::bridge
