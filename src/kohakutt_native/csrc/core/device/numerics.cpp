// Numerics scopes: a per-thread stack over the process default; a backward pass on a
// thread without a scope reads the most recently opened scope of any thread.
#include "core/device/numerics.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/device/tt_bridge.hpp"

namespace ktt::bridge::numerics {
    namespace {

        // Fidelity named `s` (LoFi|HiFi2|HiFi3|HiFi4), or nullopt.
        std::optional<MathFidelity> parse(const std::string& s) {
            if (s == "LoFi") {
                return MathFidelity::LoFi;
            }
            if (s == "HiFi2") {
                return MathFidelity::HiFi2;
            }
            if (s == "HiFi3") {
                return MathFidelity::HiFi3;
            }
            if (s == "HiFi4") {
                return MathFidelity::HiFi4;
            }
            return std::nullopt;
        }

        // The process default from the environment.
        const Numerics& process_default() {
            static const Numerics n = [] {
                Numerics d;
                for (const char* name : {"KTT_FIDELITY", "KTT_MATMUL_FIDELITY"}) {
                    const char* v = std::getenv(name);
                    if (const auto f = v ? parse(v) : std::nullopt) {
                        d.fidelity = *f;
                        break;
                    }
                }
                const char* approx = std::getenv("KTT_APPROX");
                d.approx = approx && std::string(approx) == "1";
                return d;
            }();
            return n;
        }

        std::vector<Numerics>& stack() {
            thread_local std::vector<Numerics> scopes;
            return scopes;
        }

        // Every open scope of every thread, in push order, with its thread.
        struct Open {
            std::thread::id owner;
            Numerics numerics;
        };

        std::mutex g_open_mutex;
        std::vector<Open> g_open;
        std::atomic<bool (*)()> g_backward_probe{nullptr};

    }  // namespace

    void set_backward_probe(bool (*probe)()) {
        g_backward_probe.store(probe);
    }

    int rank(MathFidelity f) {
        switch (f) {
            case MathFidelity::LoFi:
                return 0;
            case MathFidelity::HiFi2:
                return 1;
            case MathFidelity::HiFi3:
                return 2;
            default:
                return 3;
        }
    }

    MathFidelity from_rank(int r) {
        static constexpr MathFidelity order[] = {
            MathFidelity::LoFi,
            MathFidelity::HiFi2,
            MathFidelity::HiFi3,
            MathFidelity::HiFi4
        };
        return order[std::clamp(r, 0, 3)];
    }

    Numerics current() {
        const auto& s = stack();
        if (!s.empty()) {
            return s.back();
        }
        // A thread without a scope of its own that runs a backward pass (autograd's
        // device thread) reads the most recently opened scope of any thread.
        const auto probe = g_backward_probe.load();
        if (probe && probe()) {
            std::lock_guard lock(g_open_mutex);
            if (!g_open.empty()) {
                return g_open.back().numerics;
            }
        }
        return process_default();
    }

    void push(const Numerics& n) {
        stack().push_back(n);
        std::lock_guard lock(g_open_mutex);
        g_open.push_back({std::this_thread::get_id(), n});
    }

    void pop() {
        auto& s = stack();
        if (s.empty()) {
            throw std::logic_error("kohakutt: numerics pop without a push");
        }
        s.pop_back();
        std::lock_guard lock(g_open_mutex);
        const auto self = std::this_thread::get_id();
        for (auto it = g_open.rbegin(); it != g_open.rend(); ++it) {
            if (it->owner == self) {
                g_open.erase(std::next(it).base());
                break;
            }
        }
    }

    Numerics effective(const Requirements& r) {
        Numerics n = current();
        if (n.strict) {
            return n;
        }
        int f = rank(n.fidelity);
        if (r.floor) {
            f = std::max(f, rank(*r.floor));
        }
        if (r.ceiling) {
            f = std::min(f, rank(*r.ceiling));
        }
        n.fidelity = from_rank(f);
        n.approx = n.approx && !r.exact;
        return n;
    }

    ttnn::ComputeKernelConfig compute_config(const Numerics& n, bool packer_l1_acc) {
        return ttnn::ComputeKernelConfig{
            .math_fidelity = n.fidelity,
            .math_approx_mode = n.approx,
            .fp32_dest_acc_en = true,
            .packer_l1_acc = packer_l1_acc
        };
    }

}  // namespace ktt::bridge::numerics

namespace ktt::bridge {

    void numerics_push(
        int fidelity, bool approx, bool strict, int attention, int kernels
    ) {
        if (attention < 0 || attention > 2) {
            throw std::invalid_argument("kohakutt: attention mode 0..2");
        }
        if (kernels < 1 || kernels > kAttentionAll) {
            throw std::invalid_argument("kohakutt: attention kernels 1..15");
        }
        numerics::push(
            {numerics::from_rank(fidelity),
             approx,
             strict,
             static_cast<numerics::Attention>(attention),
             kernels}
        );
    }

    void numerics_pop() {
        numerics::pop();
    }

    void numerics_set_backward_probe(bool (*probe)()) {
        numerics::set_backward_probe(probe);
    }

    NumericsState numerics_current() {
        const auto n = numerics::current();
        return {
            numerics::rank(n.fidelity),
            n.approx,
            n.strict,
            static_cast<int>(n.attention),
            n.kernels
        };
    }

}  // namespace ktt::bridge
