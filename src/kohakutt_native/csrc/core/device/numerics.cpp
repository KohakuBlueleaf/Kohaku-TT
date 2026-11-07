// Numerics scopes: a per-thread stack over the process default.
#include "core/device/numerics.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
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

    }  // namespace

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
        return s.empty() ? process_default() : s.back();
    }

    void push(const Numerics& n) {
        stack().push_back(n);
    }

    void pop() {
        auto& s = stack();
        if (s.empty()) {
            throw std::logic_error("kohakutt: numerics pop without a push");
        }
        s.pop_back();
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

    void numerics_push(int fidelity, bool approx, bool strict, int attention) {
        if (attention < 0 || attention > 2) {
            throw std::invalid_argument("kohakutt: attention mode 0..2");
        }
        numerics::push({
            numerics::from_rank(fidelity),
            approx,
            strict,
            static_cast<numerics::Attention>(attention)
        });
    }

    void numerics_pop() {
        numerics::pop();
    }

    NumericsState numerics_current() {
        const auto n = numerics::current();
        return {
            numerics::rank(n.fidelity),
            n.approx,
            n.strict,
            static_cast<int>(n.attention)
        };
    }

}  // namespace ktt::bridge
