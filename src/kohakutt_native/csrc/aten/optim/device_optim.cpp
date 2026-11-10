// Bridge optimizer steps: adam_step on aten/optim/ktt_adam.
#include <array>
#include <stdexcept>

#include "aten/optim/ktt_adam/program.hpp"
#include "core/device/handle.hpp"
#include "core/device/program.hpp"
#include "core/device/tt_bridge.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ktt::bridge {

    std::vector<HandlePtr> adam_step(
        const HandlePtr& p,
        const HandlePtr& g,
        const HandlePtr& m,
        const HandlePtr& v,
        const AdamStep& s
    ) {
        if (!program::own_kernels_enabled() ||
            (p->dtype != DType::BFloat16 && p->dtype != DType::Float32) ||
            p->placement.has_partial()) {
            return {};
        }
        for (const auto& h : {g, m, v}) {
            if (h->dtype != p->dtype || h->shape != p->shape ||
                h->format != p->format || !(h->placement == p->placement)) {
                return {};
            }
        }
        const std::array<HandlePtr, 4> in = {
            local_view(p), local_view(g), local_view(m), local_view(v)
        };
        std::array<HandlePtr, 3> out;
        for (auto& h : out) {
            h = detail::make_handle(in[0]->shape, in[0]->dtype, in[0]->format);
        }
        ktt_adam::Coefficients c;
        c.b1 = static_cast<float>(s.beta1);
        c.b2 = static_cast<float>(s.beta2);
        c.scale = static_cast<float>(s.scale);
        c.eps = static_cast<float>(s.eps);
        c.step = static_cast<float>(s.step);
        c.decay = static_cast<float>(s.decay);
        c.l2 = static_cast<float>(s.l2);
        c.negate = s.negate;
        detail::counters().ops++;
        detail::submit([in, out, c] {
            const std::array<const ttnn::Tensor*, 4> pgmv = {
                &in[0]->tensor, &in[1]->tensor, &in[2]->tensor, &in[3]->tensor
            };
            if (!ktt_adam::supports(pgmv)) {
                throw std::logic_error(
                    "kohakutt_native: adam_step operands differ on the device"
                );
            }
            // Persistent values: in DRAM.
            for (const auto& h : out) {
                h->tensor = ttnn::empty(
                    pgmv[0]->logical_shape(),
                    pgmv[0]->dtype(),
                    tt::tt_metal::Layout::TILE,
                    detail::device(),
                    tt::tt_metal::MemoryConfig{}
                );
            }
            ktt_adam::run(pgmv, {&out[0]->tensor, &out[1]->tensor, &out[2]->tensor}, c);
        });
        std::vector<HandlePtr> placed;
        for (const auto& h : out) {
            placed.push_back(adopt_placed(h, p->shape, p->placement));
        }
        return placed;
    }

}  // namespace ktt::bridge
