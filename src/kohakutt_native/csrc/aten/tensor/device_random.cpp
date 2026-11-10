// Bridge random ops on device (no host draw): uniform, normal, bernoulli, and
// counter-based uniforms on the global element index (ktt_philox).
#include <cstdlib>
#include <cstring>

#include "aten/tensor/ktt_philox/program.hpp"
#include "core/device/handle.hpp"
#include "ttnn/operations/bernoulli/bernoulli.hpp"
#include "ttnn/operations/randn/randn.hpp"
#include "ttnn/operations/uniform/uniform.hpp"

namespace ktt::bridge {

    bool philox_enabled() {
        static const bool on = [] {
            const char* v = std::getenv("KTT_PHILOX");
            return v && std::strcmp(v, "1") == 0;
        }();
        return on;
    }

    HandlePtr philox_uniform(
        const std::vector<int64_t>& shape,
        DType dtype,
        uint64_t key,
        uint32_t word,
        const Placement& p
    ) {
        const auto dims = detail::device_dims(shape);
        const auto rank = static_cast<int64_t>(dims.size());
        if (rank > ktt_philox::kMaxRank) {
            return nullptr;
        }
        const auto mesh = mesh_shape();
        auto local = dims;
        for (size_t a = 0; a < p.axes.size(); ++a) {
            if (p.axes[a].kind == AxisKind::S) {
                const int64_t d = p.axes[a].dim;
                local[d] = (dims[d] + mesh[a] - 1) / mesh[a];
            }
        }
        // Params per chip (row-major over the mesh): key, word, the box's offsets.
        constexpr int64_t kRow = 16;
        const int64_t rows = mesh.empty() ? 1 : mesh[0];
        const int64_t cols = mesh.size() > 1 ? mesh[1] : 1;
        const bool placed = !p.is_replicated();
        std::vector<int32_t> host((placed ? rows * cols : 1) * kRow, 0);
        for (int64_t i = 0; i < (placed ? rows : 1); ++i) {
            for (int64_t j = 0; j < (placed ? cols : 1); ++j) {
                int32_t* row = host.data() + (i * (placed ? cols : 1) + j) * kRow;
                row[0] = static_cast<int32_t>(static_cast<uint32_t>(key));
                row[1] = static_cast<int32_t>(static_cast<uint32_t>(key >> 32));
                row[2] = static_cast<int32_t>(word);
                for (size_t a = 0; placed && a < p.axes.size(); ++a) {
                    if (p.axes[a].kind == AxisKind::S) {
                        const int64_t coord = a == 0 ? i : j;
                        row[3 + p.axes[a].dim] = (static_cast<int32_t>(
                            coord * local[p.axes[a].dim]
                        ));
                    }
                }
            }
        }
        Placement q;
        std::vector<int64_t> params_shape{1, kRow};
        if (placed) {
            q.axes.resize(mesh.size());
            q.axes[0] = AxisPlacement::shard(mesh.size() > 1 ? 0 : 1);
            if (mesh.size() > 1) {
                q.axes[1] = AxisPlacement::shard(1);
            }
            params_shape = mesh.size() > 1 ? std::vector<int64_t>{rows, cols * kRow}
                                           : std::vector<int64_t>{1, rows * kRow};
        }
        auto params = (local_view(
            upload_placed(host.data(), params_shape, DType::Int32, q)
        ));
        const auto ttnn_dtype = detail::to_ttnn(dtype);
        auto h = detail::timed_op(dtype, local, [=] {
            return (ktt_philox::run(
                local,
                dims,
                ttnn_dtype,
                params->tensor,
                detail::activation_config(detail::padded_bytes(local, dtype))
            ));
        });
        return placed ? adopt_placed(h, dims, p) : h;
    }

    HandlePtr uniform(const HandlePtr& like, double from, double to, uint32_t seed) {
        if (detail::is_empty(like)) {
            return detail::empty_of(like, like->dtype);
        }
        // Drawn per stored element: `like`'s format kept.
        return detail::timed_op(
            like->dtype,
            like->shape,
            [=] {
                return ttnn::uniform(
                    like->tensor, static_cast<float>(from), static_cast<float>(to), seed
                );
            },
            like->format
        );
    }

    HandlePtr normal(const std::vector<int64_t>& shape, DType dtype, uint32_t seed) {
        return detail::timed_op(dtype, detail::device_dims(shape), [=] {
            return (ttnn::randn(
                detail::device_shape(shape),
                *detail::device(),
                detail::to_ttnn(dtype),
                tt::tt_metal::Layout::TILE,
                ttnn::DRAM_MEMORY_CONFIG,
                std::nullopt,
                seed
            ));
        });
    }

    HandlePtr bernoulli(const HandlePtr& p, uint32_t seed) {
        if (detail::is_empty(p)) {
            return detail::empty_of(p, p->dtype);
        }
        return detail::timed_op(p->dtype, p->shape, [=] {
            return ttnn::bernoulli(p->tensor, seed);
        });
    }

}  // namespace ktt::bridge
