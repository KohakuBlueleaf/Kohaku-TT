// Helpers shared by Kohaku-TT's own Metalium program builders.
#include "core/device/program.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>

#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/creation/creation.hpp"

namespace ktt::bridge::program {

    bool own_kernels_enabled() {
        static const bool on = [] {
            const char* v = std::getenv("KTT_CUSTOM_KERNELS");
            return !(v && std::string(v) == "0");
        }();
        return on;
    }

    std::string source(const std::string& relative) {
        static const std::filesystem::path dir = [] {
            Dl_info info{};
            dladdr(reinterpret_cast<void*>(&own_kernels_enabled), &info);
            return (
                std::filesystem::path(info.dli_fname).parent_path() / "csrc" / "aten"
            );
        }();
        return (dir / relative).string();
    }

    int64_t min_tiles(int64_t measured) {
        static const char* v = std::getenv("KTT_CUSTOM_MIN_TILES");
        return v ? std::atoll(v) : measured;
    }

    int64_t tile_count(const std::vector<int64_t>& shape) {
        int64_t tiles = 1;
        const size_t r = shape.size();
        for (size_t i = 0; i + 2 < r; ++i) {
            tiles *= shape[i];
        }
        return tiles * ((shape[r - 2] + 31) / 32) * ((shape[r - 1] + 31) / 32);
    }

    Split split_tiles(uint32_t tiles, uint32_t min_per_core) {
        const auto grid = detail::device()->compute_with_storage_grid_size();
        const uint32_t total = grid.x * grid.y;
        const uint32_t wanted = (tiles + min_per_core - 1) / min_per_core;
        const uint32_t used = std::clamp<uint32_t>(wanted, 1u, total);
        Split s;
        s.range = tt::tt_metal::num_cores_to_corerangeset(used, grid, true);
        s.cores = corerange_to_cores(s.range, std::nullopt, true);
        s.base = tiles / used;
        s.extra = tiles % used;
        return s;
    }

    std::vector<HandlePtr> run_on_stream(
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes,
        std::function<void(const std::vector<ttnn::Tensor>& tensors)> run
    ) {
        std::vector<HandlePtr> outputs;
        for (size_t i = 0; i < out_shapes.size(); ++i) {
            outputs.push_back(
                detail::make_handle(detail::device_dims(out_shapes[i]), out_dtypes[i])
            );
        }
        detail::counters().ops++;
        detail::submit([inputs, outputs, run = std::move(run)] {
            const auto start = std::chrono::steady_clock::now();
            // Inputs, then outputs allocated per the activation policy.
            auto operands = [&] {
                std::vector<ttnn::Tensor> tensors;
                for (const auto& h : inputs) {
                    tensors.push_back(h->tensor);
                }
                for (const auto& h : outputs) {
                    h->tensor = (
                        ttnn::empty(
                            detail::device_shape(h->shape),
                            detail::to_ttnn(h->dtype),
                            tt::tt_metal::Layout::TILE,
                            detail::device(),
                            detail::activation_config(h->bytes)
                        )
                    );
                    tensors.push_back(h->tensor);
                }
                return tensors;
            };
            try {
                run(operands());
            } catch (const std::exception& e) {
                if (!detail::is_l1_clash(e)) {
                    throw;
                }
                // Again with L1 activations spilled and the outputs in DRAM.
                for (const auto& h : outputs) {
                    h->tensor = ttnn::Tensor();
                }
                detail::spill_l1();
                const detail::DramOnly dram;
                run(operands());
            }
            const auto elapsed = std::chrono::steady_clock::now() - start;
            using std::chrono::duration_cast, std::chrono::nanoseconds;
            detail::counters().op_ns += duration_cast<nanoseconds>(elapsed).count();
        });
        return outputs;
    }

}  // namespace ktt::bridge::program
