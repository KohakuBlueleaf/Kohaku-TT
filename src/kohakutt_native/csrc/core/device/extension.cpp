// Extension programs (include/ktt/extension/device.hpp): the named builder registry and
// the run of a registered program on the default stream.
#include "ktt/extension/device.hpp"

#include <map>
#include <mutex>
#include <stdexcept>

#include "core/device/numerics.hpp"
#include "core/device/program.hpp"
#include "ttnn/operations/generic/generic_op.hpp"

namespace ktt::extension {
    namespace {

        struct Entry {
            Builder build;
            Requirements requirements;
        };

        std::mutex& registry_mutex() {
            static std::mutex m;
            return m;
        }

        std::map<std::string, Entry>& registry() {
            static std::map<std::string, Entry> programs;
            return programs;
        }

    }  // namespace

    void register_program(
        const std::string& name, Builder build, Requirements requirements
    ) {
        std::lock_guard lock(registry_mutex());
        registry()[name] = Entry{std::move(build), requirements};
    }

    tt::tt_metal::ComputeConfigDescriptor compute_config(
        const Numerics& numerics, const std::vector<uint8_t>& unpack_fp32
    ) {
        std::vector<tt::tt_metal::UnpackToDestMode> unpack(
            NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default
        );
        for (const auto index : unpack_fp32) {
            unpack.at(index) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
        return tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = numerics.fidelity,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack,
            .math_approx_mode = numerics.approx
        };
    }

    tt::tt_metal::CoreCoord grid() {
        return bridge::detail::device()->compute_with_storage_grid_size();
    }

    uint32_t tile_count(const ttnn::Tensor& t) {
        return t.physical_volume() / tt::constants::TILE_HW;
    }

    Split split_tiles(uint32_t tiles, uint32_t min_per_core) {
        const auto s = bridge::program::split_tiles(tiles, min_per_core);
        return Split{s.cores, s.range, s.base, s.extra};
    }

}  // namespace ktt::extension

namespace ktt::bridge {

    std::vector<HandlePtr> run_registered(
        const std::string& name,
        const std::vector<HandlePtr>& inputs,
        const std::vector<std::vector<int64_t>>& out_shapes,
        const std::vector<DType>& out_dtypes
    ) {
        extension::Entry entry;
        {
            std::lock_guard lock(extension::registry_mutex());
            const auto it = extension::registry().find(name);
            if (it == extension::registry().end()) {
                throw std::invalid_argument("ktt::extension: no program named " + name);
            }
            entry = it->second;
        }
        const auto& r = entry.requirements;
        const auto n = numerics::effective({r.floor, r.ceiling, r.exact});
        const extension::Numerics numerics{n.fidelity, n.approx};
        return program::run_on_stream(
            inputs,
            out_shapes,
            out_dtypes,
            [build = entry.build, numerics](const auto& tensors) {
                program::launch(tensors, build(tensors, numerics));
            }
        );
    }

}  // namespace ktt::bridge
