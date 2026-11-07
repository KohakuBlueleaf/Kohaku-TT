// Bridge row gather: rows of a 2-D table at int32 indices through TT-NN's embedding
// kernel, which reads a row-major table; the row-major copy is cached per table
// version.
#include <memory>
#include <vector>

#include "core/device/handle.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/embedding/embedding.hpp"

namespace ktt::bridge {
    namespace {

        // Row-major copies of recent tables (worker thread only); oldest dropped first.
        struct RowMajor {
            std::weak_ptr<Handle> table;
            int64_t version, epoch;
            ttnn::Tensor tensor;
        };

        constexpr size_t kCachedTables = 8;

        std::vector<RowMajor>& row_major_cache() {
            static auto* cache = new std::vector<RowMajor>();
            return *cache;
        }

        // The row-major copy of `table` at (`version`, write `epoch`), converted on a
        // miss; `cache` false (traces alive) converts without caching.
        ttnn::Tensor row_major(
            const HandlePtr& table, int64_t version, int64_t epoch, bool cache_it
        ) {
            auto rm_of = [&] {
                return ttnn::to_layout(table->tensor, tt::tt_metal::Layout::ROW_MAJOR);
            };
            if (!cache_it) {
                return rm_of();
            }
            auto& cache = row_major_cache();
            for (auto& entry : cache) {
                if (entry.table.lock() == table && entry.version == version &&
                    entry.epoch == epoch) {
                    return entry.tensor;
                }
            }
            auto rm = rm_of();
            std::erase_if(cache, [&](const RowMajor& e) {
                return e.table.expired() || e.table.lock() == table;
            });
            if (cache.size() >= kCachedTables) {
                cache.erase(cache.begin());
            }
            cache.push_back({table, version, epoch, rm});
            return rm;
        }

    }  // namespace

    namespace detail {

        void clear_row_major() {
            post([] {
                row_major_cache().clear();
            });
        }

    }  // namespace detail

    HandlePtr embedding_rows(
        const HandlePtr& table, int64_t version, const HandlePtr& index, int64_t count
    ) {
        std::vector<int64_t> shape{count, table->shape.back()};
        const int64_t epoch = write_epoch();
        const bool cache_it = !traces_alive();
        return detail::timed_op(table->dtype, shape, [=] {
            using tt::tt_metal::Layout;
            auto idx = ttnn::typecast(index->tensor, tt::tt_metal::DataType::UINT32);
            idx = ttnn::to_layout(idx, Layout::ROW_MAJOR);
            idx = ttnn::reshape(idx, ttnn::Shape({static_cast<uint32_t>(count)}));
            return (
                ttnn::embedding(
                    idx,
                    row_major(table, version, epoch, cache_it),
                    std::nullopt,
                    Layout::TILE
                )
            );
        });
    }

}  // namespace ktt::bridge
