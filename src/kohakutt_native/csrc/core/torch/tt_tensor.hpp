// tt tensors on the torch side: storage-backed TensorImpls whose storage context holds
// a base device tensor; views are ordinary PyTorch views of that storage.
#pragma once

#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "core/device/tt_bridge.hpp"

namespace ktt {

    // The storage context: the base device tensor, its logical sizes and dtype. Bridge
    // handles are immutable; an in-place write replaces `base` for every alias. A
    // storage filled with raw bytes (deserialization) keeps them in `pending` until
    // first typed use.
    struct TtStorage {
        bridge::HandlePtr base;
        std::vector<int64_t> sizes;
        at::ScalarType dtype = at::kByte;
        at::Tensor pending;

        // q / k / v heads split from `heads_of` (a fused [B, T, 3 H D] base), reused by
        // the three views of one attention (core/torch/views.cpp).
        bridge::HandlePtr heads_of;
        int64_t heads_count = 0, heads_epoch = -1;
        std::vector<bridge::HandlePtr> heads;

        // [B, C / D, T, D] head-major copy of `major_of` (a [B, T, C] base, head dim
        // `major_d`), reused by its head views (core/torch/views.cpp).
        bridge::HandlePtr major_of, major;
        int64_t major_d = 0, major_epoch = -1;

        // Placements the last consumers of a persistent mesh-placed base wanted
        // (core/torch/mesh_rules.cpp, migration).
        std::vector<std::string> wanted;
    };

    // Allocates the base of `st` for typed use as `dtype`: uploads pending bytes, or an
    // empty 1-D base for a storage that has never been written.
    void ensure_base(TtStorage& st, at::ScalarType dtype, size_t nbytes);

    // The raw bytes [offset, offset + count) of a storage, as a uint8 CPU tensor.
    at::Tensor storage_bytes(TtStorage& st, int64_t offset, int64_t count);

    // Replaces a storage's contents with raw host bytes (whole-storage byte writes).
    void set_storage_bytes(TtStorage& st, const at::Tensor& bytes);

    // The tt allocator: storages created by size alone (UntypedStorage on tt).
    c10::Allocator* tt_allocator();

    // The physical torch dtype a logical dtype is stored as on device: bf16, f32,
    // int32; int64 -> int32 (range-checked on upload), bool -> bf16 (0/1). Others raise
    // unsupported.
    at::ScalarType physical(at::ScalarType dtype);

    // The bridge dtype of a logical torch dtype (via physical()).
    bridge::DType to_bridge(at::ScalarType dtype);

    std::vector<int64_t> to_vector(c10::IntArrayRef sizes);

    // A new contiguous tt tensor (fresh storage) of `sizes` and `dtype` holding
    // `handle`.
    at::Tensor make_tt(
        bridge::HandlePtr handle, c10::IntArrayRef sizes, at::ScalarType dtype
    );

    // The storage context of tt tensor `t`.
    TtStorage& storage_of(const at::Tensor& t);

    // The device tensor holding exactly the elements of view `t` (views.cpp); raises
    // unsupported for stride patterns it cannot express.
    bridge::HandlePtr resolve(const std::string& op, const at::Tensor& t);

    // The block-float base of 2-D `t` when t is its whole storage, or (`transposed`
    // set) the transpose of it; null otherwise (matmul operands, aten/linalg).
    bridge::HandlePtr block_base(const at::Tensor& t, bool& transposed);

    // Records `st`'s current base before its first rebinding in an active capture
    // (capture.cpp); `storage` keeps the context alive.
    void note_rebind(TtStorage& st, const c10::Storage& storage);

    // Writes `value` (dense, sizes of `dst`) into view `dst` (views.cpp).
    void commit(const std::string& op, const at::Tensor& dst, bridge::HandlePtr value);

}  // namespace ktt
