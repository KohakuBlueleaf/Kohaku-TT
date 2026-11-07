# ATen op families

Every ATen op that `tt` provides is implemented here, one folder per op family:
`eltwise/`, `linalg/`, `attention/`, `nn/`, `reduce/`, `index/`, `tensor/`, `spectral/`.
A family folder holds everything for its ops:

| file | side | contents |
|---|---|---|
| `aten_<name>.cpp` | torch | the ATen surface: registration (`TORCH_LIBRARY_IMPL`), argument validation, PyTorch semantics (dtype promotion, broadcasting, views, in-place / out=) |
| `device_<name>.cpp` | device | the implementation: TT-NN calls and composites on device handles, declared in `core/device/tt_bridge.hpp`, enqueued on the default stream |
| `ktt_<name>/` | device | one of Kohaku-TT's own Metalium programs (no TT-NN op inside): `program.{hpp,cpp}` builds it (circular buffers, work split, runtime args); `reader.cpp`, `compute.cpp`, `writer.cpp` are its device kernels, JIT-compiled by TT-Metalium |

Torch-side and device-side files never include each other's headers: an `aten_*.cpp`
calls the device side only through `core/device/tt_bridge.hpp` (no TT-NN or torch types
cross it). CMake picks the side by file name. The full rules:
[kernel contract](../../../../docs/design/kernel-contract.md).

## Template: the ATen surface (`aten_<name>.cpp`)

```cpp
// aten <ops> for tt: <what>, in-place and out= forms; implemented in device_<name>.cpp.
#include <torch/library.h>

#include "core/torch/contract.hpp"

namespace ktt {
    namespace {

        // <op>(self, alpha): <what it computes>; <supported dtypes / ranks>.
        at::Tensor tt_op(const at::Tensor& self, double alpha) {
            const char* op = "<op>";
            KTT_REQUIRE(self.scalar_type() == at::kBFloat16, op, "<why not>");
            auto out = bridge::my_op(dense(op, self), alpha);
            return wrap_output(out, self.sizes(), self.scalar_type());
        }

        at::Tensor& tt_op_(at::Tensor& self, double alpha) {
            return write_into("<op>_", self, tt_op(self, alpha));
        }

        at::Tensor& tt_op_out(const at::Tensor& self, double alpha, at::Tensor& out) {
            return into("<op>.out", out, tt_op(self, alpha));
        }

    }  // namespace

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("<op>", &tt_op);
        m.impl("<op>_", &tt_op_);
        m.impl("<op>.out", &tt_op_out);
    }

}  // namespace ktt
```

- Unsupported variants raise through `KTT_REQUIRE` (`KohakuTTUnsupportedOperator`).
- Unary / binary elementwise ops are table rows (`KTT_UNARY`, `KTT_BINARY_ALPHA`).
- An op needing autograd-aware dispatch registers at `AutogradPrivateUse1`
  (`nn/aten_norm.cpp`, `rms_norm`).

## Template: the implementation (`device_<name>.cpp`)

```cpp
// in core/device/tt_bridge.hpp: <what it computes>; <shapes / dtypes it takes>.
HandlePtr my_op(const HandlePtr& a, double alpha);

// in aten/<family>/device_<name>.cpp
HandlePtr my_op(const HandlePtr& a, double alpha) {
    if (detail::is_empty(a)) {
        return detail::empty_of(a, a->dtype);
    }
    const bool own = ktt_my_op::supports(a->dtype, a->shape);
    return detail::timed_op(a->dtype, a->shape, [=] {
        if (own) {
            return ktt_my_op::run(a->tensor, alpha);
        }
        return ttnn::my_op(a->tensor, static_cast<float>(alpha));
    });
}
```

- An own program replaces the TT-NN op only for the cases where it is measured faster
  with equal error statistics (`has_*` decides per call); `KTT_CUSTOM_KERNELS=0`
  restores TT-NN everywhere for A/B runs.

## Template: an own program (`ktt_<name>/`)

Copy `eltwise/ktt_unary/` and rename it:

```
ktt_<name>/
  program.hpp    supports(...) (when this program runs), run(...)
  program.cpp    the host side, below
  reader.cpp     DRAM -> circular buffers
  compute.cpp    circular buffers -> DST -> op -> circular buffer 16
  writer.cpp     circular buffer 16 -> DRAM
```

```cpp
// in program.cpp
ttnn::Tensor run(const ttnn::Tensor& in, double alpha) {
    auto out = ttnn::empty(
        in.logical_shape(), in.dtype(), Layout::TILE, detail::device(), in.memory_config()
    );
    const auto split = program::split_tiles(tiles, kMinTilesPerCore);
    ProgramDescriptor desc;
    // circular buffers on split.range; per-core runtime args from split.count(i);
    // kernels from program::source("<family>/ktt_<name>/reader.cpp"), ...
    return ttnn::generic_op({in, out}, desc);
}
```

Device kernel conventions (`reader.cpp`, `compute.cpp`, `writer.cpp`):

- reader (DRAM to circular buffers), compute, writer (circular buffers to DRAM);
- inputs at CB 0, 1, ..., the output at CB 16;
- tiles move in blocks (one reserve / push and one NoC barrier per block); every
  transaction on a circular buffer has one size dividing its capacity (no wrap);
- buffer addresses and tile ranges are runtime args (one compiled program serves every
  tensor of a layout); layouts are compile-time `TensorAccessorArgs`;
- the op comes from compile-time defines (e.g. TT-NN's `SFPU_OP_CHAIN_0`).

Programs: `eltwise/ktt_unary/` (blocked unary, the template), `eltwise/ktt_binary/`
(broadcast binary). A standalone walkthrough: `examples/custom_kernel/`.
