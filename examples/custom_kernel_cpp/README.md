# A tt kernel as a C++ extension: SwiGLU

The C++ counterpart of [`../custom_kernel/`](../custom_kernel/): the same reader /
compute / writer kernels, with the program built in C++ and the op registered from C++.
An extension library has two halves that never include each other's headers:

| file | side | role |
|---|---|---|
| `swiglu_program.cpp` | device (TT-NN / Metalium, `ktt/extension/device.hpp`) | builds the `ProgramDescriptor`, registered by name with `KTT_REGISTER_PROGRAM` |
| `swiglu_op.cpp` | torch (`ktt/extension/torch.hpp`) | defines `kohakutt_example_cpp::swiglu` and runs the program on tt tensors with `ktt::extension::run` |
| `CMakeLists.txt` | build | `kohakutt_add_extension(...)` from the installed kohakutt |

```bash
cmake -S examples/custom_kernel_cpp -B build/swiglu_ext -G Ninja \
    -DPython_EXECUTABLE=$(which python) -DTT_METAL_ROOT=<tt-metal tree>
ninja -C build/swiglu_ext
python examples/custom_kernel_cpp/swiglu.py --library build/swiglu_ext/swiglu_ext.so
```

```python
import kohakutt.extension

kohakutt.extension.load("build/swiglu_ext/swiglu_ext.so")
y = torch.ops.kohakutt_example_cpp.swiglu(a_tt, b_tt)
```

## API

Device side (`ktt/extension/device.hpp`):

| name | role |
|---|---|
| `register_program(name, builder, requirements)`, `KTT_REGISTER_PROGRAM(name, fn[, requirements])` | a builder `ProgramDescriptor(const std::vector<ttnn::Tensor>&, const Numerics&)` over the inputs, then the outputs; `Requirements{.floor, .ceiling, .exact}` bound the caller's numerics scope |
| `Numerics`, `compute_config(numerics, unpack_fp32)` | the effective fidelity and approximation; a `ComputeConfigDescriptor` of them (fp32 accumulation) |
| `split_tiles(tiles, min_per_core)` | `Split`: `cores`, `range`, `start(i)`, `count(i)` |
| `tile_count(t)`, `grid()` | tiles of a tensor; the compute grid |

Torch side (`ktt/extension/torch.hpp`):

| name | role |
|---|---|
| `run(name, inputs, outputs)` | runs program `name` on the default stream over the dense device tensors of `inputs` and new outputs (`{sizes, dtype}`); returns the output tt tensors |

The builder runs on the stream worker at every call, so building the descriptor costs C++
time only; `generic_op` compiles one program per descriptor structure (buffer addresses
and runtime args are not part of it). Outputs are TILE, DRAM-interleaved tensors.

The library must be built against the same tt-metal tree and torch as kohakutt;
`kohakutt.extension.cmake_module()` and `include_dir()` give the paths a build needs.
