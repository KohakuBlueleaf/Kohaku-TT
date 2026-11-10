# Writing a tt kernel: SwiGLU in TT-Metalium

The template for a hand-written TT-Metalium kernel run on `tt` tensors. `swiglu.py`
describes a program computing `silu(a) * b` with `kohakutt.kernels` and registers it as
the torch op `torch.ops.kohakutt_example.swiglu`:

```bash
python examples/kernels/metalium/swiglu.py --rows 1024 --cols 4096
```

```python
import kohakutt.kernels as kk


@kk.op("kohakutt_example::swiglu(Tensor a, Tensor b) -> Tensor")
def swiglu(a, b):
    out = kk.empty_like(a)
    split = kk.split_tiles(a)
    return kk.Program(
        cbs=[kk.CB(0, a), kk.CB(1, b), kk.CB(16, out)],
        reader=kk.Kernel("kernels/reader.cpp", [a, b], lambda c: [a, b, *split[c]]),
        compute=kk.Kernel(
            "kernels/compute.cpp", runtime_args=lambda c: [split[c][1]], fp32_dest=True
        ),
        writer=kk.Kernel("kernels/writer.cpp", [out], lambda c: [out, *split[c]]),
        outputs=[out],
        cores=split,
    )


y = torch.ops.kohakutt_example.swiglu(a.to("tt"), b.to("tt"))
```

The builder takes the op's arguments and returns a `Program`; the backend turns it
into a TT-Metal `ProgramDescriptor` over the device tensors and runs it on the default
stream, in order with every other tt op. The builder runs once per argument sizes,
dtypes, non-tensor values and `kohakutt.numerics` scope: the backend remembers the
launch, and later such calls go from the torch dispatcher to the program without
running Python.

## Anatomy of a program

| piece | file | role |
|---|---|---|
| reader | `kernels/reader.cpp` | DRAM -> circular buffers 0, 1 (inputs), via `TensorAccessor` |
| compute | `kernels/compute.cpp` | circular buffers -> DST registers -> op -> circular buffer 16 |
| writer | `kernels/writer.cpp` | circular buffer 16 -> DRAM |
| host | `swiglu()` in `swiglu.py` | circular buffers, the work split over the grid, per-core args, kernel configs |

## `kohakutt.kernels`

| name | role |
|---|---|
| `op(schema)` | registers the decorated builder as a torch op on `tt` tensors |
| `Program(cbs, reader, compute, writer, outputs, cores)` | one program; `cores` is a `Split` or a list of `(x, y)` |
| `CB(index, like, tiles=2)` / `CB(index, dtype=..., tiles=2)` | a circular buffer in the format of a tensor or of a dtype |
| `Kernel(source, compile_args, runtime_args, defines, ...)` | a kernel; `source` relative to the calling file; compute options: `fidelity` / `approx` (None: the `kohakutt.numerics` scope), `min_fidelity` / `max_fidelity` / `exact` (bounds on the scope), `fp32_dest` (default on), `unpack_fp32` |
| `empty(sizes, dtype)` / `empty_like(t)` | an output tensor of the program |
| `split_tiles(t, min_per_core=1)` | tiles of `t` split over the grid; `split[core]` = `(start, count)` |
| `tile_count(t)`, `grid()` | tile count of a tensor; the compute grid `(columns, rows)` |

Arguments are ints and tensors (inputs or outputs). A tensor in `compile_args` becomes
its `TensorAccessorArgs`; in `runtime_args` it becomes its buffer address. Tensors are
TILE layout on device, their storage format (bf16, fp32, int32 or a block float) is the
format a `CB(index, like=t)` gets. Only buffer addresses and runtime args change
between calls, so a program compiles once per layout and argument count.

Conventions shared by every Kohaku-TT kernel:

- inputs in circular buffers 0, 1, ...; the output in 16;
- buffer addresses and each core's tile range are runtime args;
- precision from the caller's `kohakutt.numerics` scope (HiFi2 by default); a kernel
  that needs more states `min_fidelity` or `exact` rather than pinning a value;
- fp32 math: `fp32_dest=True` (the default) and, for fp32 inputs, `unpack_fp32=[cb
  indices]`
  (otherwise the FPU source registers keep ~19 bits);
- every reserve / push on a circular buffer has one size dividing its capacity (a
  transaction never wraps the buffer end).

This example moves one tile per step for clarity. Kernels that reach DRAM bandwidth move
tiles in blocks (one NoC barrier per block) and process a block per DST acquire. See
the in-tree kernels:

| kernel | where |
|---|---|
| blocked unary (any SFPU op chain) | `src/kohakutt_native/csrc/aten/eltwise/ktt_unary/` |
| broadcast binary (b tile reused across its repeats) | `src/kohakutt_native/csrc/aten/eltwise/ktt_binary/` |

## A kernel in SFPU instructions: RoPE

`rope.py` rotates the column pairs `(x[..., 2i], x[..., 2i + 1])` of a bf16 tensor by
the `(cos, sin)` pairs of an fp32 table `[S, D]`, in one program:

```bash
python examples/kernels/metalium/rope.py --batch 9 --heads 16 --tokens 576 --dim 64
```

| piece | file | what it shows |
|---|---|---|
| reader | `kernels/rope_reader.cpp` | 4 tiles per NoC barrier; the table tile of x tile `t` is `t % table tiles` (the table repeats over x's leading dims) |
| compute | `kernels/rope_compute.cpp` | bf16 x and the fp32 table in fp32 DST (`unpack_fp32=[1]`, a source format switch per operand); the rotation as SFPU instructions (`SFPLOAD` of the even / odd columns of four rows, four `SFPMUL`, two `SFPADD`, two `SFPSTORE`); packed to bf16 |
| writer | `kernels/writer.cpp` | SwiGLU's |

Model code writes the same math as a cast to fp32, `view_as_complex`, a complex
multiply, `view_as_real` and a cast back. Measured on a Blackhole p150, HiFi2, per call:

| x | program | aten ops |
|---|---|---|
| (9, 16, 576, 64) | 0.113 ms | 0.325 ms |
| (1, 16, 5184, 64) | 0.120 ms | 0.311 ms |

The aten ops are three programs (cast, the backend's complex multiply on interleaved
pairs, cast).

The error against float64 equals that of the CPU doing the same fp32 math and rounding
to bf16 (rel median 1.35e-3, the bf16 output rounding); at most 0.02 % of the values
differ from the CPU's.

## Checking a kernel

- **Precision:** error statistics (max / median / mean of abs and rel error) of the
  kernel and of a same-precision CPU simulation, both against a CPU fp64 ruler; equal
  statistics mean a correct kernel.
- **Speed:** time per call over many async calls with one sync, against the op it
  replaces and against the DRAM roofline; replace only where faster.

Kernels for ATen ops live in `src/kohakutt_native/csrc/aten/<family>/`
([templates](../../../src/kohakutt_native/csrc/aten/README.md)); other kernels are user
ops built with `kohakutt.kernels` as here, or with the kernel DSL
([`../dsl_metal/`](../dsl_metal/)).
