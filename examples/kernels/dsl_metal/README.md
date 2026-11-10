# Kernels in the DSL's metal level

`swiglu_km.py` and `rope_km.py` describe the two programs of
[`../metalium/`](../metalium/) with `kohakutt.dsl.metal` (`km`) instead of hand-written
kernel files. The program is a model (buffers, the three kernels as typed bodies, the
arguments with their binding times); a verifier checks it and an emitter writes the
kernel sources.

```bash
python examples/kernels/dsl_metal/swiglu_km.py --rows 1024 --cols 4096
python examples/kernels/dsl_metal/rope_km.py --batch 9 --heads 16 --tokens 576 --dim 64
```

```python
import kohakutt.dsl.metal as km


@km.program("swiglu")
def swiglu_program(p, a: km.TensorRef, b: km.TensorRef, out: km.Out):
    start, count = p.split(p.runtime("tiles"))  # each core derives its tile range
    cb_a, cb_b, cb_out = p.cb(0, like=a), p.cb(1, like=b), p.cb(16, like=out)

    @p.dm0  # the reader
    def reader(k):
        for i in k.range(count):
            a_slot, b_slot = cb_a.reserve(1), cb_b.reserve(1)
            t = k.let("t", start + i)
            k.read(a.page(t), a_slot)
            k.read(b.page(t), b_slot)
            k.reads_done()
            cb_a.push(1)
            cb_b.push(1)

    ...  # @p.compute and @p.dm1 (the writer): see swiglu_km.py
```

| piece | role |
|---|---|
| `@km.program(name)` | a program function `fn(p, tensors..., statics...)`; traced once per kernel class |
| `km.TensorRef`, `km.Out` | an input / output tensor: its format is static, its address and size are runtime values |
| `p.runtime(name)` | a runtime value: one common argument, never part of the compiled kernel |
| `p.runtime(name, choices=(1, 2, 3))` | a runtime value from a fixed list; it may be a push / pop count, a loop step or a register slot, and every choice is checked |
| `p.runtime(name, multiple_of=n)` | a runtime value every launch gives as a multiple of `n` (an int or another runtime value); a loop over it in steps of `n` is whole |
| `p.split(total)` | `(first, count)` of the running core, computed on the device from its core index |
| `p.stride(total)` | `(first, step)`: the items dealt out in turn, core c taking c, c + cores, ...; loop with `k.range(total, start=first, step=step)` |
| `p.cb(index, like=t, pages=n)` | a circular buffer; `reserve` / `push` / `wait` / `pop` inside kernel bodies |
| `@p.dm0`, `@p.dm1`, `@p.compute` | the reader, writer and compute bodies (`k.read`, `k.write`, `c.copy`, `c.pack`, ...); a program may be one kernel that reads and writes |
| `k.let(name, value)`, `k.var(name, initial)` / `k.set(v, value)`, `with k.when(a.ne(b)):` | a named value, a 32-bit variable, and statements run only where a comparison holds |
| `slot.load32(i).bounded(low, high)` | a word of a reserved page read on the device (data), with the range the launch guarantees |
| `ref.span(page, offset, size)`, `slot.at(offset, like=src)` | a sub-page read at value offsets; `like` moves the destination to the source's alignment |
| `k.write(slot, ref.page(i))` | writes a reserved page that was filled in place |
| `km.SfpuFn`, `km.CppFn`, `k.inline(code, uses=, args=)` | inline C++ and helper functions, with the buffers, values and constants they touch declared |
| `km.sfpu_fn(name, expr)`, `c.run(fn, slots..., out=)` | a vector-unit function written from an expression over register operands (`km.sfpu`: arithmetic, `mad`, min / max, `select`, library functions), so a chain of elementwise steps is one pass |
| `km.op(schema, program)` | registers the launch builder as a torch op; it returns a `km.Call` |

What the metal level adds to `kohakutt.kernels`:

- **Sizes never compile.** Tile counts and addresses are runtime arguments and the
  split is derived on the device, so a new batch or length reuses the compiled kernels.
  The core count is the smallest entry of a fixed list (1, 2, 3, 4, 6, 8, 12, ...,
  64, 96, the grid; steps of at most 1.5x) that covers the work, so a core never takes
  more than it would on an exact count and some may take none. TT-Metal builds one
  program per core count used, not one per shape.
- **Checked before anything is emitted.** `km.RULES` lists the rules, each from a
  hazard hit on the hardware: push and pop counts divide their buffer, pushes equal
  pops, a kernel writes no further than it reserved, reader and writer use different
  NoCs, a push follows its read barrier, register slots fit the mode, and more.
  `km.VerifyError` names the rule.
- **Generated sources** are written under `KOHAKUTT_CACHE/km` (default:
  `~/.cache/kohakutt/km`, or under `XDG_CACHE_HOME`), named by content; `km.source(program, role)` returns the text.

Both examples print whether their output equals the hand-written program's (it does,
bit for bit) and both times. Measured on a Blackhole p150 over 14 RoPE shapes and 10
SwiGLU shapes: equal output on all, no kernel compiled after the first call, and warm
times within noise of the hand-written programs (RoPE 0.108 ms against 0.108 ms at
(9, 16, 576, 64)).

## A matmul

`matmul_km.py` is the backend's own matmul program (`a [..., M, K] @ w`, bf16, K and N
up to 1024) as one model: the three kernels and the launch plan in one Python file.

```bash
python examples/kernels/dsl_metal/matmul_km.py --rows 16384 --inner 768 --cols 1024
```

It uses the pieces a kernel with its own layout needs:

| piece | role |
|---|---|
| `p.layout(cores=groups * per_group)` | the cores that have work, from runtime values; the rest return at once |
| `p.define("n0", p.core // per_group * nt)` | a core's place in the layout, computed on the device |
| `p.sized("w_pages", of=kt * nt, limit=128)` | a buffer size computed per launch and rounded up to a class, so it is part of the program build and of no kernel |
| `c.matmul_init(...)`, `c.matmul(m, ia, ib, slot)` | the matrix unit: the registers sum the products of one output row tile |
| `km.select(cond, a, b)`, `km.min(a, b)` | values chosen on the device |

`plan()` picks the column tiles per core (4, 2 or 1) that fit the L1 budget from the
program's own footprint. Measured on a Blackhole p150 over 11 shapes (K 64 to 1024, N 32
to 1024, both layouts of w): output equal bit for bit to the backend's program on all,
no kernel compiled after the first call, 9 program builds, and warm times within noise
of the backend (2.99 ms against 2.97 ms per call at (16384, 768) @ (768, 1024)).

## An RMS norm

`rmsnorm_km.py` is the backend's own norm forward with one core per tile row, as one
model. Its rows travel as control words, and past one row per core each reader takes
its next row from a counter on core 0.

```bash
python examples/kernels/dsl_metal/rmsnorm_km.py --rows 4096 --width 768
```

| piece | role |
|---|---|
| `p.words(24)`, `k.put(cb, r)`, `c.take(cb)` | a row index handed from the reader to the compute and the writer |
| `with k.forever():` ... `k.exit_when(r >= rows)` | loops that end on the index past the last row |
| `p.core_ref(0)`, `k.fetch_add(counter, at=first)` | the next row from a semaphore on core 0 |
| `c.inline(...)`, `km.SfpuFn(...)` | the backend's two-pass statistics calls and vector-unit code, unchanged |

Measured on a Blackhole p150 over 7 shapes (bf16 and fp32, with and without gamma,
ragged rows and widths, fewer and more tile rows than cores): output equal bit for bit
to the backend's norm (`KTT_NORM_GROUP=1`) on all, no kernel compiled for a new shape.

## A block matmul

`gemm_km.py` is the backend's block matmul (`KTT_GEMM=1`) in its one-row-per-core
layout: each core owns a range of output row tiles and reads its own a; the top core of
each column reads w and multicasts it down the column.

```bash
python examples/kernels/dsl_metal/gemm_km.py --rows 4096 --inner 1024 --cols 768
```

| piece | role |
|---|---|
| `p.kernel("dm0", cores=p.cores.rows(0), name=...)` | the w reader variant on the top row, another on the rows below |
| `p.lane(top, below)`, `lane.send(k, slot)`, `lane.receive(k)` | the multicast of each w block with its handshake |
| `km.k_passes(k, nk, body)` | the K blocks in the form every kernel shares |
| `km.matmul_passes(c, ...)`, `p.partials(24, pages=block)` | the K blocks' products summed by packer accumulation |
| `p.choose(("sh", "sw"), SUBBLOCKS)` | the output subblock, one of the backend's shapes |

Measured on a Blackhole p150 over 9 shapes: output equal bit for bit to the backend's
block matmul run with the same plan (`KTT_GEMM_PLAN`). It needs a grid of two rows or
more.
