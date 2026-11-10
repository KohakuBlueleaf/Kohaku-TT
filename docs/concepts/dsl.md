---
title: Kernel DSL
summary: "Kernel programs written in Python: what the metal level describes, what never recompiles, what the verifier checks."
tags:
  - concepts
  - dsl
---

# Kernel DSL

`kohakutt.dsl` describes a kernel program in Python and turns it into TT-Metalium
kernel sources. A program written this way is a torch op on `tt` tensors like any
hand-written one: it runs on the default stream, in eager mode and inside traces.

The package has one level today, the **metal level** (`kohakutt.dsl.metal`, imported as
`km`). It names exactly what a Metalium program has: circular buffers, the two
data-movement kernels, the compute kernel, NoC reads and writes, register slots. It adds
no abstraction over the hardware; it adds binding times and checks.

Working examples: [`examples/kernels/dsl_metal/`](../../examples/kernels/dsl_metal/)
(SwiGLU and RoPE, each beside its hand-written twin in `examples/kernels/metalium/`).

## Why a language

Hand-written kernels share one structure, and the effort is not in the math:

| Part | Content | Where the effort goes |
|---|---|---|
| reader | tile ranges per core; DRAM -> L1 circular buffers | data movement |
| compute | per-tile unpack, math, pack over circular buffers | numerics choices |
| writer | circular buffer -> DRAM by tile range | little |
| host | program description: buffers, kernel sources, arguments, core split | boilerplate, error-prone |

The rules that make a kernel correct on the hardware (a push size divides its buffer, a
push follows its read barrier, register slots fit the mode) are not checked by any
compiler: breaking one gives a hang or wrong values. The DSL holds the program as a
model so those rules are checked before a kernel is built.

## A program

```python
import kohakutt.dsl.metal as km


@km.program("swiglu")
def swiglu_program(p, a: km.TensorRef, b: km.TensorRef, out: km.Out):
    start, count = p.split(p.runtime("tiles"))
    cb_a, cb_b, cb_out = p.cb(0, like=a), p.cb(1, like=b), p.cb(16, like=out)

    @p.dm0
    def reader(k):
        for i in k.range(count):
            a_slot, b_slot = cb_a.reserve(1), cb_b.reserve(1)
            t = k.let("t", start + i)
            k.read(a.page(t), a_slot)
            k.read(b.page(t), b_slot)
            k.reads_done()
            cb_a.push(1)
            cb_b.push(1)

    ...  # @p.compute and @p.dm1
```

The function body runs once, in Python, and records the program; the loops and
arithmetic on values become C++ in the generated kernels. `km.op(schema, program)`
registers a torch op whose builder returns a `km.Call` (the tensors and the runtime
values of this call). The builder runs once per argument sizes, dtypes, non-tensor
values and `kohakutt.numerics` scope; the backend remembers the launch and serves
later such calls without running Python.

## Building a program by code

The decorated form above is written by a person. A compiler builds the same model
through the builder underneath it, with no traced function and no torch op:

```python
p = km.Program("scale", km.Machine(cols, rows))
x, out = p.input("x", "bfloat16"), p.output("out", "bfloat16")
start, count = p.split(p.runtime("tiles"))
cb_x = p.cb(0, like=x)

k = p.kernel("dm0")  # a builder; several may be open at once
with k.loop(count) as i:
    slot = k.reserve(cb_x, 1)
    k.read(x.page(start + i), slot)
    k.reads_done()
    k.push(cb_x, 1)
...  # p.kernel("compute"), p.kernel("dm1")

run = km.Launcher(p)  # verifies, emits; a description per core count and sizes
(y,) = run({"x": t}, outputs={"out": (list(t.shape), t.dtype)}, runtime={"tiles": n})
```

| Decorated form | By code |
|---|---|
| tensors from the function's annotations | `p.input(name, dtype)`, `p.output(name, dtype)` |
| `@p.dm0` / `@p.compute` / `@p.dm1` over a function | `p.kernel(role, ...)` returns the builder |
| `cb.reserve(n)`, `cb.push(n)`, `cb.wait(n)`, `cb.pop(n)` | `k.reserve(cb, n)`, `k.push(cb, n)`, `k.wait(cb, n)`, `k.pop(cb, n)` |
| `for i in k.range(n)`, `for done, n in k.blocks(count, b)` | `with k.loop(n) as i`, `with k.block_loop(count, b) as (done, n)` |
| `with c.registers():` | the same, or `c.acquire()` ... `c.commit()` ... `c.release()` |
| `km.op(schema, template)` | `km.Launcher(program)` |

Both forms record the same model: `km.dump(program)` prints it (declarations, then
each kernel's statements as a tree), and `km.fingerprint(program)` hashes that text
and the generated sources: equal fingerprints emit the same kernels (`km.op` still
keys its kernel classes on dtypes and statics). A value has a host
side too: `value.eval({"tiles": 96, "core": 3, "cores": 16})` computes what the device
would, which is how per-core arguments written as values of `p.core` are filled.

## The compute kernel

A compute body names what each unit of the core does:

| Unit | Statements |
|---|---|
| unpacker | `c.copy(cb, i, slot)`, `c.copy_block(cb, i, slot, n)`, `c.transpose(cb, i, slot)`, `c.bcast(cb, i, slot, "row" \| "col")`; `c.format(srca=, srcb=)` switches its formats |
| matrix unit | `m = c.matmul_init(a, b, transpose, ct, rt, kt)`, then `c.matmul(m, ia, ib, slot)`: the registers sum over calls |
| vector unit | `c.op(name, slots..., value=)` for the ops of the op table (`km.ops.OPS`: unary functions, scalar arithmetic, binary arithmetic, comparisons, min / max, fill); `c.sfpu(fn, a, b, out, mode=, args=)` for an own function |
| packer | `c.pack(slot, cb)`, `c.pack(slot, cb, at=i)` (page `i` of the reserve), `c.pack_format(cb)`, `c.pack_accumulate(1, cb)` ... `c.pack_accumulate(0)` (a pack adds to its page in L1) |

A way of unpacking and a table op each take their setup first (`c.init("copy", cb)`,
`c.init("rsqrt")`); the matmul's setup is `c.matmul_init`. The op table is data: one
row (call, setup call, header, operand counts) adds an op.

A branch on a value that is fixed for the launch may hold buffer calls; both kernels of
a buffer then branch on the same value:

```python
with c.when(flag.ne(0)):
    c.init("mul")
    c.op("mul", 0, 2, 0)
with c.otherwise():
    c.init("sub_scalar")
    c.op("sub_scalar", 0, value=1.0)
```

## Binding times

Every value in a program has one of these binding times, and the binding time decides
what a change of that value costs:

| Binding time | Examples | A new value costs |
|---|---|---|
| static | dtype, the op body, buffer page counts, declared `km.Static` parameters | a kernel compile |
| size class | the number of cores: the smallest of 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, the whole grid that covers the split's total (cores past it take no work) | one TT-Metal program build, no kernel compile |
| sized | `p.sized("w_pages", of=kt * nt, limit=128)`: computed on the host from runtime values and rounded up to a class (1, 2, 3, 4, 6, 8, 12, ..., in steps of at most 1.5x); it may be a buffer's page count. `exact=True` takes the value itself (one build per distinct value) | one TT-Metal program build, no kernel compile |
| runtime | tile counts, tensor addresses, `p.runtime(name)` values | nothing |
| runtime, from a fixed list | `p.runtime("block", choices=(1, 2, 3))`: a value that may also be a push / pop count, a loop step or a register slot; the verifier checks every choice | nothing |
| runtime, a declared multiple | `p.runtime("nr", multiple_of="block")`: checked at every launch, so a loop over it in steps of `block` is whole | nothing |
| per core | the core's index, from which it derives its own range (`p.split`), its items in turn (`p.stride`), or its place in a layout (`p.define("n0", p.core // per_group * nt)`) | nothing |
| data | a word loaded from L1 on the device (`slot.load32(i)`), such as a row index; it is never a count or a loop bound, and it selects what a NoC read fetches only with a declared range (`.bounded(low, high)`) | nothing |

The range of a data value is the launch's contract: nothing checks it on the device, so
an index outside its table reads other memory.

A buffer sized from a runtime value is declared through a sized value, so the buffer's
size is part of the program description and not of any kernel:

```python
a_pages = p.sized("a_pages", of=kc, limit=32)  # rounded up to a class
cb_a = p.cb(0, like=a, pages=2 * a_pages)
...
chunk = cb_a.reserve(a_pages)  # pushes count in the same value
for i in k.range(n):  # n <= kc <= a_pages tiles are read
    k.read(a.page(first + i), chunk[i])
```

The verifier checks a sized buffer for every value it can take, and takes a page
index as inside it when it is written so: a loop variable counting to at most the
buffer's pages, or a value declared `.bounded(0, pages)`. A read of `n` pages from an
index (a block copy, a matmul's `ct` tiles of b) needs the index declared
`.bounded(0, pages - n + 1)`, or static bounds that fit.

Work that is not one even split is laid out by hand: `p.layout(cores=groups *
per_group)` states how many cores have work (the program runs on the smallest core
count class not below it, and every kernel returns at once on a core past it), and
each core computes its place from `p.core` with `p.define`.

So a new batch size, sequence length or image size compiles nothing: sizes are runtime
arguments and each core derives its share of the work on the device. Only what the
author declares static (a block size, a dtype) can cause a compile.

Measured on a Blackhole p150 with the two example programs, over 14 RoPE shapes and 10
SwiGLU shapes: no kernel compiled after the first call, 9 and 7 program builds (one per
core count used), output equal bit for bit to the hand-written kernels, warm time
within noise of them.

The matmul example (`examples/kernels/dsl_metal/matmul_km.py`: a layout of column
groups, two sized buffers) over 11 shapes: no kernel compiled after the first call, 8
program builds (9 with exact sizes), output equal bit for bit to the backend's matmul
program, warm time within noise of it.

## The verifier

`km.verify(program)` runs before anything is emitted; `km.VerifyError` names the rule.
`km.RULES` lists them. Each rule comes from a failure seen on the hardware:

| Group | Checks |
|---|---|
| binding | a value is used only where its binding time allows (a runtime value is never a buffer size, a loaded value never a count); a sized value and a layout's core count are computed from runtime values; a stepped loop runs over a multiple of its step |
| circular buffers | push and pop sizes divide the buffer; pushes equal pops across the kernels (inside a condition only when it is fixed for the launch and both kernels branch on it); a kernel writes, packs and loads no further than it reserved (in-order packs are counted from their reserve unless a loop lies between them and it), and reads only pages it waited for; the buffers fit L1 at every combination of sized values |
| NoC | reader and writer use different NoCs; a sub-page read keeps source and destination aligned alike, proved from what the offsets are multiples of; a push, a load or a write follows its read barrier; a page is flushed before it is popped or read into again |
| compute | register slots fit the mode (8 bf16 or 4 fp32 tiles per half), a matmul block's `ct * rt` slots included; math sits between acquire and commit, packs between commit and release; a way of unpacking, a table op and a matmul each run under their own setup; an unpack reads a buffer of the format its source register was set to (`c.startup`, `c.format`; a matmul reads b through the first source); a pack goes into a buffer of the packer's format; packer accumulation adds only to a page packed earlier in the same reserve and is off at exit; a setup or format made only inside a loop that may run zero times does not count after it; a changed vector-unit constant is restored at exit |
| inline C++ | the buffers, semaphores and registers it touches are declared |

Inline C++ (`k.inline`, `km.SfpuFn`) is the way to write anything the typed operations
do not cover; the verifier checks its declared effects, not its text.

## Generated sources

Sources are written under `KOHAKUTT_CACHE/km` (default:
`~/.cache/kohakutt/km`, or under `XDG_CACHE_HOME`), named by content. `km.source(program, role)` returns the text of one kernel,
which reads like a hand-written Metalium kernel.

## Relation to `kohakutt.kernels` and tt-lang

`kohakutt.kernels` runs kernel files you wrote; the DSL writes them. Both produce the
same program description and share the launch path, so they can be mixed and compared.

[tt-lang](https://github.com/tenstorrent/tt-lang) has a similar shape (compute and
data-movement functions over dataflow buffers). It brings its own compiler and runtime,
which cannot share a process with TT-NN's; this DSL lowers in process to the program
descriptions every other Kohaku-TT op uses.
