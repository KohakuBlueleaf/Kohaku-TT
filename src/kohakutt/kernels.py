"""User Metalium programs registered as torch ops on tt tensors (``op``, ``Program``).
Guide and example: examples/kernels/metalium/.
"""

import json
import math
import sys
from pathlib import Path

import torch

from kohakutt.core.numerics import Requirements, current_numerics, resolve

_SCALAR_TYPES = {torch.bfloat16: 15, torch.float32: 6, torch.int32: 3}
_LIBRARIES = {}


class Output:
    """An output tensor of a program: logical sizes and torch dtype."""

    def __init__(self, sizes, dtype):
        self.sizes = list(sizes)
        self.dtype = dtype

    def size(self):
        """The logical sizes."""
        return list(self.sizes)


def empty(sizes, dtype=torch.bfloat16):
    """A new program output of `sizes` and `dtype`."""
    if dtype not in _SCALAR_TYPES:
        raise TypeError(f"kohakutt.kernels: unsupported output dtype {dtype}")
    return Output(sizes, dtype)


def empty_like(t, dtype=None):
    """A new program output with the sizes of `t` and its dtype (or `dtype`)."""
    return empty(t.size(), dtype or t.dtype)


def tile_count(t):
    """Number of 32x32 tiles of `t` (a tensor, an Output or sizes) on device."""
    sizes = list(t.size() if hasattr(t, "size") else t)
    sizes = [1] * max(0, 2 - len(sizes)) + sizes
    lead = math.prod(sizes[:-2])
    return lead * math.ceil(sizes[-2] / 32) * math.ceil(sizes[-1] / 32)


def grid():
    """The compute grid as (columns, rows)."""
    cols, rows = torch.ops.ktt.compute_grid()
    return cols, rows


def _core_ranges(cores):
    """`cores` as [x0, y0, x1, y1] ranges: runs of consecutive x within a row."""
    out = []
    for x, y in sorted(set(cores), key=lambda c: (c[1], c[0])):
        if out and out[-1][3] == y and out[-1][2] == x - 1:
            out[-1][2] = x
        else:
            out.append([x, y, x, y])
    return out


class Split:
    """Tiles split row-major over cores; ``split[core]`` is that core's (start, count)
    and ``cores`` the cores in use. The first ``extra`` cores take one tile more."""

    def __init__(self, tiles, min_per_core=1):
        cols, rows = grid()
        n = max(1, min(cols * rows, tiles // max(1, min_per_core)))
        self.tiles = tiles
        self.cores = [(i % cols, i // cols) for i in range(n)]
        self.base, self.extra = divmod(tiles, n)
        self._ranges, start = {}, 0
        for i, core in enumerate(self.cores):
            count = self.base + (i < self.extra)
            self._ranges[core] = (start, count)
            start += count

    def __getitem__(self, core):
        return self._ranges[tuple(core)]


def split_tiles(t, min_per_core=1):
    """The Split of the tiles of `t` (a tensor, an Output or a tile count)."""
    tiles = t if isinstance(t, int) else tile_count(t)
    return Split(tiles, min_per_core)


class CB:
    """Circular buffer `index` of `tiles` tiles, in the format of tensor `like`
    (an input or output) or of `dtype` ("bfloat16", "float32", "int32", "bfloat8_b").
    """

    def __init__(self, index, like=None, tiles=2, dtype=None):
        if (like is None) == (dtype is None):
            raise ValueError(
                "kohakutt.kernels.CB: give exactly one of `like` and `dtype`"
            )
        self.index, self.like, self.tiles, self.dtype = index, like, tiles, dtype


class Kernel:
    """A kernel source with its arguments.

    `source` is resolved against the caller's directory. `runtime_args` is a callable
    core -> list, a {core: list} dict or one list for every core. `defines` maps macro
    names to values. The remaining options apply to compute kernels: `fidelity` and
    `approx` pin a value (None: the kohakutt.numerics scope); `min_fidelity`,
    `max_fidelity` and `exact` bound the scope's value (ignored in a strict scope).
    """

    def __init__(
        self,
        source,
        compile_args=(),
        runtime_args=None,
        defines=None,
        fp32_dest=True,
        fidelity=None,
        approx=None,
        min_fidelity=None,
        max_fidelity=None,
        exact=False,
        unpack_fp32=(),
    ):
        path = Path(source)
        if not path.is_absolute():
            caller = Path(sys._getframe(1).f_code.co_filename).resolve().parent
            path = caller / path
        self.source = str(path)
        self.compile_args = list(compile_args)
        self.runtime_args = runtime_args
        self.defines = dict(defines or {})
        self.fp32_dest, self.unpack_fp32 = fp32_dest, list(unpack_fp32)
        self.fidelity, self.approx = fidelity, approx
        self.requirements = Requirements(min_fidelity, max_fidelity, exact)

    def compute_options(self):
        """The compute options of the spec, with the active numerics resolved."""
        active = resolve(current_numerics(), self.requirements)
        return {
            "fp32_dest": self.fp32_dest,
            "unpack_fp32": self.unpack_fp32,
            "fidelity": self.fidelity or active.fidelity,
            "approx": active.approx if self.approx is None else self.approx,
        }

    def args_for(self, core):
        """This kernel's runtime args on `core`."""
        if self.runtime_args is None:
            return []
        if callable(self.runtime_args):
            return list(self.runtime_args(core))
        if isinstance(self.runtime_args, dict):
            return list(self.runtime_args.get(tuple(core), []))
        return list(self.runtime_args)


class Program:
    """A program: circular buffers, three kernels and its outputs.

    `cores` is a Split or a list of (x, y) cores running the program; by default the
    whole grid.
    """

    def __init__(self, cbs, reader, compute, writer, outputs, cores=None):
        self.cbs, self.outputs = list(cbs), list(outputs)
        self.kernels = {"reader": reader, "compute": compute, "writer": writer}
        if isinstance(cores, Split):
            cores = cores.cores
        if cores is None:
            cols, rows = grid()
            cores = [(x, y) for y in range(rows) for x in range(cols)]
        self.cores = [tuple(c) for c in cores]

    def _spec(self):
        """(JSON description, input tensors); tensors in the description are indices
        into the inputs followed by the outputs."""
        inputs, refs = [], {}
        for t in self._tensor_items():
            if isinstance(t, torch.Tensor) and id(t) not in refs:
                refs[id(t)] = len(inputs)
                inputs.append(t)
        for i, o in enumerate(self.outputs):
            refs[id(o)] = len(inputs) + i

        def index(t):
            if id(t) not in refs:
                raise TypeError(f"kohakutt.kernels: not an input or output: {t!r}")
            return refs[id(t)]

        def items(values, kind):
            out = []
            for v in values:
                if isinstance(v, bool) or not isinstance(
                    v, (int, torch.Tensor, Output)
                ):
                    raise TypeError(f"kohakutt.kernels: bad argument {v!r}")
                out.append(int(v) if isinstance(v, int) else {kind: index(v)})
            return out

        ranges = _core_ranges(self.cores)
        cbs = []
        for cb in self.cbs:
            entry = {"index": cb.index, "tiles": cb.tiles, "cores": ranges}
            if cb.like is not None:
                entry["tensor"] = index(cb.like)
            else:
                entry["dtype"] = cb.dtype
            cbs.append(entry)
        kernels = []
        for kind, k in self.kernels.items():
            entry = {
                "source": k.source,
                "type": kind,
                "cores": ranges,
                "compile_args": items(k.compile_args, "accessor"),
                "runtime_args": [
                    [list(core), items(k.args_for(core), "addr")] for core in self.cores
                ],
                "defines": {name: str(v) for name, v in k.defines.items()},
            }
            if kind == "compute":
                entry.update(k.compute_options())
            kernels.append(entry)
        return json.dumps({"cbs": cbs, "kernels": kernels}), inputs

    def _tensor_items(self):
        """Every tensor named by the CBs and kernel arguments, in order."""
        out = [cb.like for cb in self.cbs if cb.like is not None]
        for k in self.kernels.values():
            out += [v for v in k.compile_args if isinstance(v, (torch.Tensor, Output))]
            for core in self.cores:
                out += [
                    v for v in k.args_for(core) if isinstance(v, (torch.Tensor, Output))
                ]
        return out

    def call(self):
        """(input tensors, the other arguments of torch.ops.ktt.run_program)."""
        spec, inputs = self._spec()
        ranks = [len(o.sizes) for o in self.outputs]
        sizes = [s for o in self.outputs for s in o.sizes]
        dtypes = [_SCALAR_TYPES[o.dtype] for o in self.outputs]
        return inputs, (spec, ranks, sizes, dtypes)

    def run(self):
        """Runs the program; returns its output tensors."""
        inputs, rest = self.call()
        return torch.ops.ktt.run_program(inputs, *rest)


def _tensors(args):
    """The tensors among `args` and inside their lists / tuples, in order."""
    for a in args:
        if isinstance(a, torch.Tensor):
            yield a
        elif isinstance(a, (list, tuple)):
            yield from (t for t in a if isinstance(t, torch.Tensor))


def _program_op(lib, ns, name, signature, prepare, single):
    """Defines torch op `ns::name(signature` on `lib` for tt tensors.
    `prepare(*args, **kwargs)` returns (input tensors, the other arguments of
    torch.ops.ktt.run_program); the native backend remembers the launch and runs
    later calls with equal values, tensor sizes, dtypes and numerics without it."""

    def first(*args, **kwargs):
        inputs, rest = prepare(*args, **kwargs)
        outs = torch.ops.ktt.run_program_first(inputs, *rest)
        return outs[0] if single else tuple(outs)

    lib.define(f"{name}({signature}")
    if hasattr(torch.ops.ktt, "program_op"):
        lib.define(f"_{name}_first({signature}")
        lib.impl(f"_{name}_first", first, "PrivateUse1")
        torch.ops.ktt.program_op(f"{ns}::{name}", f"{ns}::_{name}_first")
    else:  # no native backend: the op is defined, tt tensors do not exist
        lib.impl(name, first, "PrivateUse1")


def op(schema):
    """Registers the decorated program builder as torch op `schema` ("ns::name(...)
    -> ...") for tt tensors. The builder takes the op's arguments and returns a
    Program; the op returns its outputs (one Tensor, or a tuple). The description is
    reused across calls with equal argument sizes, dtypes and non-tensor values when
    every program input is an op argument.
    """
    qualified, _, signature = schema.partition("(")
    ns, _, name = qualified.strip().partition("::")
    returns_list = schema.rsplit("->", 1)[1].strip()
    single = returns_list == "Tensor"

    def register(builder):
        if ns not in _LIBRARIES:
            _LIBRARIES[ns] = torch.library.Library(ns, "FRAGMENT")
        lib = _LIBRARIES[ns]

        def prepare(*args, **kwargs):
            return builder(*args, **kwargs).call()

        def fake(*args, **kwargs):
            device = next(t.device for t in _tensors(args))
            outs = [
                torch.empty(o.sizes, dtype=o.dtype, device=device)
                for o in builder(*args, **kwargs).outputs
            ]
            return outs[0] if single else tuple(outs)

        _program_op(lib, ns, name, signature, prepare, single)
        lib._register_fake(name, fake)
        builder.op = getattr(getattr(torch.ops, ns), name)
        return builder

    return register
