"""Runs metal-level programs as torch ops on tt tensors: the program description for
`torch.ops.ktt.run_program` (free of data sizes) and the `op` registration.
"""

import json

import torch

from kohakutt.core.numerics import Requirements, current_numerics, resolve
from kohakutt.dsl.metal.emit import emit
from kohakutt.dsl.metal.machine import Machine, ranges
from kohakutt.dsl.metal.verify import cb_ops
from kohakutt.kernels import _program_op

# torch dtype -> (storage format name, ScalarType code).
DTYPES = {
    torch.bfloat16: ("bfloat16", 15),
    torch.float32: ("float32", 6),
    torch.int32: ("int32", 3),
}
KINDS = {"dm0": "reader", "compute": "compute", "dm1": "writer"}
_LIBRARIES = {}


class Call:
    """One launch of a program: `inputs` {name: tt tensor}, `outputs` {name: (sizes,
    torch dtype)}, `runtime` {name: int} and `static` {name: value}."""

    def __init__(self, inputs, outputs, runtime=None, static=None):
        self.inputs, self.outputs = dict(inputs), dict(outputs)
        self.runtime, self.static = dict(runtime or {}), dict(static or {})


def rings(program):
    """Per pushed buffer: its index, the page counts of its recurring calls and the
    most bytes a kernel writes from a reserved pointer."""
    out = {}
    for kernel in program.kernels.values():
        for kind, cb, pages, path, times in cb_ops(kernel):
            ring = out.setdefault(
                cb.index, {"cb": cb.index, "pushes": [], "written": 0}
            )
            recurring = bool(path) or not (isinstance(times, int) and times == 1)
            counts = getattr(pages, "choices", None) or (int(pages),)
            for n in counts:
                if recurring and n not in ring["pushes"]:
                    ring["pushes"].append(n)
            if kind == "reserve":
                ring["written"] = max(ring["written"], max(counts) * cb.page_bytes)
    return [out[index] for index in sorted(out)]


def _compute_options(options, numerics):
    """The compute entries of the description for kernel `options` under `numerics`."""
    bounds = Requirements(
        options["min_fidelity"], options["max_fidelity"], options["exact"]
    )
    active = resolve(numerics, bounds)
    approx = active.approx if options["approx"] is None else options["approx"]
    return {
        "fp32_dest": options["fp32_dest"],
        "unpack_fp32": [cb.index for cb in options["unpack_fp32"]],
        "fidelity": options["fidelity"] or active.fidelity,
        "approx": approx,
    }


def _per_core(layout, index):
    """The per-core argument values of core `index` for a kernel's `layout`."""
    out = [index] if layout.core_index else []
    for arg in layout.per_core:
        value = arg.values(index)
        out.append(int(value[arg.position] if arg.count != 1 else value))
    return out


def spec(program, sources, count, numerics):
    """The JSON description of `program` on its first `count` cores with kernel
    sources `sources` ({role: path}); it holds no data size."""
    cores = program.machine.core_list(count)
    every = ranges(cores)
    tensor = {id(t): i for i, t in enumerate(program.tensors)}
    out = {"interleaved": True, "cbs": [], "semaphores": [], "kernels": []}
    for cb in program.cbs:
        entry = {"index": cb.index, "tiles": int(cb.pages), "dtype": cb.dtype}
        out["cbs"].append({**entry, "cores": every})
    for s in program.scratches:
        out["cbs"].append({"index": s.index, "bytes": int(s.bytes), "cores": every})
    for sem in program.semaphores:
        entry = {"id": int(sem.id), "initial": int(sem.initial)}
        out["semaphores"].append({**entry, "cores": ranges(sem.cores.select(cores))})
    for role, kind in KINDS.items():
        kernel = program.kernels.get(role)
        if kernel is None:
            continue
        layout = kernel.layout()
        own = set(kernel.cores.select(cores))
        if not own:
            raise ValueError(f"kohakutt.dsl.metal: {role} has no core at {count} cores")
        common = [{"addr": tensor[id(t)]} for t in layout.tensors]
        common += [{"value": program.values.index(name)} for name in layout.values]
        common += [count] if layout.core_count else []
        entry = {
            "source": str(sources[role]),
            "type": kind,
            "cores": ranges(own),
            "compile_args": [{"accessor": tensor[id(t)]} for t in layout.tensors],
            "common_args": common,
            "runtime_args": [
                [list(core), _per_core(layout, i)]
                for i, core in enumerate(cores)
                if core in own and layout.per_core_count
            ],
            "defines": {name: str(v) for name, v in kernel.defines.items()},
            "args": {"per_core": layout.per_core_count, "common": layout.common},
        }
        if kernel.noc is not None:
            entry["noc"] = int(kernel.noc)
        if role == "compute":
            entry.update(_compute_options(kernel.options, numerics))
        out["kernels"].append(entry)
    out["rings"] = rings(program)
    return json.dumps(out)


def _format(dtype):
    if dtype not in DTYPES:
        raise TypeError(f"kohakutt.dsl.metal: unsupported tensor dtype {dtype}")
    return DTYPES[dtype]


def op(schema, template):
    """Registers the decorated builder as torch op `schema` ("ns::name(...) -> ...")
    for tt tensors. The builder takes the op's arguments and returns the Call of
    `template`: it runs once per argument sizes, dtypes and non-tensor values, and
    sources and descriptions are built once per kernel class."""
    qualified, _, signature = schema.partition("(")
    ns, _, name = qualified.strip().partition("::")
    single = schema.rsplit("->", 1)[1].strip() == "Tensor"

    def register(builder):
        if ns not in _LIBRARIES:
            _LIBRARIES[ns] = torch.library.Library(ns, "FRAGMENT")
        lib = _LIBRARIES[ns]
        classes = {}

        def traced(call, numerics):
            """(program, sources, {core count: description}) of the call's class."""
            dtypes = {n: _format(t.dtype)[0] for n, t in call.inputs.items()}
            dtypes.update({n: _format(o[1])[0] for n, o in call.outputs.items()})
            key = (tuple(dtypes.items()), tuple(call.static.items()), numerics)
            entry = classes.get(key)
            if entry is None:
                cols, rows = torch.ops.ktt.compute_grid()
                program = template.trace(Machine(cols, rows), dtypes, **call.static)
                entry = classes[key] = (program, emit(program), {})
            return entry

        def launch(call, numerics):
            """(input tensors, the other arguments of torch.ops.ktt.run_program)."""
            program, sources, specs = traced(call, numerics)
            count = program.core_count(call.runtime)
            text = specs.get(count)
            if text is None:
                text = specs[count] = spec(program, sources, count, numerics)
            outputs = [call.outputs[t.name] for t in program.outputs]
            rest = (
                text,
                [len(sizes) for sizes, _ in outputs],
                [s for sizes, _ in outputs for s in sizes],
                [_format(dtype)[1] for _, dtype in outputs],
                program.value_list(call.runtime, count),
            )
            return [call.inputs[t.name] for t in program.inputs], rest

        def prepare(*args, **kwargs):
            return launch(builder(*args, **kwargs), current_numerics())

        def fake(*args, **kwargs):
            call = builder(*args, **kwargs)
            device = next(iter(call.inputs.values())).device
            outs = [
                torch.empty(list(sizes), dtype=dtype, device=device)
                for sizes, dtype in call.outputs.values()
            ]
            return outs[0] if single else tuple(outs)

        _program_op(lib, ns, name, signature, prepare, single)
        lib._register_fake(name, fake)
        builder.op = getattr(getattr(torch.ops, ns), name)
        builder.template = template
        return builder

    return register
