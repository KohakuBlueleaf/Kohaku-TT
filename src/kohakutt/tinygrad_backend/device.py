"""The KTT tinygrad device: allocator, renderer stub, program and Compiled device.
Buffers are flat bytes; a backend (`NumpyBackend`, or `executor.BridgeBackend` for
device tensors) stores them and runs kernels. The seam plans each AST once.
"""

from __future__ import annotations

import collections

import numpy as np

from kohakutt.tinygrad_backend import compat as C
from kohakutt.tinygrad_backend.compat import Allocator, Compiled, Program, Renderer

PLANS: dict[bytes, object] = {}
STATS = collections.Counter()
KERNELS = collections.Counter()
CALLS = collections.Counter()
SEAM_CALLS = collections.Counter()

_backend = None


def reset_stats():
    for c in (STATS, KERNELS, CALLS, SEAM_CALLS):
        c.clear()


class NumpyBackend:
    """Host arrays and the numpy reference evaluator: no card."""

    name = "numpy"

    def make_plan(self, ast):
        from kohakutt.tinygrad_backend.numpy_exec import Plan

        return Plan(ast)

    def alloc(self, size: int):
        return np.zeros(size, dtype=np.uint8)

    def free(self, storage):
        pass

    def copyin(self, dest, src: memoryview):
        dest[:] = np.frombuffer(src, dtype=np.uint8)

    def copyout(self, dest: memoryview, src):
        np.frombuffer(dest, dtype=np.uint8)[:] = src

    def offset(self, buf, size: int, offset: int):
        return buf[offset : offset + size]

    def host_bytes(self, arr):
        return arr

    def touch(self, arr):
        pass

    def run(self, plan, bufs: dict, var_vals: dict):
        from kohakutt.tinygrad_backend.numpy_exec import Evaluator

        if hasattr(plan, "run_host"):
            return plan.run_host(self, bufs, var_vals)

        typed = {
            p.slot: bufs[p.slot].view(C.np_dtype(p.dtype))
            for p in plan.params
            if not p.is_var
        }
        Evaluator(plan, typed, var_vals).run()

    def synchronize(self):
        pass


def get_backend():
    global _backend
    if _backend is None:
        _backend = NumpyBackend()
    return _backend


def set_backend(backend):
    global _backend
    _backend = backend


class KTTAllocator(Allocator):
    """tinygrad buffers of the KTT device; storage is whatever the backend allocates."""

    def __init__(self, dev):
        super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)

    def _alloc(self, size, options):
        return get_backend().alloc(size)

    def _free(self, opaque, options):
        get_backend().free(opaque)

    def _copyin(self, dest, src: memoryview):
        get_backend().copyin(dest, src)

    def _copyout(self, dest: memoryview, src):
        get_backend().copyout(dest, src)

    def _offset(self, buf, size: int, offset: int):
        return get_backend().offset(buf, size, offset)


class KTTRenderer(Renderer):
    """Placeholder renderer: kernels are taken at the seam, nothing is rendered."""

    has_local = has_shared = supports_float4 = False
    global_max = local_max = (1, 1, 1)

    def render(self, uops):
        raise RuntimeError(
            "the KTT device takes kernel ASTs at the seam and renders no code"
        )


class KTTProgram(Program):
    """One planned kernel: a launch hands the plan its buffers and variable values."""

    def __init__(self, dev, obj):
        self.plan = PLANS[bytes(obj.lib)]
        self.params = self.plan.params
        self.var_names = [p.name for p in self.params if p.is_var]

    def __call__(
        self,
        *bufs,
        global_size=(1, 1, 1),
        local_size=(1, 1, 1),
        vals=(),
        wait=False,
        timeout=None,
    ):
        buf_by_slot, var_vals = C.launch_args(bufs, vals, self.params, self.var_names)
        where = get_backend().run(self.plan, buf_by_slot, var_vals)
        STATS[where or "host"] += 1
        KERNELS[self.plan.kind] += 1
        for c in self.plan.calls:
            CALLS[c] += 1
        return 0.0 if wait else None


class KTTDevice(Compiled):
    def __init__(self, device: str):
        super().__init__(device, KTTAllocator(self), [KTTRenderer], KTTProgram)

    def synchronize(self, timeout=None):
        get_backend().synchronize()


def seam(ast, renderer):
    """to_program for KTT: plan the kernel AST (cached by key), return its PROGRAM."""
    if ast.op is C.Ops.PROGRAM and C.is_lib_program(ast):
        from kohakutt.tinygrad_backend import overrides

        SEAM_CALLS["to_program"] += 1
        PLANS.setdefault(ast.key, overrides.make_lib_plan(ast))
        return C.lib_program_uop(ast, renderer)
    SEAM_CALLS["to_program"] += 1
    if ast.key not in PLANS:
        PLANS[ast.key] = get_backend().make_plan(ast)
    plan = PLANS[ast.key]
    return C.program_uop(ast, renderer, f"ktt {plan.kind} {plan.calls}")
