"""The metal level (`km`): a typed model of a TT-Metalium program (cores, buffers,
semaphores, dm0 / dm1 / compute kernels, argument binding times), a verifier that
runs on the model, and a C++ emitter. `km.op` runs a program as a torch op.
"""

from kohakutt.dsl.metal.binding import Static, VerifyError, auto, select
from kohakutt.dsl.metal.binding import minimum as min
from kohakutt.dsl.metal.body import CBUse, CppFn, SfpuFn
from kohakutt.dsl.metal.dump import dump, fingerprint
from kohakutt.dsl.metal.emit import cache_dir, emit, source
from kohakutt.dsl.metal.machine import SIZE_CLASSES, CoreSet, Machine
from kohakutt.dsl.metal.model import Out, Program, Template, TensorRef, program
from kohakutt.dsl.metal.verify import RULES, verify

__all__ = [
    "RULES",
    "SIZE_CLASSES",
    "CBUse",
    "Call",
    "CppFn",
    "CoreSet",
    "Launcher",
    "Machine",
    "Out",
    "Program",
    "SfpuFn",
    "Static",
    "Template",
    "TensorRef",
    "VerifyError",
    "auto",
    "cache_dir",
    "dump",
    "emit",
    "fingerprint",
    "min",
    "op",
    "program",
    "select",
    "source",
    "verify",
]


def __getattr__(name):
    """`Call`, `Launcher` and `op` load the torch bridge on first use."""
    if name in ("Call", "Launcher", "op"):
        from kohakutt.dsl.metal import bridge

        return getattr(bridge, name)
    raise AttributeError(name)
