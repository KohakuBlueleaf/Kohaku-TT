"""Crowded-L1 runs for the card tests (a script run with KTT_L1_BUDGET set; argv:
"eager" or "trace"): a program that fit the empty chip is launched again after live L1
tensors took the space its buffers need, then once they are freed. Prints one line
per outcome."""

import math
import os
import sys

import torch

import kohakutt
import kohakutt.dsl.metal as km
from kohakutt import Captured
from kohakutt.dsl.metal.bridge import l1_free

sys.path.insert(0, os.path.dirname(__file__))
from km_chain import run

CHIP = int(os.environ.get("KTT_TEST_CHIP", "0"))
SHAPE = (2, 64, 256)
DTYPES = dict.fromkeys(("x0", "x1", "out"), "float32")


@km.program("l1_chain")
def l1_chain(p, x0: km.TensorRef, x1: km.TensorRef, out: km.Out, extra: int = 0):
    """out = x0 + x1, with `extra` bytes of scratch beside its buffers."""
    start, count = p.split(p.runtime("tiles"))
    cbs = [p.cb(0, like=x0), p.cb(1, like=x1)]
    cb_out = p.cb(16, like=out)
    if extra:
        p.scratch(extra)
    k = p.kernel("dm0")
    with k.loop(count) as i:
        tile = k.let("tile", start + i)
        slots = [k.reserve(cb, 1) for cb in cbs]
        for x, slot in zip((x0, x1), slots):
            k.read(x.page(tile), slot)
        k.reads_done()
        for cb in cbs:
            k.push(cb, 1)
    c = p.kernel("compute", fp32_dest=True, unpack_fp32=cbs, approx=False)
    c.startup(cbs[0], cbs[1], cb_out)
    with c.loop(count):
        for cb in cbs:
            c.wait(cb, 1)
        c.acquire()
        c.init("copy", cbs[0])
        c.copy(cbs[0], 0, 0)
        c.init("copy", cbs[1])
        c.copy(cbs[1], 0, 1)
        c.init("add")
        c.binary("add", 0, 1, 0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        for cb in cbs:
            c.pop(cb, 1)
    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(start + i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()


def _tiles(t):
    return math.prod(t.shape[:-2]) * math.ceil(t.shape[-2] / 32) * (t.shape[-1] // 32)


@km.op("kohakutt_test::l1_chain(Tensor a, Tensor b, int extra) -> Tensor", l1_chain)
def l1_chain_op(a, b, extra):
    return km.Call(
        inputs={"x0": a, "x1": b},
        outputs={"out": (list(a.shape), a.dtype)},
        runtime={"tiles": _tiles(a)},
        static={"extra": extra},
    )


def gap():
    """Bytes from the L1 base to the lowest live L1 buffer (None: no L1 buffer)."""
    _, lowest, base = torch.ops.ktt._debug_trace_l1(-1)
    return None if lowest < 0 else lowest - base


def crowd(below, dev):
    """Live L1 tensors leaving less than `below` bytes under the lowest one: the
    activation budget is filled, every other tensor freed, and larger ones placed
    under the rest, so the L1 in use stays inside the budget."""
    cols, rows = torch.ops.ktt.compute_grid()
    held, tiles = [], 16
    for _ in range(200):
        now = gap()
        if now is not None and now < below:
            return held
        if now is not None and tiles * 2048 > now - 96 * 1024:
            tiles = max(1, (now - 96 * 1024) // 2048)
        x = torch.zeros(32 * tiles, 32 * cols * rows, dtype=torch.bfloat16).to(dev)
        y = x + 1
        del x
        if gap() != now:
            held.append(y)
        elif not held:
            raise RuntimeError(f"crowd: no tensor goes to L1 (gap {now})")
        else:  # past the activation budget: y went to DRAM
            held, tiles = held[1::2], min(2 * tiles, 128)
    raise RuntimeError(f"crowd: the gap stayed at {gap()}")


def attempt(label, fn, want=None):
    """Runs fn() and a synchronize; prints how it ended (and whether its result
    is close to `want`)."""
    try:
        out = fn()
        torch.ops.ktt.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"{label}: {type(e).__name__}: {e}".replace("\n", " "), flush=True)
        return None
    close = "" if want is None else f", equal {torch.allclose(out.cpu(), want)}"
    print(f"{label}: ran{close}", flush=True)
    return out


def main(mode):
    kohakutt.open(CHIP)
    dev = torch.device("tt")
    machine = km.Machine(*torch.ops.ktt.compute_grid())
    free = l1_free()
    extra = free * 3 // 4 - l1_chain.trace(machine, DTYPES).footprint()
    program = l1_chain.trace(machine, DTYPES, extra=extra)
    print(f"free {free} need {program.footprint()} gap {gap()}", flush=True)
    launcher = km.Launcher(program)
    g = torch.Generator().manual_seed(0)
    cpu = [torch.rand(*SHAPE, generator=g) for _ in range(2)]
    xs = [t.to(dev) for t in cpu]
    want = cpu[0] + cpu[1]
    op = torch.ops.kohakutt_test.l1_chain
    if mode == "trace":
        step = Captured(lambda a, b: run(launcher, [a, b]), xs, 2)
        attempt("captured", lambda: step(*xs), want)
        launch = {"replay": lambda: step(*xs)}
    else:
        attempt("first launch", lambda: run(launcher, xs), want)
        attempt("first op", lambda: op(xs[0], xs[1], extra), want)
        launch = {
            "repeat launch": lambda: run(launcher, xs),
            "op replay": lambda: op(xs[0], xs[1], extra),
        }
    try:
        held = crowd(program.footprint(), dev)
    except RuntimeError as e:
        print(e, flush=True)
        held = []
    print(f"crowded: {len(held)} tensors, gap {gap()}, l1_free {l1_free()}", flush=True)
    for label, fn in launch.items():
        attempt(label, fn)
    if mode == "eager":
        other = km.Launcher(l1_chain.trace(machine, DTYPES, extra=extra + 4096))
        attempt("new launch", lambda: run(other, xs))
    del held
    torch.ops.ktt.synchronize()
    print(f"freed: gap {gap()}", flush=True)
    for label, fn in launch.items():
        attempt(f"after {label}", fn, want)
    kohakutt.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eager")
