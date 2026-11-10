"""The launch check of page indices and declared ranges (PAGE-1) on the CPU: seeded
programs whose reads leave the tensor or their declared range are refused before
anything is queued, guarded reads pass, and what it cannot prove is warned once."""

import math
import warnings

import pytest
import torch
from km_programs import MACHINE

import kohakutt.dsl.metal as km
import kohakutt.dsl.metal.bridge as bridge
import kohakutt.dsl.metal.pages as pages


@pytest.fixture(autouse=True)
def no_device(monkeypatch):
    monkeypatch.setattr(bridge, "_compute_options", lambda *_: {"fp32_dest": True})
    monkeypatch.setattr(bridge, "l1_free", lambda: None)
    monkeypatch.delenv("KTT_KM_PAGE_CHECK", raising=False)


def copy_program(index, guard=None, nested=False):
    """out = x on one core over `n` tiles; tile i is read from x.page(index(i, n)),
    under `when(guard(i, n))` when given; `nested`: per i, also pages i + j for j
    below n - i."""
    p = km.Program("seed_pages", MACHINE)
    x, out = p.input("x", "bfloat16"), p.output("out", "bfloat16")
    n = p.runtime("n")
    p.layout(cores=1)
    cb_in, cb_out = p.cb(0, like=x), p.cb(16, like=out)
    k = p.kernel("dm0")
    with k.loop(n) as i:
        slot = k.reserve(cb_in, 1)
        if nested:
            with k.loop(n - i) as j:
                k.read(x.page(i + j), slot)
                k.reads_done()
        elif guard is None:
            k.read(x.page(index(i, n)), slot)
        else:
            with k.when(guard(i, n)):
                k.read(x.page(index(i, n)), slot)
        k.reads_done()
        k.push(cb_in, 1)
    c = p.kernel("compute", fp32_dest=True, approx=False)
    c.startup(cb_in, cb_in, cb_out)
    with c.loop(n):
        c.wait(cb_in, 1)
        c.acquire()
        c.init("copy", cb_in)
        c.copy(cb_in, 0, 0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_in, 1)
    w = p.kernel("dm1")
    with w.loop(n) as i:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(i))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()
    return p


def _prepare(launcher, tiles, n=None):
    x = torch.empty(32, 32 * tiles, dtype=torch.bfloat16)
    outputs = {"out": ((32, 32 * tiles), torch.bfloat16)}
    runtime = {"n": tiles if n is None else n}
    return launcher.prepare({"x": x}, outputs, runtime, numerics="n")


def test_a_page_past_the_tensor_is_refused_before_queueing():
    launcher = km.Launcher(copy_program(lambda i, n: i + 1))
    with pytest.raises(ValueError, match=r"PAGE-1: dm0: page index i \+ 1 of x "):
        _prepare(launcher, 8)
    with pytest.raises(ValueError, match="reaches 8 on core 0; x has 8 pages"):
        _prepare(launcher, 8)


def test_a_launch_value_past_the_tensor_is_refused():
    launcher = km.Launcher(copy_program(lambda i, n: i))
    _prepare(launcher, 8)
    with pytest.raises(ValueError, match="page index i of x reaches 8 .* 8 pages"):
        _prepare(launcher, 8, n=9)


def test_a_value_outside_its_declared_range_is_refused():
    launcher = km.Launcher(copy_program(lambda i, n: i.bounded(0, 4)))
    _prepare(launcher, 4)
    with pytest.raises(ValueError, match=r"i declared in 0 \.\. 4 - 1 reaches 4"):
        _prepare(launcher, 8)


def test_a_guarded_read_is_proven_by_its_condition():
    launcher = km.Launcher(copy_program(lambda i, n: i + 1, lambda i, n: i + 1 < n))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _prepare(launcher, 8)


def test_what_it_cannot_prove_is_warned_once_and_refused_when_strict(monkeypatch):
    launcher = km.Launcher(copy_program(None, nested=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _prepare(launcher, 8)
    monkeypatch.setattr(pages, "ENUMERATE", 1)
    launcher = km.Launcher(copy_program(None, nested=True))
    with pytest.warns(pages.PageWarning, match=r"i \+ i2 of x: its bound reaches 14"):
        _prepare(launcher, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _prepare(launcher, 9)
    monkeypatch.setenv("KTT_KM_PAGE_CHECK", "strict")
    with pytest.raises(ValueError, match="KTT_KM_PAGE_CHECK=strict"):
        _prepare(launcher, 10)
    monkeypatch.setenv("KTT_KM_PAGE_CHECK", "0")
    _prepare(launcher, 11)


def test_a_repeat_launch_reuses_the_check(monkeypatch):
    launcher = km.Launcher(copy_program(lambda i, n: i))
    calls = []
    check = launcher._pages.check
    monkeypatch.setattr(
        launcher._pages, "check", lambda *a: calls.append(1) or check(*a)
    )
    for tiles in (8, 8, 8, 16, 8):
        _prepare(launcher, tiles)
    assert len(calls) == 2


def _rows_launch(launcher, shape, dynamic):
    from km_rows import runtime

    x = torch.empty(*shape)
    values = runtime(shape, MACHINE.cols * MACHINE.rows, dynamic)
    outputs = {"out": (list(shape), torch.float32)}
    return launcher.prepare({"x": x}, outputs, values, numerics="n")


def test_rows_from_control_words_are_bounded_by_the_loop_exit():
    from km_rows import rows_program

    launcher = km.Launcher(rows_program(MACHINE))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for dynamic in (0, 1):
            _rows_launch(launcher, (2, 1000, 64), dynamic)


def test_a_late_loop_exit_leaves_the_page_unproven():
    from km_rows import rows_program
    from kohakutt.dsl.metal.binding import Op
    from kohakutt.dsl.metal.body import ExitWhen, Loop

    program = rows_program(MACHINE)
    (forever,) = [s for s in program.kernels["dm0"].body if isinstance(s, Loop)]
    (stop,) = [s for s in forever.body if isinstance(s, ExitWhen)]
    stop.condition = Op(">", stop.condition.a, stop.condition.b)
    launcher = km.Launcher(program, check=False)
    with pytest.warns(km.PageWarning, match=r"page index \(r \* wt\) \+ j of x"):
        _rows_launch(launcher, (2, 1000, 64), 1)


def test_the_programs_without_a_card_test_launch_without_a_warning():
    """The lane, table and scaled-add programs (no card test launches them)."""
    from km_programs import example

    from km_lanes import lane_program
    from km_tables import table_program

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lanes = km.Launcher(lane_program())
        x = torch.empty(32, 32 * 6)
        out = {"out": ((32, 32 * 6 * 110), torch.float32)}
        lanes.prepare({"x": x}, out, {"n": 6}, numerics="n")
        with pytest.raises(ValueError, match="page index i of x reaches 6"):
            lanes.prepare({"x": x}, out, {"n": 7}, numerics="n")
        tables = km.Launcher(table_program())
        x, out = torch.empty(64, 64), {"out": ((64, 64), torch.float32)}
        jobs = {"jobs": [[[0, 2, 2]], [[2, 0, 2]]]}
        tables.prepare({"x": x}, out, {"n": 0}, numerics="n", tables=jobs)
        added = example("scaled_add_km")
        program = added.scaled_add_program.trace(
            MACHINE, dict.fromkeys(("a", "b", "out"), "bfloat16")
        )
        a = torch.empty(4, 160, 256, dtype=torch.bfloat16)
        args = ({"a": a, "b": a}, {"out": ([4, 160, 256], torch.bfloat16)})
        km.Launcher(program).prepare(*args, {"tiles": 160, "s": 0}, numerics="n")


def test_the_shipped_matmul_example_proves_its_pages_and_refuses_wrong_values():
    from km_programs import example

    gemm = example("gemm_km")
    launcher = km.Launcher(gemm.traced(MACHINE.cols, MACHINE.rows))
    for m, k, n in [(4096, 1024, 768), (16384, 1024, 3072), (1000, 256, 333)]:
        mt, kt, nt = math.ceil(m / 32), k // 32, math.ceil(n / 32)
        values = {**gemm.plan(mt, kt, nt, MACHINE.cols, MACHINE.rows), "w_nk": 0}
        a = torch.empty(m, k, dtype=torch.bfloat16)
        w = torch.empty(k, n, dtype=torch.bfloat16)
        args = ({"a": a, "w": w}, {"out": ([m, n], torch.bfloat16)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            launcher.prepare(*args, values, numerics="n")
        for wrong in ("mt", "nt", "kt"):
            bad = {**values, wrong: values[wrong] + 1}
            with pytest.raises(ValueError, match="PAGE-1: .* has .* pages"):
                launcher.prepare(*args, bad, numerics="n")
