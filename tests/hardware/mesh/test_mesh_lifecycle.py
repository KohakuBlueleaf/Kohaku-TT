"""Mesh: lifecycle in one process: open, close and open again; an exception in the
middle of a computation followed by more work; close with tensors alive; a tensor made
before a close used after reopening. Each scenario runs in a subprocess (a crash or a
hang is a recorded result), followed by a second subprocess that reopens the mesh."""

import pytest
from mesh_subproc import scenario

pytestmark = [pytest.mark.hw, pytest.mark.dual]


def check(res, marker="SCENARIO OK"):
    code, out, err = res
    assert code != "timeout", f"hang: no exit within 60 s\n{err[-1500:]}"
    assert code == 0, f"process exit {code}\n{err[-1500:]}"
    assert marker in out, out[-500:]


def test_open_close_open(native_module):
    """The mesh opens, works, closes, opens again and works with the same results."""
    body = """
dev = k.mesh([0, 1], shape=(1, 2))
try:
    first = work(dev)
finally:
    k.close()
dev = k.mesh([0, 1], shape=(1, 2))
try:
    second = work(dev)
finally:
    k.close()
assert torch.equal(first, WANT) and torch.equal(second, first)
print("SCENARIO OK")
"""
    check(scenario(body))


def test_exception_mid_computation_then_more_work(native_module):
    """A failing op (shapes that do not match) raises a Python error and the mesh keeps
    computing right afterwards."""
    body = """
dev = k.mesh([0, 1], shape=(1, 2))
try:
    a = k.shard(ints((64, 64), 1).to(dev), 0)
    bad = ints((48, 64), 3).to(dev)
    for fn in (lambda: (a @ bad).cpu(), lambda: torch.cat([a, bad], 1).cpu()):
        try:
            fn()
        except Exception as exc:
            print("raised:", type(exc).__name__)
        else:
            raise SystemExit("no error for mismatched shapes")
    after = work(dev)
finally:
    k.close()
assert torch.equal(after, WANT)
print("SCENARIO OK")
"""
    check(scenario(body))


def test_close_with_tensors_and_trace_alive(native_module):
    """Closing with placed tensors, a partial sum and a captured trace alive returns
    cleanly and the mesh opens again."""
    body = """
dev = k.mesh([0, 1], shape=(1, 2))
try:
    a = k.shard(ints((64, 64), 1).to(dev), 0)
    b = k.shard(ints((64, 64), 2).to(dev), 1)
    p = a.t() @ b
    r = k.replicate(a)
    step = k.capture(lambda t: t * 2, r)
    step(r)
finally:
    k.close()
dev = k.mesh([0, 1], shape=(1, 2))
try:
    again = work(dev)
finally:
    k.close()
assert torch.equal(again, WANT)
print("SCENARIO OK")
"""
    check(scenario(body))


@pytest.mark.xfail(
    strict=True,
    reason="a tensor used after the mesh was reopened raises a raw device check",
)
def test_stale_tensor_after_reopen(native_module):
    """A tensor made before a close raises a Python error when used after reopening
    (never wrong values, never a crash)."""
    body = """
dev = k.mesh([0, 1], shape=(1, 2))
try:
    old = k.shard(ints((64, 64), 1).to(dev), 0)
finally:
    k.close()
dev = k.mesh([0, 1], shape=(1, 2))
try:
    try:
        got = (old + 1).cpu().float()
    except Exception as exc:
        assert "TT_FATAL" not in str(exc), str(exc)[:200]
        print("raised:", type(exc).__name__, str(exc)[:120])
    else:
        assert torch.equal(got, ints((64, 64), 1).float() + 1), "wrong values"
    assert torch.equal(work(dev), WANT)
finally:
    k.close()
print("SCENARIO OK")
"""
    check(scenario(body))
