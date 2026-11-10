"""Mesh: KTT_FABRIC=0 (a mesh opened without the fabric, so without collectives):
ops that need no collective compute right, and ops that would need one raise a clear
Python error instead of hanging or giving wrong values. Each scenario runs in a
subprocess with KTT_FABRIC=0, followed by a subprocess that reopens the mesh."""

import pytest
from mesh_subproc import scenario

pytestmark = [pytest.mark.hw, pytest.mark.dual]
ENV = {"KTT_FABRIC": "0"}


def check(res):
    code, out, err = res
    assert code != "timeout", f"hang: no exit within 60 s\n{err[-1500:]}"
    assert code == 0, f"process exit {code}\n{err[-1500:]}"
    assert "SCENARIO OK" in out, out[-500:]


def test_replicated_ops_without_fabric(native_module):
    """Elementwise ops and matmul on replicated tensors compute right."""
    body = """
dev = k.mesh([0, 1], shape=(1, 2))
try:
    x = k.replicate(ints((64, 64), 1).to(dev))
    w = k.replicate(ints((64, 64), 2).to(dev))
    assert torch.equal((x * 2 + 1).cpu().float(), ints((64, 64), 1).float() * 2 + 1)
    assert torch.equal((x @ w).cpu().float(), WANT)
finally:
    k.close()
print("SCENARIO OK")
"""
    check(scenario(body, ENV))


def test_shard_without_fabric(native_module):
    """shard() of a replicated tensor reads back whole, or raises a clear error."""
    body = """
dev = k.mesh([0, 1], shape=(1, 2))
x = ints((64, 64), 3)
try:
    try:
        got = k.shard(x.to(dev), 0).cpu().float()
    except Exception as exc:
        assert "TT_FATAL" not in str(exc), str(exc)[:200]
        print("raised:", str(exc)[:120])
    else:
        assert torch.equal(got, x.float())
finally:
    k.close()
print("SCENARIO OK")
"""
    check(scenario(body, ENV))


def test_collective_without_fabric(native_module):
    """A sum over a sharded dim needs a collective: a clear error or the right value."""
    body = """
dev = k.mesh([0, 1], shape=(1, 2))
x = ints((64, 64), 4)
try:
    try:
        got = k.shard(x.to(dev), 0).sum(0).cpu().float()
    except Exception as exc:
        assert "TT_FATAL" not in str(exc), str(exc)[:200]
        print("raised:", str(exc)[:120])
    else:
        assert torch.equal(got, x.float().sum(0))
finally:
    k.close()
print("SCENARIO OK")
"""
    check(scenario(body, ENV))
