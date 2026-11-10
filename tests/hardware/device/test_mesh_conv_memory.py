"""Regression: conversion memory on a 1x2 mesh. A placed tensor converted for one
consumer is kept for the next ones, and every result equals, bit for bit, the result
of converting again (KTT_MESH_CONV_CACHE=0): for several consumers, after a write,
across collectives and across dtypes."""

import pytest
import torch
from torch import nn

ROWS, DIM = 128, 64
KINDS = (
    "all_gathers",
    "all_reduces",
    "reduce_scatters",
    "partitions",
    "all_to_alls",
    "host_conversions",
)


@pytest.fixture(scope="module")
def mesh(native_module):
    """The 1x2 mesh, open for the module."""
    device = native_module.mesh([0, 1], shape=(1, 2))
    try:
        yield device
    finally:
        native_module.close()


@pytest.fixture(autouse=True)
def knobs(monkeypatch):
    """No size floor (the tensors are tiny), memory knobs at their defaults."""
    monkeypatch.setenv("KTT_MESH_SHARD_MIN", "1")
    monkeypatch.setenv("KTT_FALLBACK_WARN", "0")
    monkeypatch.delenv("KTT_MESH_CONV_CACHE", raising=False)
    monkeypatch.delenv("KTT_MESH_CONV_FREE", raising=False)


def _pair(kohakutt, mesh, dtype):
    """A function returning a new [ROWS, DIM] partial sum (a column -> row Linear
    pair on fixed inputs) each call."""
    torch.manual_seed(0)
    up = kohakutt.colwise(nn.Linear(DIM, 4 * DIM, bias=False).to(dtype)).to(mesh)
    down = kohakutt.rowwise(nn.Linear(4 * DIM, DIM, bias=False).to(dtype)).to(mesh)
    x = kohakutt.replicate(torch.randn(ROWS, DIM).to(dtype).to(mesh))
    return lambda: down(torch.relu(up(x)))


def _host_sum(partial):
    """The fp32 sum of the two chips' partials."""
    parts = [torch.ops.ktt.local_shard(partial, c).float() for c in range(2)]
    return (parts[0] + parts[1])[:ROWS, :DIM]


def _counted(run):
    """run() and the conversions it made, by kind."""
    torch.tt.synchronize()
    before = torch.tt.mesh_stats()
    out = run()
    torch.tt.synchronize()
    after = torch.tt.mesh_stats()
    return out, {k: after[k] - before[k] for k in KINDS if after[k] != before[k]}


def _kept_and_fresh(monkeypatch, run):
    """(result, conversions) of run() with the memory on, then with every conversion
    run again."""
    kept = _counted(run)
    monkeypatch.setenv("KTT_MESH_CONV_CACHE", "0")
    fresh = _counted(run)
    monkeypatch.delenv("KTT_MESH_CONV_CACHE")
    return kept, fresh


def _assert_same_bits(kept, fresh):
    assert kept.keys() == fresh.keys()
    for name in kept:
        if isinstance(kept[name], torch.Tensor):
            assert torch.equal(kept[name], fresh[name]), name
        else:
            assert kept[name] == fresh[name], name


def _tol(dtype):
    return (
        {"rtol": 3e-2, "atol": 3e-2}
        if dtype == torch.bfloat16
        else {"rtol": 1e-3, "atol": 1e-3}
    )


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_consumers_convert_once(native_module, mesh, monkeypatch, dtype):
    kohakutt = native_module
    make = _pair(kohakutt, mesh, dtype)
    scale_host = torch.rand(ROWS, DIM).add(0.5).to(dtype)
    scale = kohakutt.shard(scale_host.to(mesh), 0)

    def run():
        with torch.no_grad():
            p = make()
            host = _host_sum(p)
            (a, b, c), counts = _counted(lambda: (p.pow(2), torch.tanh(p), p * scale))
            return {
                "host": host,
                "counts": counts,
                "placed": [kohakutt.placement(t) for t in (p, a, b, c)],
                "pow": a.cpu(),
                "tanh": b.cpu(),
                "mul": c.cpu(),
            }

    (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    # One reduce-scatter serves the three consumers; without the copies each converts.
    assert kept.pop("counts") == {"reduce_scatters": 1}
    assert fresh.pop("counts") == {"reduce_scatters": 3}
    assert kept["placed"] == ["Partial(sum)"] + ["Sharded(dim=0)"] * 3
    _assert_same_bits(kept, fresh)
    host = kept["host"]
    tol = _tol(dtype)
    torch.testing.assert_close(kept["pow"].float(), host.pow(2), **tol)
    torch.testing.assert_close(kept["tanh"].float(), host.tanh(), **tol)
    torch.testing.assert_close(kept["mul"].float(), host * scale_host.float(), **tol)


@pytest.mark.hw
@pytest.mark.dual
def test_linear_ops_keep_passing_the_partial(native_module, mesh, monkeypatch):
    kohakutt = native_module
    make = _pair(kohakutt, mesh, torch.bfloat16)

    def run():
        with torch.no_grad():
            p = make()
            host = _host_sum(p)
            a = p.pow(2)
            # A copy of the sum exists; negation and scaling do not need it.
            (negated, scaled), counts = _counted(lambda: (-p, p * 2.0))
            return {
                "host": host,
                "counts": counts,
                "placed": [kohakutt.placement(t) for t in (a, negated, scaled)],
                "negated": negated.cpu(),
                "scaled": scaled.cpu(),
            }

    (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    assert kept["counts"] == {}
    assert kept["placed"] == ["Sharded(dim=0)", "Partial(sum)", "Partial(sum)"]
    _assert_same_bits(kept, fresh)
    host = kept["host"]
    torch.testing.assert_close(kept["negated"].float(), -host, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(kept["scaled"].float(), 2.0 * host, rtol=1e-2, atol=1e-2)


def _rms_norm(x):
    """An RMS norm written out in float32, as model code does."""
    h = x.float()
    h = h * torch.pow(h.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
    return h.type_as(x)


@pytest.mark.hw
@pytest.mark.dual
def test_norm_after_pair_completes_once(native_module, mesh, monkeypatch):
    kohakutt = native_module
    make = _pair(kohakutt, mesh, torch.bfloat16)

    def run():
        with torch.no_grad():
            p = make()
            host = _host_sum(p)
            out, counts = _counted(lambda: _rms_norm(p))
            return {
                "host": host,
                "counts": counts,
                "placed": kohakutt.placement(out),
                "out": out.cpu(),
            }

    (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    monkeypatch.setenv("KTT_MESH_CONV_FREE", "0")
    before = run()
    # The product takes the copy the square made. Priced at full cost, it passes the
    # partial through, gathers the other operand, and the cast back completes again.
    assert kept["counts"] == {"reduce_scatters": 1}
    assert fresh["counts"] == {"reduce_scatters": 2}
    assert before["counts"] == {"reduce_scatters": 2, "all_gathers": 1}
    assert kept["placed"] == before["placed"] == "Sharded(dim=0)"
    assert torch.equal(kept["out"], fresh["out"])
    want = _rms_norm(kept["host"])
    for result in (kept, before):
        torch.testing.assert_close(result["out"].float(), want, rtol=2e-2, atol=2e-2)


@pytest.mark.hw
@pytest.mark.dual
def test_kept_copy_not_priced_free(native_module, mesh, monkeypatch):
    kohakutt = native_module
    monkeypatch.setenv("KTT_MESH_CONV_FREE", "0")
    make = _pair(kohakutt, mesh, torch.bfloat16)
    scale = kohakutt.shard(torch.rand(ROWS, DIM).add(0.5).bfloat16().to(mesh), 0)
    with torch.no_grad():
        p = make()
        (a, b, c), counts = _counted(lambda: (p.pow(2), torch.tanh(p), p * scale))
        placed = [kohakutt.placement(t) for t in (a, b, c)]
    # The choices from before the memory: the product passes the partial through and
    # gathers the other operand; pow and tanh still share one copy.
    assert placed == ["Sharded(dim=0)", "Sharded(dim=0)", "Partial(sum)"]
    assert counts == {"reduce_scatters": 1, "all_gathers": 1}


@pytest.mark.hw
@pytest.mark.dual
def test_write_between_consumers(native_module, mesh, monkeypatch):
    kohakutt = native_module
    gen = torch.Generator().manual_seed(2)
    y_host, q_host, q2_host = (
        torch.randn(ROWS, DIM, generator=gen, dtype=torch.bfloat16) for _ in range(3)
    )
    y = kohakutt.shard(y_host.to(mesh), 0)
    want = [y_host + q_host, y_host + 2.0 * q_host, y_host + q2_host]

    def run():
        with torch.no_grad():
            # y + q converts q (sharded on dim 1) to y's shards and keeps the copy.
            q = kohakutt.shard(q_host.to(mesh), 1)
            first, counts = _counted(lambda: (y + q).cpu())
            q.mul_(2.0)
            second = (y + q).cpu()
            q.copy_(kohakutt.shard(q2_host.to(mesh), 1))
            third = (y + q).cpu()
            return {"counts": counts, "first": first, "second": second, "third": third}

    def check(kept, fresh):
        assert kept["counts"] == {"all_to_alls": 1}
        _assert_same_bits(kept, fresh)
        for name, value in zip(("first", "second", "third"), want):
            torch.testing.assert_close(
                kept[name].float(), value.float(), rtol=2e-2, atol=2e-2, msg=name
            )

    (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    check(kept, fresh)
    # While a trace is alive an in-place write copies into the tensor's own buffer.
    small = kohakutt.replicate(torch.zeros(32, 32, dtype=torch.bfloat16).to(mesh))
    with torch.no_grad():
        step = kohakutt.capture(lambda t: t + 1.0, small)
    try:
        (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    finally:
        step.release()
    check(kept, fresh)


@pytest.mark.hw
@pytest.mark.dual
def test_replay_overwrites_converted_tensor(native_module, mesh, monkeypatch):
    kohakutt = native_module
    gen = torch.Generator().manual_seed(3)
    y_host, a_host, b_host = (
        torch.randn(ROWS, DIM, generator=gen, dtype=torch.bfloat16) for _ in range(3)
    )
    y = kohakutt.shard(y_host.to(mesh), 0)
    a = kohakutt.replicate(a_host.to(mesh))
    b = kohakutt.replicate(b_host.to(mesh))

    def run():
        with torch.no_grad():
            # The trace's own output (sharded on dim 1): each replay overwrites it.
            step = kohakutt.capture(
                lambda t: kohakutt.shard(t * 2.0, 1), a, copy_outputs=False
            )
            try:
                out = step(a)
                first = (y + out).cpu()
                out = step(b)
                second = (y + out).cpu()
            finally:
                step.release()
            return {"first": first, "second": second}

    (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    _assert_same_bits(kept, fresh)
    for name, x_host in (("first", a_host), ("second", b_host)):
        torch.testing.assert_close(
            kept[name].float(),
            (y_host + 2.0 * x_host).float(),
            rtol=2e-2,
            atol=2e-2,
            msg=name,
        )


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_each_copy_comes_from_the_tensor(native_module, mesh, monkeypatch, dtype):
    kohakutt = native_module
    make = _pair(kohakutt, mesh, dtype)

    def run():
        with torch.no_grad():
            p = make()
            # pow completes to shards (reduce-scatter); cumsum has no placed form and
            # completes to replicated (all-reduce); tanh reuses the shards.
            (a, g, b), counts = _counted(
                lambda: (p.pow(2), torch.cumsum(p, 0), torch.tanh(p))
            )
            # The same cumsum on a new partial sum nothing else has converted.
            alone = torch.cumsum(make(), 0)
            return {
                "counts": counts,
                "placed": [kohakutt.placement(t) for t in (a, g, b)],
                "pow": a.cpu(),
                "cumsum": g.cpu(),
                "tanh": b.cpu(),
                "alone": alone.cpu(),
            }

    (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    # Neither copy is made from the other: no partition, no all-gather.
    assert kept.pop("counts") == {"reduce_scatters": 1, "all_reduces": 1}
    assert fresh.pop("counts") == {"reduce_scatters": 2, "all_reduces": 1}
    assert kept["placed"] == ["Sharded(dim=0)", "Replicated", "Sharded(dim=0)"]
    _assert_same_bits(kept, fresh)
    assert torch.equal(kept["cumsum"], kept["alone"])


@pytest.mark.hw
@pytest.mark.dual
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shards_not_cut_from_replicated_copy(native_module, mesh, monkeypatch, dtype):
    kohakutt = native_module
    # Kept copies at full cost: pow asks for shards although a replicated copy exists.
    monkeypatch.setenv("KTT_MESH_CONV_FREE", "0")
    make = _pair(kohakutt, mesh, dtype)

    def run():
        with torch.no_grad():
            p = make()
            (g, a), counts = _counted(lambda: (torch.cumsum(p, 0), p.pow(2)))
            # The same pow on a new partial sum nothing else has converted.
            alone = make().pow(2)
            return {
                "counts": counts,
                "placed": [kohakutt.placement(t) for t in (g, a)],
                "cumsum": g.cpu(),
                "pow": a.cpu(),
                "alone": alone.cpu(),
            }

    (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    # The shards come from the partial sum by a reduce-scatter, not from a partition.
    assert kept.pop("counts") == {"all_reduces": 1, "reduce_scatters": 1}
    assert fresh.pop("counts") == {"all_reduces": 1, "reduce_scatters": 1}
    assert kept["placed"] == ["Replicated", "Sharded(dim=0)"]
    _assert_same_bits(kept, fresh)
    assert torch.equal(kept["pow"], kept["alone"])


@pytest.mark.hw
@pytest.mark.dual
def test_partial_stays_partial_across_dtypes(native_module, mesh, monkeypatch):
    kohakutt = native_module
    make = _pair(kohakutt, mesh, torch.bfloat16)
    scale_host = torch.rand(ROWS, DIM).add(0.5)
    scale32 = kohakutt.shard(scale_host.to(mesh), 0)
    scale16 = kohakutt.shard(scale_host.bfloat16().to(mesh), 0)

    def run():
        with torch.no_grad():
            p = make()
            host = _host_sum(p)
            a = p.pow(2)
            # A bf16 copy of the sum exists now. The fp32 consumers keep summing the
            # partials in fp32; the bf16 consumer takes the copy.
            wide, mixed, same = p.float(), p * scale32, p * scale16
            return {
                "host": host,
                "placed": [kohakutt.placement(t) for t in (a, wide, mixed, same)],
                "wide": wide.cpu(),
                "mixed": mixed.cpu(),
                "same": same.cpu(),
            }

    (kept, _), (fresh, _) = _kept_and_fresh(monkeypatch, run)
    assert kept["placed"] == [
        "Sharded(dim=0)",
        "Partial(sum)",
        "Partial(sum)",
        "Sharded(dim=0)",
    ]
    _assert_same_bits(kept, fresh)
    host = kept["host"]
    # A partial sum read back is summed on the host in fp32: exact here.
    assert torch.equal(kept["wide"], host)
    torch.testing.assert_close(kept["mixed"], host * scale_host, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        kept["same"].float(), host * scale_host, rtol=3e-2, atol=3e-2
    )
