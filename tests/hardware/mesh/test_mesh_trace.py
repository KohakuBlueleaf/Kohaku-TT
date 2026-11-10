"""Mesh: traces and size knobs: a captured trace on the mesh with a replicated,
sharded and partial-sum input replays bit for bit like the eager call on new inputs;
the placement a new tensor gets follows KTT_MESH_AUTO and KTT_MESH_SHARD_MIN (run the
file with them unset, with KTT_MESH_AUTO=0, with KTT_MESH_SHARD_MIN=0 and with it at a
huge value)."""

import os

import pytest
import torch
from mesh_support import make_partial, read

pytestmark = [pytest.mark.hw, pytest.mark.dual]

# Case ids known to fail, with the reason.
KNOWN_IDS = {}

AUTO = os.environ.get("KTT_MESH_AUTO", "1")
SHARD_MIN = int(os.environ.get("KTT_MESH_SHARD_MIN", str(1 << 20)))


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def ints(shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, shape, generator=gen).to(torch.bfloat16)


def place(k, mesh, value, spec, seed=0):
    if spec == "R":
        return k.replicate(value.to(mesh))
    if spec.startswith("S"):
        return k.shard(value.to(mesh), int(spec[1:]))
    return make_partial(k, mesh, tuple(value.shape), value.dtype, seed)


STEPS = {
    "affine": lambda w: lambda t: t * 2 + 1,
    "matmul": lambda w: lambda t: t @ w,
    "softmax": lambda w: lambda t: torch.softmax(t, 1),
    "sum": lambda w: lambda t: t.sum(0),
    "layer_norm": lambda w: lambda t: torch.nn.functional.layer_norm(t, (64,)),
}


def _cases():
    for name in STEPS:
        for spec in ("R", "S0", "S1", "P"):
            cid = f"{name}-{spec}"
            yield pytest.param(name, spec, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("name", "spec"), list(_cases()))
def test_capture_replay_equals_eager(native_module, mesh, name, spec):
    """A captured step replayed on new inputs of the same placement equals the eager
    result bit for bit."""
    k = native_module
    w = k.replicate(ints((64, 64), 20).to(mesh))
    fn = STEPS[name](w)
    x1 = place(k, mesh, ints((64, 64), 21), spec, 1)
    x2 = place(k, mesh, ints((64, 64), 22), spec, 2)
    with torch.no_grad():
        step = k.capture(fn, x1)
        try:
            replay = read(step(x2))
            eager = read(fn(x2))
            again = read(step(x1))
            eager1 = read(fn(x1))
        finally:
            step.release()
    assert torch.equal(replay, eager)
    assert torch.equal(again, eager1)


def test_capture_output_placement_matches_eager(native_module, mesh):
    """The output of a replayed trace has the placement eager gives."""
    k = native_module
    w = k.replicate(ints((64, 64), 23).to(mesh))
    x1 = place(k, mesh, ints((64, 64), 24), "S0")
    x2 = place(k, mesh, ints((64, 64), 25), "S0")
    with torch.no_grad():
        step = k.capture(lambda t: t @ w, x1)
        try:
            replayed = k.placement(step(x2))
        finally:
            step.release()
        eager = k.placement(x2 @ w)
    assert replayed == eager


# --------------------------------------------------------------------- size knobs


def _expected(nbytes):
    return "Sharded" if (AUTO != "0" and nbytes >= SHARD_MIN) else "Replicated"


@pytest.mark.parametrize(
    "shape", [(64, 64), (256, 1024), (1024, 1024), (256, 4096), (257, 4096)], ids=str
)
def test_new_tensor_placement_follows_knobs(native_module, mesh, shape):
    """A new bf16 tensor is sharded from KTT_MESH_SHARD_MIN bytes on (unless
    KTT_MESH_AUTO=0), replicated below; values are kept."""
    k = native_module
    gen = torch.Generator().manual_seed(shape[0])
    x = (torch.rand(shape, generator=gen) > 0.9).to(torch.bfloat16)
    t = x.to(mesh)
    nbytes = x.numel() * 2
    got = k.placement(t)
    want = _expected(nbytes)
    assert got.startswith(
        want
    ), f"{shape} ({nbytes} B): {got}, expected {want} (AUTO={AUTO}, MIN={SHARD_MIN})"
    assert torch.equal(read(t), x.float())
    assert torch.equal(read(t.sum(0)), x.float().sum(0))
    assert torch.equal(read(t * 2 + 1), x.float() * 2 + 1)


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_explicit_mark_wins_over_knobs(native_module, mesh, spec):
    """shard() / replicate() give the asked placement whatever the size knobs say."""
    k = native_module
    x = ints((64, 64), 26)
    t = place(k, mesh, x, spec)
    want = "Replicated" if spec == "R" else f"Sharded(dim={spec[1:]})"
    assert k.placement(t) == want
    assert torch.equal(read(t), x.float())
