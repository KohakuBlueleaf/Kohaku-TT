"""Mesh: API misuse: shard on a missing or size-1 dim, shard of a sharded tensor on
another dim, replicate and shard of a partial sum, marks on views, on parameters after
first use and on host tensors. Each case either gives the right values or raises a
Python error that names the problem (never a raw device check, never silently wrong)."""

import pytest
import torch
from mesh_support import make_partial, read
from torch import nn

pytestmark = [pytest.mark.hw, pytest.mark.dual]

# Case ids known to fail, with the reason.
KNOWN_IDS = {}

CLEAR_WORDS = (
    "dim",
    "shard",
    "device",
    "tt",
    "placement",
    "partial",
    "mesh",
    "host",
    "cpu",
)


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def ints(shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(-4, 5, shape, generator=gen).to(torch.bfloat16)


def outcome(fn):
    """('ok', result) or ('err', exception)."""
    try:
        return "ok", fn()
    except Exception as exc:  # noqa: BLE001
        return "err", exc


def assert_right_or_clear(kind, got, want):
    """`got` is outcome(...): the values equal `want`, or the error is a clear Python
    error."""
    status, value = got
    if status == "ok":
        v = read(value)
        assert v.shape == want.shape, f"shape {tuple(v.shape)} != {tuple(want.shape)}"
        assert torch.equal(v, want.float()), "wrong values"
        return
    msg = str(value)
    assert (
        "TT_FATAL" not in msg and "TT_THROW" not in msg
    ), f"raw device error: {msg[:200]}"
    assert isinstance(
        value, (ValueError, IndexError, RuntimeError, TypeError, NotImplementedError)
    )
    assert any(w in msg.lower() for w in CLEAR_WORDS), f"unclear error: {msg[:200]}"


@pytest.mark.parametrize("dim", [2, 5, -1, -3])
def test_shard_missing_dim(native_module, mesh, dim):
    """shard() of a 2-d tensor on dim 2 / 5 raises a clear error; a negative dim counts
    from the end as in torch."""
    k = native_module
    x = ints((64, 64))
    got = outcome(lambda: k.shard(x.to(mesh), dim))
    if dim >= 2 or dim < -2:
        status, value = got
        assert status == "err", f"shard on dim {dim} of a 2-d tensor returned a tensor"
        assert "TT_FATAL" not in str(value), str(value)[:200]
    else:
        assert_right_or_clear("neg", got, x)
        if got[0] == "ok":
            assert k.placement(got[1]) == "Sharded(dim=1)"


@pytest.mark.parametrize("shape", [(1, 64), (64, 1), (1, 1)])
def test_shard_size_one_dim(native_module, mesh, shape):
    """shard() on a size-1 dim gives the same values or a clear error."""
    k = native_module
    x = ints(shape)
    dim = shape.index(1)
    assert_right_or_clear("size1", outcome(lambda: k.shard(x.to(mesh), dim)), x)


@pytest.mark.parametrize("dims", [(0, 1), (1, 0)])
def test_shard_of_sharded_other_dim(native_module, mesh, dims):
    """shard(shard(x, a), b) keeps the values and is sharded on b."""
    k = native_module
    x = ints((64, 64), 1)
    s = k.shard(x.to(mesh), dims[0])
    t = k.shard(s, dims[1])
    assert torch.equal(read(t), x.float())
    assert k.placement(t) == f"Sharded(dim={dims[1]})"
    assert k.placement(s) == f"Sharded(dim={dims[0]})"
    assert torch.equal(read(s), x.float())


def test_replicate_partial_sum(native_module, mesh):
    """replicate() of a partial sum completes the sum."""
    k = native_module
    p = make_partial(k, mesh, (64, 64), torch.bfloat16, 3)
    want = read(p)
    r = k.replicate(p)
    assert k.placement(r) == "Replicated"
    assert torch.equal(read(r), want)
    assert torch.equal(read(p), want)


@pytest.mark.parametrize("dim", [0, 1])
def test_shard_partial_sum(native_module, mesh, dim):
    """shard() of a partial sum completes the sum into a shard."""
    k = native_module
    p = make_partial(k, mesh, (64, 64), torch.bfloat16, 4)
    want = read(p)
    s = k.shard(p, dim)
    assert k.placement(s) == f"Sharded(dim={dim})"
    assert torch.equal(read(s), want)


def test_replicate_twice(native_module, mesh):
    """replicate(replicate(x)) and shard of a replicated tensor keep the values."""
    k = native_module
    x = ints((64, 64), 5)
    r = k.replicate(k.replicate(x.to(mesh)))
    assert torch.equal(read(r), x.float())
    assert torch.equal(read(k.shard(r, 1)), x.float())


@pytest.mark.parametrize("fn_name", ["shard", "replicate"])
def test_mark_host_tensor(native_module, mesh, fn_name):
    """shard() / replicate() of a host tensor raise a clear error (or move it)."""
    k = native_module
    x = ints((64, 64), 6)
    call = (lambda: k.shard(x, 0)) if fn_name == "shard" else (lambda: k.replicate(x))
    assert_right_or_clear("host", outcome(call), x)


def test_mark_on_view_leaves_base_alone(native_module, mesh):
    """shard() of a slice gives a placed copy; the base keeps placement and values."""
    k = native_module
    x = ints((64, 64), 7)
    base = k.replicate(x.to(mesh))
    v = base[8:40]
    s = k.shard(v, 1)
    assert k.placement(s) == "Sharded(dim=1)"
    assert torch.equal(read(s), x[8:40].float())
    assert k.placement(base) == "Replicated"
    assert torch.equal(read(base), x.float())


def test_scalar_and_vector_marks(native_module, mesh):
    """shard() of a 1-d tensor on dim 0 keeps the values; of a 0-d tensor it raises a
    clear error or returns the value."""
    k = native_module
    v = ints((64,), 8)
    assert torch.equal(read(k.shard(v.to(mesh), 0)), v.float())
    z = torch.tensor(3.0, dtype=torch.bfloat16)
    assert_right_or_clear("scalar", outcome(lambda: k.shard(z.to(mesh), 0)), z)


def test_module_marks_after_first_use(native_module, mesh):
    """Marking a Linear that is already on the mesh and was used once moves its weight
    to the marked placement at once and the output stays right."""
    k = native_module
    torch.manual_seed(0)
    lin = nn.Linear(64, 96).bfloat16()
    for p in lin.parameters():
        p.data = ints(tuple(p.shape), 9)
    x = ints((16, 64), 10)
    want = x.float() @ lin.weight.float().t() + lin.bias.float()
    lin = lin.to(mesh)
    with torch.no_grad():
        first = lin(k.replicate(x.to(mesh)))
        assert torch.equal(read(first), want)
        k.rowwise(lin)
        assert k.placement(lin.weight) == "Sharded(dim=1)"
        assert k.placement(lin.bias) == "Replicated"
        second = lin(k.replicate(x.to(mesh)))
    assert torch.equal(read(second), want)


def test_module_marks_before_mesh(native_module, mesh):
    """Marks on a module still on the host take effect when it moves to the mesh."""
    k = native_module
    lin = k.colwise(nn.Linear(64, 96).bfloat16())
    lin = lin.to(mesh)
    assert k.placement(lin.weight) == "Sharded(dim=0)"
    assert k.placement(lin.bias) == "Sharded(dim=0)"


@pytest.mark.xfail(
    strict=True,
    reason="shard(module, d) names no tensor when a tensor lacks dim d",
)
def test_module_mark_bad_dim(native_module, mesh):
    """Marking a module sharded on a dim a tensor lacks (bias is 1-d, dim 1) raises an
    error that names the tensor (bias) or leaves the bias placed."""
    k = native_module
    lin = nn.Linear(64, 96).bfloat16()
    status, value = outcome(lambda: k.shard(lin, 1).to(mesh))
    if status == "err":
        assert "TT_FATAL" not in str(value), str(value)[:200]
        assert "bias" in str(value) and "96" in str(
            value
        ), f"error names neither the tensor nor its shape: {str(value)[:200]}"
    else:
        assert k.placement(lin.weight) == "Sharded(dim=1)"
        placed = k.placement(lin.bias)
        assert placed in ("Replicated", "Sharded(dim=0)", "Sharded(dim=1)")
