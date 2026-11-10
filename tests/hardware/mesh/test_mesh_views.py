"""Mesh: views and writes: views and slices of sharded and partial-sum tensors,
in-place writes and partial writes (also across the shard boundary), a view that must
see a write to its base, a tensor consumed by two ops that want different placements,
and a parameter used twice (tied embedding and output projection) replicated and
sharded. Values are small integers, so every result is exact."""

import pytest
import torch
from mesh_support import make_partial, read
from torch import nn

pytestmark = [pytest.mark.hw, pytest.mark.dual]

# Case ids known to fail, with the reason.
KNOWN_IDS = {}


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def ints(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(-4, 5, shape, generator=gen).to(torch.bfloat16)


def tt(k, mesh, spec, value, seed=0):
    """`value` on the mesh in placement `spec`; for P a pending sum, with its host
    value returned second."""
    if spec == "R":
        return k.replicate(value.to(mesh)), value.float()
    if spec.startswith("S"):
        return k.shard(value.to(mesh), int(spec[1:])), value.float()
    p = make_partial(k, mesh, tuple(value.shape), value.dtype, seed)
    return p, read(p)


def same(got, want):
    got = (
        read(got)
        if not isinstance(got, torch.Tensor) or got.device.type != "cpu"
        else got.float()
    )
    assert got.shape == want.shape, f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    assert torch.equal(
        got, want.float()
    ), f"max diff {float((got - want.float()).abs().max())}"


# ---------------------------------------------------------------- views and slices

VIEWS = {
    "slice_rows": lambda x: x[8:40],
    "slice_cols": lambda x: x[:, 8:40],
    "slice_across_boundary": lambda x: x[30:34],
    "row": lambda x: x[5],
    "transpose": lambda x: x.t(),
    "reshape": lambda x: x.reshape(2, 32, 64),
    "narrow_step": lambda x: x[::2],
}


def _view_cases():
    for name in VIEWS:
        for spec in ("R", "S0", "S1", "P"):
            cid = f"{name}-{spec}"
            yield pytest.param(name, spec, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("name", "spec"), list(_view_cases()))
def test_view_then_op(native_module, mesh, name, spec):
    """A view of a tensor in any placement, then an add and a copy to the host, gives
    the CPU values."""
    k = native_module
    x, ref = tt(k, mesh, spec, ints((64, 64), 1))
    v = VIEWS[name](x)
    same((v + 1).cpu(), VIEWS[name](ref) + 1)
    same(v.cpu(), VIEWS[name](ref))


# ------------------------------------------------------------------- in-place writes

WRITES = {
    "add_scalar": (lambda t: t.add_(1), lambda r: r + 1),
    "mul_scalar": (lambda t: t.mul_(2), lambda r: r * 2),
    "zero": (lambda t: t.zero_(), lambda r: r * 0),
    "fill": (lambda t: t.fill_(3), lambda r: r * 0 + 3),
    "neg": (lambda t: t.neg_(), lambda r: -r),
    "add_tensor": (lambda t: t.add_(t), lambda r: r + r),
}


def _write_cases():
    for name in WRITES:
        for spec in ("R", "S0", "S1", "P"):
            cid = f"{name}-{spec}"
            yield pytest.param(name, spec, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("name", "spec"), list(_write_cases()))
def test_inplace(native_module, mesh, name, spec):
    """An in-place op on a tensor in any placement leaves the CPU values; a partial sum
    must not apply a scalar add once per chip."""
    k = native_module
    x, ref = tt(k, mesh, spec, ints((64, 64), 2))
    fn, cpu = WRITES[name]
    fn(x)
    same(x.cpu(), cpu(ref))


PARTIAL_WRITES = {
    "rows_start": (slice(0, 4), slice(None)),
    "rows_across_boundary": (slice(30, 34), slice(None)),
    "cols_across_boundary": (slice(None), slice(30, 34)),
    "single_row": (5, slice(None)),
    "single_element": (40, 7),
}


def _pw_cases():
    for name in PARTIAL_WRITES:
        for spec in ("R", "S0", "S1", "P"):
            cid = f"{name}-{spec}"
            yield pytest.param(name, spec, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("name", "spec"), list(_pw_cases()))
def test_partial_write(native_module, mesh, name, spec):
    """x[index] = 7 changes exactly the indexed elements, wherever the shard boundary
    falls, and the rest keeps its values."""
    k = native_module
    x, ref = tt(k, mesh, spec, ints((64, 64), 3))
    idx = PARTIAL_WRITES[name]
    x[idx] = 7
    ref = ref.clone()
    ref[idx] = 7
    same(x.cpu(), ref)


@pytest.mark.parametrize(
    ("spec", "src"),
    [
        pytest.param(a, b, id=f"{a}-{b}", marks=_mark(f"{a}-{b}"))
        for a in ("R", "S0", "S1")
        for b in ("R", "S0", "S1")
    ],
)
def test_copy_between_placements(native_module, mesh, spec, src):
    """dst.copy_(src) with dst and src in different placements gives src's values and
    keeps dst's placement."""
    k = native_module
    dst, _ = tt(k, mesh, spec, ints((64, 64), 4))
    s, sref = tt(k, mesh, src, ints((64, 64), 5))
    before = k.placement(dst)
    dst.copy_(s)
    same(dst.cpu(), sref)
    assert k.placement(dst) == before


@pytest.mark.parametrize("src", ["R", "S0", "S1", "host"])
def test_copy_into_partial_sum(native_module, mesh, src):
    """dst.copy_(src) into a partial sum gives src's values; the partial sum is
    replaced, so dst takes the placement of the copied value."""
    k = native_module
    dst, _ = tt(k, mesh, "P", ints((64, 64), 16))
    if src == "host":
        s, sref = ints((64, 64), 17), ints((64, 64), 17).float()
    else:
        s, sref = tt(k, mesh, src, ints((64, 64), 17))
    dst.copy_(s)
    same(dst.cpu(), sref)
    assert not k.placement(dst).startswith("Partial")


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_copy_from_host_keeps_placement(native_module, mesh, spec):
    """dst.copy_(host tensor) keeps dst's placement, also for a tensor larger than the
    automatic sharding threshold."""
    k = native_module
    dst, _ = tt(k, mesh, spec, ints((1024, 1024), 18))
    before = k.placement(dst)
    host = ints((1024, 1024), 19)
    dst.copy_(host)
    same(dst.cpu(), host.float())
    assert k.placement(dst) == before


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_copy_into_new_tensor_takes_source_placement(native_module, mesh, spec):
    """clone() and torch.empty_like(x).copy_(x) give a tensor in x's placement."""
    k = native_module
    x, ref = tt(k, mesh, spec, ints((64, 64), 20))
    want = k.placement(x)
    for y in (x.clone(), torch.empty_like(x).copy_(x)):
        same(y.cpu(), ref)
        assert k.placement(y) == want


def test_load_state_dict_keeps_marks(native_module, mesh):
    """load_state_dict into a column-split Linear that is on the mesh keeps the weight
    sharded on dim 0 and gives the loaded values."""
    k = native_module
    lin = k.colwise(nn.Linear(64, 96).bfloat16()).to(mesh)
    sd = {"weight": ints((96, 64), 12), "bias": ints((96,), 13)}
    lin.load_state_dict(sd)
    assert k.placement(lin.weight) == "Sharded(dim=0)"
    assert k.placement(lin.bias) == "Sharded(dim=0)"
    same(lin.weight.detach().cpu(), sd["weight"])
    same(lin.bias.detach().cpu(), sd["bias"])


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_slice_assign_from_other_placement(native_module, mesh, spec):
    """x[8:24] = y with y sharded on another dim copies y's values."""
    k = native_module
    x, ref = tt(k, mesh, spec, ints((64, 64), 6))
    y, yref = tt(k, mesh, "S1", ints((16, 64), 7))
    x[8:24] = y
    ref = ref.clone()
    ref[8:24] = yref
    same(x.cpu(), ref)


# ---------------------------------------------------------- aliasing and stale copies


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_view_sees_write_to_base(native_module, mesh, spec):
    """A view made before an in-place write to its base shows the written values."""
    k = native_module
    x, ref = tt(k, mesh, spec, ints((64, 64), 8))
    v = x[4:20]
    x.add_(1)
    same(v.cpu(), ref[4:20] + 1)


@pytest.mark.parametrize("spec", ["R", "S0", "S1"])
def test_base_sees_write_to_view(native_module, mesh, spec):
    """An in-place write through a view shows in the base."""
    k = native_module
    x, ref = tt(k, mesh, spec, ints((64, 64), 9))
    v = x[4:20]
    v.add_(1)
    ref = ref.clone()
    ref[4:20] += 1
    same(x.cpu(), ref)


@pytest.mark.parametrize("op", ["flip", "matmul_rhs", "cat"])
def test_no_stale_gathered_copy(native_module, mesh, op):
    """An op that gathers its sharded input, then an in-place write, then the same op
    again, sees the write."""
    k = native_module
    x, ref = tt(k, mesh, "S0", ints((64, 64), 10))
    w, wref = tt(k, mesh, "R", ints((64, 64), 11))

    def make(weight):
        return {
            "flip": lambda t: t.flip(0),
            "matmul_rhs": lambda t: weight @ t,
            "cat": lambda t: torch.cat([t, t], 1),
        }[op]

    fn, cpu_fn = make(w), make(wref)
    first = fn(x).cpu()
    x.add_(1)
    second = fn(x).cpu()
    same(first, cpu_fn(ref))
    same(second, cpu_fn(ref + 1))


def test_two_consumers_want_different_placements(native_module, mesh):
    """One sharded tensor feeds a row-wise matmul, a reduction over the sharded dim and
    a transpose; each result is right and the tensor is unchanged afterwards."""
    k = native_module
    x, ref = tt(k, mesh, "S0", ints((64, 64), 12))
    w, wref = tt(k, mesh, "R", ints((64, 64), 13))
    a = x @ w
    b = x.sum(0)
    c = x.t().contiguous()
    d = x @ w
    same(a.cpu(), ref @ wref)
    same(b.cpu(), ref.sum(0))
    same(c.cpu(), ref.t())
    same(d.cpu(), ref @ wref)
    same(x.cpu(), ref)
    assert k.placement(x) == "Sharded(dim=0)"


# ---------------------------------------------------------- one parameter, two places


class Tied(torch.nn.Module):
    """An embedding table that is also the output projection."""

    def __init__(self, vocab, dim):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids):
        return self.head(self.emb(ids) * 2)


def _tied_cases():
    for mark in ("none", "replicate", "shard0", "shard1"):
        cid = f"tied-{mark}"
        yield pytest.param(mark, id=cid, marks=_mark(cid))


@pytest.mark.parametrize("mark", [p for p in _tied_cases()])
def test_tied_parameter(native_module, mesh, mark):
    """A weight used by the embedding and by the output projection gives the CPU logits
    under each placement of the module, and stays one tensor."""
    k = native_module
    model = Tied(64, 64)
    with torch.no_grad():
        model.emb.weight.copy_(ints((64, 64), 14).float() / 4)
    model = model.bfloat16()
    ids_h = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(15))
    want = model(ids_h).float()
    if mark == "replicate":
        k.replicate(model)
    elif mark == "shard0":
        k.shard(model, 0)
    elif mark == "shard1":
        k.shard(model, 1)
    model = model.to(mesh)
    assert model.head.weight is model.emb.weight
    with torch.no_grad():
        got = model(ids_h.to(mesh)).cpu().float()
    assert got.shape == want.shape
    assert float((got - want).abs().max()) <= float(want.abs().max()) * 2**-7
