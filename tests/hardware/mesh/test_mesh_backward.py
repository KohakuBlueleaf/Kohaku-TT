"""Mesh: gradients: the gradient of each op family with its input (and weight) in
each placement, against CPU autograd in fp32 and against the same op on replicated
inputs (what one card computes). The gradient reaches the leaf tensor with its shape
and values."""

import itertools

import pytest
import torch
import torch.nn.functional as F
from mesh_support import max_err

pytestmark = [pytest.mark.hw, pytest.mark.dual]

REASON_WGRAD = (
    "a bf16 partial sum is rounded on each card before the sum (weight gradient "
    "of a matmul with sharded rows)"
)

# Case ids known to fail, with the reason.
KNOWN_IDS = dict.fromkeys(["matmul-S0_R", "matmul-S0_S0", "matmul-S0_S1"], REASON_WGRAD)


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def vals(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen).to(torch.bfloat16)


def grads(k, mesh, fn, values, specs):
    """The gradients of sum(fn(*leaves) * weights) with each leaf (a tt tensor made from
    `values`) placed as `specs` inside the graph."""
    leaves = [v.to(mesh).requires_grad_() for v in values]
    placed = []
    for t, s in zip(leaves, specs):
        if s == "R":
            placed.append(k.replicate(t))
        else:
            placed.append(k.shard(t, int(s[1:])))
    out = fn(*placed)
    w = vals(tuple(out.shape), 99).to(mesh)
    (out * w).sum().backward()
    torch.tt.synchronize()
    return [t.grad.cpu().float() for t in leaves], w.cpu().float()


def cpu_grads(fn, values, w):
    leaves = [v.float().requires_grad_() for v in values]
    out = fn(*leaves)
    (out * w).sum().backward()
    return [t.grad for t in leaves]


OPS = {
    "mul": (lambda a, b: a * b, [(64, 64), (64, 64)]),
    "matmul": (lambda a, b: a @ b, [(64, 64), (64, 64)]),
    "linear": (lambda x, w: F.linear(x, w), [(64, 64), (96, 64)]),
    "softmax": (lambda a: torch.softmax(a, 1), [(64, 64)]),
    "layer_norm": (lambda a: F.layer_norm(a, (64,)), [(64, 64)]),
    "gelu": (lambda a: F.gelu(a), [(64, 64)]),
    "sum": (lambda a: a.sum(1), [(64, 64)]),
    "transpose": (lambda a: a.t().contiguous() * 2, [(64, 64)]),
    "cat": (lambda a, b: torch.cat([a, b], 0), [(32, 64), (32, 64)]),
}


def _cases():
    for name, (_, shapes) in OPS.items():
        for specs in itertools.product(["R", "S0", "S1"], repeat=len(shapes)):
            cid = f"{name}-{'_'.join(specs)}"
            yield pytest.param(name, specs, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("name", "specs"), list(_cases()))
def test_grad(native_module, mesh, name, specs):
    """Gradients of an op with inputs in each placement match CPU autograd with an error
    no larger than one card's."""
    k = native_module
    fn, shapes = OPS[name]
    values = [vals(s, 100 + i) for i, s in enumerate(shapes)]
    got, w = grads(k, mesh, fn, values, specs)
    base, _ = grads(k, mesh, fn, values, ["R"] * len(specs))
    want = cpu_grads(fn, values, w)
    for g, b, c in zip(got, base, want):
        assert g.shape == c.shape
        assert max_err(g, c) <= max_err(
            b, c
        ), f"{max_err(g, c):g} > one card's {max_err(b, c):g}"
