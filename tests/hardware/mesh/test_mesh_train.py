"""Mesh: training: one SGD and one Adam step on replicated and on sharded
parameters, and three training steps of a two-layer MLP and of a one-block transformer
with tiny sizes. Parameters after the steps are compared with the same steps on CPU in
fp32 and with the same steps on replicated parameters (what one card computes)."""

import pytest
import torch
from mesh_support import max_err
from torch import nn

pytestmark = [pytest.mark.hw, pytest.mark.dual]

REASON_SHARD_MODULE = "shard(module, d) names no tensor when a tensor lacks dim d"
REASON_ADAM_NOISE = (
    "three Adam steps of a one-block transformer end further from CPU than on one card"
)

# Case ids known to fail, with the reason.
KNOWN_IDS = {
    **dict.fromkeys(
        [
            "mlp1-sgd-S1",
            "mlp1-adam-S1",
            "mlp3-sgd-S1",
            "mlp3-adam-S1",
            "block3-adam-S1",
        ],
        REASON_SHARD_MODULE,
    ),
    **dict.fromkeys(["block3-adam-S0", "block3-adam-colrow"], REASON_ADAM_NOISE),
}


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


class Mlp(nn.Module):
    def __init__(self, d=64, h=128):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d=64, heads=2):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, 2 * d)
        self.down = nn.Linear(2 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = (
            self.qkv(self.ln1(x))
            .reshape(b, t, 3, self.heads, d // self.heads)
            .unbind(2)
        )
        a = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(torch.relu(self.up(self.ln2(x))))


def make(kind):
    torch.manual_seed(0)
    return Mlp() if kind == "mlp" else Block()


def data(kind, steps):
    gen = torch.Generator().manual_seed(1)
    shape = (32, 64) if kind == "mlp" else (2, 16, 64)
    return [
        (torch.randn(shape, generator=gen), torch.randn(shape, generator=gen))
        for _ in range(steps)
    ]


def run(k, mesh, kind, mark, opt_name, steps):
    """Trains a fresh model `steps` steps on the mesh with module mark `mark` ('R' or
    'S0' or 'S1' or 'colrow'), returns (losses, {name: param}) on the host."""
    model = make(kind)
    if mark == "R":
        k.replicate(model)
    elif mark == "S0":
        k.shard(model, 0)
    elif mark == "S1":
        k.shard(model, 1)
    elif mark == "colrow":
        for name, m in model.named_children():
            if name in ("fc1", "qkv", "up"):
                k.colwise(m)
            elif name in ("fc2", "proj", "down"):
                k.rowwise(m)
            else:
                k.replicate(m)
    model = model.to(mesh)
    opt = make_opt(model, opt_name)
    losses = []
    for x, y in data(kind, steps):
        opt.zero_grad()
        loss = ((model(k.replicate(x.to(mesh))) - y.to(mesh)) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.cpu()))
    return losses, {n: p.detach().cpu().float() for n, p in model.named_parameters()}


def make_opt(model, name):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=0.01)


def cpu_run(kind, opt_name, steps):
    model = make(kind)
    opt = make_opt(model, opt_name)
    losses = []
    for x, y in data(kind, steps):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses, {n: p.detach() for n, p in model.named_parameters()}


def _cases():
    for kind, steps in (("mlp", 1), ("mlp", 3), ("block", 3)):
        for opt in ("sgd", "adam"):
            if kind == "block" and opt == "sgd":
                continue
            for mark in ("S0", "S1", "colrow"):
                cid = f"{kind}{steps}-{opt}-{mark}"
                yield pytest.param(kind, steps, opt, mark, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("kind", "steps", "opt", "mark"), list(_cases()))
def test_train_steps(native_module, mesh, kind, steps, opt, mark):
    """After the steps, every parameter is within one card's error of the CPU fp32
    parameter, and the loss of each step is finite."""
    k = native_module
    want_losses, want = cpu_run(kind, opt, steps)
    base_losses, base = run(k, mesh, kind, "R", opt, steps)
    got_losses, got = run(k, mesh, kind, mark, opt, steps)
    assert all(torch.isfinite(torch.tensor(got_losses)))
    for n in want:
        assert got[n].shape == want[n].shape, n
        err, base_err = max_err(got[n], want[n]), max_err(base[n], want[n])
        assert err <= base_err, f"{n}: {err:g} > one card's {base_err:g}"
