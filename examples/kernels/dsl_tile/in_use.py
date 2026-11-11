"""Tile kernels in real use: inside kohakutt.capture and torch.compile, and as the
forward and backward of a torch.autograd.Function training a small model.
Usage: python examples/kernels/dsl_tile/in_use.py"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import kohakutt

sys.path.insert(0, str(Path(__file__).parent))
from common import builds, counted
from rms_norm import rms_norm
from rms_norm_backward import backward
from swiglu import swiglu


class RMSNormFunction(torch.autograd.Function):
    """y = x * rsqrt(mean(x^2) + eps) * weight: the forward and both gradients by
    tile kernels (rms_norm, then rms_norm_backward and rms_norm_dweight)."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rms_norm(x, weight, eps)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx, dweight = backward(x, dy.contiguous(), weight, ctx.eps, out_dtype=x.dtype)
        return dx, dweight.to(weight.dtype), None


class RMSNorm(nn.Module):
    """nn.RMSNorm's math with the tile kernels; `tile=False` runs torch ops."""

    def __init__(self, dim, eps=1e-6, tile=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps, self.tile = eps, tile

    def forward(self, x):
        if self.tile:
            return RMSNormFunction.apply(x, self.weight, self.eps)
        r = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * r * self.weight).to(x.dtype)


class Block(nn.Module):
    """A pre-norm MLP block: x + down(swiglu(gate(n), up(n))) with n = norm(x); the
    norm and (when `tile`) the gate product are tile kernels."""

    def __init__(self, dim, hidden, tile=True):
        super().__init__()
        self.norm = RMSNorm(dim, tile=tile)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.tile = tile

    def forward(self, x):
        n = self.norm(x)
        g, u = self.gate(n), self.up(n)
        h = swiglu(g, u) if self.tile and not torch.is_grad_enabled() else F.silu(g) * u
        return x + self.down(h)


def model(dim=256, hidden=512, depth=2, tile=True, seed=0):
    """A stack of blocks with a seeded initialisation (the same for both kinds)."""
    torch.manual_seed(seed)
    return nn.Sequential(*[Block(dim, hidden, tile) for _ in range(depth)])


@torch.no_grad()
def captured(shape=(4, 128, 256)):
    """The tile model replayed by kohakutt.capture against its eager calls: max
    difference, and (compiles, builds) of the replays."""
    dev = kohakutt.device(0)
    m = model(shape[-1]).to(dev).bfloat16()
    x = torch.randn(shape).bfloat16().to(dev)
    want = m(x).cpu()
    step = kohakutt.capture(m, x)
    try:
        y = torch.randn(shape).bfloat16().to(dev)
        got, grown = counted(lambda: (step(x).cpu(), step(y).cpu()))
        diff = (got[0].float() - want.float()).abs().max().item()
        again = (got[1].float() - m(y).cpu().float()).abs().max().item()
        return {"replay vs eager": diff, "new input": again, "built": builds(grown)}
    finally:
        step.release()


@torch.no_grad()
def compiled(shape=(4, 128, 256)):
    """The tile model under torch.compile(backend="ktt_tt") against eager: max
    difference, and (compiles, builds) of the compiled calls after the first three."""
    dev = kohakutt.device(0)
    m = model(shape[-1]).to(dev).bfloat16()
    x = torch.randn(shape).bfloat16().to(dev)
    want = m(x).cpu()
    fast = torch.compile(m, backend="ktt_tt")
    for _ in range(3):
        fast(x)
    got, grown = counted(lambda: fast(x).cpu())
    diff = (got.float() - want.float()).abs().max().item()
    return {"compiled vs eager": diff, "built": builds(grown)}


def train(steps=10, shape=(4, 128, 256), lr=1e-2):
    """Losses of a few AdamW steps of the same model with tile and with torch-op
    norms, on the same data (fp32 parameters, an MSE regression target)."""
    dev = kohakutt.device(0)
    g = torch.Generator().manual_seed(1)
    data = [
        (torch.randn(shape, generator=g), torch.randn(shape, generator=g) * 0.5)
        for _ in range(steps)
    ]
    losses = {}
    for tile in (True, False):
        m = model(shape[-1], tile=tile).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        out = []
        for x, y in data:
            loss = F.mse_loss(m(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            out.append(loss.item())
        losses["tile" if tile else "torch ops"] = out
    return losses


def gradients(shape=(2, 96, 320), eps=1e-6):
    """Relative errors of the Function's dx and dweight against float64 autograd,
    with a random weight and a random upstream gradient."""
    dev = kohakutt.device(0)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(shape, generator=g)
    w = torch.rand(shape[-1], generator=g) + 0.5
    dy = torch.randn(shape, generator=g)
    xd, wd = x.double().requires_grad_(), w.double().requires_grad_()
    (xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps) * wd).backward(
        dy.double()
    )
    xt, wt = x.to(dev).requires_grad_(), w.to(dev).requires_grad_()
    RMSNormFunction.apply(xt, wt, eps).backward(dy.to(dev))

    def rel(got, want):
        return ((got.cpu().double() - want).abs().max() / want.abs().max()).item()

    return {"dx": rel(xt.grad, xd.grad), "dweight": rel(wt.grad, wd.grad)}


def main():
    """Runs each use and prints what it measured."""
    try:
        for name, fn in [
            ("gradients", gradients),
            ("capture", captured),
            ("compile", compiled),
            ("training", train),
        ]:
            print(name, fn(), flush=True)
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
