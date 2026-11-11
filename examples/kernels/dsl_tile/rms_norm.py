"""RMS norm written at the tile level: one launch reading the row
once, an exact fp32 sum of squares, the row kept in L1 for the second pass. Compared
with the metal-level norm (bit for bit) and with Gemma's norm module run eagerly.
Usage: python examples/kernels/dsl_tile/rms_norm.py [--tokens T] [--width W]"""

import argparse
import sys
from pathlib import Path

import torch

import kohakutt
import kohakutt.dsl as kd
import kohakutt.dsl.metal as km

sys.path.insert(0, str(Path(__file__).parent))
from common import metal_example, timed


@kd.kernel
def rms_norm(
    x: kd.Tensor,
    scale: kd.Tensor,
    out: kd.Out,
    eps: kd.Runtime[float],
    group: kd.Static[int] = 0,
):
    """out = x * rsqrt(mean(x^2) + eps) * scale over the last dim; scale is [D]. Each
    row on `group` cores (0: the plan picks; 1: one core per row, the metal-level
    norm's sum order)."""
    g = kd.resident(scale.tiles[0, :])
    for t in kd.grid(kd.num_tile_rows(x), group=group or None):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        ms = kd.reduce_sum(row * row, dim=-1, exact=True) / kd.size(x, -1)
        r = kd.rsqrt(ms + eps)
        kd.store(out.tiles[t, :], (row * r.col() * g.row()).cast(out.dtype))


@kd.kernel
def rms_norm_split(
    x: kd.Tensor,
    scale: kd.Tensor,
    out: kd.Out,
    eps: kd.Runtime[float],
    group: kd.Static[int] = 8,
):
    """rms_norm with each tile row split over `group` cores, which exchange their
    partial sums of squares: for few, wide rows (a decode step)."""
    g = kd.resident(scale.tiles[0, :])
    for t in kd.grid(kd.num_tile_rows(x), group=group):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        ms = kd.reduce_sum(row * row, dim=-1, exact=True) / kd.size(x, -1)
        r = kd.rsqrt(ms + eps)
        kd.store(out.tiles[t, :], (row * r.col() * g.row()).cast(out.dtype))


class GemmaRMSNorm(torch.nn.Module):
    """Gemma's norm as transformers writes it: fp32 statistics, scale 1 + weight."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class MetalNorm:
    """The metal-level norm example (the backend's norm as a km program)."""

    def __init__(self):
        module = metal_example("rmsnorm_km")
        cols, rows = torch.ops.ktt.compute_grid()
        program = module.rmsnorm_program(km.Machine(cols, rows), "bfloat16", "bfloat16")
        self.module, self.launcher = module, km.Launcher(program)

    def __call__(self, x, gamma, eps):
        return self.module.rmsnorm(self.launcher, x, eps, gamma)


@kd.kernel
def rms_stat(x: kd.Tensor, out: kd.Out, eps: kd.Runtime[float]):
    """rsqrt(mean(x^2) + eps) of each row, repeated over the row (fp32 out)."""
    for t in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        ms = kd.reduce_sum(row * row, dim=-1, exact=True) / kd.size(x, -1)
        kd.store(out.tiles[t, :], row * 0.0 + kd.rsqrt(ms + eps).col())


@kd.kernel
def rms_mean_square(x: kd.Tensor, out: kd.Out):
    """mean(x^2) of each row, repeated over the row (fp32 out)."""
    for t in kd.grid(kd.num_tile_rows(x)):
        row = kd.load(x.tiles[t, :], pad=0).cast(kd.f32)
        ms = kd.reduce_sum(row * row, dim=-1, exact=True) / kd.size(x, -1)
        kd.store(out.tiles[t, :], row * 0.0 + ms.col())


def relative(got, want):
    """Largest relative error of `got` against float64 `want`."""
    got, want = got.detach().double(), want.double()
    return float(((got - want).abs() / want.abs().clamp_min(1e-30)).max())


@torch.no_grad()
def explain(shape, eps=1e-6, seed=0):
    """The tile kernel and the eager module step by step against float64: the row
    statistic, the fp32 result before the cast, and the bf16 result."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    x = torch.randn(shape, generator=g).bfloat16()
    weight = (torch.randn(shape[-1], generator=g) * 0.1).bfloat16()
    x64, s = x.double(), 1.0 + weight.float()
    r64 = torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps)
    y64 = x64 * r64 * s.double()
    x_tt, scale = x.to(dev), s.to(dev)
    xf = x_tt.float()
    ms64 = x64.pow(2).mean(-1, keepdim=True)
    eager_ms = xf.pow(2).mean(-1, keepdim=True)
    tile_ms = rms_mean_square(x_tt, out_dtype=torch.float32).cpu()[..., :1]
    eager_r = torch.rsqrt(eager_ms + eps).cpu()
    eager_ms = eager_ms.cpu()
    tile_r = rms_stat(x_tt, eps, out_dtype=torch.float32).cpu()[..., :1]
    eager32 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * scale
    eager16 = eager32.bfloat16().cpu()
    eager32 = eager32.cpu()
    tile32 = rms_norm(x_tt, scale, eps, out_dtype=torch.float32).cpu()
    tile16 = rms_norm(x_tt, scale, eps).cpu()
    differ = tile16 != eager16
    ulp = (tile16.float() - eager16.float()).abs() / (
        torch.finfo(torch.bfloat16).eps * eager16.float().abs().clamp_min(1e-30)
    )
    return {
        "mean square rel. error (tile, eager)": (
            relative(tile_ms, ms64),
            relative(eager_ms, ms64),
        ),
        "rsqrt rel. error of its own mean square (tile, eager)": (
            relative(tile_r, torch.rsqrt(tile_ms.double() + eps)),
            relative(eager_r, torch.rsqrt(eager_ms.double() + eps)),
        ),
        "statistic rel. error (tile, eager)": (
            relative(tile_r, r64),
            relative(eager_r, r64),
        ),
        "fp32 result rel. error (tile, eager)": (
            relative(tile32, y64),
            relative(eager32, y64),
        ),
        "bf16 results differing": f"{int(differ.sum())} of {differ.numel()}",
        "largest bf16 difference in ulps": (
            float(ulp[differ].max()) if differ.any() else 0
        ),
    }


def compare(shape, eps=1e-6, seed=0):
    """Times (ms) of the tile kernel and the eager module at `shape`."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    x = torch.randn(shape, generator=g).bfloat16()
    weight = (torch.randn(shape[-1], generator=g) * 0.1).bfloat16()
    module = GemmaRMSNorm(shape[-1], eps)
    with torch.no_grad():
        module.weight.copy_(weight.float())
    module = module.to(dev)
    x_tt = x.to(dev)
    scale = (1.0 + weight.float()).to(dev)
    with torch.no_grad():
        return {
            "tile ms": timed(lambda: rms_norm(x_tt, scale, eps)) * 1e3,
            "eager ms": timed(lambda: module(x_tt)) * 1e3,
        }


def same_as_metal(shape, eps=1e-6, seed=1):
    """True when the tile kernel with one core per row equals the metal-level norm on
    bf16 x and gamma, and the plan's split is no less accurate against float64."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    x = torch.randn(shape, generator=g).bfloat16().to(dev)
    gamma = torch.randn(shape[-1], generator=g).bfloat16().to(dev)
    metal = MetalNorm()(x, gamma, eps).cpu()
    if not torch.equal(rms_norm(x, gamma, eps, group=1).cpu(), metal):
        return False
    x64 = x.cpu().double()
    want = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps)
    want = want * gamma.cpu().double()
    planned = (rms_norm(x, gamma, eps).cpu().double() - want).abs()
    alone = (metal.double() - want).abs()
    return bool(planned.mean() <= alone.mean() * 1.01 and planned.max() <= alone.max())


def main():
    """Runs the norm at one shape; prints agreement, times, errors and the plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--width", type=int, default=1536)
    args = parser.parse_args()
    shape = (1, args.tokens, args.width)
    try:
        print(f"rms_norm {shape}: equal to the metal level: {same_as_metal(shape)}")
        for key, value in {**compare(shape), **explain(shape)}.items():
            print(f"  {key}: {value}")
        x = torch.empty(shape, dtype=torch.bfloat16)
        print(rms_norm.plan(x, torch.empty(args.width), 1e-6).report())
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
