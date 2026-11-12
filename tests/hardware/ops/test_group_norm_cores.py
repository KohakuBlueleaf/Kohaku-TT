"""Group norm at SDXL UNet / VAE shapes on its default routes (Nchw: the plane programs
with statistics dealt over the cores, under 1024 pixels the channels-last twin; Nhwc
storage: the row programs as stored) agrees with float64, in at most three programs."""

import pytest
import torch
import torch.nn.functional as F

import kohakutt

SHAPES = [
    (2, 320, 32, 32),
    (2, 640, 16, 16),
    (2, 1280, 8, 8),
    (1, 2560, 8, 8),
    (1, 128, 128, 128),
    (1, 512, 64, 64),
    (40, 64, 8, 8),  # 1280 (image, group) units: two rows of statistics tiles
]


def sample(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed + sum(shape))
    c = shape[1]
    return (
        (torch.randn(shape, generator=g) * 1.5 + 0.7).to(dtype),
        (torch.randn(c, generator=g) * 0.5 + 1).to(dtype),
        torch.randn(c, generator=g).to(dtype),
    )


def rel(got, ref):
    return ((got.double() - ref).norm() / ref.norm()).item()


def on_tt(x, layout):
    t = x.to("tt")
    if layout == "nhwc":
        torch.ops.ktt._debug_to_nhwc(t)
    return t


@pytest.mark.hw
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_group_norm_shapes(native, shape, dtype, layout):
    x, w, b = sample(shape, dtype)
    ref = F.group_norm(x.double(), 32, w.double(), b.double())
    cpu = F.group_norm(x.float(), 32, w.float(), b.float()).to(dtype)
    xt, wt, bt = on_tt(x, layout), w.to("tt"), b.to("tt")
    torch.tt.synchronize()
    before = kohakutt.counters()["programs"]
    with torch.no_grad():
        y = F.group_norm(xt, 32, wt, bt)
    torch.tt.synchronize()
    assert kohakutt.counters()["programs"] - before <= 3
    assert rel(y.cpu(), ref) <= max(3 * rel(cpu, ref), 2e-6)


@pytest.mark.hw
@pytest.mark.parametrize("shape", [(2, 320, 8, 8), (1, 64, 32, 32), (40, 64, 4, 4)])
def test_group_norm_gradients_and_statistics(native, shape):
    """Nchw inputs with gradients (twin and plane routes): input, weight and bias
    gradients and the saved mean / rstd against float64."""
    x, w, b = sample(shape, torch.float32)
    up = torch.randn(shape, generator=torch.Generator().manual_seed(1))
    leaf = lambda t: t.double().requires_grad_()  # noqa: E731
    xd, wd, bd = leaf(x), leaf(w), leaf(b)
    F.group_norm(xd, 32, wd, bd).backward(up.double())
    xt, wt, bt = (t.to("tt").requires_grad_() for t in (x, w, b))
    F.group_norm(xt, 32, wt, bt).backward(up.to("tt"))
    for got, want in ((xt, xd), (wt, wd), (bt, bd)):
        assert rel(got.grad.cpu(), want.grad) <= 1e-4
    n, c = shape[:2]
    with torch.no_grad():
        _, mean, rstd = torch.native_group_norm(
            x.to("tt"), w.to("tt"), b.to("tt"), n, c, shape[2] * shape[3], 32, 1e-5
        )
    grouped = x.double().reshape(n, 32, -1)
    assert rel(mean.cpu(), grouped.mean(-1)) <= 1e-5
    assert rel(rstd.cpu(), (grouped.var(-1, unbiased=False) + 1e-5).rsqrt()) <= 1e-5
