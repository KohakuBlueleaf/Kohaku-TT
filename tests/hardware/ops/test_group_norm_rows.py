"""Group norm on channels-last storage (KTT_NHWC=1) and on [N, C, L] inputs agrees with
float64, keeps the storage format with zero tile padding and runs two programs; with
gradients it writes mean / rstd and its backward stays channels-last; shapes it does not
take give the same values on the other routes."""

import copy

import pytest
import torch
import torch.nn.functional as F

import kohakutt

# (shape, groups): group sizes 1 to 32, C and the pixel count off the tile size, two and
# three images.
KEPT = [
    ((1, 64, 8, 8), 32),
    ((1, 48, 7, 9), 3),
    ((2, 64, 8, 8), 4),
    ((3, 96, 16, 16), 96),
    ((1, 32, 5, 5), 1),
    ((1, 128, 40, 24), 32),
    # Groups crossing channel tiles: 3, 10, 40 and 64 channels per group.
    ((1, 96, 8, 8), 32),
    ((2, 320, 8, 8), 32),
    ((1, 80, 8, 8), 2),
    ((1, 128, 8, 8), 2),
    ((1, 330, 5, 7), 33),
    # Images sharing pixel tiles: 63, 36 and 1023 pixels per image.
    ((2, 64, 7, 9), 8),
    ((3, 96, 6, 6), 32),
    ((4, 33, 7, 9), 3),
    ((3, 64, 33, 31), 32),
]
# Several images of under 32 pixels; groups reaching more than 8 channel tiles.
CONVERTED = [((2, 64, 5, 5), 32), ((1, 1024, 4, 8), 2)]


def layout(t):
    return torch.ops.ktt._debug_layout(t)


def sample(shape, groups, dtype, affine):
    g = torch.Generator().manual_seed(sum(shape) + groups)
    x = (torch.randn(shape, generator=g) * 1.5 + 0.7).to(dtype)
    if not affine:
        return x, None, None
    c = shape[1]
    return (
        x,
        (torch.randn(c, generator=g) * 0.5 + 1).to(dtype),
        torch.randn(c, generator=g).to(dtype),
    )


def reference(x, groups, w, b):
    """(float64 result, relative error of CPU's result in x's dtype)."""
    opt = lambda t, cast: None if t is None else cast(t)  # noqa: E731
    ref = F.group_norm(
        x.double(), groups, opt(w, torch.Tensor.double), opt(b, torch.Tensor.double)
    )
    cpu = F.group_norm(
        x.float(), groups, opt(w, torch.Tensor.float), opt(b, torch.Tensor.float)
    )
    return ref, rel(cpu.to(x.dtype), ref)


def rel(got, ref):
    return ((got.double() - ref).norm() / ref.norm()).item()


def on_device(t):
    return None if t is None else t.to("tt")


def as_nhwc(x):
    t = x.to("tt")
    torch.ops.ktt._debug_to_nhwc(t)
    return t


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("affine", [True, False], ids=["affine", "plain"])
@pytest.mark.parametrize("shape,groups", KEPT, ids=lambda v: str(v).replace(" ", ""))
def test_channels_last_group_norm(native, monkeypatch, shape, groups, affine, dtype):
    monkeypatch.setenv("KTT_NHWC", "1")
    x, w, b = sample(shape, groups, dtype, affine)
    ref, cpu_rel = reference(x, groups, w, b)
    xt, wt, bt = as_nhwc(x), on_device(w), on_device(b)
    torch.tt.synchronize()
    before = kohakutt.counters()["programs"]
    conversions = torch.ops.ktt._debug_layout_conversions()
    with torch.no_grad():
        y = F.group_norm(xt, groups, wt, bt)
    torch.tt.synchronize()
    assert kohakutt.counters()["programs"] - before == 2
    assert torch.ops.ktt._debug_layout_conversions() == conversions
    assert layout(y) == "nhwc" and torch.ops.ktt._debug_padding_max(y) == 0
    assert rel(y.cpu(), ref) <= max(3 * cpu_rel, 2e-6)
    # The result feeds a channels-last op as stored.
    assert layout(F.silu(y)) == "nhwc"


@pytest.mark.hw
@pytest.mark.parametrize(
    "shape,groups", CONVERTED, ids=lambda v: str(v).replace(" ", "")
)
def test_other_shapes_take_the_plane_kernels(native, monkeypatch, shape, groups):
    monkeypatch.setenv("KTT_NHWC", "1")
    x, w, b = sample(shape, groups, torch.bfloat16, True)
    ref, cpu_rel = reference(x, groups, w, b)
    with torch.no_grad():
        y = F.group_norm(as_nhwc(x), groups, on_device(w), on_device(b))
    assert rel(y.cpu(), ref) <= 3 * cpu_rel


@pytest.mark.hw
@pytest.mark.parametrize("shape,groups", [((1, 64, 96), 32), ((2, 32, 64), 4)])
def test_three_dim_input(native, shape, groups):
    """[N, C, L] with L and C multiples of 32: through its [N, L, C] transpose."""
    x, w, b = sample(shape, groups, torch.bfloat16, True)
    ref, cpu_rel = reference(x, groups, w, b)
    xt, wt, bt = x.to("tt"), on_device(w), on_device(b)
    torch.tt.synchronize()
    before = kohakutt.counters()["programs"]
    with torch.no_grad():
        y = F.group_norm(xt, groups, wt, bt)
    torch.tt.synchronize()
    assert kohakutt.counters()["programs"] - before == 2
    assert y.shape == x.shape
    assert rel(y.cpu(), ref) <= 3 * cpu_rel
    # The model's own transpose of the result is the stored tensor.
    assert rel(y.transpose(1, 2).contiguous().cpu(), ref.transpose(1, 2)) <= 3 * cpu_rel


@pytest.mark.hw
def test_three_dim_input_off_the_tile_size(native):
    x, w, b = sample((2, 48, 10), 3, torch.bfloat16, True)
    ref, cpu_rel = reference(x, 3, w, b)
    with torch.no_grad():
        y = F.group_norm(x.to("tt"), 3, on_device(w), on_device(b))
    assert rel(y.cpu(), ref) <= 3 * cpu_rel


@pytest.mark.hw
@pytest.mark.parametrize(
    "shape,groups",
    [((1, 64, 8, 8), 32), ((2, 320, 7, 9), 32), ((3, 48, 8, 8), 3)],
    ids=lambda v: str(v).replace(" ", ""),
)
def test_with_gradients_keeps_channels_last_and_writes_statistics(
    native, monkeypatch, shape, groups
):
    """A call that needs gradients stays channels-last, returns mean and rstd [N, G],
    and its backward matches CPU's."""
    monkeypatch.setenv("KTT_NHWC", "1")
    x, w, b = sample(shape, groups, torch.bfloat16, True)
    n, c = shape[:2]
    xc = x.float().requires_grad_()
    F.group_norm(xc, groups, w.float(), b.float()).square().sum().backward()
    xt = as_nhwc(x).requires_grad_()
    wt, bt = on_device(w), on_device(b)
    y = F.group_norm(xt, groups, wt, bt)
    assert layout(y) == "nhwc" and torch.ops.ktt._debug_padding_max(y) == 0
    y.square().sum().backward()
    assert layout(xt.grad) == "nhwc" and torch.ops.ktt._debug_padding_max(xt.grad) == 0
    assert rel(xt.grad.cpu(), xc.grad.double()) <= 3e-2
    _, mean, rstd = torch.native_group_norm(
        xt, wt, bt, n, c, shape[2] * shape[3], groups, 1e-5
    )
    grouped = x.double().reshape(n, groups, -1)
    assert rel(mean.cpu(), grouped.mean(-1)) <= 1e-2
    assert rel(rstd.cpu(), (grouped.var(-1, unbiased=False) + 1e-5).rsqrt()) <= 1e-2


@pytest.mark.hw
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("affine", [True, False], ids=["affine", "plain"])
@pytest.mark.parametrize(
    "shape,groups",
    [
        ((1, 64, 8, 8), 32),
        ((1, 48, 7, 9), 3),
        ((2, 64, 7, 9), 8),
        ((3, 96, 6, 6), 32),
        ((2, 320, 10, 10), 32),
    ],
    ids=lambda v: str(v).replace(" ", ""),
)
def test_channels_last_backward(native, monkeypatch, shape, groups, affine, dtype):
    """Input, weight and bias gradients for a random upstream gradient against float64;
    the input gradient stays channels-last, with no conversion."""
    monkeypatch.setenv("KTT_NHWC", "1")
    x, w, b = sample(shape, groups, dtype, affine)
    up = torch.randn(shape, generator=torch.Generator().manual_seed(1)).to(dtype)
    leaf = lambda t: None if t is None else t.double().requires_grad_()  # noqa: E731
    xd, wd, bd = leaf(x), leaf(w), leaf(b)
    F.group_norm(xd, groups, wd, bd).backward(up.double())
    xt, ut = as_nhwc(x).requires_grad_(), as_nhwc(up)
    wt = None if w is None else w.to("tt").requires_grad_()
    bt = None if b is None else b.to("tt").requires_grad_()
    y = F.group_norm(xt, groups, wt, bt)
    torch.tt.synchronize()
    conversions = torch.ops.ktt._debug_layout_conversions()
    y.backward(ut)
    torch.tt.synchronize()
    assert torch.ops.ktt._debug_layout_conversions() == conversions
    assert layout(xt.grad) == "nhwc" and torch.ops.ktt._debug_padding_max(xt.grad) == 0
    # The saved mean and rstd are fp32: bf16 gradients are off by their own rounding.
    tol = 2e-5 if dtype == torch.float32 else 5e-3
    assert rel(xt.grad.cpu(), xd.grad) <= tol
    if affine:
        assert rel(wt.grad.cpu(), wd.grad) <= tol
        assert rel(bt.grad.cpu(), bd.grad) <= tol


@pytest.mark.hw
def test_training_steps_channels_last(native, monkeypatch):
    """conv -> group norm -> SiLU -> conv, four SGD steps: the losses follow CPU's and
    the group norm's output and input gradient are channels-last."""
    monkeypatch.setenv("KTT_NHWC", "1")
    torch.manual_seed(0)
    cpu = torch.nn.Sequential(
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.GroupNorm(8, 64),
        torch.nn.SiLU(),
        torch.nn.Conv2d(64, 32, 3, padding=1),
    )
    # Both start from the same bf16 parameter values.
    cpu = cpu.bfloat16().float()
    dev = copy.deepcopy(cpu).bfloat16().to("tt")
    x = torch.randn(2, 32, 16, 16).bfloat16()
    target = torch.randn(2, 32, 16, 16).bfloat16()
    layouts = []
    dev[1].register_forward_hook(lambda m, i, o: layouts.append(layout(o)))
    dev[1].register_full_backward_hook(lambda m, gi, go: layouts.append(layout(gi[0])))

    def steps(model, x, target):
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        losses = []
        for _ in range(4):
            opt.zero_grad()
            loss = (model(x) - target).square().mean()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        return losses

    want = steps(cpu, x.float(), target.float())
    got = steps(dev, x.to("tt"), target.to("tt"))
    assert layouts == ["nhwc"] * 8
    assert got[-1] < got[0]
    for a, b in zip(got, want):
        assert abs(a - b) <= 5e-2 * abs(b)


@pytest.mark.hw
def test_repeat_calls_are_bit_identical(native, monkeypatch):
    monkeypatch.setenv("KTT_NHWC", "1")
    x, w, b = sample((1, 128, 40, 24), 32, torch.bfloat16, True)
    xt, wt, bt = as_nhwc(x), on_device(w), on_device(b)
    with torch.no_grad():
        first = F.group_norm(xt, 32, wt, bt).cpu()
        for _ in range(3):
            assert torch.equal(F.group_norm(xt, 32, wt, bt).cpu(), first)
