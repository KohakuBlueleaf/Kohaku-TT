"""Regression: tensors with small trailing dims (conv weights) are stored Flat, as
[d0, d1 * ... * dk] tiles: uploads and downloads are exact, DRAM holds their values, and
conv routes, gradients, AdamW and clip_grad_norm keep them Flat with no expansion, bit
for bit equal to KTT_FLAT=0; reads that need the per-plane form expand, counted."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import kohakutt


def layout(t):
    return torch.ops.ktt._debug_layout(t)


def expansions():
    return torch.ops.ktt._debug_layout_expansions()


def sample(shape, seed, dtype=torch.bfloat16, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(dtype)


def assert_near(got, want, tol, name):
    """max |got - want| at most tol times max |want|."""
    err = (got.double() - want.double()).abs().max().item()
    scale = want.double().abs().max().item()
    assert err <= tol * max(scale, 1e-6), f"{name}: error {err} at scale {scale}"


@pytest.mark.hw
@pytest.mark.parametrize(
    "shape, dtype, flat",
    [
        ((512, 512, 3, 3), torch.bfloat16, True),
        ((64, 3, 7, 7), torch.float32, True),
        ((64, 8, 1, 1), torch.bfloat16, True),
        ((32, 16, 3), torch.bfloat16, True),
        ((16, 8, 1, 1), torch.bfloat16, False),
        ((96, 1, 7, 7), torch.bfloat16, False),  # depthwise weight
        ((128, 1, 5), torch.bfloat16, False),  # depthwise 1-D weight
        ((8, 64, 3, 3), torch.bfloat16, False),
        ((2, 3, 4, 5), torch.float32, False),
        ((4, 32, 3), torch.bfloat16, False),
    ],
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
def test_flat_upload_roundtrip(native, monkeypatch, shape, dtype, flat):
    x = sample(shape, 0, dtype)
    kohakutt.synchronize()
    before = torch.ops.ktt.memory_stats()[0]
    t = x.to("tt")
    grown = torch.ops.ktt.memory_stats()[0] - before
    assert layout(t) == ("flat" if flat else "nchw")
    e = expansions()
    assert torch.equal(t.cpu(), x)
    assert expansions() == e
    if shape == (512, 512, 3, 3):
        assert grown == 16 * 144 * 2048  # [512, 4608] in tiles, not 262144 tiles
    if shape == (512, 512, 3, 3):
        return  # its KTT_FLAT=0 form is the 537 MB tile form
    monkeypatch.setenv("KTT_FLAT", "0")
    plain = x.to("tt")
    assert layout(plain) == "nchw" and torch.equal(plain.cpu(), x)


@pytest.mark.hw
def test_flat_casts_fills_and_triggers(native):
    w = sample((64, 40, 3, 3), 1)
    t = w.to("tt")
    assert layout(t) == "flat"
    e = expansions()
    f32 = t.float()
    assert layout(f32) == "flat" and torch.equal(f32.cpu(), w.float())
    assert layout(f32.bfloat16()) == "flat"
    z = torch.zeros_like(t)
    assert layout(z) == "flat" and not z.cpu().any()
    assert torch.equal(t.reshape(64, -1).cpu(), w.reshape(64, -1))
    assert_near(f32.sum().cpu(), w.double().sum(), 1e-3, "sum")
    assert torch.equal((t * 2).cpu(), w * 2) and layout(t * 2) == "flat"
    assert expansions() == e
    # Reads that need the per-plane tile form expand once per write epoch.
    assert torch.equal(t[:, :8].cpu(), w[:, :8])
    assert_near(
        t.float().sum(dim=(2, 3)).cpu(), w.double().sum(dim=(2, 3)), 1e-2, "sum23"
    )
    assert expansions() - e == 2  # t (bf16) and f32 (sum over (2, 3)) expanded


@pytest.mark.hw
@pytest.mark.parametrize("c", [40, 64])
def test_flat_tap_major(native, monkeypatch, c):
    """The conv weight matrices: tap-major [KK C, O] and Cin-block-major
    [Ct KK 32, O], from Flat and from Nchw storage."""
    o = 64
    w = sample((o, c, 3, 3), 2)
    tap = w.permute(2, 3, 1, 0).reshape(9 * c, o)
    blocks = (c + 31) // 32
    padded = torch.zeros(o, blocks * 32, 9, dtype=w.dtype)
    padded[:, :c] = w.reshape(o, c, 9)
    blocked = padded.reshape(o, blocks, 32, 9).permute(1, 3, 2, 0).reshape(-1, o)
    for env in ("1", "0"):
        monkeypatch.setenv("KTT_FLAT", env)
        t = w.to("tt")
        assert layout(t) == ("flat" if env == "1" else "nchw")
        assert torch.equal(torch.ops.ktt._debug_tap_major(t, False).cpu(), tap)
        assert torch.equal(torch.ops.ktt._debug_tap_major(t, True).cpu(), blocked)


@pytest.mark.hw
@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((64, 40, 3, 3), torch.bfloat16),
        ((192, 96, 2, 2), torch.float32),
        ((64, 33, 1, 1), torch.bfloat16),
        ((32, 48, 5), torch.float32),
    ],
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
def test_flat_from_taps(native, shape, dtype):
    """Per-tap matrices side by side, [O, taps C] (column t C + c), become the Flat
    tensor [O, C, ...] bit for bit."""
    o, c = shape[:2]
    w = sample(shape, 11, dtype)
    columns = w.reshape(o, c, -1).permute(0, 2, 1).reshape(o, -1)
    got = torch.ops.ktt._debug_flat_from_taps(columns.to("tt"), list(shape))
    assert layout(got) == "flat"
    assert torch.equal(got.cpu(), w)


# (kind, x sizes, weight sizes, conv arguments)
COMPOSITE = [
    ("downsample", (1, 96, 28, 28), (192, 96, 2, 2), dict(stride=2)),
    ("strided", (1, 64, 16, 16), (64, 64, 3, 3), dict(stride=2, padding=1)),
    ("dense", (1, 64, 16, 16), (64, 64, 3, 3), dict(padding=1)),
    ("ragged", (2, 40, 9, 11), (48, 40, 3, 3), dict(padding=1, dilation=2)),
    ("fp32", (1, 40, 9, 11), (64, 40, 3, 3), dict(padding=1)),
    ("transposed", (1, 64, 8, 8), (64, 32, 3, 3), dict(stride=2, padding=1)),
]


@pytest.mark.hw
@pytest.mark.parametrize("kind, xs, ws, args", COMPOSITE, ids=[c[0] for c in COMPOSITE])
def test_flat_composite_gradients(native, monkeypatch, kind, xs, ws, args):
    """Convs whose gradients (and, for fp32 and transposed convs, forward) run as
    per-tap matmuls, with a Flat weight: the taps are read from the Flat rows and the
    weight gradient is written as Flat rows, with no expansion; output and gradients
    match float64 and are bit for bit equal to KTT_FLAT=0."""
    dtype = torch.float32 if kind == "fp32" else torch.bfloat16
    conv = F.conv_transpose2d if kind == "transposed" else F.conv2d
    x = sample(xs, 12, dtype)
    w = sample(ws, 13, dtype, scale=(ws[1] * ws[2] * ws[3]) ** -0.5)
    xd, wd = x.double().requires_grad_(), w.double().requires_grad_()
    ref = conv(xd, wd, **args)
    gy = sample(ref.shape, 14, dtype)
    ref.backward(gy.double())
    got = {}
    for env in ("1", "0"):
        monkeypatch.setenv("KTT_FLAT", env)
        xt = x.to("tt").requires_grad_()
        wt = w.to("tt").requires_grad_()
        assert layout(wt) == ("flat" if env == "1" else "nchw")
        e = expansions()
        out = conv(xt, wt, **args)
        out.backward(gy.to("tt"))
        if env == "1":
            assert expansions() == e, f"{expansions() - e} expansions"
            assert layout(wt.grad) == "flat"
        got[env] = (out.detach().cpu(), xt.grad.cpu(), wt.grad.cpu())
    names = ("forward", "input grad", "weight grad")
    refs = (ref.detach(), xd.grad, wd.grad)
    for name, a, b, want in zip(names, got["1"], got["0"], refs, strict=True):
        assert_near(a, want, 5e-2, name)
        assert torch.equal(a, b), name


def conv_case(stride, groups, c, o, hw, seed):
    x = sample((1, c, hw, hw), seed)
    w = sample((o, c // groups, 3, 3), seed + 1, scale=(9 * c / groups) ** -0.5)
    return x, w, dict(stride=stride, padding=1, groups=groups)


@pytest.mark.hw
@pytest.mark.parametrize(
    "stride, groups, c, o, hw",
    [(1, 1, 512, 512, 8), (2, 1, 512, 512, 16), (1, 2, 64, 64, 16)],
    ids=["ktt_conv", "ttnn_conv", "grouped"],
)
def test_flat_conv_routes(native, stride, groups, c, o, hw):
    """Forward and weight gradient with a Flat weight: no expansion on the dense
    routes (one, cached per write epoch, on Kohaku-TT's grouped conv); the gradient
    is Flat; values match CPU float64."""
    x, w, args = conv_case(stride, groups, c, o, hw, 3)
    xt = x.to("tt")
    wt = w.to("tt").requires_grad_()
    assert layout(wt) == "flat"
    e = expansions()
    out = F.conv2d(xt, wt, **args)
    F.conv2d(xt, wt, **args)
    (out.float() ** 2).mean().backward()
    grown = expansions() - e
    assert grown == (1 if groups > 1 else 0)
    assert layout(wt.grad) == "flat"
    xd, wd = x.double(), w.double().requires_grad_()
    ref = F.conv2d(xd, wd, **args)
    (ref**2).mean().backward()
    assert_near(out.cpu(), ref.detach(), 3e-2, "forward")
    assert_near(wt.grad.cpu(), wd.grad, 5e-2, "weight grad")


def train(model, x, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    norms = []
    for _ in range(steps):
        loss = (model(x).float() ** 2).mean()
        loss.backward()
        norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6).cpu())
        opt.step()
        opt.zero_grad()
    return opt, norms


@pytest.mark.hw
def test_flat_adamw_clip_bit_identical(native, monkeypatch):
    """Three AdamW steps with clip_grad_norm_ on a 256 -> 256 3x3 conv: Flat weight,
    gradient and optimizer state, no expansion, parameters bit for bit equal to
    KTT_FLAT=0."""
    torch.manual_seed(0)
    cpu = nn.Conv2d(256, 256, 3, padding=1).bfloat16()
    x = sample((1, 256, 8, 8), 4).to("tt")
    runs = {}
    for env in ("1", "0"):
        monkeypatch.setenv("KTT_FLAT", env)
        model = nn.Conv2d(256, 256, 3, padding=1).bfloat16()
        model.load_state_dict(cpu.state_dict())
        model = model.to("tt")
        e = expansions()
        opt, norms = train(model, x, 3)
        runs[env] = (model, opt, norms, expansions() - e)
    flat, plain = runs["1"], runs["0"]
    weight = flat[0].weight
    state = flat[1].state[weight]
    assert layout(weight) == "flat"
    assert layout(state["exp_avg"]) == "flat" and layout(state["exp_avg_sq"]) == "flat"
    assert flat[3] == 0, f"{flat[3]} expansions"
    for name in ("weight", "bias"):
        a = getattr(flat[0], name).cpu()
        b = getattr(plain[0], name).cpu()
        assert torch.equal(a, b), name
    for a, b in zip(flat[2], plain[2], strict=True):
        assert_near(a, b.double(), 1e-2, "clip norm")


def nhwc_runs():
    """Channels-last convs run so far (program launches and 1x1 matmuls)."""
    kohakutt.synchronize()
    return torch.ops.ktt._debug_nhwc_conv_runs()


def nhwc_fallbacks():
    """Channels-last convs sent to another conv route so far."""
    kohakutt.synchronize()
    return torch.ops.ktt._debug_nhwc_conv_fallbacks()


def rel(got, ref):
    """Relative RMS error of got against float64 ref."""
    return ((got.double() - ref).norm() / ref.norm()).item()


@pytest.mark.hw
@pytest.mark.parametrize(
    "shape",
    [(1, 64, 15, 17, 64, 3), (2, 33, 7, 9, 40, 3), (1, 40, 9, 11, 40, 1)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_flat_channels_last_conv(native, monkeypatch, shape):
    """KTT_NHWC=1, x [N, C, H, W] with a Flat weight [O, C, K, K]: the forward and the
    input gradient both run on the channels-last conv with no expansion, the weight
    gradient is Flat, and all three are within 3x (weight gradient 4x) CPU bf16's error
    of float64 and bit for bit equal to KTT_FLAT=0."""
    monkeypatch.setenv("KTT_NHWC", "1")
    n, c, h, w, o, k = shape
    args = dict(padding=(k - 1) // 2)
    x = sample((n, c, h, w), 5)
    wt = sample((o, c, k, k), 6, scale=(c * k * k) ** -0.5)
    gy = sample((n, o, h, w), 7)
    grad = torch.nn.grad

    def reference(cast):
        y = F.conv2d(cast(x), cast(wt), **args)
        gx = grad.conv2d_input(x.shape, cast(wt), cast(gy), **args)
        gw = grad.conv2d_weight(cast(x), wt.shape, cast(gy), **args)
        return y, gx, gw

    refs = reference(torch.Tensor.double)
    cpu = [t.bfloat16() for t in reference(torch.Tensor.float)]
    got = {}
    for env in ("1", "0"):
        monkeypatch.setenv("KTT_FLAT", env)
        xt = x.to("tt").requires_grad_()
        wtt = wt.to("tt").requires_grad_()
        assert layout(wtt) == ("flat" if env == "1" else "nchw")
        runs, fallbacks, e = nhwc_runs(), nhwc_fallbacks(), expansions()
        y = F.conv2d(xt, wtt, **args)
        y.backward(gy.to("tt"))
        assert nhwc_runs() - runs == 2, "forward and input gradient did not both run"
        assert nhwc_fallbacks() == fallbacks
        if env == "1":
            assert expansions() == e
            assert layout(wtt.grad) == "flat"
        got[env] = (y.detach().cpu(), xt.grad.cpu(), wtt.grad.cpu())
    names = ("forward", "input grad", "weight grad")
    for name, bound, a, b, ref, c16 in zip(
        names, (3, 3, 4), got["1"], got["0"], refs, cpu, strict=True
    ):
        assert rel(a, ref) <= bound * rel(c16, ref), (name, rel(a, ref), rel(c16, ref))
        assert torch.equal(a, b), name


@pytest.mark.hw
def test_flat_channels_last_adamw(native, monkeypatch):
    """KTT_NHWC=1, three AdamW steps through two 64 -> 64 3x3 convs: each step's two
    forwards and one input gradient run on the channels-last conv, reading the weights
    the step before wrote; weights and optimizer state stay Flat with no expansion; the
    parameters are bit for bit equal to KTT_FLAT=0."""
    monkeypatch.setenv("KTT_NHWC", "1")
    torch.manual_seed(0)
    cpu = nn.Sequential(
        nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1)
    ).bfloat16()
    x = sample((1, 64, 15, 17), 8)
    results = {}
    for env in ("1", "0"):
        monkeypatch.setenv("KTT_FLAT", env)
        model = copy.deepcopy(cpu).to("tt")
        xt = x.to("tt")
        runs, fallbacks, e = nhwc_runs(), nhwc_fallbacks(), expansions()
        opt, _ = train(model, xt, 3)
        assert nhwc_runs() - runs == 9
        assert nhwc_fallbacks() == fallbacks
        out = model(xt).detach().cpu()
        results[env] = (model, opt, expansions() - e, out)
    flat, plain = results["1"], results["0"]
    for conv in (flat[0][0], flat[0][2]):
        state = flat[1].state[conv.weight]
        for t in (conv.weight, state["exp_avg"], state["exp_avg_sq"]):
            assert layout(t) == "flat"
    assert flat[2] == 0, f"{flat[2]} expansions"
    trained = {name: t.cpu() for name, t in flat[0].state_dict().items()}
    for name, t in plain[0].state_dict().items():
        assert torch.equal(trained[name], t.cpu()), name
    assert not torch.equal(trained["0.weight"], cpu.state_dict()["0.weight"])
    # The last forward read the trained weights.
    reference = copy.deepcopy(cpu)
    reference.load_state_dict(trained)
    ref = reference.double()(x.double())
    assert rel(flat[3], ref) <= 5e-2, rel(flat[3], ref)
    assert torch.equal(flat[3], plain[3])


@pytest.mark.hw
def test_flat_activation_reads(native, monkeypatch):
    """A Flat activation ([64, 4, 8, 8] from host) read by TT-NN's conv route, nearest
    upsampling and, under KTT_NHWC=1, the channels-last conv: one expansion (cached),
    values as on CPU."""
    x = sample((64, 4, 8, 8), 9)
    w = sample((8, 4, 3, 3), 10, scale=1 / 6)
    xt, wt = x.to("tt"), w.to("tt")
    assert layout(xt) == "flat" and layout(wt) == "nchw"
    e = expansions()
    ref = F.conv2d(x.double(), w.double(), padding=1)
    assert_near(F.conv2d(xt, wt, padding=1).cpu(), ref, 3e-2, "conv")
    up = F.interpolate(xt, scale_factor=2, mode="nearest").cpu()
    assert torch.equal(
        up, F.interpolate(x.float(), scale_factor=2, mode="nearest").bfloat16()
    )
    monkeypatch.setenv("KTT_NHWC", "1")
    runs = nhwc_runs() + nhwc_fallbacks()
    assert_near(F.conv2d(xt, wt, padding=1).cpu(), ref, 3e-2, "channels-last conv")
    assert nhwc_runs() + nhwc_fallbacks() == runs + 1
    assert expansions() - e == 1
