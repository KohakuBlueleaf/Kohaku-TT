"""Regression: large 1-D float tensors are stored Flat as [n / w, w] tiles, not one
tile row padded to 32 rows; elementwise ops, casts, fills and reductions keep them so,
other ops read the tile form. Values equal CPU within device precision, eager and
replayed."""

import pytest
import torch

import kohakutt

N = 65536  # 64 whole tiles
ODD = 65536 + 5 * 32  # a partly filled last tile
SIZES = [N, ODD]


def layout(t):
    return torch.ops.ktt._debug_layout(t)


def expansions():
    return torch.ops.ktt._debug_layout_expansions()


def sample(n, seed, dtype=torch.bfloat16, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) + shift).to(dtype)


def assert_near(got, want, tol, name):
    """max |got - want| at most tol times max |want|."""
    got, want = got.cpu().double(), want.double()
    assert got.shape == want.shape, f"{name}: shape {got.shape} vs {want.shape}"
    err = (got - want).abs().max().item()
    scale = want.abs().max().item()
    assert err <= tol * max(scale, 1e-6), f"{name}: error {err} at scale {scale}"


def plain(x, monkeypatch):
    """x uploaded in the tile form (KTT_FLAT_1D=0)."""
    monkeypatch.setenv("KTT_FLAT_1D", "0")
    t = x.to("tt")
    monkeypatch.delenv("KTT_FLAT_1D")
    assert layout(t) == "nchw"
    return t


@pytest.mark.hw
@pytest.mark.parametrize(
    "n, dtype, flat",
    [
        (N, torch.bfloat16, True),
        (ODD, torch.float32, True),
        (1 << 20, torch.bfloat16, True),
        (N - 32, torch.bfloat16, False),  # below the size rule
        (N + 1, torch.bfloat16, False),  # not a multiple of 32
        (N, torch.int32, False),
    ],
    ids=str,
)
def test_vector_upload_roundtrip(native, monkeypatch, n, dtype, flat):
    x = sample(n, 0).to(dtype) if dtype != torch.int32 else torch.arange(n, dtype=dtype)
    kohakutt.synchronize()
    before = torch.ops.ktt.memory_stats()[0]
    t = x.to("tt")
    grown = torch.ops.ktt.memory_stats()[0] - before
    assert layout(t) == ("flat" if flat else "nchw")
    e = expansions()
    assert torch.equal(t.cpu(), x)
    assert expansions() == e
    if flat:
        tiles = -(-n // 1024)
        assert grown == tiles * 1024 * x.element_size()  # not 32 rows of n
    assert layout(plain(x, monkeypatch)) == "nchw"


@pytest.mark.hw
@pytest.mark.parametrize("n", SIZES, ids=str)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_vector_elementwise_and_casts(native, monkeypatch, n, dtype):
    a, b = sample(n, 1, dtype), sample(n, 2, dtype, shift=3.0)
    ta, tb = a.to("tt"), b.to("tt")
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-3
    cases = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "maximum": torch.maximum,
        "scalar": lambda x, y: x * 1.5 + 2.0,
        "zero_dim": lambda x, y: x * y.sum(),
        "neg_abs": lambda x, y: (-x).abs(),
        "exp": lambda x, y: x.exp(),
        "sigmoid": lambda x, y: torch.sigmoid(x),
        "relu": lambda x, y: torch.relu(x),
        "where": lambda x, y: torch.where(x > 0, x, y),
        "addcmul": lambda x, y: torch.addcmul(x, x, y, value=0.5),
        "clamp": lambda x, y: x.clamp(-0.5, 0.5),
    }
    e = expansions()
    for name, fn in cases.items():
        got = fn(ta, tb)
        assert_near(got, fn(a, b), tol, name)
    for name in ("add", "mul", "scalar", "zero_dim", "exp", "where", "clamp"):
        assert layout(cases[name](ta, tb)) == "flat", name
    assert expansions() == e
    other = torch.float32 if dtype == torch.bfloat16 else torch.bfloat16
    cast = ta.to(other)
    assert layout(cast) == "flat"
    assert torch.equal(cast.cpu(), plain(a, monkeypatch).to(other).cpu())
    assert_near(cast, a.to(other), 1e-2, "cast")
    assert torch.equal(ta.clone().cpu(), a) and layout(ta.clone()) == "flat"


@pytest.mark.hw
@pytest.mark.parametrize("n", SIZES, ids=str)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_vector_reductions(native, n, dtype):
    x = sample(n, 3, dtype, shift=-2.0)  # mostly negative: zero padding would show
    t = x.to("tt")
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    e = expansions()
    assert_near(t.sum(), x.double().sum(), tol, "sum")
    assert_near(t.mean(), x.double().mean(), tol, "mean")
    assert_near(t.sum(0, keepdim=True), x.double().sum(0, keepdim=True), tol, "sum0")
    assert torch.equal(t.amax().cpu(), x.amax())
    assert torch.equal(t.amin().cpu(), x.amin())
    assert torch.equal(t.max().cpu(), x.max())
    assert expansions() == e
    assert int(t.argmax()) == int(x.argmax())
    assert_near(t.norm(), x.double().norm(), tol, "norm")
    assert_near(t.var(), x.double().var(), 5 * tol, "var")
    assert_near(t.view(-1, 32).sum(1), x.double().view(-1, 32).sum(1), tol, "rows")


@pytest.mark.hw
@pytest.mark.parametrize("n", SIZES, ids=str)
def test_vector_cat_stack_and_indexing(native, n):
    a, b = sample(n, 4), sample(n, 5)
    ta, tb = a.to("tt"), b.to("tt")
    assert torch.equal(torch.cat([ta, tb]).cpu(), torch.cat([a, b]))
    assert torch.equal(torch.stack([ta, tb]).cpu(), torch.stack([a, b]))
    assert torch.equal(torch.stack([ta, tb], -1).cpu(), torch.stack([a, b], -1))
    assert torch.equal(ta[100:4000].cpu(), a[100:4000])
    assert torch.equal(ta[1::2].cpu(), a[1::2])
    assert torch.equal(ta[-1].cpu(), a[-1])
    idx = torch.randint(0, n, (777,), generator=torch.Generator().manual_seed(6))
    assert torch.equal(ta[idx.to("tt")].cpu(), a[idx])
    assert torch.equal(ta.index_select(0, idx.to("tt")).cpu(), a.index_select(0, idx))
    assert torch.equal(ta.flip(0).cpu(), a.flip(0))


@pytest.mark.hw
def test_vector_views_and_reshapes(native):
    x = sample(N, 7)
    t = x.to("tt")
    for shape in [(256, 256), (-1, 32), (2048, 32), (64, 1024), (1, N), (2, 4, 8192)]:
        assert torch.equal(t.view(shape).cpu(), x.view(shape)), shape
        assert torch.equal(t.reshape(shape).cpu(), x.reshape(shape)), shape
    assert torch.equal(t.unsqueeze(0).cpu(), x.unsqueeze(0))
    assert torch.equal(t.view(256, 256).t().contiguous().cpu(), x.view(256, 256).t())
    m = x.view(256, 256).to("tt")
    assert torch.equal(m.view(-1).cpu(), x)
    mm = (t.view(256, 256) @ m).cpu()
    assert torch.equal(mm, (x.view(256, 256).to("tt") @ m).cpu())
    lin = torch.nn.functional.linear(m, m, t[:256])
    want = torch.nn.functional.linear(m, m, x[:256].to("tt"))
    assert torch.equal(lin.cpu(), want.cpu())


@pytest.mark.hw
@pytest.mark.parametrize("n", SIZES, ids=str)
def test_vector_in_place(native, n):
    a, b = sample(n, 8), sample(n, 9)
    ta, tb = a.clone().to("tt"), b.to("tt")
    ta.add_(tb).mul_(0.5)
    assert layout(ta) == "flat"
    assert_near(ta, (a.float() + b.float()) * 0.5, 1e-2, "add_ mul_")
    a = ta.cpu()  # device rounding of add_ kept; the writes below are exact
    ta[10:2000] = 1.0
    a[10:2000] = 1.0
    assert torch.equal(ta.cpu(), a)
    ta.view(-1, 32)[:, 5:9] = tb.view(-1, 32)[:, :4]
    a.view(-1, 32)[:, 5:9] = b.view(-1, 32)[:, :4]
    assert torch.equal(ta.cpu(), a)
    ta.copy_(tb)
    assert torch.equal(ta.cpu(), b)
    ta.fill_(2.0)
    assert torch.equal(ta.cpu(), torch.full_like(a, 2.0))
    ta.zero_()
    assert torch.equal(ta.cpu(), torch.zeros_like(a))
    e = torch.empty(n, dtype=torch.bfloat16, device="tt")
    e.copy_(tb)
    assert torch.equal(e.cpu(), b)


@pytest.mark.hw
def test_vector_capture_replay(native):
    xs = [sample(ODD, 10 + k).to("tt") for k in range(3)]
    w = sample(ODD, 20).to("tt")

    def make():
        acc = torch.zeros(ODD, dtype=torch.bfloat16).to("tt")

        def step(x):
            acc.mul_(0.5).add_(x * w)
            y = torch.sigmoid(acc) + x.float().bfloat16()
            return y, y.sum(), y.view(-1, 32).amax(1)

        return acc, step

    acc, step = make()
    ref_acc, ref_step = make()
    assert layout(xs[0]) == "flat" and layout(acc) == "flat"
    captured = kohakutt.capture(step, xs[0], warmup=2)
    for _ in range(2):
        ref_step(xs[0])  # the warm-up calls
    state = ref_acc.cpu().float()
    for x in xs:
        got = [v.cpu() for v in captured(x)]
        want = [v.cpu() for v in ref_step(x)]
        assert torch.equal(acc.cpu(), ref_acc.cpu())
        for g, v in zip(got, want, strict=True):
            assert_near(g, v, 1e-2, "replay")
        state = state * 0.5 + x.cpu().float() * w.cpu().float()
        y = torch.sigmoid(state) + x.cpu().float()
        assert_near(got[0], y, 2e-2, "cpu")
    captured.release()
