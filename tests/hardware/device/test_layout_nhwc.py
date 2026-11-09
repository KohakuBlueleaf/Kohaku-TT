"""Regression: Nhwc tensors through Nchw ops, views, writes and copies equal Nchw
results bit for bit; the conversion is cached per write epoch. Under KTT_NHWC=1 conv,
elementwise ops and upsampling keep Nhwc with zero tile padding, converting only at
group norm, and match KTT_NHWC=0 (exact ops bit for bit)."""

import pytest
import torch
import torch.nn.functional as F

SHAPES = [(1, 3, 17, 17), (2, 64, 56, 56), (1, 320, 8, 8), (4, 33, 7, 9)]
GROUPS = {3: 3, 64: 8, 320: 32, 33: 3}
COUT = 32
# Reads whose route depends on the input's format (an Nhwc input runs the channels-last
# conv): compared to the reference, not bit for bit to the Nchw result.
FORMAT_ROUTED = {"conv2d"}


def layout(t):
    return torch.ops.ktt._debug_layout(t)


def conversions():
    return torch.ops.ktt._debug_layout_conversions()


def sample(shape, seed, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


def as_nhwc(x):
    """A tt tensor holding x, its storage converted to Nhwc."""
    t = x.to("tt")
    torch.ops.ktt._debug_to_nhwc(t)
    assert layout(t) == "nhwc"
    return t


def assert_near(got, want, tol, name):
    """max |got - want| at most tol times max |want|."""
    err = (got.double() - want).abs().max().item()
    scale = want.abs().max().item()
    assert err <= tol * max(scale, 1e-6), f"{name}: error {err} at scale {scale}"


def reads(shape, w, b, gamma, beta):
    """(name, tt op of (a, other), float64 reference of (x, y) or None for exact).

    Exact entries compare against the reference applied to the bf16 CPU values."""
    n, c, h, wd = shape
    g = GROUPS[c]
    return [
        ("cpu", lambda a, o: a, None),
        ("add", lambda a, o: a + o, lambda x, y: x + y),
        ("add_rhs", lambda a, o: o + a, lambda x, y: y + x),
        ("add_self", lambda a, o: a + a, lambda x, y: x + x),
        ("mul", lambda a, o: a * o, lambda x, y: x * y),
        ("silu", lambda a, o: F.silu(a), lambda x, y: F.silu(x)),
        ("to_float", lambda a, o: a.float(), None),
        ("sum_hw", lambda a, o: a.sum(dim=(2, 3)), lambda x, y: x.sum(dim=(2, 3))),
        ("mean_c", lambda a, o: a.mean(dim=1), lambda x, y: x.mean(dim=1)),
        (
            "sum_chw",
            lambda a, o: a.sum(dim=(1, 2, 3)),
            lambda x, y: x.sum(dim=(1, 2, 3)),
        ),
        ("permute", lambda a, o: a.permute(0, 2, 3, 1).contiguous(), None),
        ("reshape", lambda a, o: a.reshape(n, c, h * wd), None),
        ("slice", lambda a, o: a[:, 1:, 2:, :-1], None),
        ("narrow_c", lambda a, o: a.narrow(1, c // 2, c - c // 2), None),
        ("select", lambda a, o: a[-1], None),
        (
            "as_strided",
            lambda a, o: a.as_strided((n, c, h - 1, wd - 1), a.stride(), wd + 1),
            None,
        ),
        (
            "conv2d",
            lambda a, o: F.conv2d(a, w[1], b[1], padding=1),
            lambda x, y: F.conv2d(x, w[0], b[0], padding=1),
        ),
        (
            "group_norm",
            lambda a, o: F.group_norm(a, g, gamma[1], beta[1]),
            lambda x, y: F.group_norm(x, g, gamma[0], beta[0]),
        ),
        ("cat_c", lambda a, o: torch.cat([a, o], dim=1), None),
        ("cat_n", lambda a, o: torch.cat([o, a], dim=0), None),
    ]


def exact_ref(name, x, y, shape):
    """The bf16 CPU result of exact op `name`."""
    n, c, h, wd = shape
    return {
        "cpu": lambda: x,
        "to_float": lambda: x.float(),
        "permute": lambda: x.permute(0, 2, 3, 1).contiguous(),
        "reshape": lambda: x.reshape(n, c, h * wd),
        "slice": lambda: x[:, 1:, 2:, :-1],
        "narrow_c": lambda: x.narrow(1, c // 2, c - c // 2),
        "select": lambda: x[-1],
        "as_strided": lambda: x.as_strided((n, c, h - 1, wd - 1), x.stride(), wd + 1),
        "cat_c": lambda: torch.cat([x, y], dim=1),
        "cat_n": lambda: torch.cat([y, x], dim=0),
    }[name]()


def writes(shape):
    """(name, in-place op of (t, other) returning the written tensor, bf16 CPU result
    of (x, y) or None, float64 reference of (x, y) or None)."""

    def partial_add(t, o):
        t[:, :, 1:3].add_(1.0)
        return t

    def partial_add_ref(x, y):
        z = x.clone()
        z[:, :, 1:3] += 1.0
        return z

    def partial_copy(t, o):
        t[:, :1].copy_(o[:, :1])
        return t

    def partial_copy_ref(x, y):
        z = x.clone()
        z[:, :1] = y[:, :1]
        return z

    return [
        ("add_", lambda t, o: t.add_(o), None, lambda x, y: x + y),
        ("partial_add_", partial_add, None, partial_add_ref),
        ("copy_into", lambda t, o: t.copy_(o), lambda x, y: y, None),
        ("partial_copy_into", partial_copy, partial_copy_ref, None),
        ("clone", lambda t, o: t.clone(), lambda x, y: x, None),
    ]


@pytest.mark.hw
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_nhwc_reads_match_nchw(native, shape):
    x, y = sample(shape, 0), sample(shape, 1)
    w = sample((COUT, shape[1], 3, 3), 2) * 0.2
    b = sample((COUT,), 3)
    gamma, beta = sample((shape[1],), 4), sample((shape[1],), 5)
    on_tt = lambda v: (v.double(), v.to("tt"))  # noqa: E731
    params = [on_tt(v) for v in (w, b, gamma, beta)]
    ty = y.to("tt")
    a_nhwc, a_nchw = as_nhwc(x), x.to("tt")
    for name, op, ref in reads(shape, *params):
        got = op(a_nhwc, ty).cpu()
        same = op(a_nchw, ty).cpu()
        if name in FORMAT_ROUTED:
            assert_near(same, ref(x.double(), y.double()), 3e-2, name + " (nchw)")
        else:
            assert torch.equal(got, same), f"{name}: Nhwc and Nchw results differ"
        if ref is None:
            assert torch.equal(got, exact_ref(name, x, y, shape)), name
        else:
            assert_near(got, ref(x.double(), y.double()), 3e-2, name)
        assert layout(a_nhwc) == "nhwc", f"{name} changed the Nhwc storage"

    # In-place writes, each into a fresh Nhwc tensor and the same into an Nchw one.
    for name, op, exact, ref in writes(shape):
        got = op(as_nhwc(x), ty).cpu()
        same = op(x.to("tt"), ty).cpu()
        assert torch.equal(got, same), f"{name}: Nhwc and Nchw results differ"
        if exact is not None:
            assert torch.equal(got, exact(x, y)), name
        else:
            assert_near(got, ref(x.double(), y.double()), 3e-2, name)

    # copy_ from an Nhwc source into an Nchw destination.
    dst = torch.empty(shape, dtype=torch.bfloat16, device="tt")
    dst.copy_(a_nhwc)
    assert torch.equal(dst.cpu(), x)

    # The Nchw operand is untouched, and ops on Nchw tensors convert nothing.
    assert layout(ty) == "nchw" and torch.equal(ty.cpu(), y)
    before = conversions()
    (ty * 2 + ty).sum(dim=1).cpu()
    assert conversions() == before


@pytest.mark.hw
def test_nhwc_conversion_cached_per_write_epoch(native):
    shape = (4, 33, 7, 9)
    x, z = sample(shape, 0), sample(shape, 1)
    t = as_nhwc(x)
    c0 = conversions()
    first = (t + 1).cpu()
    c1 = conversions()
    second = (t * 2).cpu()
    c2 = conversions()
    assert (c1 - c0, c2 - c1) == (1, 0), (c0, c1, c2)
    assert torch.equal(first, (x.to("tt") + 1).cpu())
    assert torch.equal(second, x * 2)

    # An in-place device write (a trace-input copy) keeps the Nhwc buffer; the next read
    # converts again and sees the new values, the read after it does not.
    torch.ops.ktt.copy_into(t, z.to("tt"))
    assert layout(t) == "nhwc"
    c3 = conversions()
    assert torch.equal(t.cpu(), z)
    c4 = conversions()
    assert torch.equal(t.cpu(), z)
    assert (c3 - c2, c4 - c3, conversions() - c4) == (1, 1, 0)


def captured_run(kohakutt, t, o, z):
    """Outputs of a captured step reading and updating static input t in place, over
    three replays, one more after z is copied into t, and t's final values."""

    def step(a, b):
        y = a * 2 + b
        a.add_(1)
        return y

    captured = kohakutt.Captured(step, (t, o), 1, static=[True, False])
    outs = [captured(t, o).cpu() for _ in range(3)]
    torch.ops.ktt.copy_into(t, z)
    outs.append(captured(t, o).cpu())
    captured.release()
    return outs, t.cpu()


@pytest.mark.hw
def test_nhwc_static_input_in_trace(native):
    shape = (4, 33, 7, 9)
    x, y, z = sample(shape, 0), sample(shape, 1), sample(shape, 2)
    t = as_nhwc(x)
    outs, final = captured_run(native, t, y.to("tt"), z.to("tt"))
    assert layout(t) == "nhwc"
    want, want_final = captured_run(native, x.to("tt"), y.to("tt"), z.to("tt"))
    for k, (got, ref) in enumerate(zip(outs, want, strict=True)):
        assert torch.equal(got, ref), f"replay {k}"
    assert torch.equal(final, want_final)
    # The prepare pass adds 1 before the first replay (the capture records only).
    assert_near(outs[0], x.double() * 2 + 2 + y.double(), 3e-2, "replay 0")
    assert_near(outs[3], z.double() * 2 + y.double(), 3e-2, "after copy_into")


@pytest.mark.hw
@pytest.mark.parametrize("shape", [SHAPES[0], SHAPES[3]], ids=["1x3x17x17", "4x33x7x9"])
def test_nhwc_fp32_exact(native, shape):
    x = sample(shape, 6, torch.float32)
    t = as_nhwc(x)
    assert torch.equal(t.cpu(), x)
    assert torch.equal(t.permute(0, 2, 3, 1).contiguous().cpu(), x.permute(0, 2, 3, 1))


# Propagation (KTT_NHWC=1): conv, elementwise ops and nearest upsampling keep Nhwc.
CHAIN_SHAPES = [(1, 64, 16, 16), (2, 32, 7, 9)]


def padding(t):
    return torch.ops.ktt._debug_padding_max(t)


def chain(x, params, group_norm, f=F):
    """conv, (group norm,) silu, conv, + the first conv's output, 2x nearest."""
    w1, b1, w2, b2 = params
    h1 = f.conv2d(x, w1, b1, padding=1)
    h = f.group_norm(h1, 8) if group_norm else h1
    h = f.conv2d(f.silu(h), w2, b2, padding=1) + h1
    return f.interpolate(h, scale_factor=2, mode="nearest")


@pytest.mark.hw
@pytest.mark.parametrize("group_norm", [False, True], ids=["no_gn", "gn"])
@pytest.mark.parametrize("shape", CHAIN_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_nhwc_chain(native, monkeypatch, shape, group_norm):
    c = shape[1]
    x = sample(shape, 0)
    params = [sample((c, c, 3, 3), 1) * 0.05, sample((c,), 2)]
    params += [sample((c, c, 3, 3), 3) * 0.05, sample((c,), 4)]
    tt_params = [p.to("tt") for p in params]
    xt = x.to("tt")
    monkeypatch.setenv("KTT_NHWC", "0")
    base = chain(xt, tt_params, group_norm).cpu()
    monkeypatch.setenv("KTT_NHWC", "1")
    before = conversions()
    out = chain(xt, tt_params, group_norm)
    # Group norm (Kohaku-TT's Nchw kernel) reads its input converted, once.
    assert conversions() - before == (1 if group_norm else 0)
    assert layout(out) == "nhwc" and padding(out) == 0
    want = chain(x.double(), [p.double() for p in params], group_norm)
    assert_near(out.cpu(), want, 3e-2, "KTT_NHWC=1 vs float64")
    assert_near(base, want, 3e-2, "KTT_NHWC=0 vs float64")


@pytest.mark.hw
@pytest.mark.parametrize("shape", [(2, 32, 7, 9), (4, 33, 7, 9), (1, 32, 4, 32)])
def test_nhwc_elementwise(native, monkeypatch, shape):
    """Ops on an Nhwc operand stay Nhwc with zero tile padding; exact ones equal the
    KTT_NHWC=0 results bit for bit, rounding ones within bf16 tolerance."""
    x, y = sample(shape, 0), sample(shape, 1)
    xt, yt = as_nhwc(x), y.to("tt")
    vt = sample((shape[1],), 2).to("tt").reshape(1, -1, 1, 1)
    one = torch.ones(1, device="tt", dtype=torch.bfloat16)
    exact = {
        "relu": F.relu,
        "neg": lambda a: -a,
        "abs": torch.abs,
        "mul_scalar": lambda a: a * 2,
        "add_self": lambda a: a + a,
        "max_nchw": lambda a: torch.maximum(yt, a),
        "upsample": lambda a: F.interpolate(a, scale_factor=2, mode="nearest"),
    }
    rounding = {
        "add_scalar": lambda a: a + 1,
        "add_nchw": lambda a: a + yt,
        "add_vector": lambda a: a + vt,
        "mul_vector": lambda a: a * vt,
        "sub_one": lambda a: a - one,
    }
    for name, op in {**exact, **rounding}.items():
        monkeypatch.setenv("KTT_NHWC", "0")
        want = op(xt).cpu()
        monkeypatch.setenv("KTT_NHWC", "1")
        got = op(xt)
        if name != "upsample" or shape[1] % 32 == 0:
            assert layout(got) == "nhwc" and padding(got) == 0, name
        if name in exact:
            assert torch.equal(got.cpu(), want), name
        else:
            assert_near(got.cpu(), want.double(), 1e-2, name)
    assert layout(yt) == "nchw" and torch.equal(yt.cpu(), y)


@pytest.mark.hw
@pytest.mark.parametrize("shape", [(2, 32, 7, 9), (4, 33, 7, 9)])
def test_nhwc_activations_zero_padding(native, monkeypatch, shape):
    """silu and gelu (0 -> 0, no fill) and exp and sigmoid (filled) leave zero tile
    padding; values match KTT_NHWC=0 within bf16 tolerance."""
    xt = as_nhwc(sample(shape, 3))
    for name, op in {
        "silu": F.silu,
        "gelu": F.gelu,
        "exp": torch.exp,
        "sigmoid": torch.sigmoid,
    }.items():
        monkeypatch.setenv("KTT_NHWC", "0")
        want = op(xt).cpu()
        monkeypatch.setenv("KTT_NHWC", "1")
        got = op(xt)
        assert layout(got) == "nhwc" and padding(got) == 0, name
        assert_near(got.cpu(), want.double(), 1e-2, name)
