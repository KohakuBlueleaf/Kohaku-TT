"""Regression: per-head tensors [.., T, H, D] with few heads are stored head-major, as
[.., H, T, D] tiles. The ops of a q / k norm and rotary chain keep them so with no
conversion, bit for bit equal to KTT_HEAD_MAJOR=0; other reads convert, counted."""

import pytest
import torch

import kohakutt

B, T, D = 1, 64, 64


def layout(t):
    return torch.ops.ktt._debug_layout(t)


def reads():
    return sum(kohakutt.head_major_reads().values())


def sample(shape, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


def chain(x, weight, cos, sin, heads):
    """A q / k norm and a rotary embedding on the [B, T, H, D] view of x [B, T, H D],
    then the head-major transpose."""
    q = x.view(B, -1, heads, D)
    f = q.float()
    norm = (f.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.5)
    q = (f * norm * weight).to(x.dtype)
    half = torch.cat((-q[..., D // 2 :], q[..., : D // 2]), dim=-1)
    return (q * cos + half * sin).transpose(1, 2)


@pytest.mark.hw
@pytest.mark.parametrize(
    "shape, dtype, major",
    [
        ((1, 64, 8, 64), torch.float32, True),
        ((1, 64, 16, 64), torch.bfloat16, True),
        ((2, 96, 1, 32), torch.float32, True),
        ((1, 40, 4, 64), torch.float32, True),
        ((1, 64, 17, 64), torch.float32, False),
        ((1, 40, 12, 64), torch.float32, False),
        ((1, 16, 4, 64), torch.float32, False),
        ((1, 64, 4, 16), torch.float32, False),
        ((1, 64, 8, 64), torch.int32, False),
    ],
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
def test_rule_roundtrip_and_bytes(native, monkeypatch, shape, dtype, major):
    b, t, h, d = shape
    x = sample((b, t, h * d), 0).mul(8).to(dtype)
    base = x.to("tt")

    def copy():
        kohakutt.synchronize()
        before = kohakutt.counters()["device_bytes"]
        y = base.view(shape) + 0
        kohakutt.synchronize()
        return y, kohakutt.counters()["device_bytes"] - before

    copy()  # the storage's head-major copy is cached from here on
    y, grown = copy()
    assert layout(y) == ("head_major" if major else "nchw")
    n = reads()
    assert torch.equal(y.cpu(), x.view(shape))
    assert reads() == n
    size = 2 if dtype == torch.bfloat16 else 4
    wide = -(-d // 32) * 32
    own, tiled = b * h * -(-t // 32) * 32 * wide, b * t * 32 * wide
    assert grown == (own if major else tiled) * size
    # Host uploads keep the tile form of their own shape.
    assert layout(x.view(shape).to("tt")) == "nchw"
    monkeypatch.setenv("KTT_HEAD_MAJOR", "0")
    plain, grown = copy()
    assert layout(plain) == "nchw" and torch.equal(plain.cpu(), x.view(shape))
    assert grown == tiled * size


@pytest.mark.hw
def test_uploads_and_activations_stay_tile_form(native):
    """A [N, C, H, W] tensor with a per-head shape that is not a view of a merged
    tensor keeps the tile form of its shape, as do the results of ops on it."""
    x = sample((1, 32, 4, 32), 8)
    t = x.to("tt")
    assert layout(t) == "nchw"
    for got in (torch.relu(t), t * 2 + 1, t + t, torch.empty_like(t), t.clone()):
        assert layout(got) == "nchw"
    w = sample((32, 32, 1, 1), 9).to("tt")
    act = torch.nn.functional.conv2d(t.bfloat16(), w.bfloat16())
    assert tuple(act.shape) == (1, 32, 4, 32)
    assert layout(act) == "nchw" and layout(torch.relu(act)) == "nchw"
    assert reads() == 0 or "conv" not in " ".join(kohakutt.head_major_reads())


@pytest.mark.hw
@pytest.mark.parametrize("heads", [1, 8])
@pytest.mark.parametrize("seq", [64, 40])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_chain_matches_tile_form(native, monkeypatch, heads, seq, dtype):
    def run():
        x = sample((B, seq, heads * D), 1, dtype).to("tt")
        weight = sample((D,), 2).to("tt")
        cos = sample((B, seq, D), 3, dtype).to("tt").unsqueeze(2)
        sin = sample((B, seq, 1, D), 4, dtype).to("tt")
        return chain(x, weight, cos, sin, heads)

    n = reads()
    out = run()
    got = out.cpu()
    assert reads() == n, kohakutt.head_major_reads()
    assert layout(out) == "head_major"
    monkeypatch.setenv("KTT_HEAD_MAJOR", "0")
    plain = run()
    assert layout(plain) == "nchw"
    assert torch.equal(got, plain.cpu())


@pytest.mark.hw
@pytest.mark.parametrize("heads", [1, 8])
def test_chain_in_a_trace(native, monkeypatch, heads):
    weight = sample((D,), 2).to("tt")
    cos = sample((B, T, D), 3, torch.bfloat16).to("tt")
    sin = sample((B, T, D), 4, torch.bfloat16).to("tt")

    def step(x):
        out = chain(x, weight, cos.unsqueeze(2), sin.unsqueeze(2), heads)
        return out.contiguous()

    xs = [sample((B, T, heads * D), 10 + k, torch.bfloat16) for k in range(3)]
    captured = kohakutt.Captured(step, (xs[0].to("tt"),), 2)
    outs = [captured(x.to("tt")).cpu() for x in xs]
    captured.release()
    monkeypatch.setenv("KTT_HEAD_MAJOR", "0")
    for x, got in zip(xs, outs, strict=True):
        assert torch.equal(got, step(x.to("tt")).cpu())


@pytest.mark.hw
@pytest.mark.parametrize("seq", [64, 40])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_heads_merge_back(native, seq, dtype):
    x = sample((B, seq, 8 * D), 7, dtype).to("tt")
    host = x.cpu() * 2
    y = x.view(B, seq, 8, D) * 2
    n = reads()
    assert layout(y) == "head_major"
    assert torch.equal(y.reshape(B, seq, 8 * D).cpu(), host)
    assert torch.equal(y.transpose(1, 2).cpu(), host.view(B, seq, 8, D).transpose(1, 2))
    z = x.clone()
    z.view(B, seq, 8, D).mul_(2)
    assert layout(z) == "nchw" and torch.equal(z.cpu(), host)
    assert reads() == n, kohakutt.head_major_reads()


@pytest.mark.hw
def test_ops_keep_the_format(native):
    x = sample((B, T, 8 * D), 5).to("tt")
    host = x.cpu().view(B, T, 8, D)
    q = x.view(B, T, 8, D)
    n = reads()
    for name, got, want in [
        ("neg", -q, -host),
        ("scalar", q * 0.5 + 1, host * 0.5 + 1),
        ("same shape", q + q, host + host),
        ("cast", q.bfloat16(), host.bfloat16()),
        ("clone", q.clone(), host),
        ("slice", -(-q)[..., 16:48], host[..., 16:48]),
        ("heads", -(-q)[:, :, 2:6], host[:, :, 2:6]),
        ("rows", -(-q)[:, 8:40], host[:, 8:40]),
        ("sum", q.sum(-1, keepdim=True), host.sum(-1, keepdim=True)),
        ("amax", q.amax(2, keepdim=True), host.amax(2, keepdim=True)),
        ("where", torch.where(q > 0, q, -q), host.abs()),
        ("addcmul", torch.addcmul(q, q, q, value=0.5), host + 0.5 * host * host),
        ("addcdiv", torch.addcdiv(q, q, q.abs() + 1), host + host / (host.abs() + 1)),
        ("lerp", torch.lerp(q, -q, 0.25), host * 0.5),
        ("cat heads", torch.cat((q, q * 2), dim=2), torch.cat((host, host * 2), 2)),
    ]:
        assert layout(got) == "head_major", name
        assert torch.allclose(
            got.cpu().float(), want.float(), rtol=1e-3, atol=1e-3
        ), name
    y = q * 2
    assert layout(y) == "head_major"
    assert torch.equal(y.transpose(1, 2).cpu(), (host * 2).transpose(1, 2))
    assert torch.equal(y.reshape(B, T, 8 * D).cpu(), (host * 2).reshape(B, T, 8 * D))
    y.mul_(2)
    assert layout(y) == "head_major" and torch.equal(y.cpu(), host * 4)
    z = x.clone()
    z.view(B, T, 8, D).mul_(2)
    assert layout(z) == "nchw" and torch.equal(z.cpu(), x.cpu() * 2)
    assert reads() == n, kohakutt.head_major_reads()


@pytest.mark.hw
def test_other_reads_convert_and_count(native):
    x = sample((B, T, 8 * D), 6).to("tt")
    host = x.cpu().view(B, T, 8, D) * 2

    def fresh():
        y = x.view(B, T, 8, D) * 2
        assert layout(y) == "head_major"
        kohakutt.reset_head_major_reads()
        return y

    y = fresh()
    assert torch.equal(y.reshape(B, T * 8, D).cpu(), host.reshape(B, T * 8, D))
    assert reads() == 1
    # The tile form is cached on the tensor: its next reader converts nothing.
    assert torch.allclose(y.mean(-1).cpu(), host.mean(-1), atol=1e-4)
    assert reads() == 1
    y = fresh()
    assert torch.equal(y[:, ::2].cpu(), host[:, ::2])
    assert reads() == 0
    assert torch.equal(y.transpose(2, 3)[:, ::2].cpu(), host.transpose(2, 3)[:, ::2])
    assert reads() == 1
    y = fresh()
    assert torch.allclose(y.mean(-1).cpu(), host.mean(-1), atol=1e-4)
    assert kohakutt.head_major_reads() == {"mean.dim": 1}
    y = fresh()
    y[:, :4] = 0
    assert layout(y) == "nchw" and not y[:, :4].cpu().any()
    assert torch.equal(y[:, 4:].cpu(), host[:, 4:])
    assert reads() >= 1
