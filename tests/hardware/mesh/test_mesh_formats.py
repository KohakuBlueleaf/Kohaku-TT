"""Mesh: storage formats with placement: a [B, T, H, D] view of a [B, T, H*D]
tensor (run with KTT_HEAD_MAJOR=0 and 1), channels-last activations and convolutions
(run with KTT_NHWC=1 and 0), and flat convolution weights, each with replicated,
sharded and partial-sum operands and passed to shard() and replicate(). Values are
small integers, so every result is exact."""

import os

import pytest
import torch
import torch.nn.functional as F
from mesh_support import make_partial, read, run_case

pytestmark = [pytest.mark.hw, pytest.mark.dual]

# Case ids known to fail, with the reason.
KNOWN_IDS = {}

HEAD_MAJOR = os.environ.get("KTT_HEAD_MAJOR", "default")
NHWC = os.environ.get("KTT_NHWC", "default")


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def ints(shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, shape, generator=gen).to(torch.bfloat16)


def place(k, mesh, value, spec, seed=0):
    if spec == "R":
        return k.replicate(value.to(mesh)), value.float()
    if spec.startswith("S"):
        return k.shard(value.to(mesh), int(spec[1:])), value.float()
    p = make_partial(k, mesh, tuple(value.shape), value.dtype, seed)
    return p, read(p)


def check(t, want):
    got = read(t)
    assert got.shape == want.shape, f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    assert torch.equal(
        got, want.float()
    ), f"max diff {float((got - want.float()).abs().max())}"


def _specs(*names):
    return [pytest.param(n, id=n, marks=_mark(f"{n}")) for n in names]


# ------------------------------------------------- [B, T, H, D] view of [B, T, H*D]


@pytest.mark.parametrize("spec", ["R", "S0", "S1", "S2", "P"])
def test_head_view_ops(native_module, mesh, spec):
    """x.view(B, T, H, D) of a [B, T, H*D] tensor in any placement, then a transpose to
    [B, H, T, D], a scalar add and a reduction over D give the CPU values."""
    k = native_module
    x, ref = place(k, mesh, ints((2, 16, 64), 1), spec)
    v = x.view(2, 16, 4, 16)
    r4 = ref.view(2, 16, 4, 16)
    check(v + 1, r4 + 1)
    check(v.transpose(1, 2).contiguous(), r4.transpose(1, 2))
    check(v.transpose(1, 2) * 2, r4.transpose(1, 2) * 2)
    check(v.sum(-1), r4.sum(-1))
    check(v[:, :, 1:3], r4[:, :, 1:3])


KNOWN_ACT_IDS = {}


@pytest.mark.parametrize(
    ("target", "spec"),
    [
        pytest.param(t, s, id=f"{t}-{s}", marks=_mark(f"{t}-{s}"))
        for t in ("R", "S1", "S2", "S3")
        for s in ("R", "S0", "S1", "S2", "P")
    ],
)
def test_head_view_marks(native_module, mesh, spec, target):
    """shard() / replicate() of the [B, T, H, D] view of a tensor in any placement give
    the CPU values in the requested placement."""
    k = native_module
    x, ref = place(k, mesh, ints((2, 16, 64), 2), spec)
    v = x.view(2, 16, 4, 16)
    out = k.replicate(v) if target == "R" else k.shard(v, int(target[1:]))
    want_place = "Replicated" if target == "R" else f"Sharded(dim={target[1:]})"
    assert k.placement(out) == want_place
    check(out, ref.view(2, 16, 4, 16))
    check(x, ref)


VIEW_OF = {
    "merge": lambda t: t.view(32, 64),
    "transpose": lambda t: t.transpose(0, 2),
    "slice": lambda t: t[:, 4:12],
    "step": lambda t: t[:, ::2],
}


@pytest.mark.parametrize("view", list(VIEW_OF))
@pytest.mark.parametrize("spec", ["R", "S0", "S1", "S2", "P"])
def test_view_marks_use_view_dims(native_module, mesh, spec, view):
    """shard() of a reshaped, transposed or sliced view of a [B, T, C] tensor in any
    placement shards the view's own last dim and gives the CPU values."""
    k = native_module
    x, ref = place(k, mesh, ints((2, 16, 64), 15), spec)
    v = VIEW_OF[view](x)
    out = k.shard(v, v.dim() - 1)
    assert k.placement(out) == f"Sharded(dim={v.dim() - 1})"
    check(out, VIEW_OF[view](ref))
    check(x, ref)


@pytest.mark.parametrize("spec", ["R", "S0", "S1", "S2", "P"])
def test_head_view_attention(native_module, mesh, spec):
    """Attention on q, k, v taken as [B, T, H, D] views and transposed to [B, H, T, D]
    gives the CPU result with an error no larger than one card's."""

    def split(t):
        return t.view(2, 16, 4, 16).transpose(1, 2)

    def attend(q, kk, v):
        return F.scaled_dot_product_attention(split(q), split(kk), split(v))

    xs = [ints((2, 16, 64), 3 + i) for i in range(3)]
    run_case(
        native_module, mesh, "formats", f"head_view_sdpa|{spec}", attend, xs, [spec] * 3
    )


@pytest.mark.parametrize("target", ["R", "S1", "S2"])
def test_head_view_write_through(native_module, mesh, target):
    """An in-place write to the [B, T, H, D] view shows in the [B, T, H*D] base."""
    k = native_module
    x, ref = place(k, mesh, ints((2, 16, 64), 5), target)
    v = x.view(2, 16, 4, 16)
    v.add_(1)
    check(x, ref + 1)


# ----------------------------------------------------------- channels-last / conv


@pytest.mark.parametrize("spec", ["R", "S0", "S1", "S2", "S3", "P"])
def test_conv_input_placements(native_module, mesh, spec):
    """conv2d of an [N, C, H, W] input in any placement with a replicated weight gives
    the CPU result with an error no larger than one card's."""

    def conv(a, b):
        return F.conv2d(a, b, padding=1)

    xs = [ints((2, 8, 16, 16), 6), ints((8, 8, 3, 3), 7)]
    run_case(
        native_module, mesh, "formats", f"conv_input|{spec}", conv, xs, [spec, "R"]
    )


@pytest.mark.parametrize("wspec", ["R", "S0", "S1"])
def test_conv_weight_placements(native_module, mesh, wspec):
    """conv2d with the weight replicated, sharded on output channels or on input
    channels gives the CPU values."""
    k = native_module
    x, ref = place(k, mesh, ints((2, 8, 16, 16), 8), "R")
    w, wr = place(k, mesh, ints((8, 8, 3, 3), 9), wspec)
    check(F.conv2d(x, w, padding=1), F.conv2d(ref, wr, padding=1))


@pytest.mark.parametrize(
    ("target", "spec"),
    [
        pytest.param(
            t,
            s,
            id=f"{t}-{s}",
            marks=(
                [pytest.mark.xfail(strict=True, reason=KNOWN_ACT_IDS[f"{t}-{s}"])]
                if f"{t}-{s}" in KNOWN_ACT_IDS
                else []
            ),
        )
        for t in ("R", "S1", "S2")
        for s in ("R", "S0", "S1", "S2", "P")
    ],
)
def test_activation_marks(native_module, mesh, spec, target):
    """shard() / replicate() of an [N, C, H, W] activation in any placement give the CPU
    values in the requested placement."""
    k = native_module
    x, ref = place(k, mesh, ints((2, 8, 16, 16), 10), spec)
    out = k.replicate(x) if target == "R" else k.shard(x, int(target[1:]))
    want_place = "Replicated" if target == "R" else f"Sharded(dim={target[1:]})"
    assert k.placement(out) == want_place
    check(out, ref)


@pytest.mark.parametrize("target", ["R", "S0", "S1"])
def test_conv_weight_marks(native_module, mesh, target):
    """shard() / replicate() of a convolution weight give the CPU values; the conv with
    the placed copy is right."""
    k = native_module
    w, wr = place(k, mesh, ints((8, 8, 3, 3), 11), "R")
    out = k.replicate(w) if target == "R" else k.shard(w, int(target[1:]))
    check(out, wr)
    x, ref = place(k, mesh, ints((2, 8, 16, 16), 12), "R")
    check(F.conv2d(x, out, padding=1), F.conv2d(ref, wr, padding=1))


def test_flat_conv_weight(native_module, mesh):
    """A convolution weight reshaped to [O, I*kh*kw] (flat) and used as a matmul on
    unfolded patches gives the CPU values for each placement of the flat weight."""
    k = native_module
    x = ints((1, 8, 8, 8), 13)
    w = ints((8, 8, 3, 3), 14)
    patches = F.unfold(x.float(), 3, padding=1)[0]
    want = w.float().reshape(8, -1) @ patches
    for spec in ("R", "S0", "S1"):
        flat, _ = place(k, mesh, w.reshape(8, -1), spec)
        pt, _ = place(k, mesh, patches.to(torch.bfloat16), "R")
        check(flat @ pt, want)
