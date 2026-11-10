"""Mesh: attention: scaled dot-product attention on [B, H, T, D] query, key and
value in the placements the heads, the sequence and a partial sum give, including mixed
ones. Each result is compared with the CPU fp32 result and with one card's error for
the same op, and its placement must be a legal one."""

import pytest
import torch
import torch.nn.functional as F
from mesh_support import host, read, run_case

pytestmark = [pytest.mark.hw, pytest.mark.dual]

# Case ids known to fail, with the reason.
KNOWN_IDS = {}

SPEC_SETS = [
    ("R", "R", "R"),
    ("S1", "S1", "S1"),
    ("S2", "S2", "S2"),
    ("S1", "R", "R"),
    ("R", "S1", "S1"),
    ("S2", "R", "R"),
    ("R", "S2", "S2"),
    ("S3", "S3", "S3"),
    ("P", "P", "P"),
    ("P", "R", "R"),
    ("S1", "S2", "S1"),
]


def _mark(cid):
    return (
        [pytest.mark.xfail(strict=True, reason=KNOWN_IDS[cid])]
        if cid in KNOWN_IDS
        else []
    )


def _cases():
    for specs in SPEC_SETS:
        for shape in ((1, 4, 32, 32), (2, 3, 33, 32)):
            cid = f"sdpa-{'_'.join(specs)}-{'x'.join(map(str, shape))}"
            yield pytest.param(specs, shape, id=cid, marks=_mark(cid))


@pytest.mark.parametrize(("specs", "shape"), list(_cases()))
def test_sdpa(native_module, mesh, specs, shape):
    """Unmasked attention gives the CPU result in a legal placement."""
    qkv = [host(shape, torch.bfloat16, seed=80 + i) for i in range(3)]
    run_case(
        native_module,
        mesh,
        "attention",
        f"sdpa|{specs}|{shape}",
        F.scaled_dot_product_attention,
        qkv,
        list(specs),
    )


@pytest.mark.parametrize("which", ["k", "v", "kv"])
def test_sdpa_ignores_key_padding(native_module, mesh, which):
    """Keys / values of 33 rows whose tile padding holds inf (a slice of a 64-row tensor
    keeps the tile data) give the result of zero padding."""
    shape = (2, 3, 33, 32)
    q, k, v = [host(shape, torch.bfloat16, seed=80 + i) for i in range(3)]

    def dirty(x):
        big = torch.full((2, 3, 64, 32), float("inf"), dtype=torch.bfloat16)
        big[:, :, :33] = x
        return native_module.replicate(big.to(mesh))[:, :, :33, :]

    def clean(x):
        return native_module.replicate(x.to(mesh))

    got = F.scaled_dot_product_attention(
        clean(q),
        dirty(k) if "k" in which else clean(k),
        dirty(v) if "v" in which else clean(v),
    )
    want = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    err = (read(got) - want).abs().max()
    assert torch.isfinite(err) and err < 0.05, f"error {float(err):g}"


@pytest.mark.parametrize("specs", [("R", "R", "R"), ("S1", "S1", "S1")])
def test_sdpa_causal(native_module, mesh, specs):
    """Causal attention gives the CPU result."""
    qkv = [host((1, 4, 32, 32), torch.bfloat16, seed=90 + i) for i in range(3)]

    def fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    run_case(
        native_module, mesh, "attention", f"sdpa_causal|{specs}", fn, qkv, list(specs)
    )
