"""Shared helpers of the mesh tests: put host values on a 1x2 mesh in a chosen
placement, run one op, compare with CPU fp32 and with the same op on replicated inputs
(what one card computes), and log what the counters say."""

import json
import os
import re
from pathlib import Path

import torch

LOG = (
    Path(os.environ["KTT_MESH_TEST_LOG"]) if "KTT_MESH_TEST_LOG" in os.environ else None
)
LEGAL = re.compile(r"^(Replicated|Sharded\(dim=(\d+)\)|Partial\((sum|max|min)\))$")

# Placement specs of one operand: R, S0, S1, ..., or P (a partial sum).
SPECS_2D = ("R", "S0", "S1", "P")


def host(shape, dtype, seed=0, low=None):
    """A reproducible host tensor; floats are normal values rounded to `dtype`, ints
    are small non-negative values."""
    gen = torch.Generator().manual_seed(seed)
    if dtype == torch.int32:
        return torch.randint(0, 50, shape, generator=gen, dtype=torch.int32)
    return torch.randn(shape, generator=gen).to(dtype)


def ternary(shape, dtype, seed):
    """Values -1, 0, 1: a K=64 product of them is an integer of at most 64 in absolute
    value, so every partial sum and every completion of it is exact in bf16 and fp32."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randint(-1, 2, shape, generator=gen)).to(dtype)


def make_partial(k, mesh, shape, dtype, seed=0):
    """A tt tensor of `shape` that holds a pending sum: the product of an activation
    sharded on K and a weight sharded on K."""
    *lead, n = shape
    m = int(torch.tensor(lead).prod()) if lead else 1
    kk = 64
    a = ternary((m, kk), dtype, seed + 100)
    w = ternary((kk, n), dtype, seed + 200)
    p = k.shard(a.to(mesh), 1) @ k.shard(w.to(mesh), 0)
    return p.reshape(shape) if lead and len(lead) > 1 else p


def place(k, mesh, value, spec, seed=0):
    """`value` (host tensor) on the mesh with placement `spec`; for P the value is what
    the device holds (a pending sum made by a K-split matmul)."""
    if spec == "R":
        return k.replicate(value.to(mesh))
    if spec.startswith("S"):
        return k.shard(value.to(mesh), int(spec[1:]))
    if spec == "P":
        return make_partial(k, mesh, tuple(value.shape), value.dtype, seed)
    raise ValueError(spec)


def read(t):
    """The host value of a tt tensor, as float32 (or int64 for int tensors)."""
    c = t.cpu()
    return c if c.dtype in (torch.int32, torch.int64, torch.bool) else c.float()


def counters_around(run):
    """(result, {mesh_stats delta}, {fallback ops since reset})."""
    torch.tt.synchronize()
    torch.tt.reset_mesh_stats()
    before = torch.tt.mesh_stats()
    out = run()
    torch.tt.synchronize()
    after = torch.tt.mesh_stats()
    delta = {k: after[k] - before[k] for k in after if after[k] != before[k]}
    return out, delta, torch.tt.mesh_fallback_ops()


def outputs(out):
    return out if isinstance(out, (tuple, list)) else (out,)


def record(section, case, **fields):
    """Appends one line to <KTT_MESH_TEST_LOG>/<section>.jsonl (conversions and
    fallbacks per case); does nothing when the variable is unset."""
    if LOG is None:
        return
    LOG.mkdir(exist_ok=True)
    with open(LOG / f"{section}.jsonl", "a") as f:
        f.write(json.dumps({"case": case, **fields}) + "\n")


def max_err(got, want):
    if got.shape != want.shape:
        return float("inf")
    if got.numel() == 0:
        return 0.0
    return float((got.double() - want.double()).abs().max())


def run_case(k, mesh, section, case, fn, host_inputs, specs, exact=False, seed=0):
    """Runs `fn` on the mesh with `host_inputs` placed as `specs`.

    Checks: the output has a legal placement and the CPU shape; its error against the
    fp32 CPU result (computed on the values the device holds) is no larger than the
    error of the same op on replicated inputs, which is one card's error; `exact` ops
    must equal the CPU result. Logs the conversions and fallbacks the op caused.
    Returns the output placements."""
    placed = [
        place(k, mesh, h, s, seed + i)
        for i, (h, s) in enumerate(zip(host_inputs, specs))
    ]
    values = [read(t) for t in placed]
    float_in = [v.float() if v.is_floating_point() else v for v in values]
    want = outputs(fn(*float_in))

    out, delta, fallbacks = counters_around(lambda: fn(*placed))
    outs = outputs(out)
    places = [k.placement(o) for o in outs]
    got = [read(o) for o in outs]

    rep = [
        k.replicate(torch.as_tensor(v).to(host_inputs[i].dtype).to(mesh))
        for i, v in enumerate(values)
    ]
    base = [read(o) for o in outputs(fn(*rep))]

    record(
        section,
        case,
        specs=list(specs),
        dtype=str(host_inputs[0].dtype),
        shape=[list(h.shape) for h in host_inputs],
        out_placement=places,
        conversions=delta,
        fallbacks=fallbacks,
    )
    for p, o, g, w in zip(places, outs, got, want):
        assert LEGAL.match(p), f"illegal placement {p!r}"
        m = LEGAL.match(p).group(2)
        if m is not None:
            assert int(m) < max(o.dim(), 1), f"{p} on a {o.dim()}-d output"
        assert g.shape == w.shape, f"shape {tuple(g.shape)} != {tuple(w.shape)}"
    for g, w, b in zip(got, want, base):
        if g.numel() == 0:
            continue
        if not w.is_floating_point() or exact:
            assert torch.equal(g.to(w.dtype), w), "result differs from CPU"
        else:
            err_m, err_r = max_err(g, w), max_err(b, w)
            assert err_m <= err_r, f"error {err_m:g} > one card's {err_r:g} ({places})"
    return places
