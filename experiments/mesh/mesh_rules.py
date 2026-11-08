"""Kohaku-TT mesh placement rules on a 1x2 mesh: for each op family with a rule, the
output placement, the conversions it ran (torch.tt.mesh_stats) and error statistics
against the CPU float64 ruler next to the CPU same-dtype baseline.

    python experiments/mesh/mesh_rules.py
"""

import torch

import kohakutt


def _host(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) + 2.0).to(dtype)


def _report(name, got, ref64, base):
    stats = kohakutt.format_stats(kohakutt.diff_stats(got, ref64))
    cpu = kohakutt.format_stats(kohakutt.diff_stats(base, ref64))
    kinds = {k: v for k, v in torch.tt.mesh_stats().items() if v}
    print(f"{name}\n    device {stats}\n    cpu    {cpu}\n    {kinds}")


CASES = {
    "exp(a)": lambda t, u: t.exp(),
    "a * 3 - 1": lambda t, u: t * 3 - 1,
    "a + b": lambda t, u: t + u,
    "a * b": lambda t, u: t * u,
    "a / b": lambda t, u: t / u,
    "a > b": lambda t, u: t > u,
}


def elementwise():
    for dtype in (torch.float32, torch.bfloat16):
        for spec in ("R,S(0)", "R,S(1)"):
            for shape in [(256, 512), (250, 70)]:
                x = _host(shape, dtype, 0)
                y = _host(shape, dtype, 1)
                a = torch.ops.ktt.distribute(x.to("tt"), spec)
                b = torch.ops.ktt.distribute(y.to("tt"), spec)
                tag = f"{shape} {dtype} {spec}"
                for name, fn in CASES.items():
                    torch.tt.reset_mesh_stats()
                    out = fn(a, b)
                    placement = kohakutt.placement(out)
                    got = out.cpu()
                    ref64 = fn(x.double(), y.double())
                    base = fn(x, y)
                    _report(f"{name} {tag} -> {placement}", got, ref64, base)


def mixed():
    x = _host((256, 512), torch.float32, 0)
    y = _host((256, 512), torch.float32, 1)
    cases = [
        ("S(0) + R", "R,S(0)", "R,R"),
        ("S(0) + S(1)", "R,S(0)", "R,S(1)"),
        ("P(sum) + P(sum)", "R,P(sum)", "R,P(sum)"),
        ("P(sum) * R", "R,P(sum)", "R,R"),
        ("P(sum) + R", "R,P(sum)", "R,R"),
    ]
    for name, sa, sb in cases:
        a = torch.ops.ktt.distribute(x.to("tt"), sa)
        b = torch.ops.ktt.distribute(y.to("tt"), sb)
        op = (lambda t, u: t * u) if "*" in name else (lambda t, u: t + u)
        torch.tt.reset_mesh_stats()
        out = op(a, b)
        placement = kohakutt.placement(out)
        got = out.cpu()
        _report(f"{name} -> {placement}", got, op(x.double(), y.double()), op(x, y))


def matmuls():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        x = (torch.randn(64, 256) / 4).to(dtype)
        w = (torch.randn(512, 256) / 16).to(dtype)
        w2 = (torch.randn(256, 512) / 16).to(dtype)
        bias = torch.randn(512).to(dtype)
        cases = [
            ("column: x R, w S(out)", "R,R", "R,S(0)", None),
            ("column + bias", "R,R", "R,S(0)", bias),
            ("row: x S(in), w S(in)", "R,S(1)", "R,S(1)", None),
            ("row + bias", "R,S(1)", "R,S(1)", bias),
            ("data: x S(batch), w R", "R,S(0)", "R,R", None),
        ]
        for name, sx, sw, b in cases:
            xd = torch.ops.ktt.distribute(x.to("tt"), sx)
            wd = torch.ops.ktt.distribute(w.to("tt"), sw)
            bd = b.to("tt") if b is not None else None
            torch.tt.reset_mesh_stats()
            out = torch.nn.functional.linear(xd, wd, bd)
            placement = kohakutt.placement(out)
            got = out.cpu()
            fn = torch.nn.functional.linear
            ref64 = fn(x.double(), w.double(), None if b is None else b.double())
            _report(f"linear {dtype} {name} -> {placement}", got, ref64, fn(x, w, b))
            # The same op on replicated inputs: the single-card kernels.
            one = fn(x.to("tt"), w.to("tt"), bd).cpu()
            print(
                f"    1-card {kohakutt.format_stats(kohakutt.diff_stats(one, ref64))}"
            )
        # MLP chain: column-parallel up projection, row-parallel down projection.
        xd = x.to("tt")
        wd = torch.ops.ktt.distribute(w.to("tt"), "R,S(0)")
        w2d = torch.ops.ktt.distribute(w2.to("tt"), "R,S(1)")
        torch.tt.reset_mesh_stats()
        h = torch.nn.functional.linear(xd, wd)
        y = torch.nn.functional.linear(h * 2, w2d)
        placement = kohakutt.placement(y)
        got = y.cpu()
        ref64 = (x.double() @ w.double().T * 2) @ w2.double().T
        _report(
            f"mlp {dtype} column -> row -> {placement}",
            got,
            ref64,
            (x @ w.T * 2) @ w2.T,
        )
        # mm, row-parallel.
        a = torch.ops.ktt.distribute(x.to("tt"), "R,S(1)")
        bm = torch.ops.ktt.distribute(w.T.contiguous().to("tt"), "R,S(0)")
        torch.tt.reset_mesh_stats()
        out = a @ bm
        placement = kohakutt.placement(out)
        _report(
            f"mm {dtype} row -> {placement}",
            out.cpu(),
            x.double() @ w.double().T,
            x @ w.T,
        )


REDUCTIONS = {
    "sum(0)": lambda t: t.sum(0),
    "sum(1)": lambda t: t.sum(1),
    "sum(1, keepdim)": lambda t: t.sum(1, keepdim=True),
    "mean(0)": lambda t: t.mean(0),
    "amax(0)": lambda t: t.amax(0),
    "amax(1)": lambda t: t.amax(1),
    "sum()": lambda t: t.sum(),
}


def reductions():
    for dtype in (torch.float32, torch.bfloat16):
        for shape in [(256, 512), (250, 70)]:
            for spec in ("R,S(0)", "R,S(1)"):
                x = _host(shape, dtype, 3)
                a = torch.ops.ktt.distribute(x.to("tt"), spec)
                for name, fn in REDUCTIONS.items():
                    torch.tt.reset_mesh_stats()
                    out = fn(a)
                    placement = kohakutt.placement(out)
                    got = out.cpu()
                    one = fn(x.to("tt")).cpu()
                    ref64 = fn(x.double())
                    _report(
                        f"{name} {shape} {dtype} {spec} -> {placement}",
                        got,
                        ref64,
                        fn(x),
                    )
                    one_stats = kohakutt.format_stats(kohakutt.diff_stats(one, ref64))
                    print(f"    1-card {one_stats}")


VIEWS = [
    ("transpose of S(1)", (64, 512), "R,S(1)", lambda t: t.t().exp()),
    (
        "split S(1) 512 -> 8 x 64",
        (64, 512),
        "R,S(1)",
        lambda t: t.view(64, 8, 64).exp(),
    ),
    ("merge led by S(0)", (8, 64, 32), "R,S(0)", lambda t: t.view(512, 32).exp()),
    ("merge of S(1) (not leading)", (64, 512), "R,S(1)", lambda t: t.view(-1).exp()),
    ("view of P(sum)", (64, 512), "R,P(sum)", lambda t: t.view(64, 8, 64) * 2),
    ("slice of S(0) on dim 1", (64, 512), "R,S(0)", lambda t: t[:, :256].exp()),
    (
        "rotary on heads S(1)",
        (2, 64, 8, 64),
        "R,S(2)",
        lambda t: (
            torch.cat((-t.transpose(1, 2)[..., 32:], t.transpose(1, 2)[..., :32]), -1)
            * 2
        ),
    ),
    ("slice cutting S(1)", (64, 512), "R,S(1)", lambda t: t[:, :100].exp()),
    ("rotary on a projection S(1)", (1, 128, 896), "R,S(1)", lambda t: _rotary(t)),
]


def _rotary(t):
    """Hugging Face rotary embedding on a (1, 128, 896) projection as 14 heads of 64,
    with cos and sin made on the input's device."""
    q = t.view(1, 128, 14, 64).transpose(1, 2)
    angle = torch.arange(128 * 64, dtype=t.dtype).view(1, 1, 128, 64) * 1e-3
    cos, sin = angle.cos().to(t.device), angle.sin().to(t.device)
    half = torch.cat((-q[..., 32:], q[..., :32]), -1)
    return q * cos + half * sin


SOFTMAX = {
    "softmax(-1)": lambda t: torch.softmax(t, -1),
    "softmax(0)": lambda t: torch.softmax(t, 0),
    "log_softmax(-1)": lambda t: torch.log_softmax(t, -1),
}


def softmaxes():
    for dtype in (torch.float32, torch.bfloat16):
        for spec in ("R,S(0)", "R,S(1)"):
            x = _host((128, 512), dtype, 7)
            a = torch.ops.ktt.distribute(x.to("tt"), spec)
            for name, fn in SOFTMAX.items():
                torch.tt.reset_mesh_stats()
                out = fn(a)
                placement = kohakutt.placement(out)
                got = out.cpu()
                one = fn(x.to("tt")).cpu()
                ref64 = fn(x.double())
                _report(f"{name} {dtype} {spec} -> {placement}", got, ref64, fn(x))
                print(
                    f"    1-card "
                    f"{kohakutt.format_stats(kohakutt.diff_stats(one, ref64))}"
                )
    # Backward through a sharded softmax dim.
    x = _host((128, 512), torch.float32, 8)
    g = _host((128, 512), torch.float32, 9)
    a = torch.ops.ktt.distribute(x.to("tt"), "R,S(1)").requires_grad_()
    torch.tt.reset_mesh_stats()
    torch.softmax(a, -1).backward(torch.ops.ktt.distribute(g.to("tt"), "R,S(1)"))
    xr = x.double().requires_grad_()
    torch.softmax(xr, -1).backward(g.double())
    xb = x.clone().requires_grad_()
    torch.softmax(xb, -1).backward(g)
    _report(
        f"softmax(-1) backward S(1) -> {kohakutt.placement(a.grad)}",
        a.grad.cpu(),
        xr.grad,
        xb.grad,
    )


def norms():
    for dtype in (torch.float32, torch.bfloat16):
        x = _host((4, 64, 512), dtype, 11)
        w = _host((512,), dtype, 12)
        b = _host((512,), dtype, 13)
        cases = {
            "layer_norm": lambda t, ww, bb: torch.nn.functional.layer_norm(
                t, (512,), ww, bb
            ),
            "rms_norm": lambda t, ww, bb: torch.nn.functional.rms_norm(t, (512,), ww),
        }
        for spec in ("R,S(0)", "R,S(2)"):
            a = torch.ops.ktt.distribute(x.to("tt"), spec)
            wd, bd = w.to("tt"), b.to("tt")
            for name, fn in cases.items():
                torch.tt.reset_mesh_stats()
                out = fn(a, wd, bd)
                placement = kohakutt.placement(out)
                got = out.cpu()
                one = fn(x.to("tt"), wd, bd).cpu()
                ref64 = fn(x.double(), w.double(), b.double())
                _report(
                    f"{name} {dtype} {spec} -> {placement}", got, ref64, fn(x, w, b)
                )
                print(
                    f"    1-card "
                    f"{kohakutt.format_stats(kohakutt.diff_stats(one, ref64))}"
                )
    # Casts keep partial sums (widening).
    x = _host((64, 512), torch.bfloat16, 14)
    p = torch.ops.ktt.distribute(x.to("tt"), "R,P(sum)")
    torch.tt.reset_mesh_stats()
    out = p.float()
    placement = kohakutt.placement(out)
    _report(f"bf16 P(sum) .float() -> {placement}", out.cpu(), x.double(), x.float())


def embeddings():
    for dtype in (torch.bfloat16, torch.float32):
        table = _host((1024, 256), dtype, 15)
        idx = torch.randint(
            0, 1024, (4, 33), generator=torch.Generator().manual_seed(16)
        )
        for spec in ("R,S(0)", "R,S(1)"):
            t = torch.ops.ktt.distribute(table.to("tt"), spec)
            torch.tt.reset_mesh_stats()
            out = torch.nn.functional.embedding(idx.to("tt"), t)
            placement = kohakutt.placement(out)
            got = out.cpu()
            ref = torch.nn.functional.embedding(idx, table)
            _report(
                f"embedding {dtype} table {spec} -> {placement}", got, ref.double(), ref
            )


def attention():
    sdpa = torch.nn.functional.scaled_dot_product_attention
    for dtype in (torch.bfloat16, torch.float32):
        q, k, v = (_host((2, 8, 128, 64), dtype, s) / 4 for s in (17, 18, 19))
        for spec, causal in (("R,S(1)", False), ("R,S(1)", True), ("R,S(0)", True)):
            qd, kd, vd = (torch.ops.ktt.distribute(t.to("tt"), spec) for t in (q, k, v))
            torch.tt.reset_mesh_stats()
            out = sdpa(qd, kd, vd, is_causal=causal)
            placement = kohakutt.placement(out)
            got = out.cpu()
            one = sdpa(q.to("tt"), k.to("tt"), v.to("tt"), is_causal=causal).cpu()
            ref64 = sdpa(q.double(), k.double(), v.double(), is_causal=causal)
            base = sdpa(q, k, v, is_causal=causal)
            _report(
                f"sdpa {dtype} {spec} causal={causal} -> {placement}", got, ref64, base
            )
            print(
                f"    1-card {kohakutt.format_stats(kohakutt.diff_stats(one, ref64))}"
            )
    # bmm with batch sharded and a 4-D matmul through it (heads sharded, batch 1).
    a = _host((1, 8, 64, 32), torch.float32, 20)
    b = _host((1, 8, 32, 48), torch.float32, 21)
    ad = torch.ops.ktt.distribute(a.to("tt"), "R,S(1)")
    bd = torch.ops.ktt.distribute(b.to("tt"), "R,S(1)")
    torch.tt.reset_mesh_stats()
    out = ad @ bd
    placement = kohakutt.placement(out)
    _report(
        f"4-D matmul heads S(1) -> {placement}",
        out.cpu(),
        a.double() @ b.double(),
        a @ b,
    )


def views():
    for name, shape, spec, fn in VIEWS:
        x = _host(shape, torch.float32, 5)
        a = torch.ops.ktt.distribute(x.to("tt"), spec)
        torch.tt.reset_mesh_stats()
        out = fn(a)
        placement = kohakutt.placement(out)
        _report(f"view {name} -> {placement}", out.cpu(), fn(x.double()), fn(x))


def main():
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        elementwise()
        mixed()
        matmuls()
        reductions()
        views()
        softmaxes()
        norms()
        embeddings()
        attention()
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
