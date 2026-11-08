"""Reductions, softmax and argmax over uneven shards on a 1x2 mesh (the last chip's
shard padded): placement, conversions and gather fallbacks per case, error statistics
against float64 next to the 1-card and CPU same-dtype results.

    python experiments/mesh/mesh_uneven.py
"""

import torch

import kohakutt

CASES = [
    ("sum over S(1)", (64, 71), "R,S(1)", lambda t: t.sum(1)),
    ("sum over S(0)", (37, 64), "R,S(0)", lambda t: t.sum(0)),
    ("mean over S(1)", (64, 71), "R,S(1)", lambda t: t.mean(1)),
    ("amax over S(1)", (64, 71), "R,S(1)", lambda t: t.amax(1)),
    ("amin over S(0)", (37, 64), "R,S(0)", lambda t: t.amin(0)),
    ("softmax over S(1)", (64, 71), "R,S(1)", lambda t: t.softmax(1)),
    ("log_softmax over S(1)", (64, 71), "R,S(1)", lambda t: t.log_softmax(1)),
    ("argmax over S(1)", (64, 71), "R,S(1)", lambda t: t.argmax(1)),
    ("sum, kept S(0)", (37, 64), "R,S(0)", lambda t: t.sum(1)),
    ("exp then sum over S(1)", (64, 71), "R,S(1)", lambda t: t.exp().sum(1)),
    ("log then amax over S(0)", (37, 64), "R,S(0)", lambda t: t.abs().log().amax(0)),
    ("mm, K sharded", (64, 71), "R,S(1)", lambda t: t @ t.t()),
]


def main():
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for name, shape, spec, fn in CASES:
                g = torch.Generator().manual_seed(7)
                x = torch.randn(shape, generator=g).to(dtype)
                ref = fn(x.double())
                xd = torch.ops.ktt.distribute(x.to("tt"), spec)
                torch.tt.reset_mesh_stats()
                out = fn(xd)
                placement = kohakutt.placement(out)
                stats = {k: v for k, v in torch.tt.mesh_stats().items() if v}
                fallbacks = torch.tt.mesh_fallback_ops()
                got = out.cpu()
                one = fn(x.to("tt")).cpu()
                print(
                    f"{name} {dtype} {shape} {spec} -> {placement} {stats} {fallbacks}"
                )
                if got.dtype in (torch.int64, torch.int32):
                    print(
                        f"    argmax equal to float64 {torch.equal(got, ref)}, "
                        f"1-card {torch.equal(one, ref)}, CPU {torch.equal(fn(x), ref)}"
                    )
                    continue
                for label, t in (("device", got), ("1-card", one), ("cpu", fn(x))):
                    s = kohakutt.format_stats(kohakutt.diff_stats(t.double(), ref))
                    print(f"    {label:6s} {s}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
