"""Trace replay of Kohaku-TT's own programs with new inputs against eager calls.

A step running own programs (unary, cast, row / head broadcast binary, a fused chain,
matmul, causal attention, embedding, cumsum, argmax, group norm, conv, nearest upsample,
pixel shuffle) is captured with kohakutt.capture on one set of inputs, then replayed
with other inputs copied in; each replay's outputs are compared with torch.equal to the
same step run eagerly on those inputs, and reported with error statistics against the
float64 CPU result next to the CPU same-dtype result's.

    python experiments/mesh/trace_replay_check.py
    KTT_DESC_CACHE=0 python experiments/mesh/trace_replay_check.py
"""

import os

os.environ.setdefault("KTT_OWN_MATMUL_MIN_ROWS", "1")

import torch
import torch.nn.functional as F

import kohakutt
from kohakutt import fusion
from kohakutt.core.stats import diff_stats, format_stats

BF = torch.bfloat16


def inputs(seed):
    g = torch.Generator().manual_seed(seed)

    def r(*s, dtype=BF, scale=1.0):
        return (torch.randn(*s, generator=g) * scale).to(dtype)

    return [
        r(4, 512, 1024),  # x: unary, cast, binaries, fusion
        r(1024),  # row
        r(512, 1024),  # head
        r(4096, 1024),  # matmul a
        r(1024, 1024, scale=0.03),  # matmul w [N, K]
        r(1, 8, 512, 64),  # q
        r(1, 8, 512, 64),  # k
        r(1, 8, 512, 64),  # v
        torch.randint(0, 1000, (512,), generator=g),  # index
        r(1000, 512, dtype=torch.float32),  # table
        r(1, 64, 64, 64),  # image
        r(64, 64, 3, 3, scale=0.05),  # conv weight
        r(1024, 1024, dtype=torch.float32),  # argmax input
    ]


def chain(x, row):
    """silu(x + row): the fused chain on tt, the composite on the CPU."""
    if x.device.type == "cpu":
        return F.silu(x + row)
    return fusion.run(
        [x, row.reshape(1, -1)],
        ["full", "row"],
        [["add", 0, 0, 1], ["silu", 0, 0, None]],
        0,
        list(x.shape),
        x.dtype,
    )


def step(x, row, head, a, w, q, k, v, idx, table, img, cw, am):
    return [
        torch.exp(x),
        x.float(),
        x + row,
        x * head,
        chain(x, row),
        F.linear(a, w),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        F.embedding(idx, table),
        torch.cumsum(x, dim=-1),
        torch.argmax(am, dim=-1),
        F.group_norm(img, 32),
        F.conv2d(img, cw, padding=1),
        F.interpolate(img, scale_factor=2, mode="nearest"),
        F.pixel_shuffle(img, 2),
    ]


NAMES = [
    "exp",
    "cast",
    "row add",
    "head mul",
    "fusion",
    "linear",
    "attention",
    "embedding",
    "cumsum",
    "argmax",
    "group_norm",
    "conv",
    "upsample",
    "pixel_shuffle",
]


def main():
    dev = kohakutt.device(0)
    print(f"KTT_DESC_CACHE={os.environ.get('KTT_DESC_CACHE', '1')}")
    captured = kohakutt.capture(step, *[t.to(dev) for t in inputs(0)])
    for seed in (1, 2):
        cpu = inputs(seed)
        replayed = [t.cpu() for t in captured(*[t.to(dev) for t in cpu])]
        eager = [t.cpu() for t in step(*[t.to(dev) for t in cpu])]
        ref64 = step(*[t.double() if t.is_floating_point() else t for t in cpu])
        base = step(*cpu)
        for name, r, e, f64, b in zip(NAMES, replayed, eager, ref64, base, strict=True):
            same = r.dtype == e.dtype and torch.equal(r, e)
            stats = format_stats(diff_stats(r, f64)) if r.is_floating_point() else ""
            bstats = format_stats(diff_stats(b, f64)) if b.is_floating_point() else ""
            print(
                f"seed {seed} {name:14s} replay vs eager: "
                f"{'bit-identical' if same else 'DIFFERENT'}"
            )
            if stats:
                print(f"    tt vs fp64: {stats}\n    cpu vs fp64: {bstats}")
    captured.release()


if __name__ == "__main__":
    main()
