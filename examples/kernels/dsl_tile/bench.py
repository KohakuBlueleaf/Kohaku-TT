"""Times of the tile-level examples against the eager torch ops (and the metal level
where it exists) at the benchmarked shapes; prints one markdown row per shape with
the memory bound (one read and one write at 512 GB/s).
Usage: python examples/kernels/dsl_tile/bench.py rms|swiglu|rope|softmax [--heads H]"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakutt

sys.path.insert(0, str(Path(__file__).parent))
import group_norm
import layer_norm
import online_softmax
import rms_norm
import rope
import swiglu
from common import timed

DRAM = 512e9


def bound(*tensors):
    """Microseconds to move every tensor once at the DRAM rate."""
    return sum(t.numel() * t.element_size() for t in tensors) / DRAM * 1e6


def row(name, shape, times, mem):
    cells = " | ".join(f"{t * 1e6:.0f}" if t else "-" for t in times)
    print(f"| {name} | {list(shape)} | {cells} | {mem:.0f} |", flush=True)


@torch.no_grad()
def bench_rms(dev, tokens):
    print("| kernel | shape | tile us | eager us | memory bound us |")
    print("|---|---|---|---|---|")
    for shape in [(1, t, 1536) for t in tokens] + [(1, t, 8, 256) for t in tokens]:
        x = torch.randn(shape).bfloat16().to(dev)
        module = rms_norm.GemmaRMSNorm(shape[-1]).to(dev)
        scale = (1.0 + module.weight.float()).to(dev)
        times = (
            timed(lambda x=x, s=scale: rms_norm.rms_norm(x, s, 1e-6)),
            timed(lambda x=x, m=module: m(x)),
        )
        row("rms_norm", shape, times, bound(x, x))


def bench_swiglu(dev, tokens):
    print(
        "| kernel | shape | tile us | tile fp32 registers us | metal us | eager us "
        "| memory bound us |"
    )
    print("|---|---|---|---|---|---|---|")
    twin = swiglu.swiglu_km()
    for t in tokens:
        a = torch.randn(1, t, 4096).bfloat16().to(dev)
        b = torch.randn(1, t, 4096).bfloat16().to(dev)
        times = (
            timed(lambda a=a: swiglu.silu(a)),
            timed(lambda a=a: swiglu.silu_fp32(a)),
            timed(lambda a=a: swiglu.silu_km.op(a)),
            timed(lambda a=a: F.silu(a)),
        )
        row("silu", a.shape, times, bound(a, a))
        times = (
            timed(lambda a=a, b=b: swiglu.swiglu(a, b)),
            timed(lambda a=a, b=b: swiglu.swiglu_fp32(a, b)),
            timed(lambda a=a, b=b: twin(a, b)),
            timed(lambda a=a, b=b: F.silu(a) * b),
        )
        row("swiglu", a.shape, times, bound(a, b, a))


def bench_rope(dev, heads, tokens):
    print("| kernel | shape | tile us | metal us | eager us | memory bound us |")
    print("|---|---|---|---|---|---|")
    rope_km = rope.metal_example("rope_km").rope.op
    for n in tokens:
        x = torch.randn(1, heads, n, 64).bfloat16().to(dev)
        cos, sin, freqs = rope.tables(n, 64)
        c, s = cos.bfloat16().to(dev), sin.bfloat16().to(dev)
        times = (
            timed(lambda x=x, c=c, s=s: rope.rope_half(x, c, s)),
            None,
            timed(lambda x=x, c=c, s=s: rope.eager_half(x, c, s)),
        )
        row("rope rotate-half", x.shape, times, bound(x, x, c, s))
        table = torch.view_as_real(freqs).reshape(n, 64).contiguous().to(dev)
        f = freqs.to(dev)
        times = (
            timed(lambda x=x, t=table: rope.rope_complex(x, t)),
            timed(lambda x=x, t=table: rope_km(x, t)),
            timed(lambda x=x, f=f: rope.eager_complex(x, f)),
        )
        row("rope complex", x.shape, times, bound(x, x, table))


@torch.no_grad()
def bench_softmax(dev, heads, tokens):
    print(
        "| kernel | scores | tile us | eager step us | softmax (no state) us "
        "| memory bound us |"
    )
    print("|---|---|---|---|---|---|")
    for n in tokens:
        for keys in (256, 1024):
            state = online_softmax.inputs(n, keys, dev, heads)
            s = state[0]
            times = (
                timed(lambda a=state: online_softmax.softmax_step(*a, 0.125)),
                timed(lambda a=state: online_softmax.step_eager(*a, 0.125)),
                timed(lambda s=s: torch.softmax(s, -1)),
            )
            row("online softmax step", s.shape, times, bound(*state, *state))


@torch.no_grad()
def bench_layer_norm(tokens):
    print("| kernel | shape | tile us | eager us | memory bound us |")
    print("|---|---|---|---|---|")
    for width in (768, 1024, 4096):
        for t in tokens:
            shape = (1, t, width)
            times = layer_norm.bench(shape)
            x = torch.empty(shape, dtype=torch.bfloat16)
            row("layer_norm", shape, (times["tile"], times["eager"]), bound(x, x))


@torch.no_grad()
def bench_group_norm():
    print("| kernel | shape | tile us | eager us | memory bound us |")
    print("|---|---|---|---|---|")
    shapes = [(2, 320, 64, 64), (2, 640, 32, 32), (2, 1280, 16, 16), (2, 640, 64, 64),
              (2, 1280, 32, 32), (2, 320, 128, 128), (1, 512, 128, 128)]  # fmt: skip
    for shape in shapes:
        times = group_norm.bench(shape)
        x = torch.empty(shape, dtype=torch.bfloat16)
        row("group_norm", shape, (times["tile"], times["eager"]), bound(x, x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kernel", choices=("rms", "swiglu", "rope", "softmax", "ln", "gn")
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--tokens", type=int, nargs="*")
    args = parser.parse_args()
    with open("/proc/loadavg") as f:
        print("load", f.read().split()[:3], flush=True)
    dev = kohakutt.device(0)
    try:
        if args.kernel == "rms":
            bench_rms(dev, args.tokens or (512, 1024, 2048, 4096))
        elif args.kernel == "swiglu":
            bench_swiglu(dev, args.tokens or (512, 1024, 2048, 4096))
        elif args.kernel == "gn":
            bench_group_norm()
        elif args.kernel == "ln":
            bench_layer_norm(args.tokens or (512, 1024, 2048, 4096))
        elif args.kernel == "softmax":
            bench_softmax(dev, args.heads, args.tokens or (256, 1024, 4096))
        else:
            bench_rope(dev, args.heads, args.tokens or (197, 576, 1024, 2304, 5184))
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
