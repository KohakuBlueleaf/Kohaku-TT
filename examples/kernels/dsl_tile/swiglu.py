"""SiLU and SwiGLU written at the tile level (`import kohakutt.dsl as kd`), each beside
its metal-level version and the eager torch ops on the tt device: prints whether
they agree bit for bit and the three times.
Usage: python examples/kernels/dsl_tile/swiglu.py [--tokens T] [--width W]"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.dsl as kd
import kohakutt.dsl.metal as km
import kohakutt.kernels as kk

sys.path.insert(0, str(Path(__file__).parent))
from common import metal_example, timed

# Tiles per NoC barrier: at [1, 4096, 4096] SwiGLU took 250, 264, 319, 331 us with
# blocks of 1, 2, 4, 8 on a Blackhole p150, so one tile per barrier.
BLOCKS = 1


@kd.kernel
def silu(x: kd.Tensor, out: kd.Out, block: kd.Static[int] = BLOCKS):
    """out = x * sigmoid(x), tile by tile."""
    for t in kd.grid(kd.num_tiles(out), block=block):
        a = kd.load(x.tiles[t])
        kd.store(out.tiles[t], a * kd.sigmoid(a))


@kd.kernel
def swiglu(x: kd.Tensor, g: kd.Tensor, out: kd.Out, block: kd.Static[int] = BLOCKS):
    """out = silu(x) * g, tile by tile."""
    for t in kd.grid(kd.num_tiles(out), block=block):
        a = kd.load(x.tiles[t])
        b = kd.load(g.tiles[t])
        kd.store(out.tiles[t], a * kd.sigmoid(a) * b)


# The same kernels with fp32 registers, as the metal-level versions run them.
silu_fp32 = kd.kernel(silu.fn, name="silu_fp32", dest="fp32")
swiglu_fp32 = kd.kernel(swiglu.fn, name="swiglu_fp32", dest="fp32")


@km.program("silu")
def silu_program(p, a: km.TensorRef, out: km.Out):
    """The metal-level SiLU: the silu tile op on each tile."""
    start, count = p.split(p.runtime("tiles"))
    cb_a, cb_out = p.cb(0, like=a), p.cb(16, like=out)

    @p.dm0
    def reader(k):
        for i in k.range(count):
            slot = cb_a.reserve(1)
            k.read(a.page(start + i), slot)
            k.reads_done()
            cb_a.push(1)

    @p.compute(fp32_dest=True)
    def compute(c):
        c.startup(cb_a, None, cb_out)
        for _ in c.range(count):
            cb_a.wait(1)
            with c.registers():
                c.init("copy", cb_a)
                c.copy(cb_a, 0, 0)
                c.init("silu")
                c.silu(0)
                cb_out.reserve(1)
                c.pack(0, cb_out)
                cb_out.push(1)
            cb_a.pop(1)

    @p.dm1
    def writer(k):
        for i in k.range(count):
            cb_out.wait(1)
            k.write(cb_out, out.page(start + i))
            k.flushed()
            cb_out.pop(1)
        k.writes_done()


@km.op("kohakutt_example::silu_km(Tensor a) -> Tensor", silu_program)
def silu_km(a):
    """The metal-level SiLU of a tt tensor."""
    return km.Call(
        inputs={"a": a},
        outputs={"out": (list(a.shape), a.dtype)},
        runtime={"tiles": kk.tile_count(a)},
    )


def swiglu_km():
    """The torch op of the metal-level SwiGLU example."""
    return metal_example("swiglu_km").swiglu.op


def compare(tokens, width, dtype=torch.bfloat16, seed=0):
    """{name: (fp32-register kernel equal to the metal level, tile s, tile fp32 s,
    metal s, eager s)} at [1, T, W]."""
    g = torch.Generator().manual_seed(seed)
    dev = kohakutt.device(0)
    a = (torch.randn(1, tokens, width, generator=g) * 3).to(dtype).to(dev)
    b = torch.randn(1, tokens, width, generator=g).to(dtype).to(dev)
    twin = swiglu_km()
    out = {}
    equal = torch.equal(silu_fp32(a).cpu(), silu_km.op(a).cpu())
    out["silu"] = (
        equal,
        timed(lambda: silu(a)),
        timed(lambda: silu_fp32(a)),
        timed(lambda: silu_km.op(a)),
        timed(lambda: F.silu(a)),
    )
    equal = torch.equal(swiglu_fp32(a, b).cpu(), twin(a, b).cpu())
    out["swiglu"] = (
        equal,
        timed(lambda: swiglu(a, b)),
        timed(lambda: swiglu_fp32(a, b)),
        timed(lambda: twin(a, b)),
        timed(lambda: F.silu(a) * b),
    )
    return out


def main():
    """Runs both kernels at one shape; prints agreement and times."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--width", type=int, default=4096)
    args = parser.parse_args()
    try:
        shape = f"[1, {args.tokens}, {args.width}]"
        for name, (equal, *times) in compare(args.tokens, args.width).items():
            ms = ", ".join(
                f"{what} {t * 1e3:.3f} ms"
                for what, t in zip(("tile", "tile fp32", "metal", "eager"), times)
            )
            print(
                f"{name} {shape}: fp32 kernel equal to the metal level: {equal}; {ms}"
            )
        print(
            swiglu.plan(
                kd.like((1, args.tokens, args.width)),
                kd.like((1, args.tokens, args.width)),
            ).report()
        )
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
