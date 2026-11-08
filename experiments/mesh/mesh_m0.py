"""Kohaku-TT mesh basics on a 1x2 mesh: placed host transfers, chip-local shards,
automatic placement, the gather fallback, strict mode, RNG replicas and mesh memory
totals.

Each case prints what it measured: bit identity of round trips and shards, the
placement chosen, mesh statistics, and error statistics (kohakutt.diff_stats) of op
results against the CPU float64 ruler next to the CPU same-dtype baseline.

    python experiments/mesh/mesh_m0.py                 # every case
    python experiments/mesh/mesh_m0.py --cases round   # cases whose name starts so
    KTT_FABRIC=0 python experiments/mesh/mesh_m0.py    # host path for conversions
"""

import argparse
import os
import time

import torch

import kohakutt

SHAPES = [
    ((64, 96), 0),
    ((64, 96), 1),
    ((50, 70), 0),
    ((3, 50, 70), 2),
    ((33,), 0),
]
DTYPES = [torch.float32, torch.bfloat16, torch.int32]


def _host(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.int32:
        return torch.randint(-1000, 1000, shape, dtype=dtype, generator=g)
    return torch.randn(shape, generator=g).to(dtype)


def round_trips():
    for shape, dim in SHAPES:
        for dtype in DTYPES:
            x = _host(shape, dtype)
            t = kohakutt.shard(x.to("tt"), dim)
            print(
                f"round trip {tuple(shape)} S({dim}) {dtype}: placement "
                f"{kohakutt.placement(t)}, bit-identical {torch.equal(t.cpu(), x)}"
            )
    x = _host((32, 64), torch.float32)
    t = torch.ops.ktt.distribute(x.to("tt"), "R,P(sum)")
    print(
        f"round trip (32, 64) P(sum): placement {kohakutt.placement(t)}, "
        f"bit-identical {torch.equal(t.cpu(), x)}"
    )
    a = kohakutt.shard(_host((48, 80), torch.float32).to("tt"), 0)
    b = kohakutt.shard(a, 1)
    print(
        f"redistribute S(0) -> S(1): placement {kohakutt.placement(b)}, "
        f"bit-identical {torch.equal(b.cpu(), a.cpu())}"
    )


def chip_shards():
    x = _host((50, 70), torch.float32)
    for dim in (0, 1):
        t = kohakutt.shard(x.to("tt"), dim)
        parts = [torch.ops.ktt.local_shard(t, c) for c in range(2)]
        split = x.split(-(-x.shape[dim] // 2), dim)
        same = all(
            torch.equal(p.narrow(dim, 0, s.shape[dim]), s)
            for p, s in zip(parts, split, strict=True)
        )
        print(
            f"chip shards (50, 70) S({dim}): local shapes "
            f"{[tuple(p.shape) for p in parts]}, valid parts bit-identical {same}"
        )


def auto_placement():
    for shape in [(1024, 512), (513, 1024), (64, 64)]:
        t = _host(shape, torch.float32).to("tt")
        print(f"auto {shape} fp32: {kohakutt.placement(t)}")
    e = torch.empty(2048, 1024, device="tt")
    e.fill_(1.0)
    print(f"auto empty (2048, 1024) then fill_: {kohakutt.placement(e)}")


def gather_fallback():
    x = _host((64, 64), torch.float32)
    t = kohakutt.shard(x.to("tt"), 0)
    torch.tt.reset_mesh_stats()
    y = (t * 2 + 1).cpu()
    ref = x.double() * 2 + 1
    print(
        f"fallback t*2+1 S(0): device "
        f"{kohakutt.format_stats(kohakutt.diff_stats(y, ref))}"
    )
    base = kohakutt.diff_stats(x * 2 + 1, ref)
    print(f"fallback t*2+1 S(0): cpu fp32 {kohakutt.format_stats(base)}")
    print(f"fallback stats after one op: {torch.tt.mesh_stats()}")
    (t - 1).cpu()
    print(
        f"fallback stats after a second op on the same input: {torch.tt.mesh_stats()}"
    )
    t = kohakutt.shard(x.to("tt"), 1)
    t[:8] = 0
    x2 = x.clone()
    x2[:8] = 0
    print(f"partial view write into S(1): bit-identical {torch.equal(t.cpu(), x2)}")


def strict_mode():
    os.environ["KTT_MESH_STRICT"] = "1"
    t = kohakutt.shard(_host((64, 64), torch.float32).to("tt"), 0)
    try:
        (t + 1).cpu()
        print("strict: no error raised")
    except RuntimeError as e:
        print(f"strict: raised: {str(e).splitlines()[0]}")
    finally:
        os.environ.pop("KTT_MESH_STRICT")


def rng_replicas():
    for draw in ("randn", "rand", "dropout"):
        torch.tt.manual_seed(7)
        if draw == "randn":
            t = torch.randn(96, 160, device="tt")
        elif draw == "rand":
            t = torch.rand(96, 160, device="tt")
        else:
            t = torch.nn.functional.dropout(torch.ones(96, 160, device="tt"), 0.5)
        same = torch.equal(
            torch.ops.ktt.local_shard(t, 0), torch.ops.ktt.local_shard(t, 1)
        )
        print(f"rng {draw}: placement {kohakutt.placement(t)}, chips identical {same}")


def memory_totals():
    torch.tt.synchronize()
    stats = kohakutt.memory_stats()
    chip = kohakutt.memory_stats(per_chip=True)
    print(f"memory mesh {stats}")
    print(f"memory per chip {chip}")
    before = torch.tt.memory_allocated()
    x = torch.empty(4096, 1024, device="tt")
    x.fill_(0.0)
    torch.tt.synchronize()
    print(
        f"memory: 16 MiB fp32 tensor ({kohakutt.placement(x)}) adds "
        f"{torch.tt.memory_allocated() - before} B over the mesh"
    )


def conversions():
    """Each conversion between placements, on the device where collectives cover it:
    bit identity against the host values and the conversion kinds counted."""
    pairs = [
        ("R,R", "R,S(0)"),
        ("R,R", "R,S(1)"),
        ("R,S(0)", "R,R"),
        ("R,S(1)", "R,R"),
        ("R,S(0)", "R,S(1)"),
        ("R,P(sum)", "R,S(0)"),
        ("R,P(sum)", "R,R"),
    ]
    for shape in [(64, 96), (128, 256), (50, 70)]:
        for dtype in (torch.float32, torch.bfloat16):
            for src, dst in pairs:
                x = _host(shape, dtype)
                # The source is built through the host, so it is exact.
                os.environ["KTT_MESH_CCL"] = "0"
                a = torch.ops.ktt.distribute(x.to("tt"), src)
                torch.tt.synchronize()
                os.environ.pop("KTT_MESH_CCL")
                torch.tt.reset_mesh_stats()
                b = torch.ops.ktt.distribute(a, dst)
                got = b.cpu()
                kinds = {k: v for k, v in torch.tt.mesh_stats().items() if v}
                line = (
                    f"convert {shape} {dtype} {src} -> {dst}: {kohakutt.placement(b)}, "
                    f"bit-identical {torch.equal(got, x)}, {kinds}"
                )
                if not torch.equal(got, x):
                    line += (
                        f"\n    {kohakutt.format_stats(kohakutt.diff_stats(got, x))}"
                    )
                print(line)


def larger_than_one_card(scale=1.25, cols=8192):
    """A bf16 tensor of `scale` x one chip's allocatable DRAM: created, filled on the
    device, read back. Needs about 1.5 x its size in host RAM."""
    chip_total = kohakutt.memory_stats(per_chip=True)["total"]
    rows = int(scale * chip_total / (cols * 2)) // 64 * 64
    size = rows * cols * 2
    print(
        f"large: chip allocatable {chip_total} B, tensor ({rows}, {cols}) bf16 {size} B"
    )
    start = time.time()
    t = torch.empty(rows, cols, dtype=torch.bfloat16, device="tt")
    t.fill_(3.0)
    torch.tt.synchronize()
    print(
        f"large: created + filled in {time.time() - start:.1f} s, placement "
        f"{kohakutt.placement(t)}, memory {kohakutt.memory_stats()}"
    )
    start = time.time()
    host = t.cpu()
    print(
        f"large: read back in {time.time() - start:.1f} s, all equal 3.0 "
        f"{bool((host == 3.0).all())}, mesh stats {torch.tt.mesh_stats()}"
    )


CASES = {
    "round_trips": round_trips,
    "chip_shards": chip_shards,
    "conversions": conversions,
    "auto_placement": auto_placement,
    "gather_fallback": gather_fallback,
    "strict_mode": strict_mode,
    "rng_replicas": rng_replicas,
    "memory_totals": memory_totals,
}
# Run only when named in --cases.
HEAVY = {"larger_than_one_card": larger_than_one_card}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="", help="run cases whose name starts so")
    args = parser.parse_args()
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        for name, case in CASES.items():
            if name.startswith(args.cases):
                case()
        for name, case in HEAVY.items():
            if args.cases and name.startswith(args.cases):
                case()
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
