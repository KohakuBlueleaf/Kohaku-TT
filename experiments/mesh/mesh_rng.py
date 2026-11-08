"""Counter-based RNG on a 1x2 mesh: torch.ops.ktt.philox_uniform against a NumPy
Philox4x32-10 reference (bit for bit), sharded draws against replicated ones, the two
chips' replicas against each other, and torch.rand / torch.randn with automatic
placement against replicated placement and against a 1-card session under the
same seed (random ops on Philox: KTT_PHILOX=1, set here).

    python experiments/mesh/mesh_rng.py
"""

import os

os.environ.setdefault("KTT_PHILOX", "1")

import numpy as np
import torch

import kohakutt

M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF


def philox(c0, c1, key):
    """The four words of Philox4x32-10 at counters (c0, c1, 0, 0), uint32 arrays."""
    x0 = c0.astype(np.uint64)
    x1 = c1.astype(np.uint64)
    x2 = np.zeros_like(x0)
    x3 = np.zeros_like(x0)
    k0, k1 = key & MASK, (key >> 32) & MASK
    for _ in range(10):
        p0 = np.uint64(M0) * x0
        p1 = np.uint64(M1) * x2
        x0, x1, x2, x3 = (
            (p1 >> np.uint64(32)) ^ x1 ^ np.uint64(k0),
            p1 & np.uint64(MASK),
            (p0 >> np.uint64(32)) ^ x3 ^ np.uint64(k1),
            p0 & np.uint64(MASK),
        )
        k0, k1 = (k0 + W0) & MASK, (k1 + W1) & MASK
    return [w.astype(np.uint32) for w in (x0, x1, x2, x3)]


def reference(sizes, dtype, key, word):
    """Uniforms of `sizes` as the kernel draws them, as a torch tensor."""
    n = int(np.prod(sizes))
    index = np.arange(n, dtype=np.uint64)
    w = philox(index & np.uint64(MASK), index >> np.uint64(32), key)[word]
    values = ((w >> np.uint32(8)).astype(np.float64) * 2.0**-24).astype(np.float32)
    t = torch.from_numpy(values.reshape(sizes))
    if dtype == torch.bfloat16:
        bits = t.view(torch.int32) >> 16
        t = (bits << 16).view(torch.float32).to(torch.bfloat16)
    return t


def same_bits(a, b):
    return torch.equal(
        a.view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32),
        b.view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32),
    )


CASES = [
    ((64, 96), "R,S(0)"),
    ((64, 96), "R,S(1)"),
    ((37, 70), "R,S(0)"),
    ((37, 70), "R,S(1)"),
    ((3, 40, 50), "R,S(0)"),
    ((2, 6, 33, 64), "R,S(1)"),
    ((1000,), "R,S(0)"),
]


def kernel_checks():
    kat = philox(np.zeros(1, np.uint32), np.zeros(1, np.uint32), 0)
    print("Philox KAT (counter 0, key 0):", [hex(int(w[0])) for w in kat])
    for dtype in (torch.float32, torch.bfloat16):
        for sizes, spec in CASES:
            key, word = 0x123456789ABCDEF0, 1
            ref = reference(sizes, dtype, key, word)
            rep = torch.ops.ktt.philox_uniform(list(sizes), dtype, key, word)
            shd = torch.ops.ktt.philox_uniform(list(sizes), dtype, key, word, spec)
            chips = [torch.ops.ktt.local_shard(rep, c).cpu() for c in (0, 1)]
            print(
                f"{dtype!s:15s} {sizes!s:16s} {spec:7s} -> {kohakutt.placement(shd)}"
                f" | replicated == reference {same_bits(rep.cpu(), ref)}"
                f" | sharded == reference {same_bits(shd.cpu(), ref)}"
                f" | replicas equal {same_bits(chips[0], chips[1])}"
            )


def draw(seed):
    torch.ops.ktt.manual_seed(seed)
    u = torch.rand(512, 1024, device="tt")
    z = torch.randn(512, 1024, device="tt", dtype=torch.bfloat16)
    return u, z


def describe(name, u, z):
    zf = z.float()
    print(
        f"{name}: rand placement {kohakutt.placement(u)} mean {u.mean().item():.5f} "
        f"var {u.var().item():.5f} (1/12 = {1 / 12:.5f}) | randn placement "
        f"{kohakutt.placement(z)} mean {zf.mean().item():.5f} std {zf.std().item():.5f}"
    )


def main():
    kohakutt.mesh([0, 1], shape=(1, 2))
    try:
        kernel_checks()
        outs = {}
        for mode in ("replicated", "auto"):
            os.environ["KTT_MESH_AUTO"] = "0" if mode == "replicated" else "1"
            u, z = draw(1234)
            describe(mode, u, z)
            outs[mode] = (u.cpu(), z.cpu())
        os.environ.pop("KTT_MESH_AUTO", None)
    finally:
        kohakutt.close()
    kohakutt.device(0)
    try:
        u, z = draw(1234)
        describe("1 card", u, z)
        outs["1 card"] = (u.cpu(), z.cpu())
    finally:
        kohakutt.close()
    for mode in ("auto", "1 card"):
        u, z = outs[mode]
        ru, rz = outs["replicated"]
        print(
            f"{mode} vs mesh replicated: rand bit-identical {same_bits(u, ru)}, "
            f"randn bit-identical {same_bits(z, rz)}"
        )


if __name__ == "__main__":
    main()
