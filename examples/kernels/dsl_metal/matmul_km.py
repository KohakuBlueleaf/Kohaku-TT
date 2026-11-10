"""a [..., M, K] @ w at the metal level: the backend's matmul program (ktt_matmul) as a
model (torch op ``kohakutt_example::matmul_km``); prints its agreement and both times.
Usage: python examples/kernels/dsl_metal/matmul_km.py [--rows M] [--inner K] [--cols N]
"""

import argparse
import functools
import math
import os
import time

import torch

import kohakutt
import kohakutt.dsl.metal as km

# Column tiles a core may take: its block of w and its register slots (fp32: 4).
COLUMNS = (4, 2, 1)
# Most inner (K) tiles per push of a, and of a product (K <= 1024).
CHUNK = 32
# Bytes of L1 per core the buffers may take.
BUDGET = 900 * 1024


@km.program("matmul")
def matmul_program(p, a: km.TensorRef, w: km.TensorRef, out: km.Out):
    """Column groups of `nt` tiles; `per_group` cores share a group's row tiles. A
    core holds its block of w (all K tiles of its columns) and sums one output row
    tile at a time in the registers. Runtime values: K tiles `kt`, N tiles `ntot`,
    row tiles `mt`, `groups`, `per_group`, the chunk `kc`, `w_nk` (w stored [N, K])."""
    kt, ntot, mt = p.runtime("kt"), p.runtime("ntot"), p.runtime("mt")
    nt = p.runtime("nt", choices=COLUMNS)
    kc = p.runtime("kc", choices=range(1, CHUNK + 1))
    groups, per_group = p.runtime("groups"), p.runtime("per_group")
    w_nk = p.runtime("w_nk")
    p.layout(cores=groups * per_group)

    # This core's place: its group's columns, its share of the row tiles.
    place = p.define("place", p.core % per_group)
    n0 = p.define("n0", p.core // per_group * nt)
    cols = p.define("cols", km.min(nt, ntot - n0))
    m0 = p.define("m0", mt * place // per_group)
    rows = p.define("rows", mt * (place + 1) // per_group - m0)

    a_pages = p.sized("a_pages", of=kc, limit=CHUNK)
    w_pages = p.sized("w_pages", of=kt * nt, limit=CHUNK * max(COLUMNS))
    cb_a = p.cb(0, like=a, pages=2 * a_pages)
    cb_w = p.cb(1, like=w, pages=w_pages)
    cb_out = p.cb(16, like=out, pages=2 * max(COLUMNS))

    @p.dm0
    def reader(k):
        block = cb_w.reserve(w_pages)
        for i in k.range(kt, name="k"):
            for j in k.range(cols, name="j"):
                n = k.let("n", n0 + j)
                page = km.select(w_nk.ne(0), n * kt + i, i * ntot + n)
                k.read(w.page(page), block[(i * cols + j).bounded(0, w_pages)])
        k.reads_done()
        cb_w.push(w_pages)
        for m in k.range(rows, name="m"):
            for k0, n in k.blocks(kt, kc, names=("k0", "count")):
                chunk = cb_a.reserve(a_pages)
                for i in k.range(n, name="k"):
                    k.read(a.page((m0 + m) * kt + k0 + i), chunk[i])
                k.reads_done()
                cb_a.push(a_pages)

    @p.compute(fp32_dest=True, approx=False)
    def compute(c):
        # The regular source order, as ktt_matmul: equal for a and w of one format
        # (UNPACK-1 refuses others); a matmul of mixed formats needs reverse=True.
        c.startup(cb_a, cb_w, cb_out)
        product = c.matmul_init(cb_a, cb_w, transpose=w_nk, ct=cols, rt=1, kt=kc)
        cb_w.wait(w_pages)
        for _ in c.range(rows, name="m"):
            cb_out.reserve(nt)
            with c.registers():
                for k0, n in c.blocks(kt, kc, names=("k0", "count")):
                    cb_a.wait(a_pages)
                    for i in c.range(n, name="k"):
                        # The cols tiles from `at` stay in the block: k0 + i < kt.
                        at = ((k0 + i) * cols).bounded(0, w_pages - cols + 1)
                        c.matmul(product, i, at, 0)
                    cb_a.pop(a_pages)
                c.commit()
                for j in c.range(cols, name="j"):
                    c.pack(j, cb_out)
            cb_out.push(nt)
        cb_w.pop(w_pages)

    @p.dm1
    def writer(k):
        for m in k.range(rows, name="m"):
            cb_out.wait(nt)
            for j in k.range(cols, name="j"):
                k.write(cb_out, out.page((m0 + m) * ntot + n0 + j), index=j)
            k.writes_done()
            cb_out.pop(nt)


def plan(a_shape, w_shape, w_nk, cores, budget=BUDGET, program=None):
    """The runtime values of matmul_program for a of device shape `a_shape` and a 2-D
    w ([K, N], or [N, K] with `w_nk`) on `cores` cores, or None when K or N is past
    1024, K is not whole tiles, or no column count fits `budget` bytes of L1."""
    k, n = a_shape[-1], w_shape[0] if w_nk else w_shape[1]
    kt, ntot = math.ceil(k / 32), math.ceil(n / 32)
    if k % 32 or kt > CHUNK or ntot > 32 or len(w_shape) != 2:
        return None
    mt = math.prod(a_shape[:-2]) * math.ceil(a_shape[-2] / 32)
    for nt in COLUMNS:
        groups = math.ceil(ntot / nt)
        values = {
            "kt": kt,
            "ntot": ntot,
            "mt": mt,
            "nt": nt,
            "kc": min(kt, CHUNK),
            "groups": groups,
            # As ktt_matmul: every core of a group reads its w block, rows or not.
            "per_group": max(1, cores // groups),
            "w_nk": int(w_nk),
        }
        fits = program is None or program.footprint(program.sizes(values)) <= budget
        if nt <= ntot and groups <= cores and fits:
            return values
    return None


DTYPES = {"a": "bfloat16", "w": "bfloat16", "out": "bfloat16"}
SCHEMA = "kohakutt_example::matmul_km(Tensor a, Tensor w, bool w_nk) -> Tensor"


@functools.cache
def traced(cols, rows):
    """The program on a grid of `cols` x `rows` cores (its buffer sizes per launch
    come from it)."""
    return matmul_program.trace(km.Machine(cols, rows), DTYPES)


def values_for(a, w, w_nk):
    """plan() for tt tensors `a` and `w` on the open device's grid."""
    cols, rows = torch.ops.ktt.compute_grid()
    return plan(a.shape, w.shape, w_nk, cols * rows, program=traced(cols, rows))


@km.op(SCHEMA, matmul_program)
def matmul(a, w, w_nk):
    """The launch computing a @ w (w_nk: a @ w^T, w stored [N, K]) for bf16 tt
    tensors a [..., M, K] and a 2-D w."""
    if a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError("matmul_km: a and w must be bfloat16")
    values = values_for(a, w, w_nk)
    if values is None:
        raise ValueError("matmul_km: K a multiple of 32, K and N up to 1024, 2-D w")
    n = w.shape[0] if w_nk else w.shape[1]
    return km.Call(
        inputs={"a": a, "w": w},
        outputs={"out": ([*a.shape[:-1], n], a.dtype)},
        runtime=values,
    )


def timed(fn, reps=20):
    """Mean seconds per call of `fn` after one warm-up call; the last result is
    read back inside the timed span."""
    fn().cpu()
    start = time.perf_counter()
    for _ in range(reps):
        out = fn()
    out.cpu()
    return (time.perf_counter() - start) / reps


def main():
    """Runs both programs on tt tensors; prints their agreement and times."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--inner", type=int, default=768)
    parser.add_argument("--cols", type=int, default=1024)
    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.randn(args.rows, args.inner).bfloat16()
    w = (torch.randn(args.inner, args.cols) / 16).bfloat16()

    dev = kohakutt.device(0)
    try:
        a_tt, w_tt = a.to(dev), w.to(dev)
        got = torch.ops.kohakutt_example.matmul_km(a_tt, w_tt, False).cpu()
        want = (a_tt @ w_tt).cpu()
        shape = f"{tuple(a.shape)} @ {tuple(w.shape)}"
        print(f"matmul_km {shape}: equal to the backend: {torch.equal(got, want)}")
        own = timed(lambda: torch.ops.kohakutt_example.matmul_km(a_tt, w_tt, False))
        backend = timed(lambda: a_tt @ w_tt)
        print(f"  time: metal level {own * 1e3:.3f} ms, backend {backend * 1e3:.3f} ms")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    # The backend runs its own matmul program from this many rows on.
    os.environ.setdefault("KTT_OWN_MATMUL_MIN_ROWS", "1")
    main()
