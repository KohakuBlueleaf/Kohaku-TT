"""a @ w at the metal level in the 1-D layout of the backend's block matmul (ktt_gemm,
`KTT_GEMM=1`): torch op ``kohakutt_example::gemm_km``; prints its agreement.
Usage: python examples/kernels/dsl_metal/gemm_km.py [--rows M] [--inner K] [--cols N]
"""

import argparse
import functools
import math
import os

import torch

import kohakutt
import kohakutt.dsl.metal as km

# Output subblocks (rows, columns) with fp32 registers, fewest unpacked tiles first.
SUBBLOCKS = ((1, 4), (4, 1), (2, 2), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1))
# Most tiles of an a block, a w block and an output block.
A_MOST, W_MOST, OUT_MOST = 512, 512, 256
# Bytes of a bf16 tile and of an fp32 partials tile.
TILE, PART_TILE = 2048, 4096
# Bytes of L1 per core the buffers may take.
BUDGET = 1024 * 1024


@km.program("gemm")
def gemm_program(p, a: km.TensorRef, w: km.TensorRef, out: km.Out):
    """Core c owns output row tiles c * pcm .. (c + 1) * pcm - 1 and reads its own a;
    the top core of each column multicasts w down it (a lane). Blocks of bm x bn
    tiles (nbh x nbw per core), K in nk blocks of kb tiles (the last kb_last), sh x
    sw subblocks; `w_nk`: w stored [N, K]."""
    kt, nt, mt = p.runtime("kt"), p.runtime("nt"), p.runtime("mt")
    w_nk, pcm = p.runtime("w_nk"), p.runtime("pcm")
    sh, sw = p.choose(("sh", "sw"), SUBBLOCKS)
    bm, bn = p.runtime("bm", multiple_of="sh"), p.runtime("bn", multiple_of="sw")
    nbh, nbw = p.runtime("nbh"), p.runtime("nbw")
    if p.machine.rows < 2:
        raise ValueError("gemm_km: the w lane needs a grid of two rows or more")
    kb, nk, kb_last = p.runtime("kb"), p.runtime("nk"), p.runtime("kb_last")
    p.limit(nk, low=1)
    p.limit(kb_last, low=1, high=kb)
    a_block = p.sized("a_block", of=bm * kb, exact=True, limit=A_MOST)
    w_block = p.sized("w_block", of=kb * bn, exact=True, limit=W_MOST)
    block = p.sized("block", of=bm * bn, exact=True, limit=OUT_MOST)
    sub = p.sized("sub", of=sh * sw, exact=True, limit=4)
    cb_a = p.cb(0, like=a, pages=2 * a_block)
    cb_w = p.cb(1, like=w, pages=2 * w_block)
    cb_out = p.cb(16, like=out, pages=2 * sub)
    part = p.partials(24, pages=block, name="part")
    blocks = nbh * nbw

    m0 = p.define("m0", p.core * pcm)
    m_end = p.define("m_end", km.min(m0 + pcm, mt))

    # w: the top row reads each block and multicasts it down its column.
    top, below = p.cores.rows(0), p.cores.rows(1, p.machine.rows - 1)
    lane = p.lane(top, below, name="w")

    def w_columns(k, blk):
        c0 = k.let("c0", blk % nbw * bn)
        return c0, k.let("cols", km.select(c0 < nt, km.min(bn, nt - c0), 0))

    send = p.kernel("dm0", cores=top, noc=0, name="w_send")
    lane.open(send)
    with send.loop(blocks, name="blk") as blk:
        c0, cols = w_columns(send, blk)

        def w_pass(kblk, first, last):
            k0, kc = kblk * kb, kb_last if last else kb
            slot = send.reserve(cb_w, w_block)
            # Pages in storage order; tile (k, c) of the block at k * bn + c.
            nk_order = send.when(w_nk.ne(0))
            with nk_order, send.loop(cols, name="c") as c, send.loop(kc, name="k") as k:
                at = (k * bn + c).bounded(0, w_block)
                send.read(w.page((c0 + c) * kt + k0 + k), slot[at])
            kn_order = send.otherwise()
            with kn_order, send.loop(kc, name="k") as k, send.loop(cols, name="c") as c:
                at = (k * bn + c).bounded(0, w_block)
                send.read(w.page((k0 + k) * nt + c0 + c), slot[at])
            send.reads_done()
            lane.send(send, slot)
            send.push(cb_w, w_block)

        km.k_passes(send, nk, w_pass)
    send.writes_done()

    recv = p.kernel("dm0", cores=below, noc=0, name="w_recv")
    with recv.loop(blocks, name="blk"):

        def w_receive(kblk, first, last):
            recv.reserve(cb_w, w_block)
            lane.receive(recv)
            recv.push(cb_w, w_block)

        km.k_passes(recv, nk, w_receive)
    recv.atomics_done()

    # a and the output: each core its own rows.
    k = p.kernel("dm1", noc=1)
    with k.loop(blocks, name="blk") as blk:
        r0 = k.let("r0", m0 + blk // nbw * bm)
        c0 = k.let("c0", blk % nbw * bn)
        rows = k.let("rows", km.select(r0 < mt, km.min(bm, mt - r0), 0))

        def a_pass(kblk, first, last):
            k0, kc = kblk * kb, kb_last if last else kb
            slot = k.reserve(cb_a, a_block)
            with k.loop(rows, name="r") as r, k.loop(kc, name="k") as i:
                at = (r * kb + i).bounded(0, a_block)
                k.read(a.page((r0 + r) * kt + k0 + i), slot[at])
            k.reads_done()
            k.push(cb_a, a_block)

        km.k_passes(k, nk, a_pass)
        with k.loop(bm, sh, name="i") as i, k.loop(bn, sw, name="j") as j:
            k.wait(cb_out, sub)
            with k.loop(sh, name="r") as r, k.loop(sw, name="c") as c:
                row, col = r0 + i + r, c0 + j + c
                with k.when(row < m_end), k.when(col < nt):
                    k.write(
                        cb_out, out.page(row * nt + col), (r * sw + c).bounded(0, sub)
                    )
            k.flushed()
            k.pop(cb_out, sub)
    k.writes_done()

    c = p.kernel("compute", fp32_dest=True, approx=False, unpack_fp32=[part])
    c.startup(cb_a, cb_w, cb_out, reverse=True)
    shape = {
        "bm": bm,
        "bn": bn,
        "sh": sh,
        "sw": sw,
        "kb": kb,
        "nk": nk,
        "kb_last": kb_last,
        "a_block": a_block,
        "w_block": w_block,
        "sub": sub,
        "transpose": w_nk,
    }
    with c.loop(blocks, name="blk"):
        km.matmul_passes(c, shape, cb_a, cb_w, part, cb_out)


def _l1(bm, bn, kb, nk, sub, w_tile=TILE):
    """Buffer bytes per core of a plan; the partials buffer counts even when nk is 1
    (the program always holds it)."""
    part = bm * bn * PART_TILE
    return 2 * bm * kb * TILE + 2 * kb * bn * w_tile + 2 * sub * TILE + part


def plan(mt, kt, nt, gx, gy, budget=BUDGET, forced=None):
    """The runtime values of gemm_program for [mt, kt] @ [kt, nt] tiles on gx x gy
    cores: `forced` (bm, bn, nbh, nbw, kb, sh, sw) when it covers the work and fits,
    else the backend's one-row-per-core search; None when nothing fits."""
    pcm = math.ceil(mt / (gx * gy))

    def values(bm, bn, nbh, nbw, kb, sh, sw):
        kb = min(kb, kt)
        nk = math.ceil(kt / kb)
        whole = bm % sh == 0 and bn % sw == 0 and (sh, sw) in SUBBLOCKS
        fits = _l1(bm, bn, kb, nk, sh * sw) <= budget
        covers = bm * nbh >= pcm and bn * nbw >= nt
        sized = bm * kb <= A_MOST and kb * bn <= W_MOST and bm * bn <= OUT_MOST
        if not (whole and fits and covers and sized):
            return None
        names = ("kt", "nt", "mt", "pcm", "bm", "bn", "nbh", "nbw", "kb", "nk")
        out = dict(zip(names, (kt, nt, mt, pcm, bm, bn, nbh, nbw, kb, nk)))
        return {**out, "kb_last": kt - (nk - 1) * kb, "sh": sh, "sw": sw}

    if forced is not None:
        return values(*forced)
    splits = []
    for nbh in range(1, pcm + 1):
        bm = math.ceil(pcm / nbh)
        if nbh > 1 and (nbh - 1) * bm >= pcm:
            continue
        for nbw in range(1, nt + 1):
            bn = math.ceil(nt / nbw)
            if (nbw > 1 and (nbw - 1) * bn >= nt) or (nt >= 8 and nbw < 2):
                continue
            splits.append((nbh * nbw, nbw * mt + nbh * nt, nbh, nbw))
    for _count, _cost, nbh, nbw in sorted(splits):
        bm, bn = math.ceil(pcm / nbh), math.ceil(nt / nbw)
        sh, sw = next(s for s in SUBBLOCKS if bm % s[0] == 0 and bn % s[1] == 0)
        for kb in (16, 8, 4, 2, 1):
            out = values(bm, bn, nbh, nbw, kb, sh, sw)
            if out is not None:
                return out
    return None


DTYPES = {"a": "bfloat16", "w": "bfloat16", "out": "bfloat16"}
SCHEMA = "kohakutt_example::gemm_km(Tensor a, Tensor w, bool w_nk) -> Tensor"


@functools.cache
def traced(cols, rows):
    """The program on a grid of `cols` x `rows` cores."""
    return gemm_program.trace(km.Machine(cols, rows), DTYPES)


def values_for(a, w, w_nk, forced=None):
    """plan() for tt tensors `a` and `w` on the open device's grid."""
    k, n = a.shape[-1], w.shape[0] if w_nk else w.shape[1]
    mt = math.prod(a.shape[:-2]) * math.ceil(a.shape[-2] / 32)
    cols, rows = torch.ops.ktt.compute_grid()
    values = plan(mt, math.ceil(k / 32), math.ceil(n / 32), cols, rows, forced=forced)
    return values and {**values, "w_nk": int(w_nk)}


@km.op(SCHEMA, gemm_program)
def gemm(a, w, w_nk):
    """The launch computing a @ w (w_nk: a @ w^T, w stored [N, K]) for bf16 tt
    tensors a [..., M, K] (K a multiple of 32) and a 2-D w."""
    if a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError("gemm_km: a and w must be bfloat16")
    values = values_for(a, w, w_nk)
    if a.shape[-1] % 32 or w.dim() != 2 or values is None:
        raise ValueError("gemm_km: K a multiple of 32, a 2-D w, a plan that fits")
    n = w.shape[0] if w_nk else w.shape[1]
    return km.Call(
        inputs={"a": a, "w": w},
        outputs={"out": ([*a.shape[:-1], n], a.dtype)},
        runtime=values,
    )


def main():
    """Runs the program and the backend's block matmul; prints their agreement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--inner", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=768)
    args = parser.parse_args()
    torch.manual_seed(0)
    a = torch.randn(args.rows, args.inner).bfloat16()
    w = (torch.randn(args.inner, args.cols) / 16).bfloat16()
    dev = kohakutt.device(0)
    try:
        a_tt, w_tt = a.to(dev), w.to(dev)
        got = torch.ops.kohakutt_example.gemm_km(a_tt, w_tt, False).cpu()
        want = (a_tt @ w_tt).cpu()
        shape = f"{tuple(a.shape)} @ {tuple(w.shape)}"
        print(f"gemm_km {shape}: equal to the backend: {torch.equal(got, want)}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    # The backend's block matmul (read at its first matmul).
    os.environ.setdefault("KTT_GEMM", "1")
    main()
