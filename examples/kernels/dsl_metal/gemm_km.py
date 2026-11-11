"""a @ w at the metal level in the backend's block matmul layouts (ktt_gemm, 1-D or
2-D) with its plan choice: torch op ``kohakutt_example::gemm_km``.
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
# The backend's plan budget: free L1 rounded down to this many bytes; the budget
# planned for without a device.
BUDGET_CLASS, BUDGET = 32 * 1024, 1024 * 1024


@km.program("gemm")
def gemm_program(
    p, a: km.TensorRef, w: km.TensorRef, out: km.Out, two_d=False, panels=None
):
    """bm x bn output blocks (nbh x nbw per core), K in nk blocks of kb tiles, sh x sw
    subblocks; `w_nk`: w stored [N, K]. 1-D: core c owns row tiles c pcm ..; 2-D: core
    (x, y) rows y pcm .., columns x pcn .., a multicast from the last column of each
    `panels` span ((first, last); None: the row). w goes down each column."""
    kt, nt, mt = p.runtime("kt"), p.runtime("nt"), p.runtime("mt")
    w_nk, pcm, pcn = p.runtime("w_nk"), p.runtime("pcm"), p.runtime("pcn")
    sh, sw = p.choose(("sh", "sw"), SUBBLOCKS)
    bm, bn = p.runtime("bm", multiple_of="sh"), p.runtime("bn", multiple_of="sw")
    nbh, nbw = p.runtime("nbh"), p.runtime("nbw")
    cols, rows = p.machine.cols, p.machine.rows
    if rows < 2 and not two_d:
        raise ValueError("gemm_km: the w lane needs a grid of two rows or more")
    kb, nk, kb_last = p.runtime("kb"), p.runtime("nk"), p.runtime("kb_last")
    for value in (nk, nbh, nbw):
        p.limit(value, low=1)
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

    if two_d:
        x, y = p.core % cols, p.core // cols
        m0, n0 = p.define("m0", y * pcm), p.define("n0", x * pcn)
    else:
        m0, n0 = p.define("m0", p.core * pcm), 0
    m_end = p.define("m_end", km.min(m0 + pcm, mt))
    n_end = p.define("n_end", km.min(n0 + pcn, nt))

    # w: the top row reads each block and multicasts it down its column.
    top = p.cores.rows(0)
    lane = p.lane(top, p.cores.rows(1, rows - 1), name="w") if rows > 1 else None

    def w_columns(k, blk):
        c0 = k.let("c0", n0 + blk % nbw * bn)
        return c0, k.let("cols", km.select(c0 < n_end, km.min(bn, n_end - c0), 0))

    send = p.kernel("dm0", cores=top, noc=0, name="w_send")
    if lane:
        lane.open(send)
    with send.loop(blocks, name="blk") as blk:
        c0, wc = w_columns(send, blk)

        def w_pass(kblk, first, last):
            k0, kc = kblk * kb, kb_last if last else kb
            slot = send.reserve(cb_w, w_block)
            # Pages in storage order; tile (k, c) of the block at k * bn + c.
            nk_order = send.when(w_nk.ne(0))
            with nk_order, send.loop(wc, name="c") as c, send.loop(kc, name="k") as k:
                at = (k * bn + c).bounded(0, w_block)
                send.read(w.page((c0 + c) * kt + k0 + k), slot[at])
            kn_order = send.otherwise()
            with kn_order, send.loop(kc, name="k") as k, send.loop(wc, name="c") as c:
                at = (k * bn + c).bounded(0, w_block)
                send.read(w.page((k0 + k) * nt + c0 + c), slot[at])
            send.reads_done()
            if lane:
                lane.send(send, slot)
            send.push(cb_w, w_block)

        km.k_passes(send, nk, w_pass)
    send.writes_done()

    if lane:
        recv = p.kernel("dm0", cores=p.cores.rows(1, rows - 1), noc=0, name="w_recv")
        with recv.loop(blocks, name="blk"):

            def w_receive(kblk, first, last):
                recv.reserve(cb_w, w_block)
                lane.receive(recv)
                recv.push(cb_w, w_block)

            km.k_passes(recv, nk, w_receive)
        recv.atomics_done()

    # a and the output; 2-D: each panel's last column reads a and multicasts it along
    # its row inside the panel.
    spans = [(0, cols - 1)] if not two_d else panels or [(0, cols - 1)]
    if two_d and [x for s in spans for x in range(s[0], s[1] + 1)] != list(range(cols)):
        raise ValueError("gemm_km: panels cover the grid's columns in order, once")

    def a_and_out(k, role, a_lane=None):
        """Per block the a passes ("read": from DRAM, "send": and multicast,
        "receive": from the lane); a block's output goes out after the next
        block's first a pass."""

        def write_block(blk):
            r0 = k.let("r0", m0 + blk // nbw * bm)
            c0 = k.let("c0", n0 + blk % nbw * bn)
            with k.loop(bm, sh, name="i") as i, k.loop(bn, sw, name="j") as j:
                k.wait(cb_out, sub)
                with k.loop(sh, name="r") as r, k.loop(sw, name="c") as c:
                    row, col = r0 + i + r, c0 + j + c
                    with k.when(row < m_end), k.when(col < n_end):
                        at = (r * sw + c).bounded(0, sub)
                        k.write(cb_out, out.page(row * nt + col), at)
                k.flushed()
                k.pop(cb_out, sub)

        def a_passes(blk, after_first=None):
            r0 = k.let("r0", m0 + blk // nbw * bm)
            a_rows = k.let("rows", km.select(r0 < m_end, km.min(bm, m_end - r0), 0))

            def a_pass(kblk, first, last):
                k0, kc = kblk * kb, kb_last if last else kb
                slot = k.reserve(cb_a, a_block)
                if role == "receive":
                    a_lane.receive(k)
                else:
                    with k.loop(a_rows, name="r") as r, k.loop(kc, name="k") as i:
                        at = (r * kb + i).bounded(0, a_block)
                        k.read(a.page((r0 + r) * kt + k0 + i), slot[at])
                    k.reads_done()
                    if role == "send":
                        a_lane.send(k, slot)
                k.push(cb_a, a_block)
                if first and after_first is not None:
                    after_first()

            km.k_passes(k, nk, a_pass)

        if role == "send":
            a_lane.open(k)
        a_passes(k.let("blk", 0))
        with k.loop(blocks, start=1, name="blk") as blk:
            a_passes(blk, lambda: write_block(blk - 1))
        write_block(blocks - 1)
        k.writes_done()
        if role == "receive":
            k.atomics_done()

    if not two_d:
        a_and_out(p.kernel("dm1", noc=1), "read")
    for n, (x0, x1) in enumerate(spans if two_d else []):
        last_col = p.cores.cols(x1)
        if x1 == x0:
            a_and_out(p.kernel("dm1", cores=last_col, noc=1, name=f"a_read{n}"), "read")
            continue
        rest = p.cores.cols(x0, x1 - 1)
        a_lane = p.lane(last_col, rest, name=f"a{n}")
        sender = p.kernel("dm1", cores=last_col, noc=1, name=f"a_send{n}")
        a_and_out(sender, "send", a_lane)
        a_and_out(
            p.kernel("dm1", cores=rest, noc=1, name=f"a_recv{n}"), "receive", a_lane
        )

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


def _l1(bm, bn, kb, nk, sub, w_tile=TILE, held=False):
    """Buffer bytes per core of a plan as the backend counts them (the partials
    block only when K takes more than one block; `held`: always, as this program)."""
    part = bm * bn * PART_TILE if nk > 1 or held else 0
    return 2 * bm * kb * TILE + 2 * kb * bn * w_tile + 2 * sub * TILE + part


# The backend's measured configs for bf16 [K, N] weights, by (Mt, Kt, Nt): (one_d,
# bm, bn, nbh, nbw, kb, sh, sw).
MEASURED = {
    (32, 32, 32): (False, 4, 3, 1, 1, 16, 4, 1),
    (64, 64, 64): (False, 7, 6, 1, 1, 8, 1, 3),
    (128, 128, 128): (False, 13, 12, 1, 1, 2, 1, 4),
    (80, 96, 128): (False, 8, 12, 1, 1, 4, 1, 4),
    (512, 32, 32): (True, 5, 16, 1, 2, 8, 1, 4),
    (512, 12, 12): (True, 5, 12, 1, 1, 4, 1, 4),
    (32, 256, 128): (False, 4, 12, 1, 1, 8, 1, 4),
    (128, 32, 256): (False, 13, 8, 1, 3, 4, 1, 4),
}


def place(one_d, mt, nt, gx, gy):
    """(pcm, pcn, cols, rows): tiles per core and the grid of a layout."""
    if one_d:
        return math.ceil(mt / (gx * gy)), nt, gx, gy
    pcm, pcn = math.ceil(mt / gy), math.ceil(nt / gx)
    return pcm, pcn, math.ceil(nt / pcn), math.ceil(mt / pcm)


def plan(mt, kt, nt, gx, gy, budget=BUDGET, forced=None, one_d=None):
    """The runtime values of gemm_program for [mt, kt] @ [kt, nt] tiles on gx x gy
    cores as the backend's plan(): `forced` (one_d, bm, bn, nbh, nbw, kb, sh, sw), else
    its measured config and its layout (or `one_d`) and block search; None when
    nothing fits. Keys "one_d", "cols", "rows" give the layout."""

    def values(one_d, bm, bn, nbh, nbw, kb, sh, sw):
        pcm, pcn, cols, rows = place(one_d, mt, nt, gx, gy)
        kb = min(kb, kt)
        nk = math.ceil(kt / kb)
        whole = bm % sh == 0 and bn % sw == 0 and (sh, sw) in SUBBLOCKS
        fits = _l1(bm, bn, kb, nk, sh * sw, held=True) <= budget
        covers = bm * nbh >= pcm and bn * nbw >= pcn
        sized = bm * kb <= A_MOST and kb * bn <= W_MOST and bm * bn <= OUT_MOST
        if not (whole and fits and covers and sized):
            return None
        names = ("kt", "nt", "mt", "pcm", "pcn", "bm", "bn", "nbh", "nbw", "kb", "nk")
        out = dict(zip(names, (kt, nt, mt, pcm, pcn, bm, bn, nbh, nbw, kb, nk)))
        out.update(kb_last=kt - (nk - 1) * kb, sh=sh, sw=sw)
        return {**out, "one_d": one_d, "cols": cols, "rows": rows}

    if forced is not None:
        return values(*forced)
    if one_d is None and (mt, kt, nt) in MEASURED:
        out = values(*MEASURED[mt, kt, nt])
        if out is not None:
            return out
    # Layout by reuse: 2-D reuses an a tile pcn times and a w tile pcm times.
    pcm, pcn, _, _ = place(False, mt, nt, gx, gy)
    pcm_1d = math.ceil(mt / (gx * gy))
    if one_d is None:
        one_d = pcm_1d >= 3 and pcm_1d > min(pcm, pcn)
    pcm, pcn, _, _ = place(one_d, mt, nt, gx, gy)
    splits = []
    for nbh in range(1, pcm + 1):
        bm = math.ceil(pcm / nbh)
        if nbh > 1 and (nbh - 1) * bm >= pcm:
            continue
        for nbw in range(1, pcn + 1):
            bn = math.ceil(pcn / nbw)
            if (nbw > 1 and (nbw - 1) * bn >= pcn) or (one_d and nt >= 8 and nbw < 2):
                continue
            splits.append((nbh * nbw, nbw * mt + nbh * nt, nbh, nbw))
    for _count, _cost, nbh, nbw in sorted(splits):
        bm, bn = math.ceil(pcm / nbh), math.ceil(pcn / nbw)
        sh, sw = next(s for s in SUBBLOCKS if bm % s[0] == 0 and bn % s[1] == 0)
        for kb in (16, 8, 4, 2, 1):
            nk = math.ceil(kt / min(kb, kt))
            if _l1(bm, bn, min(kb, kt), nk, sh * sw) <= budget:
                # The backend's pick, unless this program's buffers do not fit it
                # (the partials block it always holds); then the next candidate.
                out = values(one_d, bm, bn, nbh, nbw, kb, sh, sw)
                if out is not None:
                    return out
    return None


DTYPES = {"a": "bfloat16", "w": "bfloat16", "out": "bfloat16"}
SCHEMA = "kohakutt_example::gemm_km(Tensor a, Tensor w, bool w_nk) -> Tensor"


# Column spans ((first, last), ...) a row's multicast stays inside: the backend's panels
# of physically contiguous columns on a p150 grid (None: the whole row).
PANELS = ((0, 6), (7, 10))


def panels_for(cols):
    """PANELS cut to a grid of `cols` columns."""
    if PANELS is None:
        return None
    return tuple((x0, min(x1, cols - 1)) for x0, x1 in PANELS if x0 < cols)


@functools.cache
def traced(cols, rows, two_d=False):
    """The program on a grid of `cols` x `rows` cores in either layout."""
    return gemm_program.trace(km.Machine(cols, rows), DTYPES, two_d=two_d)


def budget():
    """The backend's plan budget: the free L1 per core in whole 32 KiB."""
    return int(torch.ops.ktt.cb_budget()) // BUDGET_CLASS * BUDGET_CLASS


def values_for(a, w, w_nk, forced=None):
    """plan() for tt tensors `a` and `w` on the open device's grid."""
    k, n = a.shape[-1], w.shape[0] if w_nk else w.shape[1]
    mt = math.prod(a.shape[:-2]) * math.ceil(a.shape[-2] / 32)
    gx, gy = torch.ops.ktt.compute_grid()
    kt, nt = math.ceil(k / 32), math.ceil(n / 32)
    values = plan(mt, kt, nt, gx, gy, budget(), forced=forced)
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
    one_d, cols, rows = values.pop("one_d"), values.pop("cols"), values.pop("rows")
    return km.Call(
        inputs={"a": a, "w": w},
        outputs={"out": ([*a.shape[:-1], n], a.dtype)},
        runtime=values,
        static={"two_d": not one_d, "panels": None if one_d else panels_for(cols)},
        grid=(cols, rows),
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
