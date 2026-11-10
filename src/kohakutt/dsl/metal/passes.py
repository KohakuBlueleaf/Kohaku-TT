"""K-block passes: the loop over a matmul's K blocks every kernel of a block matmul
runs in the same form, and the compute side that sums the blocks' products through
a partials buffer by packer accumulation."""

import contextlib

from kohakutt.dsl.metal.binding import Sized, as_expr


def k_passes(k, nk, body):
    """Runs `body(kblk, first, last)` for K blocks 0 .. `nk` - 1 of builder `k`: one
    pass when `nk` is 1, else the first, the middle ones in a loop, the last. `nk`
    is declared at least 1 (p.limit)."""
    if (getattr(nk, "at_least", None) or 0) < 1:
        raise ValueError("kohakutt.dsl.metal: k_passes needs nk declared >= 1")
    with k.when(nk.eq(1)):
        body(as_expr(0), True, True)
    with k.otherwise():
        body(as_expr(0), True, False)
        with k.loop(nk - 1, start=1, name="kblk") as kblk:
            body(kblk, False, False)
        body(nk - 1, False, True)


@contextlib.contextmanager
def _partials(c):
    c._partials = True
    try:
        yield
    finally:
        c._partials = False


def matmul_passes(c, shape, cb_a, cb_w, part, cb_out):
    """One output block of `bm` x `bn` tiles on compute builder `c`: per K block the
    products of `sh` x `sw` subblocks, summed into `part` (p.partials) by packer
    accumulation; the last K block reloads them and packs into `cb_out`. `shape`:
    {bm, bn, sh, sw, kb, nk, kb_last, a_block, w_block, sub, transpose} values."""
    bm, bn, sh, sw = shape["bm"], shape["bn"], shape["sh"], shape["sw"]
    kb, sub = shape["kb"], shape["sub"]
    if not (isinstance(part.pages, Sized) and part.pages.exact and part.partials):
        raise ValueError("kohakutt.dsl.metal: the partials buffer is p.partials(...)")
    if part.pages.of.key() != as_expr(bm * bn).key():
        raise ValueError("kohakutt.dsl.metal: the partials buffer holds bm * bn pages")
    if not (isinstance(sub, Sized) and sub.of.key() == as_expr(sh * sw).key()):
        raise ValueError("kohakutt.dsl.metal: sub is the sized value sh * sw")
    for what, of in (("a_block", bm * kb), ("w_block", kb * bn)):
        block = shape[what]
        if not (
            isinstance(block, Sized) and block.exact and block.of.key() == of.key()
        ):
            raise ValueError(
                f"kohakutt.dsl.metal: {what} is the exact sized {of.cpp()}"
            )
    if getattr(shape["kb_last"], "at_most", None) != getattr(kb, "name", 0):
        raise ValueError("kohakutt.dsl.metal: kb_last is declared at most kb (p.limit)")

    def init():
        return c.matmul_init(cb_a, cb_w, shape["transpose"], ct=sw, rt=sh, kt=kb)

    state = {}

    def one(kblk, first, last, partial):
        if first:
            state["product"] = init()
        c.wait(cb_a, shape["a_block"])
        c.wait(cb_w, shape["w_block"])
        if partial and not first and not last:
            # The previous pass's pages leave; this pass adds onto the same slots.
            with _partials(c):
                c.wait(part, part.pages)
                c.pop(part, part.pages)
        inner = shape["kb_last"] if last else kb
        with c.loop(bm, sh, name="i") as i, c.loop(bn, sw, name="j") as j:
            c.acquire()
            if last and partial:
                c.srca_format(part)
                c.init("copy", part)
                with _partials(c):
                    c.wait(part, sub)
                c.copy_block(part, 0, 0, sub)
                with _partials(c):
                    c.pop(part, sub)
                c.srca_format(cb_w)
                state["product"] = init()
            with c.loop(inner, name="k") as k:
                # The block's tiles from these stay in the waited pages.
                a_span = (sh - 1) * kb + 1
                a_at = (i * kb + k).bounded(0, shape["a_block"] - a_span + 1)
                w_at = (j + k * bn).bounded(0, shape["w_block"] - sw + 1)
                c.matmul(state["product"], a_at, w_at, 0)
            c.commit()
            target = cb_out if last else part
            with _partials(c):
                c.reserve(target, sub)
            if last and partial:
                c.pack_format(cb_out)
                c.pack_accumulate(0)
            elif not last:
                c.pack_accumulate(0 if first else 1, part)
            with c.loop(sub, name="t") as t:
                c.pack(t, target)
            c.release()
            with _partials(c):
                c.push(target, sub)
        c.pop(cb_a, shape["a_block"])
        c.pop(cb_w, shape["w_block"])

    with c.when(shape["nk"].eq(1)):
        c.pack_format(cb_out)
        one(as_expr(0), True, True, False)
    with c.otherwise():
        c.pack_format(part)
        one(as_expr(0), True, False, True)
        with c.loop(shape["nk"] - 1, start=1, name="kblk") as kblk:
            one(kblk, False, False, True)
        one(shape["nk"] - 1, False, True, True)
        c.pack_accumulate(0)
