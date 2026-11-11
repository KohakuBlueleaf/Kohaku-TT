"""What tinygrad's DEBUG report shows for a KTT kernel: a tinygrad-style name and the
operation / memory estimates of its AST. KTT runs no tinygrad codegen, so both are
derived here from the AST the seam receives (library kernels: from their spec).
"""

from __future__ import annotations

import math

from kohakutt.tinygrad_backend import compat as C
from kohakutt.tinygrad_backend.compat import Estimates, Ops, UOp, dtypes


def _ranges_in_order(topo) -> list[UOp]:
    rngs = [
        u
        for u in topo
        if u.op is Ops.RANGE and u.dtype is not dtypes.void and u.vmax > 0
    ]
    return sorted(rngs, key=lambda u: (C.axis_to_pos[u.arg[-1]],) + tuple(u.arg[:-1]))


def kernel_name(ast: UOp) -> str:
    """tinygrad's name of an unoptimised kernel: r (reduce) or E, then the sizes of
    its ranges in axis order, coloured by axis type, as DEBUG=2 prints them."""
    topo = ast.toposort()
    kind = "r" if any(u.op is Ops.REDUCE for u in topo) else "E"
    sizes = [
        C.colored(r.src[0].render(), C.axis_colors[r.arg[-1]])
        for r in _ranges_in_order(topo)
    ]
    return kind + C.colored("_", "BLACK").join([""] + sizes)


def _extent(rngs) -> int | UOp:
    out = 1
    for r in rngs:
        if r.dtype is not dtypes.void:
            out = out * (r.src[0].arg if r.src[0].op is Ops.CONST else r.src[0])
    return C.ssimplify(out)


def estimates(ast: UOp) -> Estimates:
    """Operations and bytes of a kernel AST, counted like tinygrad's estimates: ops are
    ALU ops and reduce steps, each once per iteration of the ranges it depends on (index
    math left out); lds counts every pass of the loops around an access; mem does not.
    """
    topo = ast.toposort()
    excluded: set[UOp] = set()
    roles: dict[UOp, set[bool]] = {}  # buffer INDEX -> {is a store target}
    deps: dict[UOp, frozenset] = {}  # ranges a value depends on
    for u in topo:
        if u.op is Ops.INDEX:
            for s in u.src[1:]:
                excluded.update(s.toposort())
        for i, s in enumerate(u.src):
            store = u.op is Ops.STORE and i == 0
            s = s.src[0] if store and s.op is Ops.CAST else s
            if s.op is Ops.INDEX and C.is_buffer_param(s.src[0]):
                roles.setdefault(s, set()).add(store)
        deps[u] = frozenset().union(*(deps[s] for s in u.src))
        if u.op is Ops.RANGE:
            deps[u] = frozenset({u})
        elif u.op is Ops.REDUCE:
            deps[u] = deps[u.src[0]] - set(u.src[1:])
    loops: dict[UOp, frozenset] = {}  # ranges of the loops a node runs in
    for u in reversed(topo):
        loops[u] = loops.get(u, frozenset()) | deps[u]
        inside = loops[u] | set(u.src[1:]) if u.op is Ops.REDUCE else loops[u]
        for s in u.src:
            loops[s] = loops.get(s, frozenset()) | inside
    # Nesting order of the loops; an access runs in the loops around it out to the
    # innermost one it depends on (the loops inside that one do not repeat it).
    pos = {r: i for i, r in enumerate(_ranges_in_order(topo))}
    ops, lds, mem = 0, 0, {}
    for u in topo:
        if u not in excluded and (u.op in C.GroupOp.ALU or u.op is Ops.REDUCE):
            steps = _extent(deps[u] | set(u.src[1:]) if u.op is Ops.REDUCE else deps[u])
            ops = ops + (2 * steps if u.op is Ops.MULACC else steps)
        for store in roles.get(u, ()):
            buf, item = u.src[0], u.src[0].dtype.itemsize
            last = max((pos.get(r, -1) for r in deps[u]), default=-1)
            passes = deps[u] | {r for r in loops[u] if pos.get(r, -1) <= last}
            lds = lds + _extent(passes) * item
            size = buf.src[0].arg if buf.src and buf.src[0].op is Ops.CONST else None
            total = mem.get((buf, store), 0) + _extent(deps[u]) * item
            mem[(buf, store)] = total if size is None else C.smin(total, size * item)
    return Estimates(C.ssimplify(ops), C.ssimplify(lds), C.ssimplify(sum(mem.values())))


def lib_estimates(spec: dict, params) -> Estimates:
    """Estimates of a library kernel: the bytes of its buffers; multiply-adds of the
    convolutions and attentions (the backward of attention as 2.5 forwards)."""
    shapes, n_out = spec["shapes"], spec["n_out"]
    nbytes = sum(
        math.prod(s) * p.dtype.itemsize for s, p in zip(shapes, params, strict=False)
    )
    op, flops = spec["op"], 0
    if op in ("conv2d", "conv2d_bw"):
        out, w = (shapes[0], shapes[2]) if op == "conv2d" else (shapes[1], shapes[3])
        flops = 2 * math.prod(out) * math.prod(w[1:])
    elif op in ("sdpa", "sdpa_train", "sdpa_train_bw"):
        backward = op == "sdpa_train_bw"  # inputs: gradient, output, q, k, v, stats
        q, k = shapes[n_out + 2 * backward], shapes[n_out + 2 * backward + 1]
        flops = 4 * math.prod(q) * k[-2]
        flops = flops * 5 // 2 if backward else flops
    return Estimates(flops, nbytes, nbytes)
