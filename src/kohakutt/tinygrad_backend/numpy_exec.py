"""Kernel normaliser (`Plan`: classifies a kernel AST, lists the library calls it
needs) and numpy reference executor (`Evaluator`: host fallback, oracle for the
bridge executor).
"""

from __future__ import annotations

import string

import numpy as np

from kohakutt.tinygrad_backend import compat as C
from kohakutt.tinygrad_backend.compat import AddrSpace, Invalid, Ops, UOp, dtypes

INDEX_ARITH = {Ops.ADD, Ops.MUL, Ops.RANGE, Ops.CONST, Ops.PARAM, Ops.CAST}
NOT_DATA = {
    Ops.INDEX,
    Ops.REDUCE,
    Ops.STORE,
    Ops.END,
    Ops.SINK,
    Ops.RANGE,
    Ops.CONST,
    Ops.PARAM,
    Ops.NOOP,
}
COMPARE_LOGIC = {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.AND, Ops.OR, Ops.WHERE}


def _rngs(x: UOp) -> set[UOp]:
    return {y for y in x.toposort() if y.op is Ops.RANGE}


class Plan:
    """Classification of a kernel AST: stores, ranges, buffer params, library calls."""

    def __init__(self, ast: UOp):
        self.ast = ast
        topo = list(ast.toposort())
        self.ranges = [u for u in topo if u.op is Ops.RANGE]
        self.params = C.kernel_params(ast)
        self.calls: list[str] = []
        self._count_calls(topo)

    @staticmethod
    def _index_kind(idx: UOp) -> str:
        ops = {u.op for u in idx.toposort()}
        if Ops.WHERE in ops:
            return "pad"
        if ops - INDEX_ARITH:
            return "gather"
        return "view"

    @staticmethod
    def _is_index_expr(u: UOp) -> bool:
        if u.dtype == dtypes.weakint:
            return True
        if u.dtype not in (dtypes.int32, dtypes.int64, dtypes.bool):
            return False
        allowed = INDEX_ARITH | COMPARE_LOGIC | C.INT_DIV | C.INT_MOD
        return all(x.op in allowed for x in u.toposort()) and not any(
            x.op is Ops.INDEX for x in u.toposort()
        )

    def _count_calls(self, topo):
        data_alu = [
            u
            for u in topo
            if u.op not in NOT_DATA
            and u.dtype != dtypes.weakint
            and u.dtype != dtypes.void
        ]
        data_alu = [u for u in data_alu if not self._is_index_expr(u)]
        parent = {u: u for u in data_alu}

        def find(u):
            while parent[u] is not u:
                u = parent[u]
            return u

        for u in data_alu:
            for s in u.src:
                if s in parent:
                    parent[find(s)] = find(u)
        islands = {find(u) for u in data_alu}
        reduces = [u for u in topo if u.op is Ops.REDUCE]

        def is_contraction(r):
            if not (r.arg[0] is Ops.ADD and r.src[0].op is Ops.MUL):
                return False
            a, b = r.src[0].src
            red = set(r.src[1:])
            return (
                bool(red & _rngs(a))
                and bool(red & _rngs(b))
                and (_rngs(a) - red) != (_rngs(b) - red)
            )

        contractions = [r for r in reduces if is_contraction(r)]
        for r in contractions:
            m = r.src[0]
            if m in parent and all(
                find(x) is not find(m) for x in data_alu if x is not m
            ):
                islands.discard(find(m))
        loads = [u for u in topo if u.op is Ops.INDEX and C.is_buffer_param(u.src[0])]
        stores = {u.src[0] for u in topo if u.op is Ops.STORE}
        for ld in loads:
            if ld in stores:
                continue
            kind = self._index_kind(ld.src[1]) if len(ld.src) > 1 else "view"
            if kind != "view":
                self.calls.append(kind)
        self.calls += ["matmul"] * len(contractions) + ["reduce"] * (
            len(reduces) - len(contractions)
        )
        self.calls += ["chain"] * len(islands)
        if any(self._index_kind(s.src[1]) != "view" for s in stores if len(s.src) > 1):
            self.calls.append("scatter")
        self.kind = (
            "matmul"
            if contractions
            else (
                "reduce"
                if reduces
                else (
                    "copy"
                    if not islands and not [c for c in self.calls if c != "view"]
                    else "elementwise"
                )
            )
        )
        if "gather" in self.calls or "pad" in self.calls:
            self.kind += "+gather/pad"


def _threefry2x32(x: np.ndarray, key: np.ndarray) -> np.ndarray:
    """tinygrad's THREEFRY on uint64 words (20 rounds of Threefry-2x32)."""
    u32, u64 = np.uint32, np.uint64
    x, key = np.asarray(x, u64), np.asarray(key, u64)
    x0, x1 = x.astype(u32), (x >> u64(32)).astype(u32)
    k0, k1 = key.astype(u32), (key >> u64(32)).astype(u32)
    ks = [k1, k0 ^ k1 ^ u32(0x1BD11BDA), k0]
    xr = [x0 + ks[-1], x1 + ks[0]]
    for i in range(5):
        for r in ((13, 15, 26, 6), (17, 29, 16, 24))[i % 2]:
            rot = (xr[1] << u32(r)) + (xr[1] >> u32(32 - r))
            xr[0], xr[1] = xr[0] + xr[1], (xr[0] + xr[1]) ^ rot
        xr = [xr[0] + ks[i % 3], xr[1] + ks[(i + 1) % 3] + u32(i + 1)]
    return (xr[1].astype(u64) << u64(32)) | xr[0].astype(u64)


class Evaluator:
    """Evaluate a kernel AST with numpy over named range axes (einsum for MUL-sums)."""

    NP = {
        Ops.ADD: np.add,
        Ops.MUL: np.multiply,
        Ops.SUB: np.subtract,
        Ops.MAX: np.maximum,
        Ops.CMPLT: np.less,
        Ops.CMPNE: np.not_equal,
        Ops.CMPEQ: np.equal,
        Ops.AND: np.bitwise_and,
        Ops.OR: np.bitwise_or,
        Ops.XOR: np.bitwise_xor,
        Ops.SHL: lambda a, b: np.left_shift(a, b.astype(a.dtype)),
        Ops.SHR: lambda a, b: np.right_shift(a, b.astype(a.dtype)),
        Ops.FDIV: np.divide,
        Ops.POW: np.power,
        Ops.THREEFRY: lambda x, k: _threefry2x32(x, k),
    }
    UN = {
        Ops.EXP2: np.exp2,
        Ops.LOG2: np.log2,
        Ops.SIN: np.sin,
        Ops.SQRT: np.sqrt,
        Ops.NEG: np.negative,
        Ops.TRUNC: np.trunc,
        C.RECIP: lambda x: np.divide(1.0, x),
    }

    def __init__(
        self, plan: Plan, bufs: dict[int, np.ndarray], var_vals: dict[str, int]
    ):
        self.plan, self.bufs, self.var_vals = plan, bufs, var_vals
        self.order = {r: i for i, r in enumerate(plan.ranges)}
        self.memo: dict[UOp, tuple[np.ndarray, tuple]] = {}

    def extent(self, r: UOp) -> int:
        return int(self.scalar(r.src[0]))

    def scalar(self, u: UOp):
        a, ax = self.ev(u)
        assert not ax
        return a.item() if isinstance(a, np.ndarray) else a

    @staticmethod
    def align(v, axes):
        a, ax = v
        a = np.asarray(a)
        perm = [ax.index(x) for x in axes if x in ax]
        a = np.transpose(a, perm) if a.ndim else a
        it = iter(a.shape)
        return a.reshape([next(it) if x in ax else 1 for x in axes])

    def union(self, *axs):
        return tuple(sorted(set().union(*axs), key=lambda r: self.order[r]))

    def ev(self, u: UOp):
        if u in self.memo:
            return self.memo[u]
        self.memo[u] = r = self._ev(u)
        return r

    def _ev(self, u: UOp):
        op = u.op
        if op is Ops.CONST:
            return (
                np.array(0 if u.arg is Invalid else u.arg, dtype=C.np_dtype(u.dtype)),
                (),
            )
        if op is Ops.PARAM:
            assert u.addrspace is AddrSpace.ALU, "buffer params only appear under INDEX"
            return (np.array(self.var_vals[C.param_var_name(u)]), ())
        if op is Ops.RANGE:
            return (np.arange(self.extent(u)), (u,))
        if op is Ops.INDEX:
            buf = self.bufs[C.param_slot(u.src[0])]
            if len(u.src) == 1:
                return (buf, ())
            i, ax = self.ev(u.src[1])
            return (buf[np.clip(np.asarray(i), 0, buf.size - 1)], ax)
        if op is Ops.WHERE and u.src[2].op is Ops.CONST and u.src[2].arg is Invalid:
            return self.ev(u.src[1])
        if op is Ops.CAST:
            a, ax = self.ev(u.src[0])
            return (np.asarray(a).astype(C.np_dtype(u.dtype)), ax)
        if op is Ops.BITCAST:
            a, ax = self.ev(u.src[0])
            return (np.ascontiguousarray(a).view(C.np_dtype(u.dtype)), ax)
        if op is Ops.REDUCE:
            return self._reduce(u)
        if op in self.UN:
            a, ax = self.ev(u.src[0])
            with np.errstate(all="ignore"):
                return (self.UN[op](a), ax)
        if op in C.INT_DIV or op in C.INT_MOD:
            (a, aa), (b, ba) = self.ev(u.src[0]), self.ev(u.src[1])
            ax = self.union(aa, ba)
            fn = np.floor_divide if op in C.INT_DIV else np.mod
            return (fn(self.align((a, aa), ax), self.align((b, ba), ax)), ax)
        if op is Ops.WHERE:
            vs = [self.ev(s) for s in u.src]
            ax = self.union(*[v[1] for v in vs])
            return (np.where(*[self.align(v, ax) for v in vs]), ax)
        if op in self.NP:
            vs = [self.ev(s) for s in u.src]
            ax = self.union(*[v[1] for v in vs])
            with np.errstate(all="ignore"):
                res = np.asarray(self.NP[op](*[self.align(v, ax) for v in vs]))
            if dtypes.is_int(u.dtype) and u.dtype not in dtypes.weaks:
                res = res.astype(C.np_dtype(u.dtype), copy=False)  # weak consts promote
            return (res, ax)
        raise NotImplementedError(f"op {op}")

    def _reduce(self, u: UOp):
        red = set(u.src[1:])
        rop = u.arg[0]
        if rop is Ops.ADD and u.src[0].op is Ops.MUL:
            (a, aa), (b, ba) = self.ev(u.src[0].src[0]), self.ev(u.src[0].src[1])
            allax = self.union(aa, ba, tuple(red))
            letters = {r: string.ascii_letters[i] for i, r in enumerate(allax)}
            out = tuple(r for r in allax if r not in red)
            sub = ["".join(letters[r] for r in axes) for axes in (aa, ba, out)]
            spec = f"{sub[0]},{sub[1]}->{sub[2]}"
            res = np.einsum(spec, np.asarray(a), np.asarray(b))
            for r in (r for r in red if r not in aa and r not in ba):
                res = res * self.extent(r)
            return (res, out)
        a, ax = self.ev(u.src[0])
        fn = {Ops.ADD: np.sum, Ops.MAX: np.max, Ops.MUL: np.prod}[rop]
        present = [ax.index(r) for r in ax if r in red]
        res = fn(np.asarray(a), axis=tuple(present)) if present else np.asarray(a)
        for r in red - set(ax):
            if rop is Ops.ADD:
                res = res * self.extent(r)
            elif rop is Ops.MUL:
                res = res ** self.extent(r)
        return (res, tuple(r for r in ax if r not in red))

    def run(self):
        for st in [u for u in self.plan.ast.toposort() if u.op is Ops.STORE]:
            idx = st.src[0]
            out = self.bufs[C.param_slot(idx.src[0])]
            val, vax = self.ev(st.src[1])
            if len(idx.src) == 1:
                out[0] = np.asarray(val).reshape(-1)[0]
                continue
            iv, iax = self.ev(idx.src[1])
            ax = self.union(vax, iax)
            i = np.broadcast_to(self.align((iv, iax), ax), [self.extent(r) for r in ax])
            v = np.broadcast_to(self.align((val, vax), ax), i.shape)
            out[i.reshape(-1)] = v.reshape(-1).astype(out.dtype)
