"""Bridge executor: `BridgeBackend` runs each kernel AST as bridge calls (`DevKernel`);
a buffer is host bytes and/or one device tensor (`Storage`). Kernels the bridge cannot
take raise `Unsupported` and run on the numpy evaluator, counted by reason.
"""

from __future__ import annotations

import collections
import functools
import itertools
import os
import time
import warnings
import weakref

import numpy as np

from kohakutt.tinygrad_backend import compat as C
from kohakutt.tinygrad_backend import errors
from kohakutt.tinygrad_backend.compat import Ops, UOp, dtypes
from kohakutt.tinygrad_backend.errors import DtypeError, DtypeWarning, HostKernelError
from kohakutt.tinygrad_backend.numpy_exec import Evaluator, Plan, evaluate, host_view

CHECK = os.environ.get("KTT_TINYGRAD_CHECK") == "1"
# Largest bridge gather (elements); a gather of 1048576 did not return in 100 s.
GATHER_MAX = 262144
# Largest index array analysed on the host to find a memory-op form.
ANALYZE_MAX = 1 << 22
# Host-built constants kept per plan (one per distinct variable binding).
CONST_CACHE_MAX = 256
FALLBACKS = collections.Counter()  # reason -> kernels
BRIDGE_CALLS = collections.Counter()  # bridge function -> calls
TIMES = collections.Counter()  # seconds in "device" and "fallback" kernels
ROUTES = collections.Counter()  # kernels / reads per load route (view, memory, gather)
# "route: reason" -> reads that route could not take
ROUTE_SKIPS = collections.Counter()


class Unsupported(Exception):
    """The kernel (or part of it) has no bridge mapping; the reason is the message."""


class DtypeUnsupported(Unsupported):
    """The kernel holds a value or buffer of `td`, which is not a device dtype."""

    def __init__(self, td):
        super().__init__(f"dtype {C.dtype_name(td)}")
        self.td = td


def dtype_fallback(td) -> str:
    """The fallback reason of a kernel on non-device dtype `td`, after the policy:
    DtypeError when the host cannot run it or KTT_TINYGRAD_DTYPES=error, else a
    DtypeWarning (shown once per dtype)."""
    name = C.dtype_name(td)
    if C.np_storage(td) is None:
        raise errors.not_modelled(name)
    if errors.dtype_policy() == "error":
        raise errors.not_on_device(name)
    warnings.warn(errors.host_warning(name), DtypeWarning, stacklevel=3)
    return f"dtype {name}"


def _td_codes(core):
    return {
        dtypes.float32: core.FP32,
        dtypes.bfloat16: core.BF16,
        dtypes.int32: core.INT32,
        dtypes.weakfloat: core.FP32,
        dtypes.weakint: core.INT32,
    }


def _np_of(core, code):
    return {core.FP32: np.float32, core.INT32: np.int32, core.BF16: np.uint16}[code]


def pad2(shape) -> tuple[int, ...]:
    """Device shape of logical sizes: rank at least 2, leading ones."""
    shape = tuple(int(s) for s in shape)
    return shape if len(shape) >= 2 else (1,) * (2 - len(shape)) + shape


class CountingCore:
    """Wrap a core module: count every bridge call by name, pass the rest through."""

    def __init__(self, core):
        self._core = core

    def __getattr__(self, name):
        attr = getattr(self._core, name)
        if callable(attr) and not isinstance(attr, type):

            @functools.wraps(attr)
            def counted(*a, **k):
                BRIDGE_CALLS[name] += 1
                t0 = time.perf_counter()
                try:
                    return attr(*a, **k)
                finally:
                    TIMES["core"] += time.perf_counter() - t0

            return counted
        return attr


class Canon:
    """A buffer's device form: the tensor, its device shape, the tinygrad dtype."""

    __slots__ = ("handle", "shape", "td")

    def __init__(self, handle, shape, td):
        self.handle, self.shape, self.td = handle, tuple(shape), td


class Storage:
    """A tinygrad buffer (flat bytes) living on the host, the device, or both."""

    __slots__ = ("nbytes", "host", "canon", "views", "base", "offset", "__weakref__")

    def __init__(self, nbytes, base=None, offset=0):
        self.nbytes, self.base, self.offset = nbytes, base, offset
        # uint8; valid unless canon holds newer data
        self.host: np.ndarray | None = None
        self.canon: Canon | None = None
        self.views: dict = {}


class DVal:
    """A kernel value: a device tensor over `axes` (pad2 shape), or a scalar."""

    __slots__ = ("h", "axes", "td", "c")

    def __init__(self, h, axes, td, c=None):
        self.h, self.axes, self.td, self.c = h, tuple(axes), td, c

    @property
    def is_const(self):
        return self.c is not None


class BridgeBackend:
    name = "bridge"

    def __init__(self, core=None, device_ids=(0,), rows=1, cols=1):
        if core is None:
            from kohakutt.tinygrad_backend import core as real

            core = real
        self.raw_core = core
        self.core = CountingCore(core)
        self.codes = _td_codes(core)
        self.device_args = (list(device_ids), rows, cols)
        self.opened = False
        self.owns = False
        self.live = weakref.WeakSet()
        self.fallback_log: list[tuple[str, str]] = []

    # ---- device lifetime

    def ensure_open(self):
        if not self.opened:
            # A mesh the torch side opened is shared, and stays open on close().
            self.owns = not self.raw_core.is_open()
            self.raw_core.open(*self.device_args)
            self.opened = True

    def close(self):
        """Drop every device tensor (host copies kept), then close the device."""
        for st in list(self.live):
            if st.canon is not None and st.host is None:
                self.host_bytes(st)
            st.canon, st.views = None, {}
        self.live.clear()
        if self.opened:
            self.raw_core.synchronize()
            if self.owns:
                self.raw_core.close()
            self.opened = False

    def synchronize(self):
        if self.opened:
            self.raw_core.synchronize()

    # ---- allocator side

    def alloc(self, size: int):
        st = Storage(size)
        self.live.add(st)
        return st

    def free(self, st):
        st.canon, st.views, st.host = None, {}, None

    def copyin(self, st: Storage, src: memoryview):
        data = np.frombuffer(src, dtype=np.uint8)
        if st.base is not None:
            host = self.host_bytes(st.base)
            host[st.offset : st.offset + st.nbytes] = data
            self.invalidate(st.base)
            return
        st.host = data.copy()
        self.invalidate(st)

    def copyout(self, dest: memoryview, st: Storage):
        np.frombuffer(dest, dtype=np.uint8)[:] = self.host_bytes(st)[: len(dest)]

    def offset(self, st: Storage, size: int, offset: int):
        return Storage(size, base=st, offset=offset)

    @staticmethod
    def invalidate(st: Storage):
        st.canon, st.views = None, {}

    def host_bytes(self, st: Storage) -> np.ndarray:
        """The buffer's bytes on the host (downloaded if only the device has them)."""
        if st.base is not None:
            return self.host_bytes(st.base)[st.offset : st.offset + st.nbytes]
        if st.host is None:
            st.host = np.zeros(st.nbytes, dtype=np.uint8)
            if st.canon is not None:
                self.ensure_open()
                raw = (
                    np.ascontiguousarray(self.core.download(st.canon.handle))
                    .reshape(-1)
                    .view(np.uint8)
                )
                st.host[: len(raw)] = raw[: st.nbytes]
        return st.host

    def buffer_view(self, st: Storage, td, code, shape):
        """The buffer as a device tensor of `shape` (cached until it is written)."""
        shape = pad2(shape)
        if st.base is not None:
            raise Unsupported("sub-buffer view")
        key = (shape, code)
        if key in st.views:
            return st.views[key]
        c = st.canon
        if (
            c is not None
            and c.td == td
            and int(np.prod(c.shape, dtype=np.int64))
            == int(np.prod(shape, dtype=np.int64))
        ):
            h = (
                c.handle
                if tuple(c.shape) == shape
                else self.core.reshape(c.handle, shape)
            )
        elif c is not None and c.td != td:
            raise Unsupported("buffer reinterpreted as another dtype")
        else:
            host = self.host_bytes(st)
            nbytes = td.itemsize * int(np.prod(shape, dtype=np.int64))
            if nbytes > len(host):
                raise Unsupported("read past the end of a buffer")
            h = self.core.upload(
                host[:nbytes].view(_np_of(self.raw_core, code)).reshape(shape), code
            )
            if c is None and nbytes == len(host):
                st.canon = Canon(h, shape, td)
        st.views[key] = h
        return h

    def set_canon(self, st: Storage, handle, td):
        """Make `handle` the contents of buffer `st`."""
        if st.base is not None:
            raise Unsupported("store into a sub-buffer")
        st.canon, st.views, st.host = Canon(handle, handle.shape, td), {}, None

    # ---- kernels

    def make_plan(self, ast: UOp):
        return Plan(ast)

    def run(self, plan, bufs: dict, var_vals: dict):
        self.ensure_open()
        t0 = time.perf_counter()
        try:
            if hasattr(plan, "run_bridge"):
                plan.run_bridge(self, bufs, var_vals)
            else:
                DevKernel(self, plan, bufs, var_vals).run()
            TIMES["device"] += time.perf_counter() - t0
            return "device"
        except DtypeUnsupported as e:
            reason = dtype_fallback(e.td)
        except Unsupported as e:
            reason = str(e)
        FALLBACKS[reason] += 1
        self.fallback_log.append((plan.kind, reason))
        self.run_host(plan, bufs, var_vals, reason)
        TIMES["fallback"] += time.perf_counter() - t0
        return "fallback"

    def touch(self, st: Storage):
        """Mark the host bytes of `st` as the only valid copy."""
        self.invalidate(st.base if st.base is not None else st)

    def run_host(self, plan, bufs: dict, var_vals: dict, reason: str = ""):
        """The numpy evaluator on the kernel's buffers; outputs stay on the host.
        Its failures are HostKernelError naming the kernel and `reason`."""
        try:
            if hasattr(plan, "run_host"):
                return plan.run_host(self, bufs, var_vals)
            raw = {
                p.slot: self.host_bytes(bufs[p.slot])
                for p in plan.params
                if not p.is_var
            }
            written = evaluate(plan, raw, var_vals)
        except (DtypeError, HostKernelError):
            raise
        except Exception as e:
            raise HostKernelError(
                f"kohakutt.tinygrad: kernel {plan.kind} did not run on the card "
                f"({reason or 'host executor'}) and the host evaluator failed on it: "
                f"{type(e).__name__}: {e}"
            ) from e
        for slot in written:
            st = bufs[slot]
            self.invalidate(st.base if st.base is not None else st)


class DevKernel:
    """One launch of one kernel AST on the device."""

    ALU2 = {
        Ops.ADD: "add",
        Ops.MUL: "mul",
        Ops.SUB: "sub",
        Ops.MAX: "maximum",
        Ops.FDIV: "div",
        Ops.POW: "pow",
        Ops.CMPLT: "lt",
        Ops.CMPNE: "ne",
        Ops.CMPEQ: "eq",
    }
    COMMUTATIVE = {"add", "mul", "maximum", "ne", "eq"}
    FLIP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne"}
    ALU1 = {
        Ops.EXP2: "exp2",
        Ops.LOG2: "log2",
        Ops.SIN: "sin",
        Ops.SQRT: "sqrt",
        C.RECIP: "reciprocal",
        Ops.NEG: "neg",
        Ops.TRUNC: "trunc",
    }

    def __init__(self, be: BridgeBackend, plan, bufs: dict, var_vals: dict):
        self.be, self.core, self.plan, self.bufs, self.var_vals = (
            be,
            be.core,
            plan,
            bufs,
            var_vals,
        )
        self.memo: dict[UOp, DVal] = {}
        self.ref = None
        self.routes = collections.Counter()  # load route -> reads in this launch
        self._pure_ev = None
        for name in ("pure_memo", "range_memo", "const_cache"):
            if not hasattr(plan, name):
                setattr(plan, name, {})
        self.params = {p.slot: p for p in plan.params if not p.is_var}
        self.pending: list[tuple[Storage, Canon]] = []

    # ---- helpers

    def code(self, td):
        try:
            return self.be.codes[td]
        except KeyError:
            raise DtypeUnsupported(td) from None

    def extent(self, r: UOp) -> int:
        e = r.src[0]
        if e.op is Ops.CONST:
            return int(e.arg)
        return int(self.pure_ev.scalar(e))  # a Variable bound at launch

    @property
    def pure_ev(self) -> Evaluator:
        """The numpy evaluator for load-free subtrees (ranges, constants, variables)."""
        if self._pure_ev is None:
            self._pure_ev = Evaluator(self.plan, {}, self.var_vals)
        return self._pure_ev

    def is_pure(self, u: UOp) -> bool:
        """True when `u` depends only on ranges, constants and variables."""
        memo = self.plan.pure_memo
        if u not in memo:
            memo[u] = (
                u.op is not Ops.INDEX
                and not C.is_buffer_param(u)
                and all(self.is_pure(s) for s in u.src)
            )
        return memo[u]

    def has_range(self, u: UOp) -> bool:
        memo = self.plan.range_memo
        if u not in memo:
            memo[u] = u.op is Ops.RANGE or any(self.has_range(s) for s in u.src)
        return memo[u]

    def const_entry(self, key, build):
        """A host-built constant cached on the plan by (key, variable values)."""
        cache = self.plan.const_cache
        full = (key, tuple(sorted(self.var_vals.items())))
        if full not in cache:
            if len(cache) >= CONST_CACHE_MAX:
                cache.pop(next(iter(cache)))
            cache[full] = build()
            BRIDGE_CALLS["host_constant_built"] += 1
        return cache[full]

    def host_const(self, u: UOp) -> DVal:
        """A load-free value (mask, arange, one-hot) built on the host, kept on card."""

        def build():
            arr, axes = self.pure_ev.ev(u)
            arr = np.asarray(arr)
            keep = [i for i, r in enumerate(axes) if self.extent(r) > 1]
            arr = arr.reshape([self.extent(axes[i]) for i in keep])
            td = u.dtype
            if td == dtypes.bool or dtypes.is_float(td) or td == dtypes.weakfloat:
                h = self.core.upload(
                    arr.astype(np.float32).reshape(pad2(arr.shape)),
                    self.be.raw_core.FP32,
                )
                h = (
                    self.core.typecast(h, self.be.raw_core.BF16)
                    if td == dtypes.bool
                    else h
                )
            elif dtypes.is_int(td):
                h = self.core.upload(
                    arr.astype(np.int32).reshape(pad2(arr.shape)),
                    self.be.raw_core.INT32,
                )
            else:
                raise Unsupported(f"host constant of dtype {td}")
            return h, tuple(axes[i] for i in keep)

        h, axes = self.const_entry(("c", u), build)
        if not axes:
            raise Unsupported("scalar host constant")
        return DVal(h, axes, u.dtype)

    def affine(self, u: UOp) -> tuple[dict, int]:
        """(stride per range, constant) of an index expression."""
        if u.op is Ops.CONST:
            if u.arg is C.Invalid:
                raise Unsupported("gated read")
            return {}, int(u.arg)
        if u.op is Ops.RANGE:
            return {u: 1}, 0
        if u.op is Ops.ADD:
            (ca, oa), (cb, ob) = self.affine(u.src[0]), self.affine(u.src[1])
            out = dict(ca)
            for r, s in cb.items():
                out[r] = out.get(r, 0) + s
            return out, oa + ob
        if u.op is Ops.MUL:
            (ca, oa), (cb, ob) = self.affine(u.src[0]), self.affine(u.src[1])
            if ca and cb:
                raise Unsupported("non-affine index")
            if cb:
                (ca, oa), (cb, ob) = (cb, ob), (ca, oa)
            return {r: s * ob for r, s in ca.items()}, oa * ob
        raise Unsupported(f"index op {u.op.name}")

    # ---- device tensor plumbing

    def align(self, v: DVal, order: list) -> tuple:
        """The tensor of `v` with dims in `order` (absent axes become size 1)."""
        present = [x for x in order if x in v.axes]
        h = v.h
        if len(v.axes) >= 2:
            perm = [v.axes.index(x) for x in present]
            if perm != list(range(len(perm))):
                h = self.core.permute(h, perm)
        want = pad2([self.extent(x) if x in v.axes else 1 for x in order])
        if tuple(h.shape) != want:
            h = self.core.reshape(h, want)
        return h

    def order_for(self, *vals: DVal) -> list:
        """Axis order of an op over `vals`: the widest value's, then the new axes."""
        vals = sorted((v for v in vals if not v.is_const), key=lambda v: -len(v.axes))
        order: list = []
        for v in vals:
            order += [x for x in v.axes if x not in order]
        return order

    def tensor_of(self, v: DVal, order: list, shape, like_td):
        """v aligned to `order`; a constant becomes a full tensor of `shape`."""
        if v.is_const:
            return self.core.full(shape, v.c, self.code(like_td))
        return self.align(v, order)

    def dsh(self, order: list):
        return pad2([self.extent(x) for x in order])

    def expand(self, h, shape):
        """Broadcast tensor h to `shape` (size-1 dims repeated)."""
        if tuple(h.shape) == tuple(shape):
            return h
        reps = [s // c for c, s in zip(h.shape, shape)]
        return self.core.repeat(h, reps)

    # ---- evaluation

    def ev(self, u: UOp) -> DVal:
        if u in self.memo:
            return self.memo[u]
        self.memo[u] = r = self._ev(u)
        if CHECK and not r.is_const:
            self.check_node(u, r)
        return r

    def check_node(self, u: UOp, r: DVal):
        """KTT_TINYGRAD_CHECK=1: compare each node with the numpy evaluator's value."""
        if self.ref is None:
            typed = {
                p.slot: host_view(self.be.host_bytes(self.bufs[p.slot]), p.dtype)
                for p in self.plan.params
                if not p.is_var
            }
            self.ref = Evaluator(self.plan, typed, self.var_vals)
        want, axes = self.ref.ev(u)
        want = np.asarray(want, np.float64)
        got = self.core.download(r.h)
        if got.dtype == np.uint16:
            got = (got.astype(np.uint32) << 16).view(np.float32)
        got = np.asarray(got, np.float64).reshape(
            [self.extent(x) for x in r.axes] or [1]
        )
        perm = [axes.index(x) for x in r.axes if x in axes]
        want = np.transpose(want, perm) if want.ndim == len(perm) else want
        if got.size != want.size:
            print(f"[check] {u.op.name} size mismatch {got.shape} vs {want.shape}")
            return
        err = float(
            np.max(np.abs(got.reshape(-1) - want.reshape(-1)))
            / (np.max(np.abs(want)) + 1e-12)
        )
        if err > 1e-4:
            print(
                f"[check] {self.plan.kind}: {u.op.name} {u.dtype} "
                f"err {err:.2e} shape {got.shape}"
            )

    def _ev(self, u: UOp) -> DVal:
        op = u.op
        if op is Ops.CONST:
            if u.arg is C.Invalid:
                raise Unsupported("gated read")
            return DVal(None, (), u.dtype, c=float(u.arg))
        if self.is_pure(u) and self.has_range(u):
            return self.host_const(u)
        if op is Ops.INDEX:
            return self.load(u)
        if op is Ops.CAST:
            return self.cast(u)
        if op is Ops.REDUCE:
            return self.reduce(u)
        if op is Ops.WHERE:
            if u.src[2].op is Ops.CONST and u.src[2].arg is C.Invalid:
                raise Unsupported("gated read")
            return self.where(u)
        if op in self.ALU1:
            a = self.ev(u.src[0])
            if a.is_const:
                raise Unsupported("unary op of a constant")
            return DVal(self.core.unary(self.ALU1[op], a.h), a.axes, a.td)
        if op in self.ALU2:
            return self.binary(
                self.ALU2[op], self.ev(u.src[0]), self.ev(u.src[1]), u.dtype
            )
        if op in (Ops.AND, Ops.OR, Ops.XOR):
            a, b = self.ev(u.src[0]), self.ev(u.src[1])
            if a.td != dtypes.bool or b.td != dtypes.bool:
                raise Unsupported(f"{op.name} on {a.td}")
            return self.binary(
                {Ops.AND: "mul", Ops.OR: "maximum", Ops.XOR: "ne"}[op],
                a,
                b,
                dtypes.bool,
            )
        raise Unsupported(f"op {op.name}")

    def binary(self, name: str, a: DVal, b: DVal, out_td) -> DVal:
        if a.is_const and b.is_const:
            raise Unsupported("constant folding left to tinygrad")
        if a.is_const or b.is_const:
            x, s, flipped = (b, a.c, True) if a.is_const else (a, b.c, False)
            if flipped:
                if name in self.COMMUTATIVE:
                    pass
                elif name in self.FLIP:
                    name = self.FLIP[name]
                elif name == "sub":  # c - x
                    return DVal(
                        self.core.binary_scalar("add", self.core.unary("neg", x.h), s),
                        x.axes,
                        out_td,
                    )
                elif name == "div":  # c / x
                    return DVal(
                        self.core.binary_scalar(
                            "mul", self.core.unary("reciprocal", x.h), s
                        ),
                        x.axes,
                        out_td,
                    )
                else:
                    raise Unsupported(f"constant on the left of {name}")
            return DVal(self.core.binary_scalar(name, x.h, s), x.axes, out_td)
        order = self.order_for(a, b)
        return DVal(
            self.core.binary(name, self.align(a, order), self.align(b, order)),
            tuple(order),
            out_td,
        )

    def where(self, u: UOp) -> DVal:
        return self.where_vals(*(self.ev(s) for s in u.src), u.dtype)

    def where_vals(self, c: DVal, a: DVal, b: DVal, td) -> DVal:
        if c.is_const:
            raise Unsupported("where on a constant condition")
        order = self.order_for(c, a, b)
        shape = self.dsh(order)
        h = self.core.where(
            self.expand(self.align(c, order), shape),
            self.expand(self.tensor_of(a, order, shape, td), shape),
            self.expand(self.tensor_of(b, order, shape, td), shape),
        )
        return DVal(h, tuple(order), td)

    def cast(self, u: UOp) -> DVal:
        a = self.ev(u.src[0])
        if a.is_const:
            return DVal(None, (), u.dtype, c=float(a.c))
        src_td, dst = a.td, u.dtype
        if src_td == dst:
            return a
        if dst == dtypes.bool:
            return DVal(self.core.binary_scalar("ne", a.h, 0.0), a.axes, dtypes.bool)
        if src_td == dtypes.bool:
            return DVal(self.core.typecast(a.h, self.code(dst)), a.axes, dst)
        return DVal(self.core.typecast(a.h, self.code(dst)), a.axes, dst)

    # ---- loads

    def load(self, idx: UOp) -> DVal:
        param = idx.src[0]
        if not C.is_buffer_param(param):
            raise Unsupported("load from a non-buffer")
        slot = C.param_slot(param)
        st, p = self.bufs[slot], self.params[slot]
        code = self.code(p.dtype)
        if len(idx.src) == 1:
            raise Unsupported("scalar load")
        # cheapest route first: strided view, memory-op composition, host-built gather;
        # the route taken is counted per kernel
        for route, fn in (
            ("view", self.load_view),
            ("memory", self.load_mem),
            ("gather", self.load_gather),
        ):
            try:
                r = fn(idx, st, p, code)
            except Unsupported as e:
                if route == "gather":
                    raise
                ROUTE_SKIPS[f"{route}: {e}"] += 1
                continue
            self.routes[route] += 1
            return r
        raise Unsupported("unreachable")

    def load_mem(self, idx: UOp, st: Storage, p, code) -> DVal:
        """A read made of memory ops (reshape, slice, permute, zero-padding concat):
        the gate is a box and the index is affine in the ranges or their div / mod
        pieces. The structure is found on the host and verified against the index."""
        expr, gate = idx.src[1], None
        if (
            expr.op is Ops.WHERE
            and expr.src[2].op is Ops.CONST
            and expr.src[2].arg is C.Invalid
        ):
            gate, expr = expr.src[0], expr.src[1]
        if not (self.is_pure(expr) and (gate is None or self.is_pure(gate))):
            raise Unsupported("data-dependent index")
        nelem = st.nbytes // p.dtype.itemsize

        def build():
            ev = self.pure_ev
            parts = [ev.ev(expr)] + ([ev.ev(gate)] if gate is not None else [])
            union = ev.union(*[ax for _, ax in parts])
            shape = [self.extent(r) for r in union]
            if int(np.prod(shape, dtype=np.int64)) > ANALYZE_MAX:
                raise Unsupported("index too large to analyse on the host")
            ia = np.broadcast_to(ev.align(parts[0], union), shape).astype(np.int64)
            valid = (
                np.broadcast_to(ev.align(parts[1], union), shape).astype(bool)
                if gate is not None
                else None
            )
            return self.plan_read(ia, valid, shape, nelem, union)

        taps, descs = self.const_entry(("m", idx), build)
        if taps is None:
            return self.read_desc(descs[0], st, p, code)
        tap_axes, tap_exts = taps
        vals = [self.read_desc(d, st, p, code) for d in descs]
        if any(v.axes != vals[0].axes for v in vals):
            raise Unsupported("taps read different axes")
        rest = tuple(self.extent(x) for x in vals[0].axes)
        lead_ones = [self.core.reshape(v.h, pad2((1,) + rest)) for v in vals]
        h = self.core.concat(lead_ones, 0) if len(lead_ones) > 1 else lead_ones[0]
        h = self.core.reshape(h, pad2(list(tap_exts) + list(rest)))
        return DVal(h, tuple(tap_axes) + vals[0].axes, p.dtype)

    def plan_read(self, ia, valid, shape, nelem, union):
        """(taps, descriptors) of a read: one descriptor for a padded strided view, else
        one per value of a few tap axes (extent <= 16, at most 64 taps). Each is
        replayed on the host and must reproduce the index array."""
        try:
            d = analyse_read(ia, valid, shape, nelem, union)
            self.verify_desc(d, nelem, ia, valid, shape, union)
            return None, [d]
        except Unsupported as first:
            small = sorted(
                (k for k, n in enumerate(shape) if 1 < n <= 16), key=lambda k: shape[k]
            )[:4]
            for r in (1, 2):
                for tap in itertools.combinations(small, r):
                    ext = [shape[k] for k in tap]
                    if int(np.prod(ext)) > 64:
                        continue
                    try:
                        descs = []
                        for t in itertools.product(*[range(n) for n in ext]):
                            sel = tuple(
                                t[tap.index(k)] if k in tap else slice(None)
                                for k in range(len(shape))
                            )
                            rest_shape = [
                                n for k, n in enumerate(shape) if k not in tap
                            ]
                            rest_union = [
                                u for k, u in enumerate(union) if k not in tap
                            ]
                            sub_i = ia[sel]
                            sub_v = valid[sel] if valid is not None else None
                            d = analyse_read(
                                sub_i, sub_v, rest_shape, nelem, rest_union
                            )
                            self.verify_desc(
                                d, nelem, sub_i, sub_v, rest_shape, rest_union
                            )
                            descs.append(d)
                        return (tuple(union[k] for k in tap), tuple(ext)), descs
                    except Unsupported as e:
                        ROUTE_SKIPS[f"taps {len(tap)}: {e}"] += 1
                        continue
            raise first

    def verify_desc(self, desc, nelem, ia, valid, shape, union):
        """Replay read_desc on the host; Unsupported unless it reproduces `ia`."""
        axes, widths, starts, steps, full, pieces, off = desc
        nz = [k for k, pc in enumerate(pieces) if pc[2] != 0]
        order = sorted(nz, key=lambda k: -pieces[k][2])
        dims = [pieces[k] for k in order]
        base, sl = self.resolve_view(dims, off, nelem)
        a = np.arange(nelem, dtype=np.int64).reshape(base)
        if sl is not None:
            a = a[tuple(slice(b, e) for b, e in zip(*sl))]
        a = a.reshape([d[1] for d in dims]).transpose([order.index(k) for k in nz])
        a = a.reshape([1 if pc[2] == 0 else pc[1] for pc in pieces])
        a = np.broadcast_to(a, [pc[1] for pc in pieces]).reshape(widths)
        out = np.full(full, -1, dtype=np.int64)
        out[
            np.ix_(*[s + t * np.arange(n) for s, t, n in zip(starts, steps, widths)])
        ] = a
        live = [union.index(x) for x in axes]
        sel = tuple(slice(None) if k in live else 0 for k in range(len(shape)))
        bshape = [shape[k] if k in live else 1 for k in range(len(shape))]
        ia_l = ia[sel]
        v_l = valid[sel] if valid is not None else None
        same = np.array_equal(np.broadcast_to(ia_l.reshape(bshape), shape), ia)
        if valid is not None:
            same = same and np.array_equal(
                np.broadcast_to(v_l.reshape(bshape), shape), valid
            )
        ok = (
            (np.array_equal(out[v_l], ia_l[v_l]) and bool(np.all(out[~v_l] == -1)))
            if v_l is not None
            else np.array_equal(out, ia_l)
        )
        if not (same and ok):
            raise Unsupported("memory-op replay does not reproduce the index")

    def read_desc(self, desc, st, p, code) -> DVal:
        axes, widths, starts, steps, full, pieces, off = desc
        nz = [k for k, pc in enumerate(pieces) if pc[2] != 0]
        order = sorted(nz, key=lambda k: -pieces[k][2])
        h, _ = self.strided_read(st, p, code, [pieces[k] for k in order], off)
        lead = len(h.shape) - len(order)
        # sorted position of each non-broadcast piece, in piece order
        perm = [order.index(k) for k in nz]
        if perm != list(range(len(perm))):
            h = self.core.permute(h, list(range(lead)) + [lead + x for x in perm])
        if len(nz) != len(pieces):  # broadcast pieces (stride 0): size-1 dims repeated
            h = self.core.reshape(
                h, pad2([1 if pc[2] == 0 else pc[1] for pc in pieces])
            )
            lead = len(h.shape) - len(pieces)
            h = self.core.repeat(
                h, [1] * lead + [pc[1] if pc[2] == 0 else 1 for pc in pieces]
            )
        if tuple(h.shape) != pad2(widths):
            h = self.core.reshape(h, pad2(widths))
        lead = len(h.shape) - len(widths)
        for a in range(len(widths)):
            if starts[a] == 0 and steps[a] == 1 and widths[a] == full[a]:
                continue
            h = self.place_axis(
                h, lead + a, starts[a], steps[a], widths[a], full[a], code
            )
        return DVal(h, axes, p.dtype)

    def zeros_like_dim(self, h, dim: int, n: int, code):
        shp = list(h.shape)
        shp[dim] = n
        return self.core.full(shp, 0.0, code)

    def place_axis(
        self, h, dim: int, start: int, step: int, count: int, full: int, code
    ):
        """Put `h` at start + step * arange(count) of `full` zeros along `dim`."""
        if step > 1:
            # zero-interleave: [count] -> [count, step] -> [count * step]
            o, m0 = start % step, start // step
            shape = list(h.shape)
            h = self.core.reshape(h, shape[:dim] + [count, 1] + shape[dim + 1 :])
            parts = [
                self.zeros_like_dim(h, dim + 1, n, code) if n else None
                for n in (o, step - 1 - o)
            ]
            h = self.core.concat(
                [x for x in (parts[0], h, parts[1]) if x is not None], dim + 1
            )
            h = self.core.reshape(h, shape[:dim] + [count * step] + shape[dim + 1 :])
            start, count = m0 * step, count * step
        right = full - start - count
        parts = [
            self.zeros_like_dim(h, dim, start, code) if start else None,
            h,
            self.zeros_like_dim(h, dim, right, code) if right else None,
        ]
        parts = [x for x in parts if x is not None]
        return self.core.concat(parts, dim) if len(parts) > 1 else h

    def load_gather(self, idx: UOp, st: Storage, p, code) -> DVal:
        """A read no view expresses (gates, divmod, odd strides): the flat index tensor
        and validity mask are built on the host once per plan; the read is one gather.
        """
        expr, gate = idx.src[1], None
        if (
            expr.op is Ops.WHERE
            and expr.src[2].op is Ops.CONST
            and expr.src[2].arg is C.Invalid
        ):
            gate, expr = expr.src[0], expr.src[1]
        if not (self.is_pure(expr) and (gate is None or self.is_pure(gate))):
            raise Unsupported("data-dependent index")
        nelem = st.nbytes // p.dtype.itemsize

        def build():
            ev = self.pure_ev
            parts = [ev.ev(expr)] + ([ev.ev(gate)] if gate is not None else [])
            union = ev.union(*[ax for _, ax in parts])
            shape = [self.extent(r) for r in union]
            ia = np.broadcast_to(ev.align(parts[0], union), shape).astype(np.int64)
            valid = (
                np.broadcast_to(ev.align(parts[1], union), shape).astype(bool)
                if gate is not None
                else None
            )
            if valid is not None:
                ia = np.where(valid, ia, 0)
            if ia.size == 0 or ia.min() < 0 or ia.max() >= nelem:
                raise Unsupported("gather index outside the buffer")
            if ia.size > GATHER_MAX:
                raise Unsupported(f"gather of more than {GATHER_MAX} elements")
            keep = [i for i, n in enumerate(shape) if n > 1]
            kshape = [shape[i] for i in keep]
            core_ = self.be.raw_core
            ih = self.core.upload(ia.astype(np.int32).reshape(1, -1), core_.INT32)
            mh = None
            if valid is not None:
                mh = self.core.typecast(
                    self.core.upload(
                        valid.astype(np.float32).reshape(1, -1), core_.FP32
                    ),
                    core_.BF16,
                )
            return ih, mh, tuple(union[i] for i in keep), tuple(kshape), ia.size

        ih, mh, axes, kshape, m = self.const_entry(("g", idx), build)
        src = self.buffer_view(st, p.dtype, code, (nelem,))
        h = self.core.gather(src, 1, ih)
        if mh is not None:
            h = self.core.where(mh, h, self.core.full([1, m], 0.0, code))
        want = pad2(kshape)
        if tuple(h.shape) != want:
            h = self.core.reshape(h, want)
        return DVal(h, axes, p.dtype)

    def load_view(self, idx: UOp, st: Storage, p, code) -> DVal:
        coeffs, off = self.affine(idx.src[1])
        dims = sorted(
            (
                (r, self.extent(r), s)
                for r, s in coeffs.items()
                if s != 0 and self.extent(r) > 1
            ),
            key=lambda d: -d[2],
        )
        h, _ = self.strided_read(st, p, code, dims, off)
        return DVal(h, tuple(d[0] for d in dims), p.dtype)

    def strided_read(self, st, p, code, dims, off):
        """The buffer read as dims [(key, extent, stride)] by decreasing stride from
        element `off` (reshape, then slice). Returns (tensor, extents)."""
        if any(s < 0 for _, _, s in dims):
            raise Unsupported("negative stride")
        nelem = st.nbytes // p.dtype.itemsize
        base_shape, slices = self.resolve_view(dims, off, nelem)
        h = self.buffer_view(st, p.dtype, code, base_shape)
        if slices is not None:
            begins, ends = slices
            lead = len(pad2(base_shape)) - len(base_shape)
            h = self.core.slice(h, [0] * lead + begins, [1] * lead + ends)
        want = pad2([d[1] for d in dims])
        if tuple(h.shape) != want:
            h = self.core.reshape(h, want)
        return h, [d[1] for d in dims]

    def resolve_view(self, dims, off, nelem):
        """(base shape of the buffer view, (begins, ends) of the slice) of a read."""
        if not dims:
            return ((nelem,), None) if nelem == 1 else ((nelem,), ([off], [off + 1]))
        strides = [s for _, _, s in dims]
        exts = [e for _, e, _ in dims]
        tiles = all(
            strides[i] == int(np.prod(exts[i + 1 :], dtype=np.int64))
            for i in range(len(dims))
        )
        if tiles and off == 0 and int(np.prod(exts, dtype=np.int64)) == nelem:
            return tuple(exts), None
        bases = []
        for i, s in enumerate(strides):
            prev = strides[i - 1] if i else None
            if prev is None:
                if nelem % s:
                    raise Unsupported("strided read (outer dim)")
                bases.append(nelem // s)
            else:
                if prev % s:
                    raise Unsupported("strided read (ragged strides)")
                bases.append(prev // s)
        if strides[-1] != 1:
            bases.append(strides[-1])
            exts = exts + [1]
        digits, rest = [], off
        for k in range(len(bases)):
            unit = int(np.prod(bases[k + 1 :], dtype=np.int64))
            digits.append(rest // unit)
            rest -= digits[-1] * unit
        if any(d + e > b for d, e, b in zip(digits, exts, bases)):
            raise Unsupported("read outside the buffer")
        return tuple(bases), (
            [d for d in digits],
            [d + e for d, e in zip(digits, exts)],
        )

    def buffer_view(self, st, td, code, shape):
        return self.be.buffer_view(st, td, code, shape)

    # ---- reductions

    def reduce(self, u: UOp) -> DVal:
        red = list(u.src[1:])
        rop = u.arg[0]
        if rop is Ops.ADD and u.src[0].op is Ops.MUL:
            try:
                return self.contract(u, red)
            except NotContraction:
                pass
        if rop not in (Ops.ADD, Ops.MAX):
            raise Unsupported(f"reduce {rop.name}")
        if (
            rop is Ops.ADD
            and u.src[0].op is Ops.WHERE
            and sum(self.extent(r) > 1 for r in red) == 1
        ):
            for pattern in (self.reduce_cumsum, self.reduce_embedding):
                try:
                    return pattern(u, red)
                except NoPattern:
                    pass
        body = u.src[0]
        if (
            rop is Ops.ADD
            and body.op is Ops.WHERE
            and body.src[2].op is Ops.CONST
            and body.src[2].arg is C.Invalid
        ):
            # iterations gated off by the body contribute nothing to the sum
            x = self.where_vals(
                self.ev(body.src[0]),
                self.ev(body.src[1]),
                DVal(None, (), body.dtype, c=0.0),
                body.dtype,
            )
        else:
            x = self.ev(body)
        missing = [r for r in red if r not in x.axes]
        if x.is_const:
            scale = float(np.prod([self.extent(r) for r in red], dtype=np.float64))
            return DVal(None, (), x.td, c=x.c * scale if rop is Ops.ADD else x.c)
        dims = [i for i, r in enumerate(x.axes) if r in red]
        keep = [r for r in x.axes if r not in red]
        h = x.h
        if dims:
            pad = len(h.shape) - len(x.axes)
            full_dims = [pad + i for i in dims]
            keepdim = [1 if r in red else self.extent(r) for r in x.axes]
            out_keep = pad2(keepdim)
            h = (self.core.sum if rop is Ops.ADD else self.core.amax)(
                h, full_dims, out_keep
            )
            want = pad2([self.extent(r) for r in keep])
            if tuple(h.shape) != want:
                h = self.core.reshape(h, want)
        n_missing = (
            int(np.prod([self.extent(r) for r in missing], dtype=np.int64))
            if missing
            else 1
        )
        if rop is Ops.ADD and n_missing > 1:
            h = self.core.binary_scalar("mul", h, float(n_missing))
        return DVal(h, tuple(keep), x.td)

    @staticmethod
    def zero_branch_load(body: UOp):
        """(condition, load, kept when True) of WHERE(c, 0, load) / (c, load, 0)."""
        c, a, b = body.src

        def zero(x):
            return x.op is Ops.CONST and x.arg is not C.Invalid and float(x.arg) == 0.0

        if zero(a) and b.op is Ops.INDEX:
            return c, b, False
        if zero(b) and a.op is Ops.INDEX:
            return c, a, True
        raise NoPattern

    def reduce_cumsum(self, u: UOp, red: list) -> DVal:
        """sum_r [i + r >= n - 1] * x[A + i + r - (n - 1)] (cumsum along an axis of n)
        -> bridge `cumsum` of x[A + j]. Mask and index are checked on the host;
        any mismatch is NoPattern."""
        c, ld, kept_when = self.zero_branch_load(u.src[0])
        r = next(x for x in red if self.extent(x) > 1)
        n = self.extent(r)
        if not C.is_buffer_param(ld.src[0]) or len(ld.src) == 1:
            raise NoPattern
        expr, gate = ld.src[1], None
        if (
            expr.op is Ops.WHERE
            and expr.src[2].op is Ops.CONST
            and expr.src[2].arg is C.Invalid
        ):
            gate, expr = expr.src[0], expr.src[1]
        if not all(
            self.is_pure(x) for x in (c, expr) + ((gate,) if gate is not None else ())
        ):
            raise NoPattern
        slot = C.param_slot(ld.src[0])
        st, p = self.bufs[slot], self.params[slot]
        if p.dtype != u.dtype or p.dtype not in (dtypes.float32, dtypes.bfloat16):
            raise NoPattern
        ev = self.pure_ev
        parts = [ev.ev(c), ev.ev(expr)] + ([ev.ev(gate)] if gate is not None else [])
        union = ev.union(*[ax for _, ax in parts])
        shape = [self.extent(x) for x in union]
        if r not in union or int(np.prod(shape, dtype=np.int64)) > ANALYZE_MAX:
            raise NoPattern
        nd, ri = len(shape), union.index(r)

        def full(i):
            return np.broadcast_to(ev.align(parts[i], union), shape)

        cond = full(0).astype(bool)
        keep = cond if kept_when else ~cond
        if gate is not None:
            keep = keep & full(2).astype(bool)
        ia = full(1).astype(np.int64)
        tri = np.arange(n)[:, None] + np.arange(n)[None, :] >= n - 1  # [i, r]
        for ki in range(nd):
            if ki == ri or shape[ki] != n:
                continue
            exp = tri if ki < ri else tri.T
            bshape = [n if k in (ki, ri) else 1 for k in range(nd)]
            if not np.array_equal(np.broadcast_to(exp.reshape(bshape), shape), keep):
                continue
            sel = tuple(
                n - 1 if k == ki else 0 if k == ri else slice(None) for k in range(nd)
            )
            base = ia[sel]
            step = (
                ia[
                    tuple(
                        n - 1 if k == ki else 1 if k == ri else slice(None)
                        for k in range(nd)
                    )
                ]
                - base
            )
            stride = int(step.flat[0])
            if stride <= 0 or not np.all(step == stride):
                continue
            for k in sorted((ki, ri)):
                base = np.expand_dims(base, k)
            ii = np.arange(n).reshape([n if k == ki else 1 for k in range(nd)])
            rr = np.arange(n).reshape([n if k == ri else 1 for k in range(nd)])
            if not np.array_equal(
                np.broadcast_to(base + (ii + rr - (n - 1)) * stride, shape)[keep],
                ia[keep],
            ):
                continue
            # position of the cumulative axis once the reduce axis is dropped
            kk = ki - (ki > ri)
            rest_shape = [s for k, s in enumerate(shape) if k != ri]
            lines = np.squeeze(base, axis=ri) + stride * np.arange(n).reshape(
                [n if k == kk else 1 for k in range(nd - 1)]
            )
            union2 = [x for k, x in enumerate(union) if k != ri]
            taps, descs = self.const_entry(
                ("cs", u),
                lambda lines=lines, rs=rest_shape, un=union2: self.plan_read(
                    lines, None, rs, p_nelem(st, p), un
                ),
            )
            if taps is not None:
                raise NoPattern
            h = self.read_desc(descs[0], st, p, self.code(p.dtype))
            pos = h.axes.index(union[ki])
            lead = len(h.h.shape) - len(h.axes)
            return DVal(self.core.cumsum(h.h, lead + pos), h.axes, u.dtype)
        raise NoPattern

    def reduce_embedding(self, u: UOp, red: list) -> DVal:
        """sum_v [idx[b] == v] * W[v, d] (one-hot embedding) -> bridge `embedding_rows`
        at the index values (read back to wrap negatives and check the range)."""
        c, ld, kept_when = self.zero_branch_load(u.src[0])
        if not (
            (c.op is Ops.CMPNE and not kept_when) or (c.op is Ops.CMPEQ and kept_when)
        ):
            raise NoPattern
        r = next(x for x in red if self.extent(x) > 1)
        V = self.extent(r)
        sides = [s for s in c.src if s.op is Ops.CAST and s.src[0] is r]
        if len(sides) != 1 or not C.is_buffer_param(ld.src[0]) or len(ld.src) == 1:
            raise NoPattern
        x = c.src[0] if c.src[1] is sides[0] else c.src[1]
        wrap = False
        if x.op is Ops.WHERE:  # where(l < 0, l + V, l)
            raw = x.src[2]
            lt, plus = x.src[0], x.src[1]
            if not (
                raw.op is Ops.INDEX
                and lt.op is Ops.CMPLT
                and lt.src[0] is raw
                and lt.src[1].op is Ops.CONST
                and float(lt.src[1].arg) == 0
                and plus.op is Ops.ADD
                and raw in plus.src
                and any(s.op is Ops.CONST and int(s.arg) == V for s in plus.src)
            ):
                raise NoPattern
            x, wrap = raw, True
        if x.op is not Ops.INDEX or not C.is_buffer_param(x.src[0]) or len(x.src) == 1:
            raise NoPattern
        try:
            coeffs, off = self.affine(ld.src[1])
        except Unsupported:
            raise NoPattern from None
        slot = C.param_slot(ld.src[0])
        st, p = self.bufs[slot], self.params[slot]
        dr = [k for k in coeffs if k is not r]
        if (
            off != 0
            or len(dr) != 1
            or coeffs.get(dr[0]) != 1
            or coeffs.get(r) != self.extent(dr[0])
        ):
            raise NoPattern
        D = self.extent(dr[0])
        if (
            p_nelem(st, p) != V * D
            or p.dtype not in (dtypes.float32, dtypes.bfloat16)
            or p.dtype != u.dtype
        ):
            raise NoPattern
        ids = self.ev(x)
        if ids.is_const or ids.td not in (dtypes.int32, dtypes.int, dtypes.weakint):
            raise NoPattern
        vals = np.asarray(self.core.download(ids.h)).reshape(-1).astype(np.int64)
        if wrap:
            vals = np.where(vals < 0, vals + V, vals)
        if vals.size == 0 or vals.min() < 0 or vals.max() >= V:
            raise Unsupported("embedding index outside the table")
        code = self.code(p.dtype)
        table = self.buffer_view(st, p.dtype, code, (V, D))
        index = self.core.upload(
            vals.astype(np.int32).reshape(1, -1), self.be.raw_core.INT32
        )
        out = self.core.embedding_rows(
            table, self.core.version(table), index, int(vals.size)
        )
        exts = [self.extent(a) for a in ids.axes] + [D]
        return DVal(
            self.core.reshape(out, pad2(exts)), tuple(ids.axes) + (dr[0],), p.dtype
        )

    def contract(self, u: UOp, red: list) -> DVal:
        """REDUCE_ADD(MUL(a, b)) as one `mm` on [batch, M, K] x [batch, K, N]."""
        a_u, b_u = u.src[0].src
        A, B = self.ev(a_u), self.ev(b_u)
        if A.is_const or B.is_const:
            raise NotContraction
        aa, ba = set(A.axes), set(B.axes)
        rs = {r for r in red if self.extent(r) > 1}
        if any((r in aa) != (r in ba) for r in rs):
            raise NotContraction
        K = [r for r in A.axes if r in rs]
        batch = [r for r in A.axes if r in ba and r not in rs]
        M = [r for r in A.axes if r not in ba and r not in rs]
        N = [r for r in B.axes if r not in aa and r not in rs]
        if not K or a_u is b_u or not (M or N):
            # a value with itself, or no free M / N axis: multiply and sum instead
            raise NotContraction

        def size(group):
            return int(np.prod([self.extent(x) for x in group], dtype=np.int64))

        def prep(v, outer, outer_first_default):
            """v as [batch, outer, K] or [batch, K, outer] (fewer axis moves)."""
            best = None
            for k_first in (False, True):
                order = batch + (K + outer if k_first else outer + K)
                perm = [v.axes.index(x) for x in order]
                cost = sum(1 for i, p in enumerate(perm) if i != p)
                if best is None or cost < best[0]:
                    best = (cost, k_first, perm)
            _, k_first, perm = best
            h = v.h
            if perm != list(range(len(perm))) and len(v.axes) >= 2:
                h = self.core.permute(h, perm)
            dims = ([size(batch)] if batch else []) + (
                [size(K), size(outer)] if k_first else [size(outer), size(K)]
            )
            if tuple(h.shape) != pad2(dims):
                h = self.core.reshape(h, pad2(dims))
            return h, k_first

        ha, a_k_first = prep(A, M, False)
        hb, b_k_first = prep(B, N, True)
        # A is [M, K] plain or [K, M] transposed; B is [K, N] plain or [N, K] transposed
        out = self.core.mm(ha, hb, a_k_first, not b_k_first)
        axes = tuple(batch + M + N)
        want = pad2([self.extent(r) for r in axes])
        if tuple(out.shape) != want:
            out = self.core.reshape(out, want)
        return DVal(out, axes, A.td)

    # ---- stores and kernel driver

    def run(self):
        stores = [x for x in self.plan.ast.toposort() if x.op is Ops.STORE]
        written = {}
        for st_u in stores:
            idx = st_u.src[0]
            slot = C.param_slot(idx.src[0])
            if slot in written:
                raise Unsupported("two stores to one buffer")
            written[slot] = self.prepare_store(st_u)
        for slot, canon in written.items():
            st = self.bufs[slot]
            st.canon, st.views, st.host = canon, {}, None
        ROUTES[
            (
                "kernels_with_gather"
                if self.routes["gather"]
                else (
                    "kernels_memory_route"
                    if self.routes["memory"]
                    else "kernels_view_only"
                )
            )
        ] += 1
        for route, n in self.routes.items():
            ROUTES[f"reads_{route}"] += n

    def prepare_store(self, st_u: UOp):
        idx, val_u = st_u.src
        slot = C.param_slot(idx.src[0])
        st, p = self.bufs[slot], self.params[slot]
        code = self.code(p.dtype)
        if st.base is not None:
            raise Unsupported("store into a sub-buffer")
        if len(idx.src) == 1:
            raise Unsupported("scalar store")
        v = self.ev(val_u)
        coeffs, off = self.affine(idx.src[1])
        dims = sorted(
            (
                (r, self.extent(r), s)
                for r, s in coeffs.items()
                if s != 0 and self.extent(r) > 1
            ),
            key=lambda d: -d[2],
        )
        exts = [e for _, e, _ in dims]
        nelem = st.nbytes // p.dtype.itemsize
        if (
            off != 0
            or any(
                s != int(np.prod(exts[i + 1 :], dtype=np.int64))
                for i, (_, _, s) in enumerate(dims)
            )
            or int(np.prod(exts, dtype=np.int64)) > nelem
        ):
            raise Unsupported("partial or strided store")
        axes = [d[0] for d in dims]
        if v.is_const:
            h = self.core.full(pad2(exts), v.c, code)
        else:
            if not set(v.axes) <= set(axes):
                raise Unsupported("store value has axes outside the store index")
            h = self.align(v, axes)
            h = self.expand(h, pad2(exts))
            if self.core_dtype(h) != code:
                h = self.core.typecast(h, code)
        return Canon(h, pad2(exts), p.dtype)

    def core_dtype(self, h):
        return h.dtype


def decompose(sub: np.ndarray):
    """Split the axes of an index array into pieces over which it is affine.
    Returns ([(owner axis, extent, stride)], offset) or raises Unsupported; an axis
    whose slope changes after p elements (div / mod) becomes pieces (n // p, p)."""
    arr, owners = sub, list(range(sub.ndim))
    i, splits = 0, 0
    while i < arr.ndim:
        n = arr.shape[i]
        if n == 1:
            i += 1
            continue
        d = np.diff(arr, axis=i)
        line = np.diff(
            arr[tuple(slice(None) if k == i else 0 for k in range(arr.ndim))]
        )
        if bool(np.all(d == line[0])):
            i += 1
            continue
        differs = np.flatnonzero(line != line[0])
        if not len(differs):
            raise Unsupported("stride depends on another axis")
        p = int(differs[0]) + 1
        splits += 1
        if n % p or splits > 16:
            raise Unsupported("ragged split")
        arr = arr.reshape(arr.shape[:i] + (n // p, p) + arr.shape[i + 1 :])
        owners.insert(i + 1, owners[i])
    pieces = []
    for i in range(arr.ndim):
        n = arr.shape[i]
        stride = (
            int(
                arr[tuple(1 if k == i else 0 for k in range(arr.ndim))]
                - arr[(0,) * arr.ndim]
            )
            if n > 1
            else 0
        )
        pieces.append((owners[i], n, stride))
    off = int(arr[(0,) * arr.ndim])
    check = np.full(arr.shape, off, dtype=np.int64)
    for i, (_, n, stride) in enumerate(pieces):
        check += (np.arange(n, dtype=np.int64) * stride).reshape(
            [n if k == i else 1 for k in range(arr.ndim)]
        )
    if not np.array_equal(check, arr):
        raise Unsupported("index is not affine in its pieces")
    return pieces, off


def analyse_read(ia: np.ndarray, valid, shape, nelem: int, union):
    """Memory-op form of a read (`ia` flat indices, `valid` the gate or None):
    (axes, counts, starts, steps, fulls, pieces, offset), valid positions per axis
    being start + step * arange(count) (step > 1: zero-interleave)."""
    keep = [k for k, n in enumerate(shape) if n > 1]
    ia = ia.reshape([shape[k] for k in keep]) if keep else ia.reshape(())
    if valid is not None:
        valid = valid.reshape(ia.shape)
    if not keep:
        raise Unsupported("scalar read")
    pos_sets = [np.arange(s) for s in ia.shape]
    if valid is not None:
        if not valid.any():
            raise Unsupported("read gated off everywhere")
        box = np.ones(ia.shape, dtype=bool)
        for a in range(len(keep)):
            m = valid.any(axis=tuple(k for k in range(len(keep)) if k != a))
            nz = np.flatnonzero(m)
            if len(nz) > 1 and len(set(np.diff(nz).tolist())) != 1:
                raise Unsupported("gate is not an arithmetic progression on an axis")
            pos_sets[a] = nz
            box &= m.reshape([len(m) if k == a else 1 for k in range(len(keep))])
        if not np.array_equal(box, valid):
            raise Unsupported("gate is not a product of per-axis sets")
    starts = [int(p[0]) for p in pos_sets]
    steps = [int(p[1] - p[0]) if len(p) > 1 else 1 for p in pos_sets]
    counts = [len(p) for p in pos_sets]
    sub = ia[np.ix_(*pos_sets)]
    if sub.min() < 0 or sub.max() >= nelem:
        raise Unsupported("read outside the buffer")
    pieces, off = decompose(sub)
    pieces = [pc for pc in pieces if pc[1] > 1]
    # axes the buffer varies along; the rest broadcast
    live = sorted({pc[0] for pc in pieces if pc[2] != 0})
    if not live:
        raise Unsupported("constant read")
    for a in range(len(keep)):
        if a not in live and (counts[a] != ia.shape[a] or steps[a] != 1):
            raise Unsupported("padding along a broadcast axis")
    pos = {a: i for i, a in enumerate(live)}
    pieces = [(pos[pc[0]], pc[1], pc[2]) for pc in pieces if pc[0] in pos]
    return (
        tuple(union[keep[a]] for a in live),
        [counts[a] for a in live],
        [starts[a] for a in live],
        [steps[a] for a in live],
        [ia.shape[a] for a in live],
        tuple(pieces),
        off,
    )


def p_nelem(st, p) -> int:
    return st.nbytes // p.dtype.itemsize


class NoPattern(Exception):
    """The reduce is not the pattern a fast path looks for."""


class NotContraction(Exception):
    """The REDUCE_ADD(MUL) is not a contraction with a shared reduced axis."""
