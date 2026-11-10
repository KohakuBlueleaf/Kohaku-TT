"""The launch check of page indices and declared ranges (PAGE-1): every NoC page index
of a tensor and every `.bounded(low, high)` value computed on the host is bounded, for
one launch's values, by interval arithmetic over the cores and loops."""

import os
import warnings

import numpy as np

from kohakutt.dsl.metal.binding import (
    Bounded,
    Const,
    CoreCount,
    CoreIndex,
    Load,
    Min,
    Mutable,
    Op,
    Select,
    Value,
    Var,
    Word,
    as_expr,
)
from kohakutt.dsl.metal.body import (
    CBOp,
    ExitWhen,
    Let,
    Loop,
    Read,
    When,
    Write,
    exprs,
)
from kohakutt.dsl.metal.model import CoreArg, PerCore, TableArg, TableWord

# Elements (cores times enumerated loop iterations) a loop is enumerated up to; past
# it, a loop variable is the interval of its range.
ENUMERATE = 1 << 14
# Device values are 32-bit: interval ends are kept within this.
_CLIP = 1 << 40
_FLIP = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


class PageWarning(UserWarning):
    """A page index or declared range the launch check could not prove."""


def page_mode():
    """KTT_KM_PAGE_CHECK: "warn" (default), "strict" (an unproven access is refused)
    or "0" (no check)."""
    return os.environ.get("KTT_KM_PAGE_CHECK", "warn")


class _Unknown(Exception):
    """A value known on the device only."""


def _arr(value, n):
    return np.full(n, int(value), dtype=np.int64)


def _clip(lo, hi):
    return np.clip(lo, -_CLIP, _CLIP), np.clip(hi, -_CLIP, _CLIP)


class _Context:
    """The elements (a core and a choice of the enumerated loop variables) reaching a
    statement: `possible` may reach it, `certain` reach it for sure."""

    def __init__(self, cores, possible):
        self.n = len(cores)
        self.core = np.asarray(cores, dtype=np.int64)
        self.possible = np.asarray(possible, dtype=bool)
        self.certain = self.possible.copy()
        self.loops, self.narrow = {}, {}

    def copy(self):
        out = _Context.__new__(_Context)
        out.n, out.core = self.n, self.core
        out.possible, out.certain = self.possible.copy(), self.certain.copy()
        out.loops, out.narrow = dict(self.loops), dict(self.narrow)
        return out

    def repeat(self, times):
        """This context with each element repeated `times` times."""
        out = self.copy()
        out.n = self.n * times
        out.core = np.repeat(self.core, times)
        out.possible = np.repeat(self.possible, times)
        out.certain = np.repeat(self.certain, times)
        out.loops = {
            k: tuple(np.repeat(a, times) for a in v) for k, v in out.loops.items()
        }
        out.narrow = {
            k: tuple(np.repeat(a, times) for a in v) for k, v in out.narrow.items()
        }
        return out


class _Eval:
    """Intervals (lo, hi) of expressions over a context's elements; raises _Unknown
    for a value known on the device only."""

    def __init__(self, check, ctx):
        self.check, self.ctx, self.memo, self.busy = check, ctx, {}, set()

    def __call__(self, expr):
        expr = as_expr(expr)
        got = self.memo.get(id(expr))
        if got is None:
            if id(expr) in self.busy:
                raise _Unknown()
            self.busy.add(id(expr))
            bound = (
                self.ctx.narrow.get(self.check.key(expr)) if self.ctx.narrow else None
            )
            try:
                got = self._raw(expr)
            except _Unknown:
                # A device value the conditions around bound: a 32-bit unsigned.
                if bound is None or np.any(self.ctx.possible & (bound[1] >= _CLIP)):
                    raise
                got = (np.zeros(self.ctx.n, np.int64), np.full(self.ctx.n, _CLIP))
            finally:
                self.busy.discard(id(expr))
            if bound is not None:
                got = (np.maximum(got[0], bound[0]), np.minimum(got[1], bound[1]))
            self.memo[id(expr)] = got
        return got

    def _scalar(self, value):
        a = _arr(value, self.ctx.n)
        return a, a

    def _raw(self, e):
        env = self.check.env
        if isinstance(e, Const):
            return self._scalar(e.number)
        if isinstance(e, CoreIndex):
            return self.ctx.core, self.ctx.core
        if isinstance(e, CoreCount):
            return self._scalar(env["cores"])
        if isinstance(e, Bounded):
            try:
                return self(e.inner)
            except _Unknown:
                return self(e.low)[0], self(e.high)[1] - 1
        if isinstance(e, (Load, Word, TableWord, TableArg)):
            raise _Unknown()
        if isinstance(e, Value):
            if e.name not in env:
                raise _Unknown()
            return self._scalar(env[e.name])
        if isinstance(e, PerCore):
            return self._per_core(e)
        if isinstance(e, Mutable):
            parts = [self(v) for v in e.assigned]
            return (
                np.minimum.reduce([p[0] for p in parts]),
                np.maximum.reduce([p[1] for p in parts]),
            )
        if isinstance(e, Var):
            if id(e) in self.ctx.loops:
                return self.ctx.loops[id(e)]
            given = e.source if e.source is not None else e.definition
            given = given if given is not None else self.check.lets.get(id(e))
            if given is None:
                raise _Unknown()
            return self(given)
        if isinstance(e, Min):
            (al, ah), (bl, bh) = self(e.a), self(e.b)
            return np.minimum(al, bl), np.minimum(ah, bh)
        if isinstance(e, Select):
            cl, ch = self(e.when)
            (al, ah), (bl, bh) = self(e.a), self(e.b)
            lo = np.where(cl >= 1, al, np.where(ch < 1, bl, np.minimum(al, bl)))
            hi = np.where(cl >= 1, ah, np.where(ch < 1, bh, np.maximum(ah, bh)))
            return lo, hi
        if isinstance(e, Op):
            return _clip(*self._op(e.op, self(e.a), self(e.b)))
        raise _Unknown()

    def _per_core(self, e):
        if isinstance(e, CoreArg):
            values = []
            for c in self.ctx.core:
                value = e.fn(*self.check.cores[c], self.check.env["cores"], 0)
                if not isinstance(value, int):
                    raise _Unknown()
                values.append(value)
        else:
            values = [e.at(int(c), self.check.env["cores"]) for c in self.ctx.core]
        a = np.asarray(values, dtype=np.int64)
        return a, a

    def _op(self, op, a, b):
        (al, ah), (bl, bh) = a, b
        if op == "+":
            return al + bl, ah + bh
        if op == "-":
            return al - bh, ah - bl
        if op == "*":
            corners = [x.astype(float) * y for x in (al, ah) for y in (bl, bh)]
            lo, hi = np.minimum.reduce(corners), np.maximum.reduce(corners)
            lo, hi = _clip(lo, hi)
            return lo.astype(np.int64), hi.astype(np.int64)
        # Unsigned on the device: a negative operand wraps.
        negative = np.any(self.ctx.possible & ((al < 0) | (bl < 0)))
        if op in ("/", "%", ">>", "<<", "&") and negative:
            raise _Unknown()
        if op == "/":
            if np.any(self.ctx.possible & (bl < 1)):
                raise _Unknown()
            safe_l, safe_h = np.maximum(bl, 1), np.maximum(bh, 1)
            return al // safe_h, ah // safe_l
        if op == "%":
            if np.any(self.ctx.possible & (bl < 1)):
                raise _Unknown()
            # Exact where b is one value and a stays within one period of it.
            b = np.maximum(bl, 1)
            one = (bl == bh) & (al // b == ah // b)
            lo = np.where(one, al % b, 0)
            return lo, np.where(one, ah % b, np.minimum(ah, bh - 1))
        if op == ">>":
            return al >> np.minimum(bh, 62), ah >> np.minimum(bl, 62)
        if op == "<<":
            return al << np.minimum(bl, 40), ah << np.minimum(bh, 40)
        if op == "&":
            return np.zeros_like(al), np.minimum(ah, bh)
        return _compare(op, al, ah, bl, bh)


def _compare(op, al, ah, bl, bh):
    """Interval (0 / 1) of comparison `op`."""
    if op in (">", ">="):
        return _compare({">": "<", ">=": "<="}[op], bl, bh, al, ah)
    if op == "<":
        sure, maybe = ah < bl, al < bh
    elif op == "<=":
        sure, maybe = ah <= bl, al <= bh
    elif op == "==":
        sure = (al == ah) & (bl == bh) & (al == bl)
        maybe = (al <= bh) & (bl <= ah)
    else:
        sure = (ah < bl) | (bh < al)
        maybe = ~((al == ah) & (bl == bh) & (al == bl))
    return sure.astype(np.int64), maybe.astype(np.int64)


class Access:
    """One checked value: a page index of tensor `ref`, or a declared range."""

    def __init__(self, kernel, expr, ref=None, bounded=None):
        self.kernel, self.expr, self.ref, self.bounded = kernel, expr, ref, bounded

    def describe(self):
        if self.ref is not None:
            return f"{self.kernel}: page index {self.expr.cpp()} of {self.ref.name}"
        b = self.bounded
        return (
            f"{self.kernel}: {b.inner.cpp()} declared in {b.low.cpp()} .. "
            f"{b.high.cpp()} - 1"
        )


class PageCheck:
    """PAGE-1 for `program`: `check(env, count, pages)` returns the accesses it could
    not prove (an Access and the reason) and raises ValueError on a proven one."""

    def __init__(self, program):
        self.program, self.keys, self.lets = program, {}, {}
        for kernel in program.kernels.values():
            self._lets(kernel.body)

    def key(self, expr):
        got = self.keys.get(id(expr))
        if got is None:
            got = self.keys[id(expr)] = (expr, expr.key())
        return got[1]

    def _lets(self, body):
        for stmt in body:
            if isinstance(stmt, Let) and stmt.var.source is None:
                self.lets[id(stmt.var)] = stmt.expr
            for inner in (getattr(stmt, "body", None), getattr(stmt, "orelse", None)):
                if isinstance(inner, list):
                    self._lets(inner)

    def check(self, env, count, pages):
        """`env`: the launch's runtime, sized and derived values; `count`: its core
        count; `pages`: {tensor name: pages}."""
        self.env, self.pages = {**env, "cores": count}, pages
        self.cores = self.program.machine.core_list(count)
        self.found = {}
        for name, kernel in self.program.kernels.items():
            own = set(kernel.cores.select(self.cores))
            index = [i for i, c in enumerate(self.cores) if c in own]
            if not index:
                continue
            ctx = _Context(index, np.ones(len(index), dtype=bool))
            if kernel.active is not None:
                try:
                    lo, _ = _Eval(self, ctx)(kernel.active)
                    ctx.possible &= ctx.core < lo
                    ctx.certain = ctx.possible.copy()
                except _Unknown:
                    pass
            self._walk(name, kernel.body, ctx)
        return list(self.found.values())

    def _walk(self, name, body, ctx):
        for stmt in body:
            if not ctx.possible.any():
                return
            ctx = _forget(ctx, self._renewed(stmt))
            if isinstance(stmt, Loop):
                changed = self._assigned(stmt.body)
                inner = self._loop(stmt, _forget(ctx, changed))
                if inner is not None:
                    self._walk(name, stmt.body, inner)
                ctx = _forget(ctx, changed)
                continue
            if isinstance(stmt, When):
                self._walk(name, stmt.body, self._branch(stmt.condition, True, ctx))
                if stmt.orelse is not None:
                    other = self._branch(stmt.condition, False, ctx)
                    self._walk(name, stmt.orelse, other)
                changed = self._assigned(stmt.body + (stmt.orelse or []))
                ctx = _forget(ctx, changed)
                continue
            if isinstance(stmt, ExitWhen):
                # The rest of the loop body runs where the exit was not taken.
                ctx = self._branch(stmt.condition, False, ctx)
                continue
            self._statement(name, stmt, ctx)

    def _renewed(self, stmt):
        """The keys of the values `stmt` gives a new value: an assigned variable, a
        taken or fetched word, the words loaded from a buffer it reserves or waits."""
        var = None if isinstance(stmt, (Loop, Let)) else getattr(stmt, "var", None)
        if var is not None:
            return {self.key(var)}
        if isinstance(stmt, CBOp):
            return {f"load({stmt.cb.name},"}
        return set()

    def _assigned(self, body):
        """The keys of the values `body` renews (`_renewed`), nested bodies included."""
        out = set()
        for stmt in body:
            out |= self._renewed(stmt)
            for inner in (getattr(stmt, "body", None), getattr(stmt, "orelse", None)):
                if isinstance(inner, list):
                    out |= self._assigned(inner)
        return out

    def _loop(self, loop, ctx):
        """The context inside `loop` (None: it never runs)."""
        ev = _Eval(self, ctx)
        if loop.count is None:
            return ctx
        try:
            cl, ch = ev(loop.count)
            sl, sh = (np.zeros(ctx.n, np.int64),) * 2
            if loop.start is not None:
                sl, sh = ev(loop.start)
            step = as_expr(loop.step)
            tl, th = ev(step)
        except _Unknown:
            inner = ctx.copy()
            inner.loops.pop(id(loop.var), None)
            inner.certain[:] = False
            return inner
        live = ctx.possible
        exact = np.all(((cl == ch) & (sl == sh) & (tl == th) & (tl >= 1)) | ~live)
        if exact:
            steps = np.where(live, -(-(ch - sh) // np.maximum(tl, 1)), 0)
            most = int(max(0, steps.max()))
            if most and ctx.n * most <= ENUMERATE:
                inner = ctx.repeat(most)
                j = np.tile(np.arange(most, dtype=np.int64), ctx.n)
                v = np.repeat(sh, most) + j * np.repeat(tl, most)
                inside = v < np.repeat(ch, most)
                inner.possible &= inside
                inner.certain &= inside
                inner.loops[id(loop.var)] = (v, v)
                return inner
        inner = ctx.copy()
        inner.possible &= ch > sl
        inner.certain &= cl > sh
        top = ch - 1
        if np.all(((sl == sh) & (tl == th) & (tl >= 1)) | ~live):
            # The last value a stepped loop takes from its start.
            t = np.maximum(tl, 1)
            top = np.where(top >= sh, sh + (top - sh) // t * t, top)
        inner.loops[id(loop.var)] = (sl, top)
        if loop.remainder is not None:
            inner.loops.pop(id(loop.remainder), None)
        return inner

    def _branch(self, condition, taken, ctx):
        """The context of the branch of a `when` on `condition` (taken or else)."""
        out = ctx.copy()
        ev = _Eval(self, ctx)
        try:
            lo, hi = ev(condition)
            if taken:
                out.possible &= hi >= 1
                out.certain &= lo >= 1
            else:
                out.possible &= lo < 1
                out.certain &= hi < 1
        except _Unknown:
            out.certain[:] = False
        op = condition.op if taken else _FLIP[condition.op]
        big = np.full(ctx.n, _CLIP, dtype=np.int64)
        sides = []
        for side in (condition.a, condition.b):
            try:
                sides.append(ev(side))
            except _Unknown:
                sides.append((-big, big))
        a, b = sides
        bounds = {
            "<": ((-big, b[1] - 1), (a[0] + 1, big)),
            "<=": ((-big, b[1]), (a[0], big)),
            ">": ((b[0] + 1, big), (-big, a[1] - 1)),
            ">=": ((b[0], big), (-big, a[1])),
            "==": ((b[0], b[1]), (a[0], a[1])),
        }.get(op)
        if bounds is not None:
            for side, (lo, hi) in zip((condition.a, condition.b), bounds):
                key = self.key(side)
                old = out.narrow.get(key)
                if old is not None:
                    lo, hi = np.maximum(lo, old[0]), np.minimum(hi, old[1])
                out.narrow[key] = (lo, hi)
        return out

    def _statement(self, name, stmt, ctx):
        ev = _Eval(self, ctx)
        if isinstance(stmt, (Read, Write)) and stmt.ref.name in self.pages:
            access = self._access(name, stmt.page, ref=stmt.ref)
            self._judge(access, ev, 0, self.pages[stmt.ref.name] - 1)
        for expr in exprs(stmt):
            for b in _bounded(expr, set()):
                try:
                    ev(b.inner)
                    (ll, lh), (hl, hh) = ev(b.low), ev(b.high)
                except _Unknown:
                    continue
                access = self._access(name, b.inner, bounded=b)
                self._judge(access, ev, lh, hl - 1, (ll, hh - 1))

    def _access(self, name, expr, ref=None, bounded=None):
        key = (name, id(expr), id(ref), id(bounded))
        if key not in self.found:
            self.found[key] = (Access(name, expr, ref, bounded), None)
            self.found[key][0].ident = key
        return self.found[key][0]

    def _judge(self, access, ev, low, high, wide=None):
        """Raises on a value of `access` surely outside `wide` (default: low ..
        high), records one that may leave low .. high."""
        ctx, key = ev.ctx, access.ident
        try:
            lo, hi = ev(access.expr)
        except _Unknown:
            if access.bounded is None and not _table_form(access.expr):
                self.found[key] = (access, "it reads a value known on the device only")
            return

        def spread(v):
            return np.broadcast_to(np.asarray(v, dtype=np.int64), lo.shape)

        low, high = spread(low), spread(high)
        far_low, far_high = (low, high) if wide is None else map(spread, wide)
        out = (lo > far_high) | (hi < far_low)
        proven = ctx.certain & out
        if proven.any():
            i = int(np.argmax(proven))
            value = int(lo[i] if lo[i] > far_high[i] else hi[i])
            core, top = int(ctx.core[i]), int(far_high[i])
            raise ValueError(_message(self, access, value, core, top))
        maybe = ctx.possible & ((hi > high) | (lo < low))
        if maybe.any():
            i = int(np.argmax(maybe))
            reach = int(hi[i]) if hi[i] > high[i] else int(lo[i])
            self.found[key] = (
                access,
                f"its bound reaches {reach} on core {int(ctx.core[i])}, the range is "
                f"{int(low[i])} .. {int(high[i])}",
            )


def _message(check, access, value, core, high):
    name = check.program.name
    if access.ref is not None:
        return (
            f"kohakutt.dsl.metal: {name}: PAGE-1: {access.describe()} reaches {value} "
            f"on core {core}; {access.ref.name} has {high + 1} pages"
        )
    return (
        f"kohakutt.dsl.metal: {name}: PAGE-1: {access.describe()} reaches {value} on "
        f"core {core}"
    )


def _forget(ctx, keys):
    """`ctx` without the narrowing of the values keyed `keys` and of every value
    whose key contains one of them."""
    stale = [k for k in ctx.narrow if any(key in k for key in keys)]
    if not stale:
        return ctx
    out = ctx.copy()
    for key in stale:
        del out.narrow[key]
    return out


def _bounded(expr, seen):
    """The Bounded values inside `expr`."""
    if id(expr) in seen:
        return []
    seen.add(id(expr))
    found = [expr] if isinstance(expr, Bounded) else []
    parts = list(expr.parts())
    if isinstance(expr, Bounded):
        parts = [expr.inner]
    for part in parts:
        found += _bounded(part, seen)
    return found


def _table_form(expr):
    """True when `expr` reads a table word (checked against the tensor's pages when
    the table is filled)."""
    return any(isinstance(leaf, (TableWord, TableArg)) for leaf in expr.leaves())


def report_pages(program, unproven, reported):
    """Warns once per access of `unproven` not in `reported`; refuses them under
    KTT_KM_PAGE_CHECK=strict."""
    fresh = [(a, why) for a, why in unproven if why and a.ident not in reported]
    texts = [f"{a.describe()}: {why}" for a, why in unproven if why]
    if texts and page_mode() == "strict":
        raise ValueError(
            f"kohakutt.dsl.metal: {program.name}: PAGE-1 (KTT_KM_PAGE_CHECK=strict): "
            f"not proven: {'; '.join(texts)}"
        )
    for access, why in fresh:
        reported.add(access.ident)
        warnings.warn(
            f"kohakutt.dsl.metal: {program.name}: PAGE-1 could not prove "
            f"{access.describe()}: {why}",
            PageWarning,
            stacklevel=4,
        )
