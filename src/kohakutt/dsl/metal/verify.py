"""The verifier: rules a program must pass before any source is emitted. Each rule
comes from a hazard hit on the hardware or documented by TT-Metal; RULES lists them.
"""

import itertools
import math
import re
from dataclasses import dataclass

from kohakutt.dsl.metal.binding import (
    PER_CORE,
    Bounded,
    CoreCount,
    CoreIndex,
    Expr,
    Load,
    Min,
    Mutable,
    Op,
    Select,
    Sized,
    Value,
    Var,
    VerifyError,
    _upper_and,
    as_expr,
)
from kohakutt.dsl.metal.body import (
    Apply,
    Assign,
    Barrier,
    Binary,
    CBOp,
    Copy,
    Else,
    ExitWhen,
    Format,
    Init,
    Inline,
    Let,
    Loop,
    Matmul,
    MatmulInit,
    Mcast,
    McastSet,
    Pack,
    PackAccumulate,
    PackFormat,
    Put,
    Read,
    Reduce,
    ReduceDone,
    ReduceInit,
    Registers,
    SemOp,
    Send,
    SfpuCall,
    Slot,
    SlotAt,
    SrcaFormat,
    Startup,
    Take,
    Tiles,
    TilesInit,
    Unary,
    Waypoint,
    When,
    Write,
    Zone,
    conditions,
    exprs,
    loops,
    walk,
)
from kohakutt.dsl.metal.model import TableWord
from kohakutt.dsl.metal.ops import OPS, UNPACK_SETUP
from kohakutt.dsl.metal.sfpu import GeneratedFn

# The statements that run on the math thread, between acquire and commit.
MATH = (Copy, Unary, Binary, SfpuCall, Apply, Matmul, Tiles, Reduce)

# Runtime arguments per core and in common (TT-Metal's max_runtime_args).
MAX_ARGS = 341
# Combinations of listed runtime values and sized values one check may walk.
MAX_CASES = 50000
# Circular buffer indices of a core.
MAX_BUFFERS = 64
# The NoC a reader (dm0) and a writer (dm1) use unless pinned.
DEFAULT_NOC = {"dm0": 0, "dm1": 1}
# The shared vector-unit constant register an SFPCONFIG write may change.
SHARED_CONSTANT = "lreg11"
KINDS = ("reserve", "push", "wait", "pop")


def _fail(rule, message):
    raise VerifyError(rule, message)


def _int(value, what):
    """`value` as a static int, or a BIND-1 error naming `what`."""
    if isinstance(value, bool) or not isinstance(value, (int, Expr)):
        _fail("BIND-1", f"{what} is {value!r}, not a static integer")
    return as_expr(value)._static(what)


def _cases(what, *values):
    """Every {name: value} over the listed values' leaves: runtime values from a
    list and sized values. A value computed from anything else is BIND-1."""
    listed = {}
    for value in values:
        for node in _nodes(as_expr(value), set()):
            if isinstance(node, Value) and node.choices:
                listed[node.name] = node.choices
            elif isinstance(node, (Load, Mutable)) or (
                node.value() is None and not node.parts()
            ):
                node._static(what)
    if math.prod(len(choices) for choices in listed.values()) > MAX_CASES:
        _fail("BIND-1", f"{what} has more than {MAX_CASES} combinations to check")
    names = list(listed)
    for picked in itertools.product(*listed.values()):
        yield dict(zip(names, picked))


def _counts(value, what):
    """The values a page count, loop step or register slot can take, ascending: a
    static's one value, or what a value computed from runtime values from a list and
    sized values comes to; anything else is BIND-1."""
    if isinstance(value, bool) or not isinstance(value, (int, Expr)):
        _fail("BIND-1", f"{what} is {value!r}, not a static integer")
    value = as_expr(value)
    if value.value() is not None:
        return (value.value(),)
    return tuple(sorted({value.eval(env) for env in _cases(what, value)}))


def _at_most(count, pages, positive=False):
    """True when `count` cannot exceed `pages` by how it is written: the same value,
    the smaller of it and another, or the items of a block loop stepping by it.
    `positive`: `pages` is known to be at least 1 (a sized value's source)."""
    if count is None:
        return False
    if count.value() is not None and count.value() <= (1 if positive else 0):
        return True
    if count.key() == pages.key():
        return True
    if isinstance(count, Min):
        return _at_most(count.a, pages, positive) or _at_most(count.b, pages, positive)
    if isinstance(count, Var) and count.block_of is not None:
        return _at_most(as_expr(count.block_of.step), pages, positive)
    if isinstance(count, Var) and count.parts() and not isinstance(count, Mutable):
        return _at_most(count.parts()[0], pages, positive)
    # A sized value is at least what it is computed from, which the host takes only
    # from 1 on; a product of unsigned factors that is at least 1 bounds each factor.
    if isinstance(pages, Sized):
        return _at_most(count, pages.of, True)
    if isinstance(pages, Op) and pages.op == "*":
        for factor, other in ((pages.a, pages.b), (pages.b, pages.a)):
            whole = positive or (factor.value() or 0) >= 1
            if whole and _at_most(count, other, positive):
                return True
    return False


def _lowest(expr):
    """The smallest value `expr` takes as integer arithmetic, before the device's
    unsigned wrap (negative: it can wrap), or None when not known."""
    expr = as_expr(expr)
    if expr.value() is not None:
        return expr.value()
    if isinstance(expr, Bounded):
        return expr.low.value()
    if isinstance(expr, Value):
        return min(expr.choices) if expr.choices else 0
    if isinstance(expr, Mutable):
        return expr._over_assigned(lambda values: _least(map(_lowest, values)), 0)
    if isinstance(expr, Var) and expr.parts():
        return _lowest(expr.parts()[0])
    if isinstance(expr, Select):
        return _least((_lowest(expr.a), _lowest(expr.b)))
    if not isinstance(expr, Op):
        return 0
    la, lb = _lowest(expr.a), _lowest(expr.b)
    if expr.op == "-":
        top = expr.b.upper()
        return None if la is None or top is None else la - top
    if la is not None and la < 0 or lb is not None and lb < 0:
        return min(la or 0, lb or 0)
    if expr.op in ("+", "*"):
        return None if la is None or lb is None else _FOLD_LOW[expr.op](la, lb)
    return 0


_FOLD_LOW = {"+": lambda a, b: a + b, "*": lambda a, b: a * b}


def _least(values):
    values = list(values)
    return None if None in values else min(values)


def _below_zero(index):
    """True when page index `index` can be negative, so that it wraps."""
    low = _lowest(index)
    return low is not None and low < 0


def _top(index):
    """The largest value of page index `index`: a loop variable stepping by s from 0
    over a static count n ends at the last multiple of s below n."""
    count, step = getattr(index, "count", None), getattr(index, "step", None)
    stepped = isinstance(index, Var) and isinstance(step, int) and count is not None
    if stepped and (count.value() or 0) >= 1:
        return (count.value() - 1) // step * step
    return index.upper()


def _within(index, pages):
    """True when page `index` lies inside `pages` by how it is written: a loop
    variable counting to at most `pages`, or a value declared `.bounded(0, pages)`."""
    pages = as_expr(pages)
    if isinstance(index, Bounded):
        return index.low.value() == 0 and _at_most(index.high, pages)
    return _at_most(getattr(index, "count", None), pages)


def _by_multiples(extent, pages):
    """True when `extent` is at most `pages`, an exact sized value of f * g, because a
    runtime value f is declared a multiple of n: extent n (g >= 1), or (n - 1) * g + 1.
    """
    if not (isinstance(pages, Sized) and pages.exact and isinstance(pages.of, Op)):
        return False
    if pages.of.op != "*":
        return False
    pairs = ((pages.of.a, pages.of.b), (pages.of.b, pages.of.a))
    exact = isinstance(extent, Sized) and extent.exact
    if exact and isinstance(extent.of, Op) and extent.of.op == "*":
        # u * v within f * g where f and g are declared multiples of u and v.
        units = {getattr(f, "multiple_of", None) for f in (pages.of.a, pages.of.b)}
        names = {getattr(x, "name", None) for x in (extent.of.a, extent.of.b)}
        if len(names) == 2 and names == units:
            return True
    for f, g in pairs:
        unit = getattr(f, "multiple_of", None)
        if isinstance(f, Value) and isinstance(unit, str):
            n = Value(unit)
            if extent.key() in (n.key(), ((n - 1) * g + 1).key()):
                return True
    return False


def _inside(index, extent, pages):
    """True when the `extent` pages from page `index` lie inside `pages`: one page
    as in _within, several from a value declared `.bounded(0, pages - extent + 1)`
    with `extent` at most `pages`, or by the static bounds of both."""
    pages, extent = as_expr(pages), as_expr(extent)
    if _below_zero(index):
        return False
    if extent.value() == 1 and _within(index, pages):
        return True
    declared = isinstance(index, Bounded) and index.low.value() == 0
    fits = _at_most(extent, pages) or _by_multiples(extent, pages)
    fits = fits or (
        extent.upper() is not None and extent.upper() <= min(_counts(pages, "pages"))
    )
    if declared and fits and index.high.key() == ((pages - extent) + 1).key():
        return True
    if as_expr(index).value() == 0 and fits:
        return True
    # Declared against a part of the buffer: .bounded(0, part - extent + 1).
    high = index.high if declared else None
    if isinstance(high, Op) and high.op == "+" and high.b.value() == 1:
        rest = high.a
        if isinstance(rest, Op) and rest.op == "-" and rest.b.key() == extent.key():
            part = rest.a
            in_part = _at_most(extent, part) or _by_multiples(extent, part)
            if in_part and _at_most(part, pages):
                return True
    top, most = _top(index), extent.upper()
    return (
        top is not None
        and most is not None
        and top + most <= min(_counts(pages, "pages"))
    )


def _may_skip(loop):
    """True when `loop` may run zero times: its count is not a static count of at
    least one from 0 (a split's count is 0 on the cores past the work)."""
    if loop.count is None:
        return True
    count = as_expr(loop.count).value()
    return loop.start is not None or count is None or count < 1


def _exits(loop):
    """The ExitWhen statements directly in endless loop `loop`'s body."""
    return [s for s in loop.body if isinstance(s, ExitWhen)]


def _ending(loop):
    """The comparison that ends endless loop `loop`."""
    exits = _exits(loop)
    if len(exits) != 1:
        _fail("CTL-1", "an endless loop has one exit_when directly in its body")
    return exits[0].condition


def _streams(program):
    """{id(endless loop): name of the loop whose words drive it}: a loop ending on a
    word it takes is driven by the loop that puts that word, else by itself."""
    found = []
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            if isinstance(stmt, Loop) and stmt.count is None:
                found.append((f"{role}.{len(found)}", stmt))
    feeds, out = {}, {}
    for _name, loop in found:
        if len(_exits(loop)) == 1:
            for name in _taken(loop)[1]:
                feeds.setdefault(name, loop)
    names = {id(loop): name for name, loop in found}
    for _name, loop in found:
        seen, at = {id(loop)}, loop
        while len(_exits(at)) == 1:
            condition = _ending(at)
            ends = [n for n, t in _taken(at)[0].items() if t.var is condition.a]
            source = feeds.get(ends[0]) if ends else None
            if source is None or id(source) in seen:
                break
            seen.add(id(source))
            at = source
        out[id(loop)] = names[id(at)]
    return out


def _phase(loop, path, stmt):
    """Where `stmt` (under `path`) runs in an iteration of endless loop `loop`:
    "before" its exit (once more often than "after" it)."""
    at = path.index(loop)
    child = path[at + 1] if at + 1 < len(path) else stmt
    if isinstance(child, Else):
        child = child.when
    index = next(i for i, s in enumerate(loop.body) if s is child)
    exit_at = next(i for i, s in enumerate(loop.body) if isinstance(s, ExitWhen))
    return "before" if index < exit_at else "after"


def _trips(path, times=1, stmt=None, streams=None):
    """How often a statement `stmt` under loops `path` runs: (static factor, symbolic
    factors). An endless loop counts by the stream that drives it (`streams`)."""
    path, full = loops(path), path
    number, streams = 1, streams or {}
    factors = []
    for loop in path:
        if loop.count is None:
            _ending(loop)
            stream = streams.get(id(loop), f"loop{id(loop)}")
            factors.append(("until", stream, _phase(loop, full, stmt)))
    path = tuple(loop for loop in path if loop.count is not None)
    blocked = {
        id(loop.count.block_of)
        for loop in path
        if isinstance(loop.count, Var) and loop.count.block_of in path
    }
    for loop in path:
        if id(loop) in blocked:
            continue
        if loop.start is not None:
            keys = (loop.count.key(), loop.start.key(), as_expr(loop.step).key())
            factors.append(("from", *keys))
            continue
        count, steps = loop.count, _counts(loop.step, "a loop step")
        step = steps[0]
        if isinstance(count, Var) and count.block_of in path:
            factors.append(("each", count.block_of.count.key()))
        elif len(steps) > 1:
            factors.append(("blocks", count.key(), loop.step.key()))
        elif count.value() is not None:
            number *= math.ceil(count.value() / step)
        elif step == 1:
            factors.append(("each", count.key()))
        else:
            factors.append(("blocks", count.key(), step))
    if isinstance(times, Expr) and times.value() is None:
        factors.append(("each", times.key()))
    else:
        number *= int(times)
    return number, tuple(sorted(factors))


def _fixed(expr):
    """True when `expr` keeps one value per core over a launch: it reads argument
    values and the core's place only (no loop variable, assigned variable or load)."""
    for node in _nodes(expr, set()):
        named = isinstance(node, Var) and not node.parts()
        if named or isinstance(node, (Mutable, Load)):
            return False
    return True


def cb_ops(kernel):
    """(kind, buffer, pages, loops, runs, statement) of every buffer call of
    `kernel`, inline stages' and control words' included; `pages` is an int or a
    bounded runtime value."""
    for stmt, path in walk(kernel.body):
        if isinstance(stmt, CBOp):
            _counts(stmt.pages, "a page count")
            yield stmt.kind, stmt.cb, stmt.pages, path, 1, stmt
        elif isinstance(stmt, (Put, Take)):
            for kind in ("reserve", "push") if isinstance(stmt, Put) else KINDS[2:]:
                yield kind, stmt.cb, 1, path, 1, stmt
        elif isinstance(stmt, Inline):
            for use in stmt.uses:
                for kind in KINDS:
                    if getattr(use, kind):
                        yield kind, use.cb, getattr(use, kind), path, stmt.times, stmt


def _totals(kernel, kind, streams):
    """{buffer name: {symbolic factors: pages}} of `kind` calls over one launch."""
    out = {}
    for k, cb, pages, path, times, stmt in cb_ops(kernel):
        if k == kind:
            number, factors = _trips(path, times, stmt, streams)
            for condition, taken in conditions(path):
                if not _fixed(condition):
                    _fail(
                        "CB-2",
                        f"a {kind} on buffer {cb.name} sits inside a condition that "
                        "changes within a launch; its count cannot be proved",
                    )
                factors = (*factors, ("when", condition.key(), taken))
            counts = _counts(pages, "a page count")
            if len(counts) > 1:
                factors = (*factors, ("pages", pages.key()))
            else:
                number *= counts[0]
            per_cb = out.setdefault(cb.name, {})
            per_cb[factors] = per_cb.get(factors, 0) + number
    return {name: _merged(per_cb) for name, per_cb in out.items()}


def _merged(totals):
    """`totals` ({factors: pages}) with the two branches of a condition folded into
    one term where both take the same count."""
    totals = dict(totals)
    changed = True
    while changed:
        changed = False
        for factors, number in list(totals.items()):
            for f in factors:
                if f[0] != "when" or not f[2]:
                    continue
                rest = tuple(x for x in factors if x != f)
                other = tuple(x if x != f else (f[0], f[1], False) for x in factors)
                if totals.get(other) == number and factors in totals:
                    del totals[factors], totals[other]
                    totals[rest] = totals.get(rest, 0) + number
                    changed = True
                    break
    return totals


def _nodes(expr, seen):
    """Every value `expr` is computed from, itself included, once each."""
    if id(expr) in seen:
        return
    seen.add(id(expr))
    yield expr
    for part in expr.parts():
        yield from _nodes(part, seen)


def _loads(stmt):
    """The L1 loads the values of statement `stmt` are computed from."""
    seen, found = set(), list(exprs(stmt))
    if isinstance(stmt, Inline):
        found = [v for v in found if not isinstance(v, Slot)]
    return [n for e in found for n in _nodes(e, seen) if isinstance(n, Load)]


def _users(program, kinds):
    """{buffer name: roles of the kernels making a call of `kinds` on it}."""
    out = {}
    for role, kernel in program.kernels.items():
        for kind, cb, *_ in cb_ops(kernel):
            if kind in kinds:
                out.setdefault(cb.name, set()).add(role)
    return out


def _built(value, what):
    """The values a buffer size can take: static, or computed from sized values
    (a runtime value, even one from a list, is BIND-1)."""
    if isinstance(value, Expr):
        for node in _nodes(value, set()):
            if isinstance(node, Value) and not isinstance(node, Sized):
                node._static(what)
    return _counts(value, what)


def bind_1(program, _):
    for cb in program.cbs:
        if _built(cb.pages, f"the page count of buffer {cb.name}")[0] < 1:
            _fail("BIND-1", f"buffer {cb.name} has no pages")
    for s in program.scratches:
        _built(s.bytes, f"the size of scratch {s.name}")
    for sem in program.semaphores:
        _int(sem.initial, f"the initial value of semaphore {sem.name}")
    for role, kernel in program.kernels.items():
        for name, value in kernel.defines.items():
            if isinstance(value, Expr):
                value._static(f"define {name} of {role}")
        for stmt, _path in walk(kernel.body):
            strided = isinstance(stmt, Loop) and stmt.start is not None
            if strided and not isinstance(stmt.step, CoreCount):
                _counts(stmt.step, f"a loop step of {role}")
            if isinstance(stmt, Loop) and not strided:
                _counts(stmt.step, f"a loop step of {role}")
            if isinstance(stmt, CBOp):
                _counts(stmt.pages, f"a {stmt.kind} count on buffer {stmt.cb.name}")


def bind_2(program, _):
    for name, values, count in program.per_core_args:
        given = values if isinstance(values, (list, tuple)) else [values]
        listed = isinstance(values, (list, tuple)) and len(values) == count
        if all(isinstance(v, Expr) for v in given) and (listed or count == 1):
            for value in given:
                if any(
                    not isinstance(leaf, (CoreIndex, CoreCount))
                    for leaf in value.leaves()
                ):
                    _fail(
                        "BIND-2",
                        f"per-core argument {name} is computed from {value.cpp()}; "
                        "only the core index and the core count are known per core",
                    )
            continue
        if values is not None and not callable(values):
            _fail(
                "BIND-2",
                f"per-core argument {name} takes its value from {values!r}; a per-core "
                "value is a function of the core index, or derived on the device",
            )


def _table_words(expr):
    """The table words value `expr` reads."""
    return [
        n
        for n in _nodes(as_expr(expr), set())
        if getattr(n, "table", None) is not None and hasattr(n, "col")
    ]


def _table_page(page, ref):
    """Whether page index `page` of tensor `ref` is `t[j, c]` or `t[j, c] + v`, with
    column c declared as pages of `ref` counted by column n and v walking `t[j, n]`."""
    word, var = page, None
    if isinstance(page, Op) and page.op == "+":
        word, var = page.a, page.b
    if not (isinstance(word, TableWord) and word.col is not None):
        return False
    given = word.table.pages.get(word.col)
    if given is None or given[0] is not ref:
        return False
    if var is None:
        return True
    count = getattr(var, "count", None)
    return (
        getattr(var, "step", None) == 1
        and isinstance(count, TableWord)
        and count.table is word.table
        and count.col == given[1]
        and count.row.key() == word.row.key()
    )


def _table_page_uses(program, role, stmt):
    """Refuses a table word anywhere but a loop count or a declared page index."""
    if isinstance(stmt, Loop):
        return
    pages = []
    if isinstance(stmt, (Read, Write)):
        pages = [stmt.page]
        if _table_page(stmt.page, stmt.ref):
            pages = []
    rest = [
        e for e in exprs(stmt) if e is not None and e is not getattr(stmt, "page", 0)
    ]
    if any(_table_words(e) for e in pages + rest):
        _fail(
            "ARG-2",
            f"{role} uses a table word outside a loop count or a page index "
            "t[j, c] (+ a loop over its count column) declared in the table's pages",
        )


def arg_2(program, _):
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            _table_page_uses(program, role, stmt)
            for word in (w for e in exprs(stmt) for w in _table_words(e)):
                table, row = word.table, word.row
                if word.col is None or table not in program.tables:
                    if table not in program.tables:
                        _fail("ARG-2", f"{role} reads a table of another program")
                    continue
                static = row.value()
                walks = (
                    getattr(row, "count", None) is table.used
                    and getattr(row, "step", None) == 1
                )
                if not (walks or (static is not None and 0 <= static < table.rows)):
                    _fail(
                        "ARG-2",
                        f"{role} reads row {row.cpp()} of table {table.name}; a row is "
                        f"a static index below {table.rows} or the variable of a loop "
                        f"over {table.name}.used",
                    )
            counts = []
            if isinstance(stmt, Loop):
                counts = [stmt.count, stmt.step, stmt.start]
            elif isinstance(stmt, Inline):
                counts = [stmt.times]
            for value in counts:
                for word in _table_words(value) if value is not None else []:
                    if word.upper() is None:
                        _fail(
                            "ARG-2",
                            f"{role} counts with column {word.col} of table "
                            f"{word.table.name}, which declares no most",
                        )
            if isinstance(stmt, CBOp) and _table_words(stmt.pages):
                _fail(
                    "ARG-2",
                    f"{role} takes a page count on buffer {stmt.cb.name} from a table",
                )


def arg_1(program, _):
    for name, _values, count in program.per_core_args:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            _fail(
                "ARG-1",
                f"per-core argument {name} has {count!r} entries; argument counts "
                "are fixed per kernel class",
            )
    for role, kernel in program.kernels.items():
        layout = kernel.layout()
        if layout.common > MAX_ARGS or layout.per_core_count > MAX_ARGS:
            _fail("ARG-1", f"{role} takes more than {MAX_ARGS} runtime arguments")


def _most(value, what):
    """The largest value a tile count can take: a static's, or the known upper
    bound of a value; a value without one is DST-1."""
    top = value.upper() if isinstance(value, Expr) else value
    if isinstance(top, bool) or not isinstance(top, int):
        _fail("DST-1", f"compute uses {value!r} as {what}, which has no known bound")
    return top


def _page_reads(stmt):
    """(buffer, page index, pages from it) of every read `stmt` makes of the waited
    pages of a buffer; the page count is an int or a value."""
    if isinstance(stmt, Copy):
        return [(stmt.cb, stmt.index, stmt.count)]
    if isinstance(stmt, Write) and stmt.slot is None:
        return [(stmt.cb, stmt.index, 1)]
    if isinstance(stmt, Tiles):
        return [(stmt.init.a, stmt.ia, 1), (stmt.init.b, stmt.ib, 1)]
    if isinstance(stmt, Reduce):
        init = stmt.init
        return [(init.cb, stmt.index, 1), (init.scaler, stmt.scaler_index, 1)]
    unbounded = isinstance(stmt, Matmul) and as_expr(stmt.init.kt).upper() is None
    if unbounded and isinstance(stmt.init.rt, Value):
        # Rows chosen per launch: the a tiles of the block span (rt - 1) * kt + 1.
        init = stmt.init
        span = (init.rt - 1) * as_expr(init.kt) + 1
        return [(init.a, stmt.ia, span), (init.b, stmt.ib, init.ct)]
    if isinstance(stmt, Matmul):
        init = stmt.init
        rows = _most(init.rt, "a block's row tiles") - 1
        inner = init.kt.upper() if isinstance(init.kt, Expr) else init.kt
        if rows and inner is None:
            _fail("CB-3", "a matmul block of several rows needs a bounded inner count")
        return [(init.a, stmt.ia, rows * (inner or 0) + 1), (init.b, stmt.ib, init.ct)]
    return []


def cb_4(program, _):
    seen = {}
    for b in (*program.cbs, *program.scratches):
        if not 0 <= b.index < MAX_BUFFERS:
            _fail("CB-4", f"buffer {b.name} has index {b.index}")
        if b.index in seen:
            _fail("CB-4", f"buffers {seen[b.index]} and {b.name} share index {b.index}")
        seen[b.index] = b.name
    producers = _users(program, ("reserve", "push"))
    consumers = _users(program, ("wait", "pop"))
    for what, users in (("filled", producers), ("drained", consumers)):
        for name, roles in users.items():
            ks = [program.kernels[r] for r in sorted(roles)]
            if any(a.cores.overlaps(b.cores) for i, a in enumerate(ks) for b in ks[:i]):
                _fail("CB-4", f"buffer {name} is {what} by {sorted(roles)}")
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            name = getattr(getattr(stmt, "cb", None), "name", None)
            held = isinstance(stmt, Write) and stmt.slot is not None
            if held and role not in producers.get(name, ()):
                _fail("CB-4", f"{role} writes from buffer {name} unreserved")
            for cb, _index, _extent in _page_reads(stmt):
                if role not in consumers.get(cb.name, ()):
                    _fail("CB-4", f"{role} reads buffer {cb.name} without a wait")
            if isinstance(stmt, Pack) and role not in producers.get(name, ()):
                _fail("CB-4", f"{role} packs into buffer {name} unreserved")


def cb_1(program, _):
    for role, kernel in program.kernels.items():
        for kind, cb, pages, path, times, _stmt in cb_ops(kernel):
            if cb.partials:
                continue
            recurring = bool(loops(path)) or not (isinstance(times, int) and times == 1)
            sizes, count = as_expr(cb.pages), as_expr(pages)
            for env in _cases("a page count", sizes, count):
                total, n = sizes.eval(env), count.eval(env)
                if n < 1 or n > total or (recurring and total % n):
                    _fail(
                        "CB-1",
                        f"{role} makes a {kind} of {n} on buffer {cb.name} of "
                        f"{total} pages; a recurring count must divide the buffer",
                    )


def _align(ref):
    """The byte alignment NoC reads of `ref` keep: 64 from DRAM, 16 from L1."""
    return 64 if ref.kind == "dram" else 16


def _held(slot):
    """Bytes of the smallest reserve `slot` can be."""
    return _counts(slot.pages, "pages")[0] * slot.cb.page_bytes


def _slot_end(role, dst, size, align):
    """Checks that `size` bytes written (or sent) at slot address `dst` stay in its
    reserve."""
    cb = dst.slot.cb
    held = _held(dst.slot)
    if _below_zero(dst.index) or _below_zero(dst.offset):
        _fail("CB-3", f"{role} writes buffer {cb.name} below its reserve")
    top = _top(dst.index)
    offset = dst.offset.upper()
    if dst.like is not None and offset is not None:
        offset += _upper_and(dst.like, dst.like.upper(), align - 1)
    fits = top is not None and offset is not None
    fits = fits and top * cb.page_bytes + offset + size <= held
    if not fits and _within(dst.index, dst.slot.pages):
        # Any page of the reserve, the last included: the write stays in one page.
        top, held = 0, cb.page_bytes
    if top is None or offset is None:
        _fail("CB-3", f"{role} writes buffer {cb.name} at an unbounded address")
    end = top * cb.page_bytes + offset + size
    if end > held:
        _fail(
            "CB-3",
            f"{role} writes up to byte {end} of a {held}-byte reserve (or page) on "
            f"buffer {cb.name}",
        )


def cb_3(program, _):
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            if isinstance(stmt, Read):
                size = stmt.size or stmt.ref.page_bytes
                _slot_end(role, stmt.dst, size, _align(stmt.ref))
            if isinstance(stmt, Mcast):
                slot = stmt.src.slot
                whole = (as_expr(slot.pages) * slot.cb.page_bytes).key()
                at = (stmt.src.index.value(), stmt.src.offset.value(), stmt.src.like)
                if at != (0, 0, None) or stmt.size.key() != whole:
                    _fail(
                        "CB-3",
                        f"{role} multicasts other than the whole reserve of buffer "
                        f"{slot.cb.name}",
                    )
            if isinstance(stmt, Send):
                if stmt.size <= 0:
                    _fail("CB-3", f"{role} sends {stmt.size} bytes")
                _slot_end(role, stmt.src, stmt.size, 16)
            held = isinstance(stmt, Write) and stmt.slot is not None
            if held and stmt.ref.page_bytes > _held(stmt.slot):
                _fail(
                    "CB-3",
                    f"{role} writes a {stmt.ref.page_bytes}-byte page from a "
                    f"{_held(stmt.slot)}-byte reserve on buffer {stmt.cb.name}",
                )
            for load in _loads(stmt):
                top = load.word.upper()
                low = _below_zero(load.word)
                if low or top is None or (top + 1) * 4 > _held(load.slot):
                    _fail(
                        "CB-3",
                        f"{role} loads a word of buffer {load.slot.cb.name} past its "
                        "reserve, or at an unbounded index",
                    )
            for cb, index, extent in _page_reads(stmt):
                if _below_zero(index):
                    _fail(
                        "CB-3",
                        f"{role} reads buffer {cb.name} at page {_text(index)}, which "
                        "can go below 0 and wrap (indices are unsigned)",
                    )
                if not _inside(index, extent, cb.pages):
                    _fail(
                        "CB-3",
                        f"{role} reads past buffer {cb.name}, or at an index with no "
                        "known bound (declare one with .bounded(low, high))",
                    )
            if isinstance(stmt, Inline):
                for use in stmt.uses:
                    held = (use.reserve or _counts(use.cb.pages, "pages")[0]) * (
                        use.cb.page_bytes
                    )
                    if use.written > held:
                        _fail(
                            "CB-3",
                            f"inline code of {role} writes {use.written} bytes into "
                            f"{held} reserved on buffer {use.cb.name}",
                        )
        _front(role, kernel.body, {})


def _meet(mine, theirs):
    """The waited pages {buffer: count} known when either `mine` or `theirs`
    happened, into `mine`."""
    for name in list(mine):
        if name not in theirs:
            del mine[name]
        elif mine[name].key() != theirs[name].key():
            least = min(_counts(mine[name], "pages") + _counts(theirs[name], "pages"))
            mine[name] = as_expr(least)


def _popped(waited, name, pages):
    """`waited` after a pop of `pages` from buffer `name`: what is left when both
    counts are static, else nothing known."""
    have, gone = waited.pop(name, None), as_expr(pages).value()
    left = None if have is None or None in (have.value(), gone) else have.value() - gone
    if left is not None and left > 0:
        waited[name] = as_expr(left)


def _front(role, stmts, waited):
    """Checks that statements `stmts` of `role` read only pages they waited for,
    updating `waited` ({buffer name: pages waited at the front})."""
    for stmt in stmts:
        if isinstance(stmt, Loop):
            # Twice: the second pass meets what the first left.
            before = dict(waited)
            _front(role, stmt.body, waited)
            _front(role, stmt.body, waited)
            if _may_skip(stmt):
                _meet(waited, before)
        elif isinstance(stmt, When):
            other = dict(waited)
            _front(role, stmt.body, waited)
            _front(role, stmt.orelse or [], other)
            _meet(waited, other)
        elif isinstance(stmt, CBOp) and stmt.kind == "wait":
            waited[stmt.cb.name] = as_expr(stmt.pages)
        elif isinstance(stmt, CBOp) and stmt.kind == "pop":
            _popped(waited, stmt.cb.name, stmt.pages)
        elif isinstance(stmt, Inline):
            for use in stmt.uses:
                if use.wait:
                    waited[use.cb.name] = as_expr(use.wait)
                if use.pop:
                    _popped(waited, use.cb.name, use.pop)
        for cb, index, extent in _page_reads(stmt):
            pages = waited.get(cb.name)
            if pages is None or not _inside(index, extent, pages):
                _fail(
                    "CB-3",
                    f"{role} reads buffer {cb.name} at page {_text(index)} past the "
                    "pages it waited for",
                )


def _rows(program, cores):
    """The grid rows of core set `cores`."""
    if cores.first is None:
        return set(range(program.machine.rows))
    return set(range(cores.first, cores.last + 1))


def cb_2(program, _):
    streams, kernels = _streams(program), list(program.kernels.values())
    pushes = [(k, _totals(k, "push", streams)) for k in kernels]
    pops = [(k, _totals(k, "pop", streams)) for k in kernels]
    partials = {cb.name for cb in program.cbs if cb.partials}
    names = {n for _, t in pushes + pops for n in t} - partials
    for name in names:
        for mine, theirs in ((pushes, pops), (pops, pushes)):
            for k, totals in mine:
                if name not in totals:
                    continue
                other = [
                    (q, t) for q, t in theirs if q.cores.overlaps(k.cores) and name in t
                ]
                rows = set().union(*(_rows(program, q.cores) for q, _ in other))
                if not other or not _rows(program, k.cores) <= rows:
                    _fail(
                        "CB-2",
                        f"buffer {name}: {k.name} pushes or pops it on cores where no "
                        "kernel does the other",
                    )
                for q, t in other:
                    push, pop = (totals, t) if mine is pushes else (t, totals)
                    if push[name] != pop[name]:
                        _fail(
                            "CB-2",
                            f"buffer {name}: pushes {push[name]} and pops {pop[name]} "
                            f"per launch differ ({k.name}, {q.name})",
                        )


def cb_5(program, _):
    kernel = program.kernels.get("compute")
    if kernel is None:
        return
    stmts = [s for s, _ in walk(kernel.body)]
    packed = {s.cb.name for s in stmts if isinstance(s, Pack) and not s.cb.partials}
    # A buffer the compute both pushes and waits on hands its pages over by a token,
    # when every page it pushes was packed.
    ops = [(s.kind, s.cb.name) for s in stmts if isinstance(s, CBOp)]
    state = _Units()
    try:
        _units(kernel.body, state)
    except VerifyError:
        state.partial = set(packed)
    handed = {n for k, n in ops if k == "push"} & {n for k, n in ops if k == "wait"}
    packed -= handed - state.partial
    for stmt in stmts:
        for cb, _index, _extent in _page_reads(stmt):
            if cb.name in packed:
                _fail(
                    "CB-5",
                    f"compute packs into buffer {cb.name} and copies from it; the "
                    "unpacker may read the slot before the packer wrote it",
                )


def _dm_pairs(program):
    """(dm0 kernel, dm1 kernel) of every pair of variants sharing a core."""
    ks = program.kernels.values()
    return [
        (a, b)
        for a in ks
        if a.role == "dm0"
        for b in ks
        if b.role == "dm1" and a.cores.overlaps(b.cores)
    ]


def noc_1(program, _):
    for dm0, dm1 in _dm_pairs(program):
        nocs = [
            DEFAULT_NOC[k.role] if k.noc is None else _int(k.noc, "a NoC")
            for k in (dm0, dm1)
        ]
        if nocs[0] == nocs[1] or not set(nocs) <= {0, 1}:
            _fail(
                "NOC-1",
                f"{dm0.name} and {dm1.name} of a core use NoC {nocs[0]} and NoC "
                f"{nocs[1]}",
            )


def noc_2(program, _):
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            if not isinstance(stmt, Read):
                continue
            align = _align(stmt.ref)
            cb, src, dst = stmt.dst.slot.cb, stmt.offset, stmt.dst
            aligned = cb.page_bytes % align == 0 and stmt.ref.page_bytes % align == 0
            if dst.like is not None:
                # The destination was built from the source offset modulo `align`.
                congruent = (
                    dst.like.key() == src.key() and dst.offset.multiple() % align == 0
                )
            elif src.value() is not None and dst.offset.value() is not None:
                congruent = (src.value() - dst.offset.value()) % align == 0
            else:
                congruent = (
                    src.multiple() % align == 0 and dst.offset.multiple() % align == 0
                )
            if not aligned or not congruent:
                _fail(
                    "NOC-2",
                    f"{role} reads {stmt.ref.name} at byte {src.cpp()} into byte "
                    f"{dst.offset.cpp()} of buffer {cb.name}; source and destination "
                    f"must be congruent modulo {align}",
                )


class _Flight:
    """NoC work a kernel has issued and not waited for: buffers with reads landing
    (`reading`), waited pages being written (`writing`), reserved slots being
    written (`sending`), whether a write is unacknowledged, and whether a semaphore
    increment is (`atomic`)."""

    def __init__(self):
        self.reading, self.writing, self.sending = set(), set(), set()
        self.unacked = self.sent = self.atomic = False

    def copy(self):
        out = _Flight()
        out.reading, out.writing = set(self.reading), set(self.writing)
        out.sending, out.unacked = set(self.sending), self.unacked
        out.sent, out.atomic = self.sent, self.atomic
        return out

    def join(self, other):
        """What is in flight when either this or `other` happened."""
        self.reading |= other.reading
        self.writing |= other.writing
        self.sending |= other.sending
        self.unacked = self.unacked or other.unacked
        self.sent = self.sent or other.sent
        self.atomic = self.atomic or other.atomic


def _noc_3(role, stmts, f):
    """Checks statements `stmts` of `role` in order, updating flight state `f`."""
    for stmt in stmts:
        for load in _loads(stmt):
            if load.slot.cb.name in f.reading:
                _fail(
                    "NOC-3",
                    f"{role} loads from buffer {load.slot.cb.name} before its reads "
                    "land",
                )
        if isinstance(stmt, Loop):
            # Twice: the second pass meets what the first left in flight.
            before = f.copy()
            _noc_3(role, stmt.body, f)
            _noc_3(role, stmt.body, f)
            if _may_skip(stmt):
                f.join(before)
        elif isinstance(stmt, When):
            other = f.copy()
            _noc_3(role, stmt.body, f)
            _noc_3(role, stmt.orelse or [], other)
            f.join(other)
        elif isinstance(stmt, Read):
            name = stmt.dst.slot.cb.name
            if name in f.sending:
                _fail("NOC-3", f"{role} reads into buffer {name} while it is written")
            f.reading.add(name)
        elif isinstance(stmt, Write):
            name = stmt.cb.name
            if stmt.slot is not None and name in f.reading:
                _fail("NOC-3", f"{role} writes buffer {name} before its reads land")
            (f.writing if stmt.slot is None else f.sending).add(name)
            f.unacked = True
        elif isinstance(stmt, Send):
            name = stmt.src.slot.cb.name
            if name in f.reading:
                _fail("NOC-3", f"{role} sends buffer {name} before its reads land")
            f.sending.add(name)
            f.unacked = f.sent = True
        elif isinstance(stmt, Mcast):
            name = stmt.src.slot.cb.name
            if name in f.reading:
                _fail("NOC-3", f"{role} multicasts buffer {name} before its reads land")
            f.sending.add(name)
            f.unacked = f.sent = True
        elif isinstance(stmt, McastSet):
            if f.sent:
                _fail(
                    "NOC-4",
                    f"{role} multicasts flag {stmt.sem.name} before the data it sent "
                    "left (k.flushed() first)",
                )
            f.unacked = True
        elif isinstance(stmt, SemOp) and stmt.kind in ("inc", "fetch_add"):
            if f.sent and stmt.at is not None:
                _fail(
                    "NOC-4",
                    f"{role} signals semaphore {stmt.sem.name} on another core before "
                    "the data it sent there left (k.flushed() first)",
                )
            if stmt.kind == "fetch_add" and f.atomic:
                _fail(
                    "SEM-3",
                    f"{role} fetches from semaphore {stmt.sem.name} with an increment "
                    "in flight; its return lands on the same word (k.atomics_done())",
                )
            f.atomic = stmt.kind == "inc"
        elif isinstance(stmt, Barrier):
            if stmt.kind == "atomics_done":
                f.atomic = False
            if stmt.kind == "reads_done":
                f.reading.clear()
            elif stmt.kind != "atomics_done":
                f.writing.clear()
                f.sending.clear()
                f.sent = False
                f.unacked = f.unacked and stmt.kind != "writes_done"
        elif isinstance(stmt, CBOp):
            name = stmt.cb.name
            if stmt.kind == "push" and name in f.reading:
                _fail("NOC-3", f"{role} pushes buffer {name} before its reads land")
            if stmt.kind == "pop" and name in f.writing:
                _fail("NOC-3", f"{role} pops buffer {name} before its write left")
        elif isinstance(stmt, Inline):
            for use in stmt.uses:
                name = use.cb.name
                if name in f.reading or (use.written and name in f.sending):
                    _fail(
                        "NOC-3",
                        f"inline code of {role} touches buffer {name} with NoC work "
                        "on it in flight",
                    )


def noc_3(program, _):
    for role, kernel in program.kernels.items():
        flight = _Flight()
        _noc_3(role, kernel.body, flight)
        if flight.reading:
            _fail("NOC-3", f"{role} ends with reads not waited for")
        if flight.unacked:
            _fail("NOC-3", f"{role} ends without waiting for its writes")
        if flight.atomic:
            _fail(
                "NOC-3",
                f"{role} ends without waiting for its semaphore increments "
                "(k.atomics_done())",
            )


def noc_4(program, _):
    for role, kernel in program.kernels.items():
        try:
            _noc_3(role, kernel.body, _Flight())
        except VerifyError as e:
            if e.rule == "NOC-4":
                raise


def _atomics(kernel):
    """The semaphore increments and fetches of `kernel`, its inline code's included
    (any semaphore it declares)."""
    ops = [s for s, _ in walk(kernel.body) if isinstance(s, SemOp)]
    ops = [s.kind for s in ops if s.kind in ("inc", "fetch_add")]
    return ops + ["inline" for s in kernel.inlines() if s.sems]


def sem_3(program, _):
    for role, kernel in program.kernels.items():
        try:
            _noc_3(role, kernel.body, _Flight())
        except VerifyError as e:
            if e.rule == "SEM-3":
                raise
    for dm in _dm_pairs(program):
        for mine, other in (dm, dm[::-1]):
            if "fetch_add" in _atomics(mine) and _atomics(other):
                _fail(
                    "SEM-3",
                    f"{mine.name} fetches from a semaphore while {other.name} of the "
                    "core increments one; both returns land on the same word",
                )


def data_1(program, _):
    def refuse(role, value, what):
        if isinstance(value, Expr) and value.raw_data():
            _fail(
                "DATA-1",
                f"{role} uses {value.cpp()}, loaded from L1, as {what}; declare its "
                "range with .bounded(low, high) where it only selects what is read",
            )

    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            for load in _loads(stmt):
                refuse(role, load.word, "the index of a load")
            if isinstance(stmt, Loop):
                for value in (stmt.count, stmt.step, stmt.start):
                    refuse(role, value, "a loop bound")
            elif isinstance(stmt, CBOp):
                refuse(role, stmt.pages, "a page count")
            elif isinstance(stmt, Read):
                dst = stmt.dst
                for value in (stmt.page, stmt.offset, dst.index, dst.offset, dst.like):
                    refuse(role, value, "a NoC page or address")
            elif isinstance(stmt, Write):
                refuse(role, stmt.page, "a NoC page")
                refuse(role, stmt.index, "a buffer page index")
            elif isinstance(stmt, Inline):
                refuse(role, stmt.times, "a run count")
            elif isinstance(stmt, Pack):
                refuse(role, stmt.at, "a buffer page index")
            for _cb, index, _extent in _page_reads(stmt):
                refuse(role, index, "a buffer page index")
            for slot, _extent in _slots(stmt):
                refuse(role, slot, "a register slot")


def loop_1(program, _):
    for role, kernel in program.kernels.items():
        for stmt, _path in walk(kernel.body):
            plain = isinstance(stmt, Loop) and stmt.start is None
            plain = plain and stmt.count is not None
            if not plain or stmt.remainder is not None:
                continue
            step, count = stmt.step, stmt.count
            if isinstance(step, Value) and step.choices:
                whole = isinstance(count, Value) and count.multiple_of == step.name
                steps = step.choices
            else:
                whole, steps = False, (_int(step, "a loop step"),)
            if not whole and any(count.multiple() % s for s in steps):
                _fail(
                    "LOOP-1",
                    f"{role} loops over {count.cpp()} in steps of {_text(step)}; "
                    "declare the count a multiple of the step (p.runtime(name, "
                    "multiple_of=...)) or take the remainder with k.blocks",
                )


def _text(value):
    return value.cpp() if isinstance(value, Expr) else str(value)


def sem_1(program, _):
    sems = program.semaphores
    for i, a in enumerate(sems):
        _int(a.id, f"the id of semaphore {a.name}")
        for b in sems[i + 1 :]:
            if a.id == b.id and a.cores.overlaps(b.cores):
                _fail("SEM-1", f"semaphores {a.name} and {b.name} share id {a.id}")
    for role, kernel in program.kernels.items():
        used = [sem for stmt in kernel.inlines() for sem in stmt.sems]
        used += [s.sem for s, _ in walk(kernel.body) if isinstance(s, SemOp)]
        for stmt, _path in walk(kernel.body):
            if isinstance(stmt, SemOp) and stmt.at is not None:
                first, last = stmt.sem.cores.first, stmt.sem.cores.last
                rows = (
                    [stmt.at.core[1]]
                    if stmt.at.place is None
                    else [y for _x, y in _targets(program, kernel, stmt.at)]
                )
                if first is not None and not all(first <= y <= last for y in rows):
                    _fail(
                        "SEM-1",
                        f"{role} signals semaphore {stmt.sem.name} on core "
                        f"{stmt.at.name}, which does not hold it",
                    )
        for sem in used:
            if sem not in sems or not sem.cores.covers(kernel.cores):
                _fail(
                    "SEM-1",
                    f"{role} uses semaphore {sem.name}, which does not exist on "
                    "every core the kernel runs on",
                )
        if role == "compute" and any(
            isinstance(s, SemOp) for s, _ in walk(kernel.body)
        ):
            _fail(
                "SEM-1", "compute has no NoC: semaphore operations belong to dm0, dm1"
            )


def _static_runs(path):
    """How often a statement under `path` runs, when no condition holds it and the
    loops around it have static counts and steps from 0; else None."""
    if any(True for _ in conditions(path)):
        return None
    runs = 1
    for loop in loops(path):
        count = None if loop.count is None else as_expr(loop.count).value()
        if count is None or loop.start is not None or not isinstance(loop.step, int):
            return None
        runs *= math.ceil(count / loop.step)
    return runs


def _signals(program, sem, role):
    """(statement, path) of what can raise semaphore `sem` for a wait of `role`: an
    increment of another core, or another kernel's increment or non-zero set
    (statement None: inline code declaring it)."""
    out = []
    for other, kernel in program.kernels.items():
        out += [(None, ()) for stmt in kernel.inlines() if sem in stmt.sems]
        for stmt, path in walk(kernel.body):
            if not isinstance(stmt, SemOp) or stmt.sem is not sem:
                continue
            if stmt.kind == "set" and stmt.value.value() == 0:
                continue
            if stmt.kind != "wait_min" and (stmt.at is not None or other != role):
                out.append((stmt, path))
    return out


def sem_2(program, _):
    for role, kernel in program.kernels.items():
        for wait, _path in walk(kernel.body):
            if not isinstance(wait, SemOp) or wait.kind not in ("wait_min", "wait_eq"):
                continue
            if wait.lane is not None:
                continue
            found = _signals(program, wait.sem, role)
            if not found:
                _fail(
                    "SEM-2",
                    f"{role} waits on semaphore {wait.sem.name}, which no other kernel "
                    "and no other core increments or sets",
                )
            total = _int(wait.sem.initial, "a semaphore's initial value")
            for stmt, path in found:
                local = stmt is not None and stmt.kind == "inc" and stmt.at is None
                runs = _static_runs(path) if local else None
                by = None if runs is None else stmt.value.value()
                if by is None:
                    _fail(
                        "SEM-2",
                        f"{role} waits on semaphore {wait.sem.name}, raised where the "
                        "verifier cannot count it (another core, a set, inline code, "
                        "a condition or a runtime count); only counted increments of "
                        "the own core by the other kernel are accepted",
                    )
                total += by * runs
            need = wait.value.value()
            if need is None or total < need:
                _fail(
                    "SEM-2",
                    f"{role} waits for semaphore {wait.sem.name} to reach "
                    f"{_text(wait.value)}; the program raises it to {total}",
                )


def _taken(loop):
    """{buffer name: Take} of the control words endless loop `loop` takes before its
    exit, and the same for its puts ({buffer name: Put})."""
    exit_at = next(i for i, s in enumerate(loop.body) if isinstance(s, ExitWhen))
    before = loop.body[:exit_at]
    takes = {s.cb.name: s for s in before if isinstance(s, Take)}
    puts = {s.cb.name: s for s in before if isinstance(s, Put)}
    return takes, puts


def _ends_on(loop, put):
    """True when endless loop `loop` ends on the very value its `put` wrote: the
    compared value is the put's, and nothing sets it between the put and the exit."""
    condition = _ending(loop)
    if put.value is not condition.a:
        return False
    at = next(i for i, s in enumerate(loop.body) if s is put)
    exit_at = next(i for i, s in enumerate(loop.body) if isinstance(s, ExitWhen))
    between = [s for s, _ in walk(loop.body[at + 1 : exit_at])]
    return not any(isinstance(s, Assign) and s.var is put.value for s in between)


def ctl_1(program, _):
    endless = {}
    for role, kernel in program.kernels.items():
        for stmt, path in walk(kernel.body):
            if isinstance(stmt, ExitWhen) and not (
                path and isinstance(path[-1], Loop) and path[-1].count is None
            ):
                _fail("CTL-1", f"{role} has an exit_when outside an endless loop")
            if isinstance(stmt, Loop) and stmt.count is None:
                if path:
                    _fail(
                        "CTL-1",
                        f"{role} has an endless loop inside a loop or a condition; "
                        "endless loops are accepted at the top of a kernel only",
                    )
                condition = _ending(stmt)
                if not _fixed(condition.b):
                    _fail(
                        "CTL-1",
                        f"{role} ends an endless loop on {condition.cpp()}; its right "
                        "side must keep one value over the launch",
                    )
                endless.setdefault(role, []).append(stmt)
    feeds = {}
    for role, found in endless.items():
        for loop in found:
            condition = _ending(loop)
            takes, puts = _taken(loop)
            for name, put in puts.items():
                if sum(isinstance(s, Put) and s.cb.name == name for s in loop.body) > 1:
                    _fail("CTL-1", f"{role} puts two words into {name} per iteration")
                if name in feeds:
                    _fail("CTL-1", f"two endless loops put words into {name}")
                feeds[name] = (condition, _ends_on(loop, put))
            ended = [t for t in takes.values() if t.var is condition.a]
            ended += [p for p in puts.values() if _ends_on(loop, p)]
            if not ended:
                _fail(
                    "CTL-1",
                    f"{role} ends an endless loop on {condition.cpp()}, not on a word "
                    "it takes, or puts before the exit and keeps unchanged up to it",
                )
    for role, found in endless.items():
        for loop in found:
            condition = _ending(loop)
            takes, _puts = _taken(loop)
            for name, take in takes.items():
                if take.var is not condition.a:
                    continue
                source = feeds.get(name)
                same = source is not None and (
                    source[0].op == condition.op
                    and source[0].b.key() == condition.b.key()
                    and source[1]
                )
                if not same:
                    _fail(
                        "CTL-1",
                        f"{role} ends on words of buffer {name} that no kernel puts "
                        "before ending its own loop on the same comparison",
                    )


def _inline_code(kernel):
    """(code, declared clobbers) of every inline stage and helper of `kernel`."""
    parts = [(s.code, s.clobbers) for s in kernel.inlines()]
    return parts + [(fn.code, fn.clobbers) for fn in kernel.helpers]


def clobbers(kernel):
    """The shared vector-unit constants the inline code of `kernel` declares."""
    return sorted({c for _, declared in _inline_code(kernel) for c in declared})


def sfpu_threads(kernel):
    """The compute threads ("math", "pack") that issue vector-unit work in `kernel`:
    typed vector-unit calls run on the math thread, inline stages where declared."""
    typed = (Unary, Binary, SfpuCall, Apply)
    threads = (
        {"math"} if any(isinstance(s, typed) for s, _ in walk(kernel.body)) else set()
    )
    return threads | {s.sfpu for s in kernel.inlines() if s.sfpu}


def sfpu_1(program, _):
    for role, kernel in program.kernels.items():
        for stmt in kernel.inlines():
            if stmt.sfpu not in (None, "math", "pack") or (
                stmt.sfpu and role != "compute"
            ):
                _fail("SFPU-1", f"{role} declares vector-unit work on {stmt.sfpu!r}")
        for code, declared in _inline_code(kernel):
            if "SFPCONFIG" in code and SHARED_CONSTANT not in declared:
                _fail(
                    "SFPU-1",
                    f"inline code of {role} writes a vector-unit constant; declare "
                    f'clobbers=["{SHARED_CONSTANT}"] so it is restored at exit',
                )
            if declared and role != "compute":
                _fail("SFPU-1", f"{role} declares vector-unit constants; it has none")
            unknown = set(declared) - {SHARED_CONSTANT}
            if unknown:
                _fail("SFPU-1", f"{role} declares unknown constants {sorted(unknown)}")


# Vector-unit instructions whose result takes two cycles: (name, argument naming the
# register written; None: unknown).
SLOW = {
    "SFPMAD": 3,
    "SFPMUL": 3,
    "SFPADD": 3,
    "SFPMUL24": 3,
    "SFPMULI": 1,
    "SFPADDI": 1,
    "SFPLUTFP32": 0,
    "SFPLUT": None,
}
# Instructions the hardware does not hold back when they read such a result.
UNSTALLED = (
    "SFPSWAP",
    "SFPSHFT",
    "SFPIADD",
    "SFPSETEXP",
    "SFPSETMAN",
    "SFPSETSGN",
    "SFPSHFT2",
    "SFPAND",
    "SFPOR",
    "SFPARECIP",
    "SFPLOADI",
    "SFPLOAD",
)
REGISTER_0 = ("0", "L0", "LREG0", "p_sfpu::LREG0")
INSTRUCTION = re.compile(r"\bTTI?_(SFP\w+)")


def _instructions(code):
    """(name, argument texts) of the vector-unit instruction macros in `code`, in
    the order written; comments are skipped."""
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S)
    out = []
    for match in INSTRUCTION.finditer(code):
        args, depth, start = [], 0, match.end()
        rest = code[start:]
        if rest.lstrip().startswith("("):
            begin = start + rest.index("(") + 1
            piece = begin
            for at in range(begin, len(code)):
                char = code[at]
                if char == "(":
                    depth += 1
                elif char == ")" and depth == 0:
                    args.append(code[piece:at])
                    break
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 0:
                    args.append(code[piece:at])
                    piece = at + 1
        out.append((match.group(1), ["".join(a.split()) for a in args]))
    return out


def _unstalled(written, name, args):
    """Whether instruction `name(args)` reads register `written` (None: unknown)
    through a form the hardware does not hold back."""
    if name == "SFPCONFIG":
        return written is None or written in REGISTER_0
    if name not in UNSTALLED:
        return False
    if name == "SFPSWAP" and args and (args[-1] == "0" or args[-1].endswith("_SWAP")):
        return False
    if name in ("SFPLOADI", "SFPLOAD"):
        return written is None or args[:1] == [written]
    return written is None or written in args


def sfpu_2(program, _):
    for role, kernel in program.kernels.items():
        for code, _declared in _inline_code(kernel):
            stream = _instructions(code)
            for (name, args), (after, uses) in itertools.pairwise(stream):
                if name not in SLOW:
                    continue
                at = SLOW[name]
                written = args[at] if at is not None and at < len(args) else None
                if _unstalled(written, after, uses):
                    _fail(
                        "SFPU-2",
                        f"inline code of {role}: {after} reads the result of {name} "
                        "on the next instruction, which the hardware does not hold "
                        "back; put TTI_SFPNOP between them",
                    )


def _slots(stmt):
    """(first register slot, slots from it) of every slot range `stmt` names."""
    if isinstance(stmt, Copy):
        return [(stmt.dst, _most(stmt.count, "a tile count"))]
    if isinstance(stmt, (Unary, Pack, Tiles, Reduce)):
        return [(stmt.dst, 1)]
    if isinstance(stmt, SfpuCall):
        # A generated function's operands past the second are its first arguments.
        more = stmt.args[: max(0, getattr(stmt.fn, "operands", 2) - 2)]
        return [(slot, 1) for slot in (stmt.a, stmt.b, stmt.out, *more)]
    if isinstance(stmt, Binary):
        return [(slot, 1) for slot in (stmt.a, stmt.b, stmt.out)]
    if isinstance(stmt, Apply):
        return [(slot, 1) for slot in stmt.slots]
    if isinstance(stmt, Matmul):
        ct, rt = stmt.init.ct, stmt.init.rt
        joint = getattr(ct, "joint", None)
        if joint is not None and joint == getattr(rt, "joint", None):
            names, table = joint
            at = names.index(ct.name), names.index(rt.name)
            return [(stmt.dst, max(row[at[0]] * row[at[1]] for row in table))]
        block = _most(ct, "a block's column tiles") * _most(rt, "a block's row tiles")
        return [(stmt.dst, block)]
    return []


def dst_1(program, _):
    kernel = program.kernels.get("compute")
    if kernel is None:
        return
    capacity = 4 if kernel.options.get("fp32_dest", True) else 8
    for stmt, _path in walk(kernel.body):
        for slot, extent in _slots(stmt):
            top = slot.upper() if isinstance(slot, Expr) else slot
            if isinstance(top, bool) or not isinstance(top, int):
                _fail(
                    "DST-1", f"compute names register slot {slot!r}, which is unbounded"
                )
            if top < 0 or top + extent > capacity:
                slot = slot.cpp() if isinstance(slot, Expr) else slot
                _fail(
                    "DST-1",
                    f"compute names register slot {slot} (up to {top + extent - 1}); "
                    f"the registers hold {capacity} tiles in this mode",
                )


def reg_1(program, _):
    kernel = program.kernels.get("compute")
    if kernel is None:
        return
    state, scope = None, None
    for stmt, path in walk(kernel.body):
        if isinstance(stmt, Registers):
            wanted = {"acquire": None, "commit": "math", "release": "pack"}[stmt.step]
            if state != wanted:
                _fail("REG-1", f"compute: register {stmt.step} out of order")
            loops = [id(loop) for loop in path]
            if stmt.step != "acquire" and loops != scope:
                _fail(
                    "REG-1", f"compute: register {stmt.step} inside a loop of its scope"
                )
            scope = loops
            state = {"acquire": "math", "commit": "pack", "release": None}[stmt.step]
        elif isinstance(stmt, MATH) and state != "math":
            _fail("REG-1", "compute: a math call outside acquire .. commit")
        elif isinstance(stmt, Pack) and state != "pack":
            _fail("REG-1", "compute: a pack outside commit .. release")
    if state is not None:
        _fail("REG-1", "compute ends inside a register scope")


def _slot_reads_writes(stmt):
    """(slots read, slots written) by compute statement `stmt` as static ints, None
    for a slot that is not one; a hand-written vector function reads nothing known."""
    if isinstance(stmt, Copy):
        count = stmt.count if isinstance(stmt.count, int) else None
        if not isinstance(stmt.dst, int) or count is None:
            return (), (None,)
        return (), tuple(range(stmt.dst, stmt.dst + count))
    if isinstance(stmt, Matmul):
        ((dst, extent),) = _slots(stmt)
        if not isinstance(dst, int):
            return (), (None,)
        return (), tuple(range(dst, dst + extent))
    if isinstance(stmt, Unary):
        return (stmt.dst,), (stmt.dst,)
    if isinstance(stmt, Tiles):
        return ((stmt.dst,) if stmt.init.acc else ()), (stmt.dst,)
    if isinstance(stmt, Reduce):
        return (), (stmt.dst,)
    if isinstance(stmt, Pack):
        return (stmt.dst,), ()
    if isinstance(stmt, Binary):
        return (stmt.a, stmt.b), (stmt.out,)
    if isinstance(stmt, Apply):
        if stmt.name == "fill":
            return (), stmt.slots
        return (stmt.slots if len(stmt.slots) == 1 else stmt.slots[:-1]), stmt.slots[
            -1:
        ]
    if isinstance(stmt, SfpuCall):
        if not isinstance(stmt.fn, GeneratedFn):
            return (), (None,)
        operands = (stmt.a, stmt.b, *stmt.args[: max(0, stmt.fn.operands - 2)])
        return operands[: stmt.fn.operands], (stmt.out,)
    if isinstance(stmt, Inline):
        return (), (None,)
    return (), ()


def reg_2(program, _):
    kernel = program.kernels.get("compute")
    if kernel is None:
        return

    def run(body, written):
        # `written`: static slots written since acquire; None once a write is unknown.
        for stmt in body:
            if isinstance(stmt, Registers):
                if stmt.step == "acquire":
                    written = set()
                continue
            if isinstance(stmt, Loop):
                written = run(stmt.body, written)
                continue
            if isinstance(stmt, When):
                taken = run(stmt.body, None if written is None else set(written))
                other = written if stmt.orelse is None else run(stmt.orelse, written)
                written = None if taken is None or other is None else taken & other
                continue
            reads, writes = _slot_reads_writes(stmt)
            if written is not None:
                for slot in reads:
                    if isinstance(slot, int) and slot not in written:
                        _fail(
                            "REG-2",
                            f"compute: {type(stmt).__name__.lower()} reads register "
                            f"slot {slot} before anything in its scope wrote it",
                        )
            if None in writes:
                written = None
            elif written is not None:
                written |= set(writes)
        return written

    run(kernel.body, set())


class _Units:
    """Compute unit state: unpacker (`unpack`, `srca`, `srcb`), vector unit (`sfpu`),
    packer format (`pack`), accumulation (`acc`), reduction masks (`reduce`: output
    buffer name); per buffer: pages packed since its reserve (`written`: {page key:
    `k.var` names read}), pages still reserved (`held`), next in-order page (`next`)."""

    def __init__(self):
        self.unpack = self.sfpu = self.pack = self.srca = self.srcb = None
        self.reduce, self.tags = None, {}
        self.acc, self.written, self.held, self.next = False, {}, {}, {}
        # The page count of each buffer's last reserve while none of it is pushed.
        self.reserved = {}
        # Buffers pushed with a page the packer did not write.
        self.partial = set()

    def copy(self):
        out = _Units()
        out.__dict__.update(self.__dict__)
        out.written = {name: dict(pages) for name, pages in self.written.items()}
        out.held, out.next = dict(self.held), dict(self.next)
        out.reserved, out.partial = dict(self.reserved), set(self.partial)
        out.tags = dict(self.tags)
        return out

    def join(self, other):
        """The setup known to hold when either this or `other` happened."""
        for name in ("unpack", "sfpu", "pack", "acc", "srca", "srcb", "reduce"):
            if getattr(self, name) != getattr(other, name):
                setattr(self, name, "?")
        self.tags = {k: v for k, v in self.tags.items() if other.tags.get(k) == v}
        mine, theirs = self.written, other.written
        self.written = {
            n: {page: v for page, v in mine[n].items() if page in theirs[n]}
            for n in mine.keys() & theirs.keys()
        }
        mine, theirs = self.held, other.held
        self.held = {n: min(mine[n], theirs[n]) for n in mine.keys() & theirs.keys()}
        mine, theirs = self.next, other.next
        self.next = {
            n: mine[n] for n in mine.keys() & theirs.keys() if mine[n] == theirs[n]
        }
        self.partial |= other.partial
        mine, theirs = self.reserved, other.reserved
        self.reserved = {
            n: mine[n]
            for n in mine.keys() & theirs.keys()
            if mine[n].key() == theirs[n].key()
        }


# Formats the unpacker moves straight into the registers, through its first source.
UNPACK_TO_DEST = ("float32", "int32", "uint32")


def _source(s):
    """The unpacker source ("srca" or "srcb") copy statement `s` reads through: a
    broadcast of a 16-bit format goes through the second."""
    broadcast = s.kind.startswith("bcast")
    return "srcb" if broadcast and s.cb.dtype not in UNPACK_TO_DEST else "srca"


def _formats(s, u):
    """UNPACK-1 for an unpack of `s` (a copy or a matmul) under unit state `u`."""
    if isinstance(s, Matmul):
        reads = [("srca", s.init.b), ("srcb", s.init.a)]
    elif isinstance(s, Tiles):
        reads = [("srca", s.init.a), ("srcb", s.init.b)]
    elif isinstance(s, Reduce):
        # A row sum takes the scaler through the first source (a matrix product).
        i = s.init
        swap = i.dim == "row" and i.kind == "sum"
        reads = (
            [("srca", i.scaler), ("srcb", i.cb)]
            if swap
            else [
                ("srca", i.cb),
                ("srcb", i.scaler),
            ]
        )
    else:
        reads = [(_source(s), s.cb)]
    for source, cb in reads:
        if getattr(u, source) != cb.dtype:
            _fail(
                "UNPACK-1",
                f"compute unpacks buffer {cb.name} ({cb.dtype}) with the unpacker's "
                f"{source} set to {getattr(u, source)}; set it with c.startup or "
                "c.format first",
            )


def _reduction(s, u):
    """RED-1 for statement `s` under unit state `u` (reduction masks), updating it."""
    setup = isinstance(s, (Startup, MatmulInit, TilesInit)) or (
        isinstance(s, Init) and s.name in UNPACK_SETUP
    )
    if setup and u.reduce is not None:
        _fail(
            "RED-1",
            "compute changes the unpacker's setup with a reduction's packer masks in "
            "effect; call c.reduce_done() first",
        )
    if isinstance(s, ReduceInit):
        u.reduce = (s.out.name, s.dim)
    elif isinstance(s, ReduceDone):
        u.reduce = None
    elif isinstance(s, Registers) and s.step == "acquire":
        u.tags = {}
    elif isinstance(s, Pack) and u.reduce is not None:
        out, dim = u.reduce if u.reduce != "?" else ("?", "?")
        if out != s.cb.name or u.tags.get(s.dst) != dim:
            _fail(
                "RED-1",
                f"compute packs slot {s.dst!r} into buffer {s.cb.name} under the "
                f"packer masks of a reduction along {dim} into {out}: only results "
                "of reductions along that dim pack under them; call c.reduce_done() "
                "or pack in a register scope of one reduction dim",
            )
    elif isinstance(s, MATH):
        writes = _slot_reads_writes(s)[1]
        if None in writes:
            u.tags = {}
        for slot in writes:
            tag = s.init.dim if isinstance(s, Reduce) else "other"
            u.tags[slot] = tag


def _units(stmts, u):
    """Checks compute statements `stmts` in order, updating unit state `u`."""
    for s in stmts:
        _reduction(s, u)
        if isinstance(s, Loop):
            # Twice: the second pass meets the setup the first left. In-order packs
            # into a buffer reserved outside it land on pages not known here.
            before = u.copy()
            for _ in range(2):
                for p, _path in walk(s.body):
                    if isinstance(p, Pack) and p.at is None:
                        u.next[p.cb.name] = None
                _units(s.body, u)
            # A pack at the loop's variable into a reserve made before the loop has,
            # by the loop's end, packed each page below the count (none if it is 0).
            again = {p.cb.name for p, _ in walk(s.body) if isinstance(p, CBOp)}
            every = {
                name: pages[s.var.key()]
                for name, pages in u.written.items()
                if s.var.key() in pages and name not in again
            }
            if _may_skip(s):
                u.join(before)
            plain = s.count is not None and s.start is None and s.step == 1
            spread = _spread(s.var)
            for name, read in every.items():
                if plain and name in u.written:
                    u.written[name][f"<{s.count.key()}"] = read
                if spread and name in u.written:
                    u.written[name].update(dict.fromkeys(spread, read))
        elif isinstance(s, When):
            other = u.copy()
            _units(s.body, u)
            _units(s.orelse or [], other)
            u.join(other)
        elif isinstance(s, Startup):
            u.pack = s.out.dtype
            in1 = s.in0 if s.in1 is None else s.in1
            first, second = (in1, s.in0) if s.reverse else (s.in0, in1)
            u.srca, u.srcb = first.dtype, second.dtype
        elif isinstance(s, Format):
            u.srca = u.srca if s.srca is None else s.srca.dtype
            u.srcb = u.srcb if s.srcb is None else s.srcb.dtype
        elif isinstance(s, SrcaFormat):
            u.srca = s.cb.dtype
        elif isinstance(s, PackFormat):
            u.pack = s.cb.dtype
        elif isinstance(s, Init) and s.name in UNPACK_SETUP:
            u.unpack = (s.name, s.cb.dtype)
        elif isinstance(s, Init):
            u.sfpu = OPS[s.name].init
        elif isinstance(s, MatmulInit):
            u.unpack = ("matmul", id(s))
        elif isinstance(s, Matmul) and u.unpack != ("matmul", id(s.init)):
            _fail(
                "MM-1",
                "compute runs a matmul whose setup is not the unpacker's last one; "
                "call c.matmul_init again after another way of unpacking",
            )
        elif isinstance(s, (TilesInit, ReduceInit)):
            u.unpack = (type(s).__name__, id(s))
        elif isinstance(s, (Tiles, Reduce)) and u.unpack != (
            type(s.init).__name__,
            id(s.init),
        ):
            call = "c.tiles_init" if isinstance(s, Tiles) else "c.reduce_init"
            _fail(
                "INIT-1",
                f"compute runs a {type(s).__name__.lower()} statement whose setup is "
                f"not the unpacker's last one; call {call} again after another setup",
            )
        elif isinstance(s, Copy) and u.unpack != (UNPACK_SETUP[s.kind], s.cb.dtype):
            _fail(
                "INIT-1",
                f"compute unpacks buffer {s.cb.name} ({s.kind}) without "
                f'c.init("{UNPACK_SETUP[s.kind]}", ...) of a {s.cb.dtype} buffer as '
                "the unpacker's last setup",
            )
        elif isinstance(s, (Unary, Binary, Apply)) and u.sfpu != OPS[s.name].init:
            _fail(
                "INIT-1",
                f'compute runs tile op {s.name} without c.init("{s.name}") as the '
                "vector unit's last setup",
            )
        elif (
            isinstance(s, SfpuCall)
            and getattr(s.fn, "init", None)
            and u.sfpu != OPS[s.fn.init].init
        ):
            _fail(
                "INIT-1",
                f'compute runs function {s.fn.name} without c.init("{s.fn.init}") as '
                "the vector unit's last setup",
            )
        elif isinstance(s, PackAccumulate):
            _accumulate(s, u)
        elif isinstance(s, CBOp) and s.kind == "reserve":
            u.written[s.cb.name], u.next[s.cb.name] = {}, 0
            u.reserved[s.cb.name] = as_expr(s.pages)
            u.held[s.cb.name] = min(_counts(s.pages, "a page count"))
        elif isinstance(s, CBOp) and s.kind == "push":
            pages, written = as_expr(s.pages), u.written.get(s.cb.name, {})
            whole = pages.value() is not None and all(
                str(i) in written for i in range(pages.value())
            )
            if not whole and f"<{pages.key()}" not in written:
                u.partial.add(s.cb.name)
            _pushed(u, s.cb.name, pages.value())
        elif isinstance(s, Assign):
            for pages in u.written.values():
                for page in [p for p, read in pages.items() if s.var.name in read]:
                    del pages[page]
        elif isinstance(s, Pack):
            _packs(s, u)
        if isinstance(s, (Copy, Matmul, Tiles, Reduce)):
            _formats(s, u)


def _spread(index):
    """The page keys a loop variable `index` takes over its loop (from 0 in steps of
    1 over a static count), else None."""
    count = getattr(index, "count", None)
    whole = isinstance(index, Var) and index.step == 1 and count is not None
    if not whole or count.value() is None:
        return None
    return [str(page) for page in range(count.value())]


def _pushed(u, name, pages):
    """Unit state `u` after a push of `pages` (None: a value) of buffer `name`: the
    write pointer moves on, page i becomes page i - pages, in-order packs restart."""
    u.next[name] = 0
    held, written = u.held.pop(name, None), u.written.pop(name, {})
    u.reserved.pop(name, None)
    if pages is None or held is None:
        return
    if held > pages:
        u.held[name] = held - pages
    u.written[name] = {
        str(int(page) - pages): read
        for page, read in written.items()
        if page.isdigit() and int(page) >= pages
    }


def _accumulate(s, u):
    on = s.on.value() if isinstance(s.on, Expr) else int(s.on)
    if on is None:
        _fail(
            "ACC-1",
            "packer accumulation is switched by a runtime value; the pages it adds "
            "to cannot be proved written",
        )
    if on and s.cb is None:
        _fail("ACC-1", "c.pack_accumulate(1, cb) names the buffer it adds into")
    u.acc = s.cb.name if on else False


def _packs(s, u):
    name = s.cb.name
    if u.pack != s.cb.dtype:
        _fail(
            "PACK-1",
            f"compute packs into buffer {name} ({s.cb.dtype}) with the packer set to "
            f"{u.pack}; call c.pack_format(cb) first",
        )
    held, read = u.held.get(name, 0), frozenset()
    if s.at is not None:
        top = _top(s.at)
        reserved = u.reserved.get(name)
        inside = reserved is not None and _within(s.at, reserved)
        if not inside and (top is None or top >= held) or _below_zero(s.at):
            _fail("CB-3", f"compute packs outside its reserve on buffer {name}")
        page = s.at.key() if s.at.value() is None else str(s.at.value())
        nodes = _nodes(s.at, set())
        read = frozenset(n.name for n in nodes if isinstance(n, Mutable))
    else:
        # The next page of the reserve; not known after in-order packs in a loop.
        page = u.next.get(name)
        if page is not None and page >= held:
            _fail(
                "CB-3",
                f"compute packs {page + 1} pages in order into {held} reserved on "
                f"buffer {name}",
            )
        if page is not None:
            u.next[name] = page + 1
            page = str(page)
    written = u.written.setdefault(name, {})
    spread = _spread(s.at) if s.at is not None else None
    packed = page in written or (spread and all(p in written for p in spread))
    # A partials buffer adds onto its slots of the pass before (matmul_passes).
    if u.acc and (u.acc != name or page is None or not packed) and not s.cb.partials:
        _fail(
            "ACC-1",
            f"compute packs into page {page} of buffer {name} with packer accumulation "
            f"on for {u.acc}; it adds only to pages packed earlier in the same reserve",
        )
    if not u.acc and page is not None:
        written[page] = read


def _compute_flow(program, rules):
    """Runs the unit-state checks; only failures of `rules` are raised."""
    kernel = program.kernels.get("compute")
    if kernel is None:
        return
    state = _Units()
    try:
        _units(kernel.body, state)
        if state.acc:
            _fail("ACC-1", "compute ends with packer accumulation on")
        if state.reduce is not None:
            _fail("RED-1", "compute ends with a reduction's packer masks in effect")
    except VerifyError as e:
        if e.rule in rules:
            raise


def init_1(program, _):
    _compute_flow(program, ("INIT-1",))


def mm_1(program, _):
    _compute_flow(program, ("MM-1",))


def pack_1(program, _):
    _compute_flow(program, ("PACK-1", "CB-3"))


def acc_1(program, _):
    _compute_flow(program, ("ACC-1",))


def unpack_1(program, _):
    _compute_flow(program, ("UNPACK-1",))


def red_1(program, _):
    _compute_flow(program, ("RED-1",))


_BUFFER_CALL = re.compile(
    r"\b(?:cb_\w+|get_(?:write|read)_ptr|get_tile_size)\(\s*(\w+)"
)
_SEMAPHORE_CALL = re.compile(r"\bget_semaphore\(\s*(\w+)")


def inl_1(program, _):
    for role, kernel in program.kernels.items():
        for stmt in kernel.inlines():
            names = {f"cb_{u.cb.name}" for u in stmt.uses}
            names |= {str(u.cb.index) for u in stmt.uses}
            for found in _BUFFER_CALL.findall(stmt.code):
                if found not in names:
                    _fail(
                        "INL-1",
                        f"inline code of {role} uses buffer {found}, which it does "
                        "not declare",
                    )
            sems = {f"sem_{s.name}" for s in stmt.sems} | {str(s.id) for s in stmt.sems}
            for found in _SEMAPHORE_CALL.findall(stmt.code):
                if found not in sems:
                    _fail(
                        "INL-1",
                        f"inline code of {role} uses semaphore {found}, which it does "
                        "not declare",
                    )


ZONE_NAME = re.compile(r"[A-Za-z0-9_ .:/-]{1,48}")
WAYPOINT_NAME = re.compile(r"[A-Za-z0-9]{1,4}")


def _declared(stmt):
    """Ids of the variables and reserved pages statement `stmt` declares."""
    if isinstance(stmt, Assign) and not stmt.declares:
        return set()
    if isinstance(stmt, CBOp):
        return set() if stmt.slot is None else {id(stmt.slot)}
    var = getattr(stmt, "var", None)
    found = {id(var)} if isinstance(stmt, (Let, Assign, Take, SemOp)) else set()
    return found - {id(None)}


def _referenced(stmt):
    """Ids of the values, variables and reserved pages statement `stmt` uses."""
    seen = set()
    found = {id(n) for e in exprs(stmt) for n in _nodes(e, seen)}
    if isinstance(stmt, Assign):
        found.add(id(stmt.var))
    for part in (getattr(stmt, "dst", None), getattr(stmt, "src", None)):
        if isinstance(part, SlotAt):
            found.add(id(part.slot))
    if isinstance(stmt, Write) and stmt.slot is not None:
        found.add(id(stmt.slot))
    if isinstance(stmt, Inline):
        found |= {id(v) for v in stmt.args.values() if isinstance(v, Slot)}
    return found


def _zones(role, stmts):
    """DBG-1 over statement list `stmts` of kernel `role` and the lists inside it."""
    opened = []
    for i, s in enumerate(stmts):
        if isinstance(s, Loop):
            _zones(role, s.body)
        elif isinstance(s, When):
            _zones(role, s.body)
            _zones(role, s.orelse or [])
        elif isinstance(s, Waypoint) and not WAYPOINT_NAME.fullmatch(s.name):
            _fail(
                "DBG-1", f"{role}: waypoint {s.name!r} is not 1 to 4 letters or digits"
            )
        elif isinstance(s, Zone) and not ZONE_NAME.fullmatch(s.name):
            _fail("DBG-1", f"{role}: zone name {s.name!r} is not plain text")
        elif isinstance(s, Zone) and s.step == "begin":
            opened.append(i)
        elif isinstance(s, Zone):
            first = opened.pop()
            inside = {d for t, _ in walk(stmts[first + 1 : i]) for d in _declared(t)}
            for t, _ in walk(stmts[i + 1 :]):
                if _referenced(t) & inside:
                    _fail(
                        "DBG-1",
                        f"{role}: a value declared inside zone {s.name!r} is used "
                        "after its end",
                    )


def dbg_1(program, _):
    for role, kernel in program.kernels.items():
        _zones(role, kernel.body)


def _launch_value(expr):
    """True when `expr` is computed from runtime and sized values only."""
    return all(
        isinstance(node, Value)
        or node.value() is not None
        or (node.parts() and not isinstance(node, (Load, Mutable)))
        for node in _nodes(expr, set())
    )


def size_1(program, _):
    for name, value in program.sized_values.items():
        if not _launch_value(value.of):
            _fail(
                "SIZE-1",
                f"sized value {name} is computed from {value.of.cpp()}; the host "
                "computes it from runtime values before the launch",
            )
    kernel = program.kernels.get("compute")
    for stmt, _path in walk(kernel.body if kernel else []):
        for slot, _extent in _slots(stmt):
            nodes = _nodes(slot, set()) if isinstance(slot, Expr) else ()
            if any(isinstance(node, Sized) for node in nodes):
                _fail(
                    "SIZE-1",
                    f"compute names register slot {slot.cpp()}, a sized value; a "
                    "sized value sizes buffers and counts pages",
                )


def lay_1(program, _):
    cores = program.layout_of
    if cores is not None and not _launch_value(cores):
        _fail(
            "LAY-1",
            f"the layout's core count is {cores.cpp()}; the host picks the size "
            "class from it, so it reads runtime and sized values only",
        )
    count = None if cores is None else cores.value()
    whole = program.layout_of is None and program.split_of is None
    if whole:
        count = _launched(program)
    for ref in program.core_refs:
        if ref.place is not None:
            if not whole:
                _fail(
                    "LAY-1",
                    f"core {ref.name} is relative to each core; it needs every "
                    "launched core to run (no p.layout or p.split)",
                )
            for kernel in program.kernels.values():
                _targets(program, kernel, ref)
            continue
        index = ref.core[1] * program.machine.cols + ref.core[0]
        if index > 0 and (count is None or index >= count):
            _fail(
                "LAY-1",
                f"core {ref.name} is core {index} of the program; only core 0, or a "
                "core below a static layout's count, is sure to run",
            )


def kern_1(program, _):
    rows = program.machine.rows
    for role in ("dm0", "dm1", "compute"):
        found = [k for k in program.kernels.values() if k.role == role]
        if len(found) < 2:
            continue
        expect = 0
        for first, last, name in sorted(
            (k.cores.first, k.cores.last, k.name) for k in found
        ):
            if first != expect or last < first:
                _fail(
                    "KERN-1",
                    f"variants of {role} overlap or leave rows out at row {expect} "
                    f"({name})",
                )
            expect = last + 1
        if expect != rows:
            _fail(
                "KERN-1", f"variants of {role} cover rows 0 .. {expect - 1} of {rows}"
            )


def _everywhere(expr):
    """True when `expr` has one value on every core of a launch: no core place,
    per-core argument, loaded word or assigned variable."""
    for node in _nodes(as_expr(expr), set()):
        per_core = getattr(node, "binding", None) == PER_CORE
        if per_core or isinstance(node, (CoreIndex, Load, Mutable)):
            return False
    return True


def _lane_parts(program, lane):
    """{kernel name: [(statement, path)]} of the statements of `lane`."""
    out = {}
    for name, kernel in program.kernels.items():
        for stmt, path in walk(kernel.body):
            if getattr(stmt, "lane", None) is lane:
                out.setdefault(name, []).append((stmt, path))
    return out


def _lane_kernels(program, lane):
    """(sending kernel, receiving kernels, parts) of `lane`; MC-1 unless one kernel
    sends and the others only receive."""
    parts = _lane_parts(program, lane)
    sends = [n for n, ps in parts.items() if any(isinstance(x, Mcast) for x, _ in ps)]
    if len(sends) != 1 or len(parts) < 2:
        _fail(
            "MC-1",
            f"lane {lane.name} has one sending kernel and at least one receiving one",
        )
    send = program.kernels[sends[0]]
    if any(isinstance(x, SemOp) and x.kind == "inc" for x, _ in parts[send.name]):
        _fail("MC-1", f"{send.name} both sends and receives on lane {lane.name}")
    receivers = [program.kernels[n] for n in parts if n != send.name]
    return send, receivers, parts


def mc_1(program, _):
    if program.lanes and (
        program.layout_of is not None or program.split_of is not None
    ):
        _fail(
            "MC-1", "a lane needs every launched core to run (no p.layout or p.split)"
        )
    owner = {id(sem): lane for lane in program.lanes for sem in (lane.ready, lane.flag)}
    for name, kernel in program.kernels.items():
        if any(id(sem) in owner for stmt in kernel.inlines() for sem in stmt.sems):
            _fail("MC-1", f"inline code of {name} touches a lane's semaphore")
        for stmt, _path in walk(kernel.body):
            if (
                isinstance(stmt, SemOp)
                and owner.get(id(stmt.sem), stmt.lane) is not stmt.lane
            ):
                _fail(
                    "MC-1",
                    f"{name} operates on semaphore {stmt.sem.name} outside its lane",
                )
    launched = set(program.machine.core_list(_launched(program)))
    for lane in program.lanes:
        if not _lane_parts(program, lane):
            continue
        send, receivers, parts = _lane_kernels(program, lane)
        top, (low, high) = lane.senders.first, (
            lane.receivers.first,
            lane.receivers.last,
        )
        if not 0 <= low <= high < program.machine.rows or low <= top <= high:
            _fail(
                "MC-1",
                f"lane {lane.name}: its rectangle (rows {low} .. {high}) is outside "
                f"the grid or holds its sender's row {top}",
            )
        if (send.cores.first, send.cores.last) != (top, top):
            _fail("MC-1", f"lane {lane.name} sends from {send.name}, not its row {top}")
        rows = set()
        for r in receivers:
            if not lane.receivers.covers(r.cores):
                _fail("MC-1", f"{r.name} receives on lane {lane.name} outside its rows")
            rows |= set(range(r.cores.first, r.cores.last + 1))
        if rows != set(range(low, high + 1)):
            _fail("MC-1", f"lane {lane.name}: a receiver row runs no receiving kernel")
        for x, y in launched:
            if y == top and not any((x, r) in launched for r in range(low, high + 1)):
                _fail("MC-1", f"lane {lane.name}: column {x} has no receiver")
            if low <= y <= high and (x, top) not in launched:
                _fail("MC-1", f"lane {lane.name}: core {(x, y)} has no sender")
        kinds = {
            n: [x.kind if isinstance(x, SemOp) else type(x).__name__ for x, _ in ps]
            for n, ps in parts.items()
        }
        opened = kinds[send.name][:1] == ["set"] and kinds[send.name].count("set") == (
            1 + kinds[send.name].count("Mcast")
        )
        first, path = parts[send.name][0]
        top_level = not path and any(x is first for x in send.body)
        if not (opened and top_level):
            _fail("MC-1", f"lane {lane.name} is opened once, before its first send")
        for _x, path in (pair for ps in parts.values() for pair in ps):
            if any(isinstance(q, Loop) and q.count is None for q in path):
                _fail("MC-1", f"lane {lane.name} works inside an endless loop")
            around = [c for c, _ in conditions(path)]
            for loop in loops(path):
                around += [loop.count, loop.step, loop.start]
            if not all(_everywhere(e) for e in around if isinstance(e, Expr)):
                _fail(
                    "MC-1",
                    f"lane {lane.name} runs under a count or condition that differs "
                    "between cores",
                )


def _siblings(kernel, path):
    """The statement list holding a statement under `path` of `kernel`."""
    if not path:
        return kernel.body
    last = path[-1]
    return last.when.orelse if isinstance(last, Else) else last.body


def _around(kernel, stmt, path, name):
    """(reserve, push) on buffer `name` around `stmt` in its statement list, with
    nothing else on that buffer between them; NOC-5 when there are none."""
    body = _siblings(kernel, path)
    at = next(i for i, x in enumerate(body) if x is stmt)
    ops = [
        (i, x) for i, x in enumerate(body) if isinstance(x, CBOp) and x.cb.name == name
    ]
    before = [(i, x) for i, x in ops if i < at]
    after = [(i, x) for i, x in ops if i > at]
    if (
        not before
        or before[-1][1].kind != "reserve"
        or not after
        or (after[0][1].kind != "push")
    ):
        _fail(
            "NOC-5",
            f"{kernel.name}: a lane's data moves outside a reserve of buffer {name}",
        )
    return before[-1], after[0]


def _key(value):
    return None if value is None else as_expr(value).key()


def _trace(body, name, lanes):
    """The calls on buffer `name` and the sends and receives of `lanes` in `body`,
    in order, nested in the loops and conditions around them."""
    out = []
    for x in body:
        if isinstance(x, Loop):
            inner = _trace(x.body, name, lanes)
            if inner:
                out.append(("loop", _key(x.count), _key(x.start), _key(x.step), inner))
        elif isinstance(x, When):
            a, b = _trace(x.body, name, lanes), _trace(x.orelse or [], name, lanes)
            if a or b:
                out.append(("when", x.condition.key(), a, b))
        elif isinstance(x, CBOp) and x.cb.name == name:
            out.append((x.kind, _key(x.pages)))
        elif getattr(x, "lane", None) in lanes and (
            isinstance(x, Mcast) or isinstance(x, SemOp) and x.kind == "inc"
        ):
            out.append(("lane", x.lane.name))
    return tuple(out)


def _one_each(trace, who):
    """NOC-5 unless every reserve .. push span of `trace` holds at most one send or
    receive per lane and none inside a loop or condition."""
    for entry in trace:
        if entry[0] in ("loop", "when"):
            for part in entry[2:]:
                if isinstance(part, tuple):
                    _one_each(part, who)
    span = None
    for entry in trace:
        if entry[0] == "reserve":
            span = []
        elif entry[0] == "push":
            span = None
        elif span is not None and entry[0] == "lane":
            if entry in span:
                _fail("NOC-5", f"{who} sends or receives twice on one reserve")
            span.append(entry)
        elif (
            span is not None
            and entry[0] in ("loop", "when")
            and "'lane'" in repr(entry)
        ):
            _fail(
                "NOC-5", f"{who} sends or receives in a loop or condition of a reserve"
            )


def noc_5(program, _):
    for lane in program.lanes:
        if not _lane_parts(program, lane):
            continue
        send, receivers, parts = _lane_kernels(program, lane)
        casts = [(x, path) for x, path in parts[send.name] if isinstance(x, Mcast)]
        names = {x.src.slot.cb.name for x, _ in casts}
        if len(names) != 1:
            _fail("NOC-5", f"lane {lane.name} carries one buffer, not {sorted(names)}")
        name = names.pop()
        carried = {
            x.lane
            for k in program.kernels.values()
            for x, _ in walk(k.body)
            if isinstance(x, Mcast) and x.src.slot.cb.name == name
        }
        sent = _trace(send.body, name, carried)
        _one_each(sent, send.name)
        for x, path in casts:
            (j, reserve), (m, _) = _around(send, x, path, name)
            if reserve.slot is not x.src.slot:
                _fail("NOC-5", f"lane {lane.name} sends another reserve than its own")
            body = _siblings(send, path)
            after = body[body.index(x) + 1 : m]
            if any(
                isinstance(y, Read) and y.dst.slot.cb.name == name
                for y, _ in walk(after)
            ):
                _fail("NOC-5", f"{send.name} fills the reserve after sending it")
        for r in receivers:
            incs = [(x, path) for x, path in parts[r.name] if x.kind == "inc"]
            for x, path in incs:
                (j, _), (m, _) = _around(r, x, path, name)
                inside = _siblings(r, path)[j + 1 : m]
                for y, _ in walk(inside):
                    touches = (
                        isinstance(y, Read)
                        and y.dst.slot.cb.name == name
                        or isinstance(y, (Write, Send, Mcast, Inline))
                        and name
                        in {
                            getattr(getattr(y, "cb", None), "name", None),
                            getattr(
                                getattr(getattr(y, "src", None), "slot", None),
                                "cb",
                                None,
                            )
                            and y.src.slot.cb.name,
                            *(u.cb.name for u in getattr(y, "uses", ())),
                        }
                    )
                    if touches or _loads(y):
                        _fail(
                            "NOC-5",
                            f"{r.name} touches the pages lane {lane.name} fills",
                        )
            got = _trace(r.body, name, carried)
            _one_each(got, r.name)
            if got != sent:
                _fail(
                    "NOC-5",
                    f"{r.name} does not reserve and push buffer {name} in step with "
                    f"{send.name}, once per send of lane {lane.name}",
                )


def _launched(program):
    """The number of cores a program without a layout or a split runs on."""
    return program.machine.classes(program.size_classes)[-1]


def _uses(kernel, ref):
    """True when the body of `kernel` names core reference `ref`."""
    ids = {id(ref.x), id(ref.y)}
    return any(
        id(leaf) in ids
        for stmt, _ in walk(kernel.body)
        for e in exprs(stmt)
        for leaf in e.leaves()
    )


def _targets(program, kernel, ref):
    """The logical cores relative reference `ref` names from the launched cores of
    `kernel` that use it; LAY-1 when one is not a launched core."""
    if not _uses(kernel, ref):
        return []
    launched = program.machine.core_list(_launched(program))
    out = []
    for x, y in kernel.cores.select(launched):
        target = tuple(ref.place(x, y))
        if target not in launched:
            _fail(
                "LAY-1",
                f"core {ref.name} of core {(x, y)} is {target}, not a core the program "
                "runs on",
            )
        out.append(target)
    return out


def l1_1(program, l1_budget):
    if l1_budget is not None and program.footprint() > l1_budget:
        _fail(
            "L1-1",
            f"buffers and scratch take {program.footprint()} bytes per core, the "
            f"budget is {l1_budget}",
        )


@dataclass(frozen=True)
class Rule:
    """A verifier rule: what it checks, the hazard it comes from, and its check
    (None: a reserved slot, not checked yet)."""

    id: str
    checks: str
    origin: str
    check: object = None


RULES = (
    Rule(
        "BIND-1",
        "no runtime value in a compile-time slot (pages, defines, steps, Python "
        "control flow)",
        "a new shape never compiles",
        bind_1,
    ),
    Rule(
        "BIND-2",
        "a per-core argument is a function of the core index",
        "the launch patches addresses and common values only",
        bind_2,
    ),
    Rule(
        "SIZE-1",
        "a sized value is computed from runtime values and is never a register slot",
        "a buffer size is fixed when the program is built: the host must know it "
        "before the launch",
        size_1,
    ),
    Rule(
        "LAY-1",
        "a layout's core count is computed from runtime and sized values; a core "
        "named as a NoC target runs at every launch",
        "the host picks the core count's size class before the launch",
        lay_1,
    ),
    Rule(
        "DATA-1",
        "a value loaded from L1 is no count, loop bound or register slot, and selects "
        "a NoC page or address only through a declared range; the range is the "
        "launch's contract, not checked on the device",
        "an index tensor's values are known on the device only; an index outside "
        "its table reads another tensor's memory",
        data_1,
    ),
    Rule(
        "ARG-1",
        "runtime argument counts are fixed per kernel class and within the limit",
        "programs differing in per-core argument length share a cache entry",
        arg_1,
    ),
    Rule(
        "ARG-2",
        "a table is read at a static row below its size or at the variable of a loop "
        "over its used rows; a column used as a count declares its most; no page count "
        "comes from a table",
        "a row past the table reads another argument; an unbounded count overruns a "
        "buffer",
        arg_2,
    ),
    Rule(
        "KERN-1",
        "variants of a role run on row sets that are disjoint and together cover the "
        "grid; compute has none",
        "a core with no kernel of a role, or two, never balances its buffers",
        kern_1,
    ),
    Rule(
        "MC-1",
        "a lane has one sending kernel on its sender row and receiving kernels on "
        "every receiver row; its rectangle is in the grid, without the sender, and "
        "every core of it runs; its semaphores are touched only by its statements; "
        "it is opened once before it sends and works under counts equal on all cores",
        "a multicast to a core that does not take part, or a receiver count that is "
        "wrong, hangs the sender or overwrites a running kernel's L1",
        mc_1,
    ),
    Rule(
        "NOC-5",
        "a lane moves data only inside a reserve of its buffer on both sides; the "
        "receivers reserve and push it in step with the sender (same calls, same "
        "counts) and do not touch the pages it fills",
        "a write landing in pages the receiver did not reserve overwrites live data",
        noc_5,
    ),
    Rule(
        "CB-4",
        "one filling and one draining kernel per buffer; reads follow a wait, packs "
        "a reserve",
        "only the unpacker waits and pops, only the packer reserves and pushes",
        cb_4,
    ),
    Rule(
        "CB-1",
        "every recurring reserve / push / wait / pop count divides the buffer",
        "a non-dividing push wraps differently on the two sides: wrong data",
        cb_1,
    ),
    Rule(
        "CB-3",
        "a kernel writes no further than the pages it reserved, and reads only pages "
        "it waited for",
        "a buffer smaller than what its kernel wrote overran L1 and hung the card",
        cb_3,
    ),
    Rule(
        "CTL-1",
        "an endless loop has one exit, on a control word it takes or puts before it; "
        "a loop ending on taken words is fed by a kernel that puts one word per "
        "iteration before ending its own loop on the same comparison",
        "a consumer waiting for a word its producer never puts hangs the core",
        ctl_1,
    ),
    Rule(
        "CB-2",
        "pushes equal pops on every buffer over a launch",
        "unpaired pushes and pops stall the consumer or the producer",
        cb_2,
    ),
    Rule(
        "LOOP-1",
        "a loop in steps of n runs over a count that is a multiple of n",
        "the last step of a stepped loop reads and pushes whole steps past the count",
        loop_1,
    ),
    Rule(
        "CB-5",
        "a page the packer wrote is not copied back without a pushed token",
        "nothing orders the packer and the unpacker on a reserved slot",
        cb_5,
    ),
    Rule(
        "NOC-1",
        "the reader and the writer of a core use different NoCs",
        "two processors on one NoC share its command buffers and counters: a hang",
        noc_1,
    ),
    Rule(
        "NOC-2",
        "a DRAM read keeps source and destination congruent modulo 64 bytes (L1: 16)",
        "misaligned sub-page reads return another row's data",
        noc_2,
    ),
    Rule(
        "NOC-3",
        "a push, a load or a write follows the reads into its pages, a pop or a new "
        "read the flush of its write, and a writing kernel ends with its writes "
        "acknowledged",
        "a kernel ending with pending NoC transactions races the next one",
        noc_3,
    ),
    Rule(
        "NOC-4",
        "a semaphore increment or fetch on another core follows the flush of the "
        "data this kernel sent before it",
        "a flag that overtakes its data lets the receiver read stale L1",
        noc_4,
    ),
    Rule(
        "SEM-3",
        "a fetch from a semaphore has no other increment of its core in flight: none "
        "of the same kernel since its last atomics_done, none of the other "
        "data-movement kernel",
        "every increment returns the old value to one L1 word of the core",
        sem_3,
    ),
    Rule(
        "SEM-1",
        "a semaphore exists on every core that uses it or is signalled; ids are "
        "distinct per core; only data-movement kernels operate on semaphores",
        "a missing semaphore reads another program's word",
        sem_1,
    ),
    Rule(
        "SEM-2",
        "every semaphore a kernel waits on is raised by another kernel or another "
        "core, and, where every raise is counted, to at least the value waited for",
        "a wait on a semaphore nothing signals never returns",
        sem_2,
    ),
    Rule(
        "SFPU-1",
        "a kernel that writes the shared vector-unit constant declares it (and it is "
        "restored at exit)",
        "a changed -1.0 constant made later kernels on the core return infinities",
        sfpu_1,
    ),
    Rule(
        "SFPU-2",
        "inline vector-unit code: a multiply-add result is not read by the next "
        "instruction through a form the stall logic misses",
        "an unspaced multiply-add fed a stale value into a clamp",
        sfpu_2,
    ),
    Rule(
        "DST-1",
        "register slots stay within the capacity of the mode (4 fp32 or 8 tiles)",
        "the destination register holds 8 tiles at 16 bits, 4 at fp32",
        dst_1,
    ),
    Rule(
        "REG-1",
        "math calls sit between acquire and commit, packs between commit and release",
        "the math and pack threads hand the registers over at commit",
        reg_1,
    ),
    Rule(
        "REG-2",
        "a register slot an op, a generated vector function or a pack reads was "
        "written earlier in its register scope (a loop is taken to run at least once)",
        "a statistics slot never initialised accumulates onto what the registers last "
        "held",
        reg_2,
    ),
    Rule(
        "INIT-1",
        "a way of unpacking or a tile op runs with its own setup the last one made "
        "for its unit (the unpacker, the vector unit)",
        "an op under another op's setup computes with that op's constants and "
        "address steps",
        init_1,
    ),
    Rule(
        "MM-1",
        "a matmul runs under its own setup, with no other unpacker setup since",
        "the matrix unit's block shape is part of the setup, not of the call",
        mm_1,
    ),
    Rule(
        "PACK-1",
        "a pack goes into a buffer of the format the packer is set to, at a page "
        "inside its reserve",
        "the packer converts to the format it was last set to, whatever the buffer",
        pack_1,
    ),
    Rule(
        "ACC-1",
        "packer accumulation adds only to pages of its buffer packed earlier in the "
        "same reserve, and is off at kernel exit",
        "accumulation onto a fresh page adds to what the page last held; left on, "
        "it corrupts the next kernel's packs",
        acc_1,
    ),
    Rule(
        "UNPACK-1",
        "an unpack reads a buffer of the format its source register is set to (a "
        "matmul: b through the first source, a through the second; a whole-tile op: "
        "a, b; a row sum: scaler, buffer; another reduction: buffer, scaler)",
        "the setup of a way of unpacking keeps the source formats; a buffer of "
        "another format is read as the format set",
        unpack_1,
    ),
    Rule(
        "RED-1",
        "while a reduction's setup is in effect, a pack goes into its output buffer "
        "and packs a slot last written by a reduction along the same dim, and no "
        "other way of unpacking is set up; c.reduce_done() ends it before the kernel "
        "does",
        "the reduction's packer masks write zeros over everything but the reduced "
        "row or column of every tile packed, until they are cleared",
        red_1,
    ),
    Rule(
        "INL-1",
        "inline code declares every buffer and semaphore its text names",
        "the verifier sees inline code only through its declaration",
        inl_1,
    ),
    Rule(
        "L1-1",
        "buffers and scratch fit the L1 budget given",
        "a program's buffers must not clash with L1 tensors",
        l1_1,
    ),
    Rule(
        "DBG-1",
        "a profiler zone's name is plain text and a value, variable or reserved page "
        "declared inside a zone is not used after its end; a watcher waypoint names "
        "1 to 4 letters or digits",
        "a zone is a C++ block, so a name declared in it ends with it; the watcher "
        "keeps a waypoint as one 32-bit word",
        dbg_1,
    ),
)


def verify(program, l1_budget=None):
    """Raises VerifyError for the first rule of RULES `program` breaks."""
    for rule in RULES:
        if rule.check is None:
            continue
        rule.check(program, l1_budget)
