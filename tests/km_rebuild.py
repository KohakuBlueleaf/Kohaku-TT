"""A generator for tests: `rebuild(program)` reads a metal-level program's model and
constructs an equal program through the explicit builder only (`km.Program`,
`p.input`, `p.kernel`, `with k.loop(...)`): no decorator, no traced function."""

import operator

import kohakutt.dsl.metal as km
from kohakutt.dsl.metal import binding as kb
from kohakutt.dsl.metal import body as ir

OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "%": operator.mod,
    "/": operator.floordiv,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    ">>": operator.rshift,
    "<<": operator.lshift,
    "&": operator.and_,
    "!=": lambda a, b: a.ne(b),
    "==": lambda a, b: a.eq(b),
}


class _Rebuild:
    """The state of one rebuild: the new program and old -> new objects."""

    def __init__(self, old):
        self.old = old
        self.new = km.Program(old.name, old.machine, old.size_classes)
        self.new.statics = dict(old.statics)
        self.memo, self.split = {}, {}
        self.refs = {
            id(t): (self.new.output if t.is_output else self.new.input)(
                t.name, t.dtype, t.kind
            )
            for t in old.tensors
        }
        self._values()
        self.per_core = {}
        for name, values, count in old.per_core_args:
            made = self.new.per_core(name, values, count)
            for arg in made if isinstance(made, list) else [made]:
                self.per_core[name, arg.position] = arg
        self.cbs = {
            id(cb): self.new.cb(cb.index, dtype=cb.dtype, pages=cb.pages, name=cb.name)
            for cb in old.cbs
        }
        for s in old.scratches:
            made = self.new.scratch(s.bytes, s.align, s.name)
            assert made.index == s.index
            self.cbs[id(s)] = made
        self.sems = {
            id(s): self.new.semaphore(s.initial, s.cores, s.name, s.id)
            for s in old.semaphores
        }

    def _values(self):
        old, new = self.old, self.new
        total = old.split_of[0] if old.split_of else None
        for name in old.values:
            if total is not None and name == f"{total}_base":
                least = old.split_of[1]
                least = new.runtime(least) if isinstance(least, str) else least
                start, count = new.split(self._value(total), least)
                self.split = {"start": start, "count": count}
            elif name not in new.values:
                self._value(name)
        if old.strided:
            new.stride(self._value(total))
        if old.layout_of is not None:
            new.layout(self.value(old.layout_of))

    def _value(self, name):
        sized = self.old.sized_values.get(name)
        if sized is None:
            return self.new.runtime(
                name, self.old.choices.get(name), self.old.multiples.get(name)
            )
        if name in self.new.sized_values:
            return self.new.sized_values[name]
        of = self.value(sized.of)
        if sized.exact:
            return self.new.sized(name, of, exact=True, limit=sized.choices[-1])
        return self.new.sized(name, of, classes=sized.choices)

    def value(self, e):
        """Old value `e` (an int or a kernel value) in the new program."""
        if not isinstance(e, kb.Expr):
            return e
        if id(e) in self.memo:
            return self.memo[id(e)]
        if e.value() is not None:
            return e.value()
        if isinstance(e, kb.Value):
            return self._value(e.name)
        if isinstance(e, kb.CoreIndex):
            return self.new.core
        if isinstance(e, kb.CoreCount):
            return self.new.core_count_value
        if isinstance(e, kb.Var) and e.name in self.split:
            return self.split[e.name]
        if isinstance(e, kb.Var):
            made = self.memo[id(e)] = self.new.define(e.name, self.value(e.definition))
            return made
        if isinstance(e, kb.Load):
            return self.memo[id(e.slot)].load32(self.value(e.word))
        if isinstance(e, kb.Bounded):
            low, high = self.value(e.low), self.value(e.high)
            return self.value(e.inner).bounded(low, high)
        if isinstance(e, kb.Min):
            return km.min(self.value(e.a), self.value(e.b))
        if isinstance(e, kb.Select):
            return km.select(self.value(e.when), self.value(e.a), self.value(e.b))
        if isinstance(e, kb.Op):
            return OPERATORS[e.op](kb.as_expr(self.value(e.a)), self.value(e.b))
        if isinstance(e, kb.PerCore):
            return self.per_core[e.name, e.position]
        raise TypeError(f"rebuild: no rule for {e!r}")

    def address(self, at):
        slot = self.memo[id(at.slot)]
        if at.index.value() == 0:
            like = None if at.like is None else self.value(at.like)
            return slot.at(self.value(at.offset), like)
        assert at.offset.value() == 0 and at.like is None
        return slot[self.value(at.index)]

    def kernel(self, role, old):
        options = dict(old.options)
        if role == "compute":
            options["unpack_fp32"] = [
                self.cbs[id(cb)] for cb in old.options["unpack_fp32"]
            ]
        k = self.new.kernel(role, cores=old.cores, noc=old.noc, **options)
        for name, value in old.defines.items():
            k.define(name, value)
        for header in old.includes:
            k.include(header)
        for fn in old.helpers:
            k.helper(fn)
        self.body(k, old.body)

    def body(self, k, stmts):
        for s in stmts:
            self.statement(k, s)

    def statement(self, k, s):
        v, cb = self.value, lambda old: self.cbs[id(old)]
        if isinstance(s, ir.Loop) and s.remainder is not None:
            names = (s.var.name, s.remainder.name)
            with k.block_loop(v(s.count), v(s.step), names) as (done, n):
                self.memo[id(s.var)], self.memo[id(s.remainder)] = done, n
                self.body(k, s.body[1:])
        elif isinstance(s, ir.Loop):
            start = None if s.start is None else v(s.start)
            with k.loop(v(s.count), v(s.step), s.var.name, start) as var:
                self.memo[id(s.var)] = var
                self.body(k, s.body)
        elif isinstance(s, ir.When):
            with k.when(v(s.condition)):
                self.body(k, s.body)
            if s.orelse is not None:
                with k.otherwise():
                    self.body(k, s.orelse)
        elif isinstance(s, ir.Let):
            self.memo[id(s.var)] = k.let(s.var.name, v(s.expr))
        elif isinstance(s, ir.Assign) and s.declares:
            self.memo[id(s.var)] = k.var(s.var.name, v(s.expr))
        elif isinstance(s, ir.Assign):
            k.set(self.memo[id(s.var)], v(s.expr))
        elif isinstance(s, ir.CBOp):
            made = getattr(k, s.kind)(cb(s.cb), v(s.pages))
            if s.slot is not None:
                self.memo[id(s.slot)] = made
        elif isinstance(s, ir.Read):
            ref = self.refs[id(s.ref)]
            if s.size is None:
                source = ref.page(v(s.page))
            else:
                source = ref.span(v(s.page), v(s.offset), s.size)
            k.read(source, self.address(s.dst))
        elif isinstance(s, ir.Write):
            target = self.refs[id(s.ref)].page(v(s.page))
            if s.slot is None:
                k.write(cb(s.cb), target, index=v(s.index))
            else:
                k.write(self.memo[id(s.slot)], target)
        elif isinstance(s, ir.Barrier):
            getattr(k, s.kind)()
        elif isinstance(s, ir.Startup):
            in1 = None if s.in1 is None else cb(s.in1)
            k.startup(cb(s.in0), in1, cb(s.out), s.reverse)
        elif isinstance(s, ir.Init):
            k.init(s.name, None if s.cb is None else cb(s.cb))
        elif isinstance(s, ir.SrcaFormat):
            k.srca_format(cb(s.cb))
        elif isinstance(s, ir.Format):
            k.format(*(None if b is None else cb(b) for b in (s.srca, s.srcb)))
        elif isinstance(s, ir.PackFormat):
            k.pack_format(cb(s.cb))
        elif isinstance(s, ir.PackAccumulate):
            k.pack_accumulate(v(s.on), None if s.cb is None else cb(s.cb))
        elif isinstance(s, ir.Registers):
            getattr(k, s.step)()
        elif isinstance(s, ir.Copy) and s.kind == "copy":
            k.copy(cb(s.cb), v(s.index), v(s.dst))
        elif isinstance(s, ir.Copy) and s.kind == "copy_block":
            k.copy_block(cb(s.cb), v(s.index), v(s.dst), v(s.count))
        elif isinstance(s, ir.Copy) and s.kind == "transpose":
            k.transpose(cb(s.cb), v(s.index), v(s.dst))
        elif isinstance(s, ir.Copy):
            k.bcast(cb(s.cb), v(s.index), v(s.dst), s.kind.removeprefix("bcast_"))
        elif isinstance(s, ir.MatmulInit):
            shape = (v(s.transpose), v(s.ct), v(s.rt), v(s.kt))
            self.memo[id(s)] = k.matmul_init(cb(s.a), cb(s.b), *shape)
        elif isinstance(s, ir.Matmul):
            k.matmul(self.memo[id(s.init)], v(s.ia), v(s.ib), v(s.dst))
        elif isinstance(s, ir.Unary):
            getattr(k, s.name)(v(s.dst))
        elif isinstance(s, ir.Binary):
            k.binary(s.name, v(s.a), v(s.b), v(s.out))
        elif isinstance(s, ir.Apply):
            k.op(s.name, *(v(x) for x in s.slots), values=[v(x) for x in s.values])
        elif isinstance(s, ir.SfpuCall):
            args = [v(a) for a in s.args]
            k.sfpu(s.fn, v(s.a), v(s.b), v(s.out), s.mode, args)
        elif isinstance(s, ir.Pack):
            k.pack(v(s.dst), cb(s.cb), None if s.at is None else v(s.at))
        elif isinstance(s, ir.Inline):
            uses = [
                km.CBUse(cb(u.cb), u.reserve, u.push, u.wait, u.pop, u.written)
                for u in s.uses
            ]
            args = {
                name: self.memo[id(a)] if isinstance(a, ir.Slot) else v(a)
                for name, a in s.args.items()
            }
            sems = [self.sems[id(sem)] for sem in s.sems]
            k.inline(s.code, uses, sems, s.clobbers, v(s.times), s.sfpu, args)
        else:
            raise TypeError(f"rebuild: no rule for {s!r}")


def rebuild(program):
    """A new Program equal to `program`, constructed through the explicit builder
    from its model; verified before it is returned."""
    state = _Rebuild(program)
    for role, kernel in program.kernels.items():
        state.kernel(role, kernel)
    km.verify(state.new)
    return state.new
