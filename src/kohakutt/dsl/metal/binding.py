"""Values of a metal-level program and when each is bound: static (part of the kernel
class), size class (the core count), runtime (a common argument), per core, derived
on the device, or data (loaded from L1 on the device).
"""

import math

STATIC, RUNTIME, PER_CORE, DERIVED = "static", "runtime", "per_core", "derived"
SIZE_CLASS, DATA = "size class", "data"


class VerifyError(Exception):
    """A program breaks verifier rule `rule`."""

    def __init__(self, rule, message):
        super().__init__(f"{rule}: {message}")
        self.rule = rule


class Static:
    """Annotation of a program parameter fixed per kernel class: ``Static[int]``."""

    def __class_getitem__(cls, item):
        return cls


class Auto:
    """Candidate values of a static parameter; the first that verifies is used."""

    def __init__(self, *candidates):
        if not candidates:
            raise ValueError("kohakutt.dsl.metal: auto() needs a candidate")
        self.candidates = candidates


def auto(*candidates):
    """A static parameter default tried in the order of `candidates`."""
    return Auto(*candidates)


class Expr:
    """An unsigned 32-bit value of a kernel; operators build device arithmetic."""

    binding = STATIC

    def cpp(self):
        """The C++ text of the value."""
        raise NotImplementedError

    def key(self):
        """A text equal for equal values across the kernels of one program."""
        return self.cpp()

    def value(self):
        """The Python int of a static value, else None."""
        return None

    def upper(self):
        """The largest value it can take when known statically, else None."""
        return self.value()

    def leaves(self):
        """The argument values and named variables it reads."""
        return []

    def parts(self):
        """The values it is computed from."""
        return []

    def multiple(self):
        """An integer every value it takes is a multiple of (0: it is always 0)."""
        value = self.value()
        return 1 if value is None else value

    def raw_data(self):
        """True when it depends on a value loaded from L1 with no declared bound."""
        return any(part.raw_data() for part in self.parts())

    def bounded(self, low, high):
        """This value, declared to lie in low .. high - 1: the launch's contract,
        not checked on the device."""
        return Bounded(self, low, high)

    def ne(self, other):
        """The comparison `self != other`."""
        return Op("!=", self, other)

    def eq(self, other):
        """The comparison `self == other`."""
        return Op("==", self, other)

    def _static(self, what):
        if self.value() is None:
            raise VerifyError(
                "BIND-1",
                f"{self.cpp()} is a {self.binding} value used as {what}; only "
                "static values decide the program's shape",
            )
        return self.value()

    def __bool__(self):
        return bool(self._static("a Python condition"))

    def __index__(self):
        return self._static("a Python integer")

    __int__ = __index__

    def __add__(self, other):
        return Op("+", self, other)

    def __radd__(self, other):
        return Op("+", other, self)

    def __sub__(self, other):
        return Op("-", self, other)

    def __rsub__(self, other):
        return Op("-", other, self)

    def __mul__(self, other):
        return Op("*", self, other)

    def __rmul__(self, other):
        return Op("*", other, self)

    def __mod__(self, other):
        return Op("%", self, other)

    def __floordiv__(self, other):
        return Op("/", self, other)

    def __lt__(self, other):
        return Op("<", self, other)

    def __rshift__(self, other):
        return Op(">>", self, other)

    def __lshift__(self, other):
        return Op("<<", self, other)

    def __and__(self, other):
        return Op("&", self, other)

    def __rand__(self, other):
        return Op("&", other, self)


class Const(Expr):
    """A static integer."""

    def __init__(self, number):
        self.number = int(number)

    def cpp(self):
        return str(self.number)

    def value(self):
        return self.number


class Value(Expr):
    """Runtime value `name`: one common argument for every core. With `choices`
    (the only values it takes) it may be a page count, a loop step or a register
    slot; `multiple_of` (an int or the name of a runtime value) is what every launch
    value is a multiple of."""

    binding = RUNTIME

    def __init__(self, name, choices=None, multiple_of=None):
        self.name, self.multiple_of = name, multiple_of
        self.choices = tuple(sorted(choices)) if choices else None
        if self.choices and not all(isinstance(c, int) and c >= 1 for c in choices):
            raise ValueError(f"kohakutt.dsl.metal: choices of {name} must be ints >= 1")

    def cpp(self):
        return self.name

    def upper(self):
        return self.choices[-1] if self.choices else None

    def multiple(self):
        return self.multiple_of if isinstance(self.multiple_of, int) else 1

    def leaves(self):
        return [self]


class CoreIndex(Expr):
    """The index of the running core within the program's cores."""

    binding = PER_CORE

    def cpp(self):
        return "core"

    def multiple(self):
        return 1

    def leaves(self):
        return [self]


class CoreCount(Expr):
    """The number of cores the program runs on: fixed per size class."""

    binding = SIZE_CLASS

    def cpp(self):
        return "cores"

    def multiple(self):
        return 1

    def leaves(self):
        return [self]


class Var(Expr):
    """Named device variable. `definition`: computed at kernel start; `source`: the
    value a `let` gave it; `step`: a loop variable counting from 0 in such steps."""

    binding = DERIVED

    def __init__(self, name, definition=None, upper=None):
        self.name, self.definition, self._upper = name, definition, upper
        self.block_of, self.count = None, None
        self.source, self.step = None, None

    def cpp(self):
        return self.name

    def key(self):
        return self.definition.key() if self.definition is not None else self.name

    def upper(self):
        return self._upper

    def parts(self):
        given = self.source if self.source is not None else self.definition
        return [] if given is None else [given]

    def multiple(self):
        if isinstance(self.step, int):
            return self.step
        parts = self.parts()
        return parts[0].multiple() if parts else 1

    def leaves(self):
        return [self]


class Mutable(Var):
    """A 32-bit device variable assigned in the kernel body (`k.var`)."""

    def __init__(self, name, initial):
        super().__init__(name)
        self.assigned, self._visiting = [as_expr(initial)], False

    def parts(self):
        return list(self.assigned)

    def _over_assigned(self, fn, inside):
        """fn(assigned values); `inside` when an assignment reads the variable."""
        if self._visiting:
            return inside
        self._visiting = True
        try:
            return fn(self.assigned)
        finally:
            self._visiting = False

    def multiple(self):
        return self._over_assigned(
            lambda values: math.gcd(*(e.multiple() for e in values)), 0
        )

    def raw_data(self):
        return self._over_assigned(
            lambda values: any(e.raw_data() for e in values), False
        )


class Load(Expr):
    """32-bit word `word` of the pages reserved as `slot`, read from L1: data."""

    binding = DATA

    def __init__(self, slot, word):
        self.slot, self.word = slot, as_expr(word)

    def cpp(self):
        pointer = f"reinterpret_cast<volatile uint32_t*>({self.slot.name})"
        return f"{pointer}[{self.word.cpp()}]"

    def key(self):
        return f"load({self.slot.cb.name},{self.word.key()})"

    def multiple(self):
        return 1

    def parts(self):
        return [self.word]

    def raw_data(self):
        return True

    def leaves(self):
        return [*self.word.leaves(), self]


class Bounded(Expr):
    """`inner`, declared to lie in low .. high - 1; emitted as `inner`."""

    binding = DERIVED

    def __init__(self, inner, low, high):
        self.inner, self.low, self.high = inner, as_expr(low), as_expr(high)

    def cpp(self):
        return self.inner.cpp()

    def key(self):
        return self.inner.key()

    def upper(self):
        top = self.high.upper()
        return None if top is None else top - 1

    def multiple(self):
        return self.inner.multiple()

    def parts(self):
        return [self.inner]

    def raw_data(self):
        return False

    def leaves(self):
        return self.inner.leaves() + self.low.leaves() + self.high.leaves()


_FOLD = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "%": lambda a, b: a % b,
    "/": lambda a, b: a // b,
    "<": lambda a, b: int(a < b),
    "!=": lambda a, b: int(a != b),
    "==": lambda a, b: int(a == b),
    ">>": lambda a, b: a >> b,
    "<<": lambda a, b: a << b,
    "&": lambda a, b: a & b,
}
COMPARISONS = ("<", "!=", "==")


def _power_of_two(n):
    """The largest power of two dividing `n` (n itself for 0)."""
    return n & -n


def _upper(op, a, b):
    """The upper bound of `a op b` from the operands' bounds, or None."""
    ua, ub = a.upper(), b.upper()
    if op == "%":
        return None if ub is None else ub - 1
    if op in ("-", "/"):
        return ua
    if op in COMPARISONS:
        return 1
    if op == "&":
        return _upper_and(a, ua, b.value())
    if op == ">>":
        return None if ua is None or b.value() is None else ua >> b.value()
    if ua is None or ub is None:
        return None
    return _FOLD[op](ua, ub)


def _upper_and(a, ua, mask):
    """The upper bound of `a & mask`: for a mask of low bits and `a` a multiple of a
    power of two, the largest such multiple inside the mask."""
    if mask is None:
        return ua
    top = mask
    if mask & (mask + 1) == 0:
        step = _power_of_two(a.multiple())
        top = 0 if step == 0 or step > mask else mask + 1 - step
    return top if ua is None else min(top, ua)


def _multiple(op, a, b):
    """An integer every value of `a op b` is a multiple of."""
    ma, mb, vb = a.multiple(), b.multiple(), b.value()
    if op == "*":
        return ma * mb
    if op in ("+", "-"):
        return math.gcd(ma, mb)
    if op == "<<" and vb is not None:
        return ma << vb
    if op == "&":
        return _power_of_two(ma) if ma else 0
    if op == "%" and vb is not None:
        return math.gcd(ma, vb)
    return 1


class Op(Expr):
    """`a op b`."""

    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, as_expr(a), as_expr(b)
        static = self.a.value() is not None and self.b.value() is not None
        self.binding = STATIC if static else DERIVED

    def _side(self, e, right):
        nested = isinstance(e, (Op, Select))
        chained = (
            isinstance(e, Op) and e.op == self.op and self.op in "+*" and not right
        )
        return f"({e.cpp()})" if nested and not chained else e.cpp()

    def cpp(self):
        if self.value() is not None:
            return str(self.value())
        return f"{self._side(self.a, False)} {self.op} {self._side(self.b, True)}"

    def key(self):
        return f"({self.a.key()}{self.op}{self.b.key()})"

    def value(self):
        a, b = self.a.value(), self.b.value()
        return None if a is None or b is None else _FOLD[self.op](a, b)

    def upper(self):
        value = self.value()
        return _upper(self.op, self.a, self.b) if value is None else value

    def multiple(self):
        value = self.value()
        return _multiple(self.op, self.a, self.b) if value is None else value

    def parts(self):
        return [self.a, self.b]

    def leaves(self):
        return self.a.leaves() + self.b.leaves()


class Select(Expr):
    """`when ? a : b`."""

    binding = DERIVED

    def __init__(self, when, a, b):
        self.when, self.a, self.b = as_expr(when), as_expr(a), as_expr(b)

    def cpp(self):
        return f"{self.when.cpp()} ? {self.a.cpp()} : {self.b.cpp()}"

    def key(self):
        return f"({self.when.key()}?{self.a.key()}:{self.b.key()})"

    def upper(self):
        ua, ub = self.a.upper(), self.b.upper()
        return None if ua is None or ub is None else max(ua, ub)

    def multiple(self):
        return math.gcd(self.a.multiple(), self.b.multiple())

    def parts(self):
        return [self.when, self.a, self.b]

    def leaves(self):
        return self.when.leaves() + self.a.leaves() + self.b.leaves()


class Min(Select):
    """The smaller of `a` and `b`; bounded when either side is."""

    def __init__(self, a, b):
        super().__init__(Op("<", a, b), a, b)

    def upper(self):
        bounds = [u for u in (self.a.upper(), self.b.upper()) if u is not None]
        return min(bounds) if bounds else None


def minimum(a, b):
    """The smaller of `a` and `b`."""
    a, b = as_expr(a), as_expr(b)
    if a.value() is not None and b.value() is not None:
        return Const(min(a.value(), b.value()))
    return Min(a, b)


def as_expr(value):
    """`value` (an int or an Expr) as an Expr."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"kohakutt.dsl.metal: not a kernel value: {value!r}")
    return Const(value)
