"""Vector-unit functions from expressions: `sfpu_fn(name, expr)` writes one inline
function computing `expr` over up to four register operands, run once per face by
`c.run`. Every `+`, `-` and `*` rounds on its own; `mad` is the fused multiply-add.
"""

import functools
import struct
from dataclasses import dataclass

from kohakutt.dsl.metal.body import SfpuFn

LLK = "ckernel_sfpu_"
ONE = "sfpi::vFloat(1.0f)"
# Most register operands of one function: two broadcasts may share a slot.
OPERANDS = 6
# Most output tiles of one call (the fp32 registers hold four tiles).
TILES = 4
ARGS = ("a", "b", "out", "c", "d", "e", "f")
# Most fp32 scalar arguments of one function.
SCALARS = 4
CONVERTER = "ckernel::sfpu::Converter::as_float"


@dataclass(frozen=True)
class Function:
    """A library function of one vector: its C++ call on `{x}`, its header, the
    table op whose setup (`c.init`) it runs under (None: any), and whether it is
    for fp32 registers only."""

    call: str
    header: str
    init: str | None = None
    fp32: bool = False


S = "ckernel::sfpu::"
FUNCTIONS = {
    "sigmoid": Function(S + "_sfpu_sigmoid_<DST_ACCUM_MODE>({x})", "sigmoid.h", "silu"),
    "recip": Function(S + "sfpu_reciprocal<false>({x})", "recip.h", "silu"),
    "exp": Function(S + "_sfpu_exp_accurate_<DST_ACCUM_MODE>({x})", "exp.h"),
    "tanh": Function(S + "_sfpu_tanh_fp32_accurate_({x})", "tanh.h", "tanh", True),
    "sqrt": Function(S + "_calculate_sqrt_body_<APPROX, false>({x})", "sqrt.h", "sqrt"),
    "rsqrt": Function(
        S + "_calculate_sqrt_body_<APPROX, true>({x})", "sqrt.h", "rsqrt"
    ),
}
# ktt_flash's exp: p(m) ~ 2^(m - 1) on [1, 2), and the argument's floor (2^-127).
FLASH_B2, FLASH_B1, FLASH_B0 = 0.33718854, -0.01674134, 0.68127801
FLASH_FLOOR = 512.0
COMPARES = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}


def _wrap(value):
    if isinstance(value, V):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return V("const", (), float(value))
    raise TypeError(f"kohakutt.dsl.metal: {value!r} is not a vector value")


class V:
    """A value of the vector unit (32 lanes of fp32): an operand, a constant, or an
    operation on others. `a < b` and the like give a condition for `select`."""

    __hash__ = object.__hash__
    along = None

    def __init__(self, kind, args=(), value=None):
        self.kind, self.args, self.value = kind, tuple(args), value

    def __add__(self, other):
        return V("add", (self, _wrap(other)))

    def __radd__(self, other):
        return V("add", (_wrap(other), self))

    def __sub__(self, other):
        return V("sub", (self, _wrap(other)))

    def __rsub__(self, other):
        return V("sub", (_wrap(other), self))

    def __mul__(self, other):
        return V("mul", (self, _wrap(other)))

    def __rmul__(self, other):
        return V("mul", (_wrap(other), self))

    def __neg__(self):
        return V("neg", (self,))

    def __abs__(self):
        return V("abs", (self,))

    def __bool__(self):
        raise TypeError(
            "kohakutt.dsl.metal: a vector value has no Python truth value; write "
            "f.maximum / f.minimum, not max / min, and branch with f.select"
        )

    def __lt__(self, other):
        return Cond("lt", self, _wrap(other))

    def __le__(self, other):
        return Cond("le", self, _wrap(other))

    def __gt__(self, other):
        return Cond("gt", self, _wrap(other))

    def __ge__(self, other):
        return Cond("ge", self, _wrap(other))

    def __eq__(self, other):
        return Cond("eq", self, _wrap(other))

    def __ne__(self, other):
        return Cond("ne", self, _wrap(other))

    def _part(self, part):
        if self.kind != "operand" or self.value[1] is not None or self.along:
            raise ValueError("kohakutt.dsl.metal: .even / .odd are of a whole operand")
        return V("operand", (), (self.value[0], part))

    @property
    def even(self):
        """The even columns of this operand (a function of column pairs)."""
        return self._part(0)

    @property
    def odd(self):
        """The odd columns of this operand (a function of column pairs)."""
        return self._part(1)


@dataclass(frozen=True, eq=False)
class Cond:
    """A lanewise comparison of two vector values."""

    kind: str
    a: V
    b: V

    def __bool__(self):
        raise TypeError(
            "kohakutt.dsl.metal: a comparison of vector values has no Python truth "
            "value: write `lo < a < hi` as nested f.select of `lo < a` and `a < hi`, "
            "and max / min as f.maximum / f.minimum"
        )


def operand(index, along=None, half=0, band=0, step=0):
    """Register operand `index` (0 .. 5) of a function: the slot `c.run` gives it.
    `along="col"`: one value per row, read from the faces of column half `half`;
    `"row"`: one per column, read from row band `band + step * t` (4 rows) for tile t.
    """
    if not 0 <= index < OPERANDS:
        raise ValueError(
            f"kohakutt.dsl.metal: a function has operands 0 .. {OPERANDS - 1}"
        )
    out = V("operand", (), (index, None))
    if along == "col" and half in (0, 1):
        out.along = ("col", half)
    elif along == "row" and 0 <= band < 4 and step >= 0:
        out.along = ("row", band, step)
    elif along is not None:
        raise ValueError(
            'kohakutt.dsl.metal: an operand is along "col" (half 0 or 1) or "row" '
            "(bands 0 .. 3)"
        )
    return out


def operands(count):
    """Register operands 0 .. count - 1."""
    return tuple(operand(i) for i in range(count))


def scalar(index):
    """Scalar argument `index` (0 .. 3) of a function: an fp32 value `c.run` passes
    as its bits (`values=`), the same in every lane; it takes no register slot."""
    if not 0 <= index < SCALARS:
        raise ValueError(
            f"kohakutt.dsl.metal: a function has scalars 0 .. {SCALARS - 1}"
        )
    return V("scalar", (), index)


def bf16(x):
    """x rounded to bf16 (to nearest, ties to even), kept as an fp32 value."""
    return V("bf16", (_wrap(x),))


def mad(a, b, c):
    """a * b + c in one instruction with one rounding (the product keeps more than
    fp32 precision), unlike `a * b + c`, which rounds the product first."""
    return V("mad", (_wrap(a), _wrap(b), _wrap(c)))


def minimum(a, b):
    """The lanewise smaller of two values."""
    return V("min", (_wrap(a), _wrap(b)))


def maximum(a, b):
    """The lanewise larger of two values."""
    return V("max", (_wrap(a), _wrap(b)))


def select(cond, then, other):
    """`then` in the lanes where `cond` holds, `other` elsewhere."""
    if not isinstance(cond, Cond):
        raise TypeError(
            "kohakutt.dsl.metal: select takes a comparison of vector values"
        )
    return V("select", (_wrap(then), _wrap(other)), cond)


def call(name, x):
    """Library function `name` (a key of FUNCTIONS) of `x`."""
    if name not in FUNCTIONS:
        raise ValueError(f"kohakutt.dsl.metal: no vector function {name!r}")
    return V("call", (_wrap(x),), name)


def sigmoid(x):
    """1 / (1 + exp(-x)), as the sigmoid tile op computes it."""
    return call("sigmoid", x)


def silu(x):
    """x * sigmoid(x), as the silu tile op computes it."""
    x = _wrap(x)
    return x * call("sigmoid", x)


def recip(x):
    """1 / x: the hardware's estimate refined by two Newton steps."""
    return call("recip", x)


def exp(x):
    """e^x, as the exp tile op computes it without approximation."""
    return call("exp", x)


def flash_exp2(u):
    """2^(u - 639) as ktt_flash's softmax exp computes it: the fraction of u through
    its degree-2 polynomial (max rel. error 1.75e-3), u below 512 gives 0; u < 768."""
    return V("flash_exp2", (_wrap(u),))


def flash_exp(x, a, b):
    """2^(x a + b - 639) with x a + b one multiply-add, as ktt_flash's exp of a score
    x (a: scale log2 e, b: 639 - a m for row max m)."""
    return flash_exp2(mad(x, a, b))


def tanh(x):
    """tanh(x), as the tanh tile op computes it on fp32 registers."""
    return call("tanh", x)


def sqrt(x):
    """The square root of x, as the sqrt tile op computes it."""
    return call("sqrt", x)


def rsqrt(x):
    """1 / sqrt(x), as the rsqrt tile op computes it."""
    return call("rsqrt", x)


def relu(x):
    """max(x, 0)."""
    return maximum(x, 0.0)


def square(x):
    """x * x."""
    x = _wrap(x)
    return x * x


def _literal(value):
    """The C++ fp32 literal with exactly the bits of `value` rounded to fp32."""
    exact = struct.unpack("<f", struct.pack("<f", value))[0]
    if exact != exact or exact in (float("inf"), float("-inf")):
        raise ValueError("kohakutt.dsl.metal: a vector constant is a finite number")
    return f"{exact.hex()}f"


def _product(node):
    """Whether `node` is a product the compiler would fuse into a following add."""
    return node.kind == "mul" or (node.kind == "neg" and _product(node.args[0]))


class _Lowering:
    """The statements computing a set of values, each node once."""

    def __init__(self, banded=False):
        self.lines, self.names, self.used, self.calls = [], {}, set(), set()
        self.scalars, self.alongs, self.banded = set(), {}, banded

    def _new(self, text):
        name = f"t{len(self.names)}"
        self.lines.append(f"sfpi::vFloat {name} = {text};")
        return name

    def _text(self, node):
        args = [self.value(a) for a in node.args]
        kind = node.kind
        if kind == "operand":
            index, part = node.value
            self.used.add(index)
            if self.alongs.setdefault(index, node.along) != node.along:
                raise ValueError(
                    f"kohakutt.dsl.metal: operand {index} is read with two layouts"
                )
            if self.banded:
                return f"o{index}"
            arg = ARGS[index] if index < 2 else ARGS[index + 1]
            return f"sfpi::dst_reg[{arg} * 32{' + 1' if part else ''}]"
        if kind == "const":
            return _literal(node.value)
        if kind == "scalar":
            self.scalars.add(node.value)
            return f"sfpi::vFloat({CONVERTER}(s{node.value}))"
        if kind == "bf16":
            return f"float32_to_bf16_rne({args[0]})"
        if kind in ("add", "sub"):
            a, b = args
            if any(_product(n) for n in node.args):
                # Written as 1 * a + b so that the product is rounded before the sum.
                b = b if kind == "add" else f"(-{b})"
                return f"__builtin_rvtt_sfpmad({ONE}.get(), {a}.get(), {b}.get(), 0)"
            return f"{a} {'+' if kind == 'add' else '-'} {b}"
        if kind == "mul":
            return f"{args[0]} * {args[1]}"
        if kind == "mad":
            a, b, c = args
            return f"__builtin_rvtt_sfpmad({a}.get(), {b}.get(), {c}.get(), 0)"
        if kind == "neg":
            return f"-{args[0]}"
        if kind == "abs":
            return f"sfpi::abs({args[0]})"
        if kind in ("min", "max"):
            return f"sfpi::{kind}({args[0]}, {args[1]})"
        if kind == "flash_exp2":
            return self._flash_exp2(args[0])
        function = FUNCTIONS[node.value]
        self.calls.add(node.value)
        return function.call.format(x=args[0])

    def _flash_exp2(self, u):
        """The statements of ktt_flash's 2^(u - 639); the text of the last value."""
        n = len(self.names) + len(self.lines)
        s, m, p = f"e{n}s", f"e{n}m", f"e{n}p"
        clamped = f"sfpi::max({u}, sfpi::vFloat({_literal(FLASH_FLOOR)}))"
        self.lines += [
            f"sfpi::vUInt {s} = sfpi::shft(sfpi::as<sfpi::vUInt>({clamped}), 9);",
            f"sfpi::vFloat {m} = sfpi::setexp(sfpi::as<sfpi::vFloat>({s}), 127);",
            f"sfpi::vFloat {p} = __builtin_rvtt_sfpmad({m}.get(), "
            f"sfpi::vFloat({_literal(FLASH_B2)}).get(), "
            f"sfpi::vFloat({_literal(FLASH_B1)}).get(), 0);",
            f"{p} = __builtin_rvtt_sfpmad({p}.get(), {m}.get(), "
            f"sfpi::vFloat({_literal(FLASH_B0)}).get(), 0);",
        ]
        return f"sfpi::copyexp({p}, sfpi::as<sfpi::vFloat>({s}))"

    def value(self, node):
        """The name of the variable holding `node` (an operand or a constant is
        loaded once, any other node is computed once)."""
        shared = node.kind in ("operand", "const", "scalar")
        # A constant is keyed by its bits: 0.0 == -0.0 in Python.
        value = _literal(node.value) if node.kind == "const" else node.value
        key = (node.kind, value if shared else id(node))
        if node.kind == "operand" and self.alongs.get(node.value[0], node.along) != (
            node.along
        ):
            raise ValueError(
                f"kohakutt.dsl.metal: operand {node.value[0]} is read with two layouts"
            )
        if key in self.names:
            return self.names[key]
        if node.kind == "select":
            cond = node.value
            a, b = self.value(cond.a), self.value(cond.b)
            then, other = (self.value(arg) for arg in node.args)
            name = self.names[key] = self._new(other)
            self.lines.append(
                f"v_if({a} {COMPARES[cond.kind]} {b}) {{ {name} = {then}; }}"
            )
            self.lines.append("v_endif;")
            return name
        text = self._text(node)
        name = self.names[key] = self._new(text)
        return name


class GeneratedFn(SfpuFn):
    """A vector-unit function written from expressions: `operands` register operands
    and `scalars` fp32 scalar arguments, run under the setup of table op `init`
    (`c.init(fn.init)`); `fp32`: it needs fp32 registers."""

    scalars = 0

    def __init__(self, name, code, operands, init, headers, fp32=False):
        super().__init__(name, code)
        self.operands, self.init, self.headers = operands, init, tuple(headers)
        self.fp32 = fp32
        self.along, self.tiles_of, self.result = {}, None, None
        self.whole, self.reduces = False, None

    def tiles(self, n):
        """This function over `n` output tiles in one call: operand slot s (and the
        output) at s + t for tile t; "col" / "row" operands fixed, loaded once per
        row band (col) or per load (row)."""
        if self.result is None or isinstance(self.result, dict):
            raise ValueError(
                f"kohakutt.dsl.metal: {self.name} is not a whole-tile function"
            )
        if not isinstance(n, int) or not 1 <= n <= TILES:
            raise ValueError("kohakutt.dsl.metal: a function runs over 1 .. 4 tiles")
        for along in self.along.values():
            if along and along[0] == "row" and along[1] + along[2] * (n - 1) > 3:
                raise ValueError(
                    f"kohakutt.dsl.metal: {self.name} reads row bands past 3 over "
                    f"{n} tiles"
                )
        cache = self.__dict__.setdefault("_tiled", {})
        if n in cache:
            return cache[n]
        out = cache[n] = GeneratedFn(
            f"{self.name}_t{n}",
            _banded(self, n),
            self.operands,
            self.init,
            self.headers,
            self.fp32,
        )
        out.scalars, out.along, out.tiles_of = self.scalars, self.along, n
        return out


def _banded(fn, n):
    """The code of `fn` over `n` output tiles: per row band (16 per tile, each 4 rows
    of even or odd columns) the column operands once, then each tile's two halves."""
    low = _Lowering(banded=True)
    (out,) = [low.value(_wrap(fn.result))]
    names = [*ARGS[: max(3, fn.operands + 1)], *(f"s{i}" for i in range(fn.scalars))]
    body = []
    for h in range(2):
        for d in range(8):
            unit = 16 * h + d
            body.append("{")
            for i in range(fn.operands):
                along = fn.along.get(i)
                if along and along[0] == "col":
                    at = f"{_arg(i)} * 32 + {unit + 8 * along[1]}"
                    body.append(f"    sfpi::vFloat o{i} = sfpi::dst_reg[{at}];")
            for t in range(n):
                for half in range(2):
                    body.append("    {")
                    for i in range(fn.operands):
                        along = fn.along.get(i)
                        if along and along[0] == "col":
                            continue
                        if along:
                            rows = 2 * (along[1] + along[2] * t)
                            at = f"{_arg(i)} * 32 + {8 * half + rows + d % 2}"
                        else:
                            at = f"({_arg(i)} + {t}) * 32 + {unit + 8 * half}"
                        body.append(f"        sfpi::vFloat o{i} = sfpi::dst_reg[{at}];")
                    body += [f"        {line}" for line in low.lines]
                    body.append("        if constexpr (!DST_ACCUM_MODE) {")
                    body.append(f"            {out} = float32_to_bf16_rne({out});")
                    body.append("        }")
                    at = f"(out + {t}) * 32 + {unit + 8 * half}"
                    body.append(f"        sfpi::dst_reg[{at}] = {out};")
                    body.append("    }")
            body.append("}")
    params = ", ".join(f"uint32_t {arg}" for arg in names)
    return "\n".join(
        [
            "#ifdef TRISC_MATH",
            f"inline void {fn.name}_t{n}({params}) {{",
            *(f"    {line}" for line in body),
            "}",
            "#endif",
        ]
    )


def _moves(name, pairs):
    """A whole-registers function (run once, VectorMode::None) storing
    `dst_reg[source]` to `dst_reg[target]` for each (target, source) of `pairs`."""
    lines = [
        f"    {{ sfpi::vFloat v = sfpi::dst_reg[{s}]; sfpi::dst_reg[{t}] = v; }}"
        for t, s in pairs
    ]
    code = "\n".join(
        [
            "#ifdef TRISC_MATH",
            f"inline void {name}(uint32_t a, uint32_t b, uint32_t out) {{",
            *lines,
            "}",
            "#endif",
        ]
    )
    fn = GeneratedFn(name, code, 2 if "b * 32" in code else 1, "mul", ["sfpi.h"])
    fn.whole = True
    return fn


@functools.cache
def join_cols():
    """Slot `out` = faces 0 / 2 (columns 0-15) of slot `a` and faces 1 / 3 of slot
    `b`: two column broadcasts in one slot (`operand(i, along="col", half=h)`)."""
    pairs = [
        (f"out * 32 + {8 * f + d}", f"{'b' if f % 2 else 'a'} * 32 + {8 * f + d}")
        for f in range(4)
        for d in range(8)
    ]
    return _moves("km_join_cols", pairs)


@functools.cache
def put_rows(band):
    """Slot `out` rows 4 band .. 4 band + 3 (every column) = rows 0 .. 3 of slot
    `a`, the rest of `out` kept: row broadcasts side by side in one slot."""
    pairs = [
        (f"out * 32 + {8 * h + 2 * band + e}", f"a * 32 + {8 * h + e}")
        for h in range(2)
        for e in range(2)
    ]
    return _moves(f"km_put_rows{band}", pairs)


@functools.cache
def row_reduce(op, n, fresh=False):
    """Function: the max or sum `op` of each row of slots `a` .. `a + n - 1`, with
    slot `b` (a column broadcast) unless `fresh`, into slot `out` as a column
    broadcast; lanes combine tile by tile, then by rotations 4, 2, 1 in each row."""
    if op not in ("max", "sum") or not 1 <= n <= TILES:
        raise ValueError("kohakutt.dsl.metal: a row reduce is max or sum of 1 .. 4")

    def combine(x, y):
        return f"sfpi::max({x}, {y})" if op == "max" else f"{x} + {y}"

    lines = []
    for h in range(2):
        for k in range(4):
            units = [
                f"(a + {t}) * 32 + {16 * h + 8 * half + 2 * k + e}"
                for t in range(n)
                for half in range(2)
                for e in range(2)
            ]
            lines.append("{")
            lines.append(f"    sfpi::vFloat v = sfpi::dst_reg[{units[0]}];")
            for unit in units[1:]:
                lines.append(f"    {{ sfpi::vFloat w = sfpi::dst_reg[{unit}];")
                lines.append(f"      v = {combine('v', 'w')}; }}")
            for shift in (4, 2, 1):
                lines.append("    { sfpi::vFloat w = v;")
                rotate = "      w = sfpi::vFloat(sfpi::subvec_shflror1(w));"
                lines += [rotate] * shift
                lines.append(f"      v = {combine('v', 'w')}; }}")
            if not fresh:
                at = f"b * 32 + {16 * h + 2 * k}"
                lines.append(f"    {{ sfpi::vFloat w = sfpi::dst_reg[{at}];")
                lines.append(f"      v = {combine('v', 'w')}; }}")
            for half in range(2):
                for e in range(2):
                    at = f"out * 32 + {16 * h + 8 * half + 2 * k + e}"
                    lines.append(f"    sfpi::dst_reg[{at}] = v;")
            lines.append("}")
    name = f"km_row_{op}{n}{'f' if fresh else ''}"
    code = "\n".join(
        [
            "#ifdef TRISC_MATH",
            f"inline void {name}(uint32_t a, uint32_t b, uint32_t out) {{",
            *(f"    {line}" for line in lines),
            "}",
            "#endif",
        ]
    )
    fn = GeneratedFn(name, code, 1 if fresh else 2, "mul", ["sfpi.h"])
    fn.whole, fn.reduces = True, n
    return fn


def _arg(index):
    """The parameter naming the slot of operand `index`."""
    return ARGS[index] if index < 2 else ARGS[index + 1]


def sfpu_fn(name, result):
    """The function `name` storing `result` into its output slot: one value, or
    {"even": value, "odd": value} for the even and odd columns (operands by
    `.even` / `.odd`). Run it with `c.run(fn, *slots, out=slot)`."""
    pairs = isinstance(result, dict)
    if pairs and set(result) != {"even", "odd"}:
        raise ValueError('kohakutt.dsl.metal: a pair function gives "even" and "odd"')
    results = (
        [_wrap(result["even"]), _wrap(result["odd"])] if pairs else [_wrap(result)]
    )
    low = _Lowering()
    outs = [low.value(r) for r in results]
    parts = {key[1][1] for key in low.names if key[0] == "operand"}
    if not (parts <= {0, 1} if pairs else parts <= {None}):
        raise ValueError(
            "kohakutt.dsl.metal: a pair function reads operands by .even / .odd, "
            "any other reads them whole"
        )
    if not low.used or max(low.used) + 1 != len(low.used):
        raise ValueError("kohakutt.dsl.metal: a function uses operands 0 .. n - 1")
    inits = {FUNCTIONS[c].init for c in low.calls} - {None}
    if len(inits) > 1:
        raise ValueError(
            f"kohakutt.dsl.metal: functions {sorted(low.calls)} need different setups "
            f"({sorted(inits)}); one vector-unit function runs under one"
        )
    count, scalars = len(low.used), len(low.scalars)
    if low.scalars and max(low.scalars) + 1 != scalars:
        raise ValueError("kohakutt.dsl.metal: a function uses scalars 0 .. n - 1")
    names = [*ARGS[: max(3, count + 1)], *(f"s{i}" for i in range(scalars))]
    params = ", ".join(f"uint32_t {arg}" for arg in names)
    body = list(low.lines)
    for i, out in enumerate(outs):
        body.append("if constexpr (!DST_ACCUM_MODE) {")
        body.append(f"    {out} = float32_to_bf16_rne({out});")
        body.append("}")
        body.append(f"sfpi::dst_reg[out * 32{' + 1' if i else ''}] = {out};")
    body.append("sfpi::dst_reg += 2;" if pairs else "sfpi::dst_reg++;")
    headers = sorted({LLK + FUNCTIONS[c].header for c in low.calls})
    headers = ["sfpi.h", LLK + "conversions.h", *headers]
    if scalars:
        headers.append("sfpu/ckernel_sfpu_converter.h")
    # Only the math thread runs (and can compile) vector-unit code.
    lines = [
        "#ifdef TRISC_MATH",
        f"inline void {name}({params}) {{",
        f"    for (int rows = 0; rows < {4 if pairs else 8}; rows++) {{",
        *(f"        {line}" for line in body),
        "    }",
        "}",
        "#endif",
    ]
    init = next(iter(inits), "mul")
    fp32 = any(FUNCTIONS[c].fp32 for c in low.calls)
    fn = GeneratedFn(name, "\n".join(lines), count, init, headers, fp32)
    fn.scalars, fn.along, fn.result = scalars, dict(low.alongs), result
    return fn
