"""RoPE as one program at the metal level (torch op ``ktt_fusion::rope``): out = x * cos
+ rotate(x) * sin, rotate(x) = x @ R on the matrix unit (R holds one +-1 per column:
exact for bf16 x), the rest one multiply-add in fp32 registers, rounded once.
Forms: rotate-half (cat(-x2, x1)) and interleaved pairs ((-x1, x0) per pair)."""

import math

import torch

import kohakutt.dsl.metal as km

f = km.sfpu
_x, _y, _c, _s = f.operands(4)
# x * cos + rotate(x) * sin: the product rotate(x) * sin is exact for bf16 operands.
ROTATED = km.sfpu_fn("rope_rotated", f.mad(_x, _c, _y * _s))

# R [dt * 32, dt * 32] in the reserved tiles: zeros, then -1 / +1 at (source row,
# column) for each of the 2 * half columns (form 0 rotate-half, 1 interleaved pairs).
BUILD_R = """
auto* r = reinterpret_cast<volatile uint32_t*>(r_tiles);
for (uint32_t i = 0; i < DT * DT * 512; ++i) {
    r[i] = 0;
}
auto* e = reinterpret_cast<volatile uint16_t*>(r_tiles);
for (uint32_t col = 0; col < 2 * half; ++col) {
    const bool low = FORM == 0 ? col < half : (col & 1) == 0;
    const uint32_t row = FORM == 0 ? (low ? col + half : col - half) : (col ^ 1);
    const uint32_t tr = row & 31, tc = col & 31;
    const uint32_t at = ((row >> 5) * DT + (col >> 5)) * 1024 +
        ((tr >> 4) * 2 + (tc >> 4)) * 256 + (tr & 15) * 16 + (tc & 15);
    e[at] = low ? 0xBF80 : 0x3F80;
}
"""

# Columns tail .. 31 of the last column tile of x zeroed (head dims not whole tiles:
# padding times R's zero rows must not bring a NaN into the sums).
ZERO_TAIL = """
if (tail != 0) {
    auto* t = reinterpret_cast<volatile uint16_t*>(x_tiles + (DT - 1) * 2048);
    for (uint32_t row = 0; row < 32; ++row) {
        for (uint32_t col = tail; col < 32; ++col) {
            t[((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15)] = 0;
        }
    }
}
"""


@km.program("rope")
def rope_program(
    p,
    x: km.TensorRef,
    cos: km.TensorRef,
    sin: km.TensorRef,
    out: km.Out,
    dt: km.Static[int] = 2,
    form: km.Static[int] = 0,
):
    """A unit is one row tile of one slice of x (its dt column tiles). Runtime values:
    `units`, the row tiles `st` of a slice, `rep` (consecutive x slices per cos / sin
    slice) and `half` (head dim / 2)."""
    units, st, rep = p.runtime("units"), p.runtime("st"), p.runtime("rep")
    half, tail = p.runtime("half"), p.runtime("tail")
    start, count = p.split(units)
    cb_x = p.cb(0, like=x, pages=2 * dt)
    cb_r = p.cb(1, like=x, pages=dt * dt, name="r")
    cb_cos = p.cb(2, like=cos, pages=2 * dt)
    cb_sin = p.cb(3, like=sin, pages=2 * dt)
    cb_out = p.cb(16, like=out, pages=2 * dt)

    @p.dm0
    def reader(k):
        r_tiles = cb_r.reserve(dt * dt)
        k.macro("DT", dt)
        k.macro("FORM", form)
        k.inline(
            BUILD_R,
            uses=[km.CBUse(cb_r, written=dt * dt * cb_r.page_bytes)],
            args={"r_tiles": r_tiles, "half": half},
        )
        cb_r.push(dt * dt)
        for i in k.range(count, name="u"):
            u = k.let("unit", start + i)
            crow = k.let("crow", u // (rep * st) * st + u % st)
            xs, cs, ss = cb_x.reserve(dt), cb_cos.reserve(dt), cb_sin.reserve(dt)
            for j in k.range(dt, name="j"):
                k.read(x.page(u * dt + j), xs[j])
                k.read(cos.page(crow * dt + j), cs[j])
                k.read(sin.page(crow * dt + j), ss[j])
            k.reads_done()
            k.inline(
                ZERO_TAIL,
                uses=[km.CBUse(cb_x, written=dt * cb_x.page_bytes)],
                args={"x_tiles": xs, "tail": tail},
            )
            cb_x.push(dt)
            cb_cos.push(dt)
            cb_sin.push(dt)

    @p.compute(fp32_dest=True, unpack_fp32=[cb_cos, cb_sin], min_fidelity="HiFi4")
    def compute(c):
        c.startup(cb_x, cb_r, cb_out)
        cb_r.wait(dt * dt)
        for _ in c.range(count, name="u"):
            cb_x.wait(dt)
            cb_cos.wait(dt)
            cb_sin.wait(dt)
            cb_out.reserve(dt)
            for j in c.range(dt, name="j"):
                with c.registers():
                    # Slot 0: rotate(x) column tile j = sum over k of x_k @ R_kj.
                    c.format(srca=cb_x, srcb=cb_r)
                    product = c.matmul_init(cb_x, cb_r, ct=1, rt=1, kt=dt)
                    for kk in c.range(dt, name="k"):
                        c.matmul(product, kk, (kk * dt + j).bounded(0, dt * dt), 0)
                    c.srca_format(cb_x)
                    c.init("copy", cb_x)
                    c.copy(cb_x, j, 1)
                    c.srca_format(cb_cos)
                    c.init("copy", cb_cos)
                    c.copy(cb_cos, j, 2)
                    c.srca_format(cb_sin)
                    c.init("copy", cb_sin)
                    c.copy(cb_sin, j, 3)
                    c.init(ROTATED.init)
                    c.run(ROTATED, 1, 0, 2, 3, out=0)
                    c.pack(0, cb_out, at=j)
            cb_out.push(dt)
            cb_x.pop(dt)
            cb_cos.pop(dt)
            cb_sin.pop(dt)
        cb_r.pop(dt * dt)

    @p.dm1
    def writer(k):
        for i in k.range(count, name="u"):
            u = k.let("unit", start + i)
            cb_out.wait(dt)
            for j in k.range(dt, name="j"):
                k.write(cb_out, out.page(u * dt + j), index=j)
            k.writes_done()
            cb_out.pop(dt)


def plan(x, cos):
    """(runtime values, statics) of rope_program for x [..., S, D] and cos / sin of
    shape [..., S, D] broadcasting to x by whole leading slices; None when they do
    not (or D is odd or past 256)."""
    d, s = x.shape[-1], x.shape[-2]
    if x.dim() < 2 or cos.dim() < 2 or d % 2 or d > 256:
        return None
    if tuple(cos.shape[-2:]) != (s, d) or cos.dim() > x.dim():
        return None
    lead = list(x.shape[:-2])
    clead = [1] * (len(lead) - (cos.dim() - 2)) + list(cos.shape[:-2])
    # cos's leading dims: the outer ones equal x's, the inner ones 1.
    inner = len(lead)
    while inner > 0 and clead[inner - 1] == 1:
        inner -= 1
    if clead[:inner] != lead[:inner]:
        return None
    st = math.ceil(s / 32)
    values = {
        "units": math.prod(lead) * st,
        "st": st,
        "rep": math.prod(lead[inner:]),
        "half": d // 2,
        "tail": d % 32,
    }
    return values, {"dt": math.ceil(d / 32)}


SCHEMA = "ktt_fusion::rope(Tensor x, Tensor cos, Tensor sin, int form) -> Tensor"


@km.op(SCHEMA, rope_program)
def rope(x, cos, sin, form):
    """The launch of x * cos + rotate(x) * sin for tt tensors (x bf16; cos, sin bf16
    or fp32 of one shape); form 0 rotates halves, 1 interleaved pairs."""
    planned = plan(x, cos) if tuple(sin.shape) == tuple(cos.shape) else None
    if planned is None or x.dtype != torch.bfloat16 or form not in (0, 1):
        raise ValueError("ktt_fusion.rope: x bf16 [..., S, D], cos / sin [..., S, D]")
    values, statics = planned
    return km.Call(
        inputs={"x": x, "cos": cos, "sin": sin},
        outputs={"out": (list(x.shape), x.dtype)},
        runtime=values,
        static={**statics, "form": int(form)},
    )
