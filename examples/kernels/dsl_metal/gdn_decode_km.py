"""One decode step of a gated delta net (Qwen3.5 / 3.8) at the metal level, on the
model's tensors (op `gdn_decode_km`): the fp32 state updated in place, o returned;
errors against float64 and transformers' step. Usage: python
examples/kernels/dsl_metal/gdn_decode_km.py [--heads HK HV] [--seqs B] [--steps N]"""

import argparse
import struct
import time

import torch
import torch.nn.functional as F

import kohakutt
import kohakutt.dsl.metal as km

EPS = 1e-6
f = km.sfpu

# S' = S d + k u (the delta rule's rank-1 update): S, d, k (per row), u (per column).
UPDATE = km.sfpu_fn(
    "gdn_update", f.mad(f.operand(0), f.operand(1), f.operand(2) * f.operand(3))
)

# A bf16 page of zeros with ones in row 0 (row 0 of faces 0 and 1).
ROW0 = """
auto* words = reinterpret_cast<volatile uint32_t*>(page);
for (uint32_t i = 0; i < 512; ++i) {
    words[i] = 0;
}
for (uint32_t i = 0; i < 8; ++i) {
    words[i] = 0x3f803f80u;
    words[128 + i] = 0x3f803f80u;
}
"""


def element_bits(slot, ref, col):
    """The fp32 bits of element (0, col) of the tile in `slot` (format of `ref`)."""
    if ref.dtype == "float32":
        return slot.load32(((col // 16) * 256 + col % 16).bounded(0, 512))
    word = slot.load32(((col // 16) * 128 + (col % 16) // 2).bounded(0, 256))
    return km.select((col & 1).eq(1), word & 0xFFFF0000, word << 16)


@km.program("gdn_decode")
def gdn_decode_program(
    p,
    state: km.TensorRef,
    qkv: km.TensorRef,
    a: km.TensorRef,
    b: km.TensorRef,
    a_log: km.TensorRef,
    dt_bias: km.TensorRef,
    o: km.Out,
    kt: int = 4,
):
    """A unit is (sequence, value head, block of 32 V columns): its state block is kt
    fp32 tiles [32 K x 32 V]. Runtime values: `units`, `vt` (V / 32), `hv`, `group`
    (HV / HK), the qkv row's tiles `ct` and its k / v offsets `koff` / `voff`, the
    gate rows' tiles `gt`, `scale` (fp32 bits of K^-0.5)."""
    units, vt, hv, group = (p.runtime(n) for n in ("units", "vt", "hv", "group"))
    ct, koff, voff, gt = (p.runtime(n) for n in ("ct", "koff", "voff", "gt"))
    scale = p.runtime("scale")
    first, step = p.stride(units)
    cb_s = p.cb(0, like=state, pages=kt)
    cb_k = p.cb(1, like=qkv, pages=kt, name="k")
    cb_q = p.cb(2, like=qkv, pages=kt, name="q")
    cb_v = p.cb(3, like=qkv, pages=1, name="v")
    gates = [p.cb(4 + i, like=t, pages=1) for i, t in enumerate((a, dt_bias, a_log, b))]
    cb_mask = p.cb(8, dtype="bfloat16", pages=1, name="mask")
    cb_c = p.cb(9, dtype="float32", pages=3, name="consts")  # d, e, f1
    cb_g = p.cb(10, dtype="float32", pages=2, name="outs")  # g1, g2
    cb_u = p.cb(11, dtype="float32", pages=1, name="update")
    cb_o = p.cb(16, like=o, pages=1)
    cb_so = p.cb(17, like=state, pages=2, name="state_out")
    words = p.words(24, name="gate_words")

    def place(k, u):
        """(sequence, value head, V block) of unit `u`."""
        vb = k.let("vb", u % vt)
        head = k.let("head", u // vt % hv)
        return k.let("seq", u // vt // hv), head, vb

    def state_page(seq, head, i, vb):
        return ((seq * hv + head) * kt + i) * vt + vb

    @p.dm0
    def reader(k):
        mask = cb_mask.reserve(1)
        k.inline(
            ROW0,
            uses=[km.CBUse(cb_mask, written=cb_mask.page_bytes)],
            args={"page": mask},
        )
        cb_mask.push(1)
        held = [cb.reserve(1) for cb in gates]
        for u in k.range(units, start=first, step=step, name="u"):
            seq, head, vb = place(k, u)
            col, gtile = k.let("col", head % 32), k.let("gtile", head // 32)
            # The step's gate inputs as words: a, dt_bias, A_log, b of this head.
            k.read(a.page(seq * gt + gtile), held[0])
            k.read(dt_bias.page(gtile), held[1])
            k.read(a_log.page(gtile), held[2])
            k.read(b.page(seq * gt + gtile), held[3])
            k.reads_done()
            for slot, ref in zip(held, (a, dt_bias, a_log, b), strict=True):
                k.put(words, element_bits(slot, ref, col))
            kh = k.let("kh", head // group)
            s, kk, qq, v = cb_s.reserve(kt), cb_k.reserve(kt), cb_q.reserve(kt), None
            for i in range(kt):
                k.read(state.page(state_page(seq, head, i, vb)), s[i])
                k.read(qkv.page(seq * ct + koff + kh * kt + i), kk[i])
                k.read(qkv.page(seq * ct + kh * kt + i), qq[i])
            v = cb_v.reserve(1)
            k.read(qkv.page(seq * ct + voff + head * vt + vb), v)
            k.reads_done()
            for cb, n in ((cb_s, kt), (cb_k, kt), (cb_q, kt), (cb_v, 1)):
                cb.push(n)

    fp32_cbs = (cb_s, cb_c, cb_g, cb_u)

    @p.compute(fp32_dest=True, unpack_fp32=fp32_cbs, exact=True)
    def compute(c):
        def unpack(kind, cb, index, slot):
            c.format(srca=cb, srcb=cb)
            c.init(kind, cb)
            if kind == "copy":
                c.copy(cb, index, slot)
            else:
                c.bcast(cb, index, slot, "row")

        def op(name, *slots, value=None):
            c.init(name)
            c.op(name, *slots, value=value)

        def dot(x, y, into):
            """Slot `into` = the sum over K of bf16 rows x * y (x, y: kt tiles each
            holding a vector in row 0), in every element; slot 0 (and 1) as scratch."""
            for i in range(kt):
                unpack("bcast_row", x, i, 0)
                if y is not x:
                    unpack("bcast_row", y, i, 1)
                op("mul", 0, 1 if y is not x else 0, 0)
                c.init("mul")
                c.row_reduce("sum", 0, 1, into=into, fresh=i == 0)

        def packed(cb, slots):
            c.commit()
            c.reserve(cb, len(slots))
            c.pack_format(cb)
            for slot in slots:
                c.pack(slot, cb)
            c.release()
            c.push(cb, len(slots))
            c.wait(cb, len(slots))

        c.startup(cb_s, None, cb_so)
        c.wait(cb_mask, 1)
        for _ in c.range(units, start=first, step=step, name="u"):
            wa, wdt, walog, wb = (
                c.take(words, n) for n in ("wa", "wdt", "walog", "wb")
            )
            for cb, n in ((cb_s, kt), (cb_k, kt), (cb_q, kt), (cb_v, 1)):
                c.wait(cb, n)
            # rk = |k|^-1 (2); d = exp(-exp(A_log) softplus(a + dt_bias)) (0);
            # e = rk sigmoid(b) (1); f1 = d rk (3).
            c.acquire()
            dot(cb_k, cb_k, 2)
            op("add_scalar", 2, value=EPS)
            op("rsqrt", 2)
            op("fill", 0, value=wa)
            op("fill", 1, value=wdt)
            op("add", 0, 1, 0)
            op("exp", 0)
            op("add_scalar", 0, value=1.0)
            op("log", 0)
            op("fill", 1, value=walog)
            op("exp", 1)
            op("mul", 0, 1, 0)
            op("neg", 0)
            op("exp", 0)
            op("fill", 1, value=wb)
            op("sigmoid", 1)
            op("mul", 2, 1, 1)
            op("mul", 0, 2, 3)
            packed(cb_c, (0, 1, 3))
            # rqs = |q|^-1 K^-0.5 (3); g1 = rqs d (3); g2 = rqs (k . q) (2).
            c.acquire()
            dot(cb_k, cb_q, 2)
            dot(cb_q, cb_q, 3)
            op("add_scalar", 3, value=EPS)
            op("rsqrt", 3)
            op("mul_scalar", 3, value=scale)
            op("mul", 3, 2, 2)
            unpack("copy", cb_c, 0, 0)
            op("mul", 3, 0, 3)
            packed(cb_g, (3, 2))
            # hk = S^T k (2), hq = S^T q (3) per V row from the transposed state tiles;
            # u = e (v - f1 hk), o = g1 hq + g2 u; both turned to rows of V.
            c.acquire()
            for i in range(kt):
                unpack("copy", cb_s, i, 0)
                op("transpose_dest", 0)
                for x, into in ((cb_k, 2), (cb_q, 3)):
                    unpack("bcast_row", x, i, 1)
                    op("mul", 0, 1, 1)
                    c.init("mul")
                    c.row_reduce("sum", 1, 1, into=into, fresh=i == 0)
            unpack("bcast_row", cb_v, 0, 0)
            op("transpose_dest", 0)
            unpack("copy", cb_c, 2, 1)
            op("mul", 1, 2, 1)
            op("sub", 0, 1, 0)
            unpack("copy", cb_c, 1, 1)
            op("mul", 0, 1, 0)
            unpack("copy", cb_g, 0, 1)
            op("mul", 3, 1, 3)
            unpack("copy", cb_g, 1, 1)
            op("mul", 1, 0, 1)
            op("add", 3, 1, 3)
            op("transpose_dest", 3)
            op("transpose_dest", 0)
            unpack("copy", cb_mask, 0, 1)
            op("mul", 3, 1, 3)
            c.commit()
            c.reserve(cb_o, 1)
            c.pack_format(cb_o)
            c.pack(3, cb_o)
            c.reserve(cb_u, 1)
            c.pack_format(cb_u)
            c.pack(0, cb_u)
            c.release()
            c.push(cb_o, 1)
            c.push(cb_u, 1)
            c.wait(cb_u, 1)
            # S' = S d + k u, tile by tile (k per K row, u per V column).
            for i in range(kt):
                c.acquire()
                unpack("copy", cb_s, i, 0)
                unpack("bcast_row", cb_k, i, 1)
                op("transpose_dest", 1)
                unpack("copy", cb_u, 0, 2)
                unpack("copy", cb_c, 0, 3)
                c.init(UPDATE.init)
                c.run(UPDATE, 0, 3, 1, 2, out=0)
                c.commit()
                c.reserve(cb_so, 1)
                c.pack_format(cb_so)
                c.pack(0, cb_so)
                c.release()
                c.push(cb_so, 1)
            for cb, n in ((cb_u, 1), (cb_g, 2), (cb_c, 3), (cb_v, 1)):
                c.pop(cb, n)
            for cb in (cb_q, cb_k, cb_s):
                c.pop(cb, kt)
        c.pop(cb_mask, 1)

    @p.dm1
    def writer(k):
        for u in k.range(units, start=first, step=step, name="u"):
            seq, head, vb = place(k, u)
            cb_o.wait(1)
            k.write(cb_o, o.page(seq * hv * vt + head * vt + vb))
            k.flushed()
            cb_o.pop(1)
            for i in range(kt):
                cb_so.wait(1)
                k.write(cb_so, state.page(state_page(seq, head, i, vb)))
                k.flushed()
                cb_so.pop(1)
        k.writes_done()


def bits(value):
    """The fp32 bits of `value`, as a runtime value."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


@km.op(
    "kohakutt_example::gdn_decode_km(Tensor(a!) state, Tensor qkv, Tensor a, Tensor b, "
    "Tensor a_log, Tensor dt_bias) -> Tensor",
    gdn_decode_program,
)
def gdn_decode(state, qkv, a, b, a_log, dt_bias):
    """One step: `state` [B, HV, K, V] fp32 becomes the step's state in place; returns
    o [B, 1, HV V] in qkv's dtype. qkv [B, 1, 2 HK K + HV V] (q | k | v after the conv),
    a, b [B, 1, HV], a_log, dt_bias [HV]."""
    n, hv, dk, dv = state.shape
    hk, rest = divmod(qkv.shape[-1] - hv * dv, 2 * dk)
    if state.dtype != torch.float32 or dk % 32 or dv % 32 or rest or hv % hk:
        raise ValueError(
            "gdn_decode_km: an fp32 state [B, HV, K, V] of whole tiles, qkv "
            "[B, 1, 2 HK K + HV V] with HV a multiple of HK"
        )
    if qkv.shape[:2] != (n, 1) or a.shape != (n, 1, hv) or b.shape != (n, 1, hv):
        raise ValueError("gdn_decode_km: one token: qkv [B, 1, C], a and b [B, 1, HV]")
    if a_log.shape != (hv,) or dt_bias.shape != (hv,):
        raise ValueError("gdn_decode_km: a_log and dt_bias are [HV]")
    kt, vt = dk // 32, dv // 32
    return km.Call(
        inputs={
            "state": state,
            "qkv": qkv,
            "a": a,
            "b": b,
            "a_log": a_log,
            "dt_bias": dt_bias,
        },
        outputs={"o": ([n, 1, hv * dv], qkv.dtype)},
        runtime={
            "units": n * hv * vt,
            "vt": vt,
            "hv": hv,
            "group": hv // hk,
            "ct": qkv.shape[-1] // 32,
            "koff": hk * kt,
            "voff": 2 * hk * kt,
            "gt": -(-hv // 32),
            "scale": bits(dk**-0.5),
        },
        static={"kt": kt},
    )


def split(qkv, hk, hv, dk, dv, dtype):
    """q, k [B, HV, K] (each key head repeated for its value heads) and v [B, HV, V]."""
    n = qkv.shape[0]
    x = qkv.reshape(n, -1).to(dtype)
    q, k = (x[:, i * hk * dk : (i + 1) * hk * dk].reshape(n, hk, dk) for i in (0, 1))
    v = x[:, 2 * hk * dk :].reshape(n, hv, dv)
    return q.repeat_interleave(hv // hk, 1), k.repeat_interleave(hv // hk, 1), v


def reference(state, qkv, a, b, a_log, dt_bias, hk):
    """The step in float64 (transformers' recurrent form, every input as given):
    (state [B, HV, K, V], o [B, 1, HV V])."""
    n, hv, dk, dv = state.shape
    q, k, v = split(qkv, hk, hv, dk, dv, torch.float64)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + EPS) * dk**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + EPS)
    x = a.double().reshape(n, hv) + dt_bias.double()
    d = (-a_log.double().exp() * F.softplus(x)).exp()
    beta = b.double().reshape(n, hv).sigmoid()
    s = state.double() * d[..., None, None]
    delta = (v - (s * k[..., None]).sum(-2)) * beta[..., None]
    s = s + k[..., None] * delta[..., None, :]
    return s, (s * q[..., None]).sum(-2).reshape(n, 1, hv * dv)


def hf_step(state, qkv, a, b, a_log, dt_bias, hk):
    """The step as transformers' Qwen3.5 GDN layer runs it on the CPU (its gates, its
    `torch_recurrent_gated_delta_rule`): (state, o [B, 1, HV V] in qkv's dtype)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_recurrent_gated_delta_rule,
    )

    n, hv, dk, dv = state.shape
    q, k, v = split(qkv, hk, hv, dk, dv, qkv.dtype)
    beta = b.sigmoid()
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)
    o, s = torch_recurrent_gated_delta_rule(
        q[:, None],
        k[:, None],
        v[:, None],
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    return s, o.reshape(n, 1, hv * dv)


def inputs(shape, seed=0):
    """A state and the steps' model tensors for `shape` (B, HK, HV, K, V, steps): qkv,
    a, b bf16 per step; A_log, dt_bias bf16 (the checkpoint's dtype)."""
    n, hk, hv, dk, dv, steps = shape
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(n, hv, dk, dv, generator=g) * 0.05
    per_step = [
        (
            torch.randn(n, 1, 2 * hk * dk + hv * dv, generator=g).bfloat16(),
            torch.randn(n, 1, hv, generator=g).bfloat16(),
            torch.randn(n, 1, hv, generator=g).bfloat16(),
        )
        for _ in range(steps)
    ]
    a_log = (torch.rand(hv, generator=g) * 15.99 + 0.01).log().bfloat16()
    dt_bias = (torch.randn(hv, generator=g) * 0.5 + 1).bfloat16()
    return state, per_step, a_log, dt_bias


def relative(got, want):
    """Max error relative to the largest magnitude of `want`."""
    return ((got.cpu().double() - want).abs().max() / want.abs().max()).item()


def compare(shape, seed=0):
    """Errors against float64 after each of the steps (the state carried) of the
    program on tt and of transformers' step on the CPU: {name: [per step]}."""
    n, hk, hv, dk, dv, steps = shape
    state, per_step, a_log, dt_bias = inputs(shape, seed)
    dev = kohakutt.device(0)
    op = torch.ops.kohakutt_example.gdn_decode_km
    tt_state = state.to(dev)
    params = (a_log.to(dev), dt_bias.to(dev))
    want, hf = state.double(), state.clone()
    out = {key: [] for key in ("o", "state", "hf o", "hf state")}
    for qkv, a, b in per_step:
        got_o = op(tt_state, qkv.to(dev), a.to(dev), b.to(dev), *params)
        want, want_o = reference(want, qkv, a, b, a_log, dt_bias, hk)
        hf, hf_o = hf_step(hf, qkv, a, b, a_log, dt_bias, hk)
        out["o"].append(relative(got_o, want_o))
        out["state"].append(relative(tt_state, want))
        out["hf o"].append(relative(hf_o, want_o))
        out["hf state"].append(relative(hf, want))
    return out


def timed(fn, reps=20):
    """Mean seconds per call of `fn` after one warm-up call."""
    fn()
    kohakutt.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    kohakutt.synchronize()
    return (time.perf_counter() - start) / reps


def main():
    """Runs the step at one shape; prints the errors per step and the time per call."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", type=int, default=1)
    parser.add_argument("--heads", type=int, nargs=2, default=(16, 32))
    parser.add_argument("--dims", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    shape = (args.seqs, *args.heads, *args.dims, args.steps)
    try:
        print(f"gdn_decode_km B, HK, HV, K, V = {shape[:5]}")
        for key, errors in compare(shape).items():
            print(f"  {key} error per step: {', '.join(f'{e:.3g}' for e in errors)}")
        state, ((qkv, a, b), *_), a_log, dt_bias = inputs(shape)
        dev = kohakutt.device(0)
        tensors = [x.to(dev) for x in (state, qkv, a, b, a_log, dt_bias)]
        op = torch.ops.kohakutt_example.gdn_decode_km
        print(f"  time per call: {timed(lambda: op(*tensors)) * 1e6:.1f} us")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
