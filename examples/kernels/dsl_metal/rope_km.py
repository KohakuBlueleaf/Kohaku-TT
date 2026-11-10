"""Rotary position embedding written at the metal level (`kohakutt.dsl.metal`): the
program of rope.py as a model, with generated kernel sources (torch op
``kohakutt_example::rope_km``). Prints whether it equals rope.py and both times.
Usage: python examples/kernels/dsl_metal/rope_km.py (the arguments of rope.py)
"""

import argparse
import importlib.util
import time
from pathlib import Path

import torch

import kohakutt
import kohakutt.dsl.metal as km
import kohakutt.kernels as kk

# Tiles per NoC barrier.
BLOCK = 4

# One face (16 register rows) of tile `x` rotated by tile `table` into tile `out`.
ROTATE_FACE = km.SfpuFn(
    "rotate_face",
    """
    // An SFPU load or store moves the even or the odd columns of four register rows.
    inline void rotate_face(uint32_t x, uint32_t table, uint32_t out) {
        using namespace ckernel;
        constexpr uint32_t x0 = p_sfpu::LREG0, x1 = p_sfpu::LREG1;
        constexpr uint32_t c = p_sfpu::LREG2, s = p_sfpu::LREG3;
        constexpr uint32_t t0 = p_sfpu::LREG4, t1 = p_sfpu::LREG5;
        constexpr uint32_t t2 = p_sfpu::LREG6, t3 = p_sfpu::LREG7;
        constexpr uint32_t zero = p_sfpu::LCONST_0, one = p_sfpu::LCONST_1;
        // A tile is 64 register rows; +2 selects the odd columns.
        const uint32_t xa = x * 64, ta = table * 64, oa = out * 64;
        for (uint32_t rows = 0; rows < 4; ++rows) {
            TT_SFPLOAD(x0, 0, ADDR_MOD_7, xa);
            TT_SFPLOAD(x1, 0, ADDR_MOD_7, xa + 2);
            TT_SFPLOAD(c, 0, ADDR_MOD_7, ta);
            TT_SFPLOAD(s, 0, ADDR_MOD_7, ta + 2);
            // even = x0 c - x1 s, odd = x0 s + x1 c (SFPMUL's modifier 1 negates).
            TTI_SFPMUL(x0, c, zero, t0, 0);
            TTI_SFPMUL(x1, s, zero, t1, 1);
            TTI_SFPMUL(x0, s, zero, t2, 0);
            TTI_SFPMUL(x1, c, zero, t3, 0);
            TTI_SFPADD(one, t0, t1, t0, 0);
            TTI_SFPADD(one, t2, t3, t2, 0);
            TT_SFPSTORE(t0, 0, ADDR_MOD_7, oa);
            TT_SFPSTORE(t2, 0, ADDR_MOD_7, oa + 2);
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
    """,
)


@km.program("rope")
def rope_program(
    p, x: km.TensorRef, table: km.TensorRef, out: km.Out, block: km.Static[int] = BLOCK
):
    """Rotates x (bf16) by the (cos, sin) table (fp32). Runtime values: `tiles` of x
    and `period`, the table's tile count (it repeats over x's leading dims)."""
    period = p.runtime("period")
    start, count = p.split(p.runtime("tiles"))
    cb_x = p.cb(0, like=x, pages=2 * block)
    cb_table = p.cb(1, like=table, pages=2 * block)
    cb_out = p.cb(16, like=out, pages=2)

    @p.dm0
    def reader(k):
        for done, n in k.blocks(count, block):
            x_slot, table_slot = cb_x.reserve(block), cb_table.reserve(block)
            for i in k.range(n):
                t = k.let("t", start + done + i)
                k.read(x.page(t), x_slot[i])
                k.read(table.page(t % period), table_slot[i])
            k.reads_done()
            cb_x.push(block)
            cb_table.push(block)

    @p.compute(fp32_dest=True, unpack_fp32=[cb_table])
    def compute(c):
        c.startup(cb_x, cb_table, cb_out)
        c.init("mul")
        for _, n in c.blocks(count, block):
            cb_x.wait(block)
            cb_table.wait(block)
            for i in c.range(n):
                with c.registers():
                    # x in slot 0 (bf16 held exactly in fp32), the table tile in slot 1.
                    c.srca_format(cb_x)
                    c.init("copy", cb_x)
                    c.copy(cb_x, i, 0)
                    c.srca_format(cb_table)
                    c.init("copy", cb_table)
                    c.copy(cb_table, i, 1)
                    c.sfpu_binary(ROTATE_FACE, 0, 1, 0)
                    cb_out.reserve(1)
                    c.pack(0, cb_out)
                    cb_out.push(1)
            cb_x.pop(block)
            cb_table.pop(block)

    @p.dm1
    def writer(k):
        for i in k.range(count):
            cb_out.wait(1)
            k.write(cb_out, out.page(start + i))
            k.flushed()
            cb_out.pop(1)
        k.writes_done()


@km.op("kohakutt_example::rope_km(Tensor x, Tensor table) -> Tensor", rope_program)
def rope(x, table):
    """The launch rotating x [..., S, D] (bf16, D even) by table [S, D] (fp32, cos at
    even columns and sin at odd ones)."""
    if x.dtype != torch.bfloat16 or table.dtype != torch.float32:
        raise TypeError("rope: x must be bfloat16 and the table float32")
    if tuple(table.shape) != tuple(x.shape[-2:]) or x.shape[-1] % 2:
        raise ValueError("rope: the table must be [S, D] of x [..., S, D] with D even")
    return km.Call(
        inputs={"x": x, "table": table},
        outputs={"out": (list(x.shape), x.dtype)},
        runtime={"tiles": kk.tile_count(x), "period": kk.tile_count(table)},
    )


def hand_written():
    """The torch op of rope.py (the same program with hand-written kernels)."""
    path = Path(__file__).parents[1] / "metalium" / "rope.py"
    spec = importlib.util.spec_from_file_location("rope_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rope.op


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
    """Runs both programs on tt tensors; prints their agreement and times."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=9)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=576)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.batch, args.heads, args.tokens, args.dim).bfloat16()
    angles = torch.rand(args.tokens, args.dim // 2) * 6.2831853
    freqs = torch.polar(torch.ones_like(angles), angles)
    table = torch.view_as_real(freqs).reshape(args.tokens, args.dim).contiguous()

    dev = kohakutt.device(0)
    try:
        x_tt, table_tt = x.to(dev), table.to(dev)
        reference = hand_written()
        got = torch.ops.kohakutt_example.rope_km(x_tt, table_tt).cpu()
        want = reference(x_tt, table_tt).cpu()
        print(f"rope_km {tuple(x.shape)}: equal to rope.py: {torch.equal(got, want)}")
        own = timed(lambda: torch.ops.kohakutt_example.rope_km(x_tt, table_tt))
        hand = timed(lambda: reference(x_tt, table_tt))
        print(f"  time: metal level {own * 1e3:.3f} ms, rope.py {hand * 1e3:.3f} ms")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
