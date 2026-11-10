"""Generated vector-unit functions (`km.sfpu_fn`) on the CPU: recorded sources, what
an expression becomes, what is refused, and (where the kernel compiler is installed)
the instructions the compiler makes of them."""

import os
import re
import subprocess
from pathlib import Path

import pytest
from km_programs import MACHINE

import kohakutt.dsl.metal as km
from km_fused import cases, fused_program
from km_rebuild import rebuild

GOLDEN = Path(__file__).resolve().parent / "km_golden"
RECORDED = ("silu", "swiglu", "complex_mul", "chain6")
f = km.sfpu


def _program(case):
    inputs, _, body = cases()[case]
    row = 2 if case == "chain6" else None
    return fused_program(MACHINE, f"{case}_fn", inputs, body, row=row)


@pytest.mark.parametrize("case", RECORDED)
def test_source_equals_recorded(case):
    text = km.source(_program(case), "compute")
    assert text == (GOLDEN / f"fused_{case}_compute.cpp.txt").read_text()


@pytest.mark.parametrize("case", sorted(cases()))
def test_both_forms_verify_and_rebuild(case):
    inputs, ops, function = cases()[case]
    for tag, body in (("ops", ops), ("fn", function)):
        program = fused_program(MACHINE, f"{case}_{tag}", inputs, body)
        km.verify(program)
        built = rebuild(program)
        assert km.source(built, "compute") == km.source(program, "compute")


def test_a_chain_is_one_pass_over_the_registers():
    text = km.source(_program("chain6"), "compute")
    body = text[text.index("void kernel_main") :]
    assert body.count("_llk_math_eltwise_binary_sfpu_params_(chain6_fn, 0, 1, 0") == 1
    assert "VectorMode::RC, 2, 3)" in body
    assert "mul_binary_tile_init();" in body
    for call in ("add_binary_tile(", "mul_binary_tile(", "abs_tile(", "_unary_tile("):
        assert call not in body
    assert '#ifdef TRISC_MATH\n#include "ckernel_sfpu_conversions.h"' in text


def test_what_an_expression_becomes():
    a, b, c = f.operands(3)
    fn = km.sfpu_fn("fn", f.select(a < b, f.mad(a, b, c), a * b + c) - 1.5)
    assert (fn.operands, fn.init) == (3, "mul")
    code = fn.code
    assert "uint32_t a, uint32_t b, uint32_t out, uint32_t c)" in code
    assert code.count("sfpi::dst_reg[a * 32]") == 1
    assert "= __builtin_rvtt_sfpmad(t0.get(), t1.get(), t2.get(), 0);" in code
    # A product then a sum: the sum is written as 1 * product + c.
    assert (
        "__builtin_rvtt_sfpmad(sfpi::vFloat(1.0f).get(), t4.get(), t2.get(), 0)" in code
    )
    assert "v_if(t0 < t1) { t6 = t3; }" in code
    assert "0x1.8000000000000p+0f" in code
    assert "sfpi::dst_reg[out * 32] = " in code and "sfpi::dst_reg++;" in code
    pair = km.sfpu_fn("pair", {"even": a.odd, "odd": -a.even})
    assert "sfpi::dst_reg[a * 32 + 1]" in pair.code
    assert "sfpi::dst_reg[out * 32 + 1] = " in pair.code
    assert "sfpi::dst_reg += 2;" in pair.code and "rows < 4" in pair.code


def test_library_functions_name_their_setup():
    a, b = f.operands(2)
    assert km.sfpu_fn("fn", f.silu(a) * f.recip(b)).init == "silu"
    assert km.sfpu_fn("fn", f.exp(a) + b).init == "mul"
    assert km.sfpu_fn("fn", f.rsqrt(a)).init == "rsqrt"
    fn = km.sfpu_fn("fn", f.tanh(a))
    assert fn.init == "tanh" and "ckernel_sfpu_tanh.h" in fn.headers


def test_what_is_refused():
    a, b = f.operands(2)
    with pytest.raises(ValueError, match="different setups"):
        km.sfpu_fn("fn", f.sigmoid(a) + f.tanh(b))
    with pytest.raises(ValueError, match="operands 0 .. n - 1"):
        km.sfpu_fn("fn", b * 2.0)
    with pytest.raises(ValueError, match="operands 0 .. 3"):
        f.operand(4)
    with pytest.raises(ValueError, match="reads operands by .even / .odd"):
        km.sfpu_fn("fn", a.even + b)
    with pytest.raises(ValueError, match="reads operands by .even / .odd"):
        km.sfpu_fn("fn", {"even": a, "odd": b})
    with pytest.raises(ValueError, match="finite"):
        km.sfpu_fn("fn", a * float("inf"))
    with pytest.raises(TypeError, match="comparison"):
        f.select(a, a, b)
    with pytest.raises(TypeError, match="not a vector value"):
        a + "1"
    program = km.Program("p", MACHINE)
    with pytest.raises(ValueError, match="takes 2 operand slots"):
        program.kernel("compute").run(km.sfpu_fn("fn", a + b), 0, out=0)
    half = km.Program("q", MACHINE).kernel("compute", fp32_dest=False)
    with pytest.raises(ValueError, match="needs fp32 registers"):
        half.run(km.sfpu_fn("fn", f.tanh(a)), 0, out=0)


PRELUDE = "\n".join(
    [
        "namespace ckernel { extern volatile unsigned long* instrn_buffer; }",
        "#include <cstdint>",
        "#include <sfpi.h>",
        "#define TRISC_MATH",
        "constexpr bool DST_ACCUM_MODE = true;",
        "inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat x) { return x; }",
        "",
    ]
)
FLAGS = (
    "-mcpu=tt-bh-tensix -O3 -std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval "
    "-ffast-math -fno-finite-math-only -fsigned-zeros -fno-associative-math "
    "-fno-exceptions -fno-rtti"
)


def _instructions(fn, tmp_path):
    """The vector-unit instructions ("name operands") the kernel compiler makes of
    generated function `fn`, which calls no library function."""
    root = Path(os.environ.get("TT_METAL_HOME", "/nonexistent")) / "runtime/sfpi"
    compiler = root / "compiler/bin/riscv-tt-elf-g++"
    if not compiler.exists():
        pytest.skip("the kernel compiler is not installed")
    source = tmp_path / f"{fn.name}.cpp"
    source.write_text(PRELUDE + fn.code.replace("inline void", "void") + "\n")
    obj = source.with_suffix(".o")
    command = [
        compiler,
        *FLAGS.split(),
        f"-I{root / 'include'}",
        "-c",
        source,
        "-o",
        obj,
    ]
    subprocess.run(command, check=True, capture_output=True)
    dump = subprocess.run(
        [compiler.with_name("riscv-tt-elf-objdump"), "-d", obj],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return re.findall(r"\t(sfp\w+(?:\t\S+)?)", dump)


def test_complex_multiply_compiles_to_the_hand_written_instructions(tmp_path):
    fn = cases()["complex_mul"][2].fn
    got = [
        i
        for i in _instructions(fn, tmp_path)
        if not i.startswith(("sfpload", "sfpstore"))
    ]
    names = [i.split("\t")[0] for i in got]
    assert names[:6] == ["sfpmul", "sfpmul", "sfpmad", "sfpmul", "sfpmul", "sfpmad"]
    # Each sum is 1 * product + product (LReg 10 holds 1): no product is fused.
    sums = [i for i in got if i.startswith("sfpmad")]
    assert sums and all(i.split("\t")[1].split(",")[1] == "L10" for i in sums)
    assert (
        sum(i.endswith(",1") for i in got if i.startswith("sfpmul")) == len(sums) // 2
    )


def test_a_multiply_add_is_spaced_from_a_min(tmp_path):
    a, b, c = f.operands(3)
    fn = km.sfpu_fn("spaced", f.minimum(f.mad(a, b, c), b))
    names = [i.split("\t")[0] for i in _instructions(fn, tmp_path)]
    assert "sfpmad" in names and "sfpswap" in names
    for first, second in zip(names, names[1:], strict=False):
        assert (first, second) != ("sfpmad", "sfpswap")
    assert names[names.index("sfpmad") + 1] == "sfpnop"
