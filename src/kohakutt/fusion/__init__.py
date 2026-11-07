"""Fused pointwise chains: up to 4 inputs and a list of SFPU ops on DST slots, run as
one Metalium program (kernels/) through kohakutt.kernels as torch.ops.ktt_fusion.chain.

A chain spec is JSON:
    {"inputs": [layout, ...], "ops": [[op, dst, a, b], ...], "out": slot}
Input i is loaded into slot i; layouts: "full", "matrix", "row", "column", "scalar"
(see kernels/reader.cpp). Ops: binary "add" "sub" "mul" "div" (dst = a op b, slots);
scalar "adds" "subs" "rsubs" "muls" "divs" (dst = a op b, b a float); unary "exp" "tanh"
"sigmoid" "rsqrt" "sqrt" "recip" "neg" "abs" "silu" "square" (dst = op(a)); "copy"
(dst = a). Slots 0..3 (fp32 DST, half sync).
"""

import hashlib
import json
import math
import os
import struct
from pathlib import Path

import torch

from kohakutt import kernels as kk

KERNELS = Path(__file__).resolve().parent / "kernels"
# Generated chain sources: KOHAKUTT_CACHE, else beside the package.
GENERATED = Path(
    os.environ.get("KOHAKUTT_CACHE", Path(__file__).resolve().parent / "generated")
)
LAYOUTS = {"full": 0, "matrix": 1, "row": 2, "column": 3, "scalar": 4}
SLOTS = 4
BLOCK = 4
DTYPES = {15: torch.bfloat16, 6: torch.float32}
BINARY = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}
SCALAR = {"adds": "add", "subs": "sub", "rsubs": "rsub", "muls": "mul", "divs": "div"}
UNARY = {
    "exp": "exp",
    "tanh": "tanh",
    "sigmoid": "sigmoid",
    "rsqrt": "rsqrt",
    "sqrt": "sqrt",
    "recip": "recip",
    "neg": "negative",
    "abs": "abs",
    "silu": "silu",
    "square": "square",
}


def _bits(value):
    """The fp32 bit pattern of `value` as an unsigned integer literal."""
    return f"0x{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}u"


def body(ops):
    """FUSED_BODY: the SFPU calls of `ops` (each [op, dst, a, b])."""
    lines = []
    for op, dst, a, b in ops:
        if not all(0 <= s < SLOTS for s in (dst, a) if s is not None):
            raise ValueError(f"fusion: slot out of range in {op}")
        if op in BINARY:
            name = BINARY[op]
            lines.append(
                f"{name}_binary_tile_init(); {name}_binary_tile({a}, {b}, {dst});"
            )
            continue
        if a != dst:
            lines.append(f"copy_dest_values_init(); copy_dest_values({a}, {dst});")
        if op == "copy":
            continue
        if op in SCALAR:
            call = f"{SCALAR[op]}_unary_tile({dst}, {_bits(b)});"
            lines.append(f"binop_with_scalar_tile_init(); {call}")
        elif op in UNARY:
            name = UNARY[op]
            lines.append(f"{name}_tile_init(); {name}_tile({dst});")
        else:
            raise ValueError(f"fusion: unknown op {op}")
    return " ".join(lines)


def _compute_source(layouts, batched, out_slot, fused_body):
    """Path of the compute source for one chain: kernels/compute.cpp with NUM_INPUTS,
    LAYOUTS, BATCHED, OUT_SLOT and FUSED_BODY defined, written once under GENERATED
    (named by content)."""
    template = (KERNELS / "compute.cpp").read_text()

    def table(values):
        return "{" + ", ".join(str(v) for v in [*values, 0, 0, 0, 0][:4]) + "}"

    text = (
        f"#define NUM_INPUTS {len(layouts)}\n#define LAYOUTS {table(layouts)}\n"
        f"#define BATCHED {table(batched)}\n#define OUT_SLOT {out_slot}\n"
        f"#define FUSED_BODY {fused_body}\n{template}"
    )
    path = GENERATED / f"chain_{hashlib.sha1(text.encode()).hexdigest()[:16]}.cpp"
    if not path.exists():
        GENERATED.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(text)
        partial.replace(path)
    return path


@kk.op("ktt_fusion::chain(Tensor[] inputs, str spec, int[] sizes, int dtype) -> Tensor")
def chain(inputs, spec, sizes, dtype):
    """The program running chain `spec` over `inputs` into a new tensor of `sizes`
    (ScalarType code `dtype`: 15 bf16, 6 fp32)."""
    s = json.loads(spec)
    if not 1 <= len(inputs) <= SLOTS or len(s["inputs"]) != len(inputs):
        raise ValueError("fusion: 1 to 4 inputs, one layout each")
    out = kk.empty(sizes, DTYPES[dtype])
    rows = sizes[-2] if len(sizes) > 1 else 1
    rt, ct = math.ceil(rows / 32), math.ceil(sizes[-1] / 32)
    split = kk.split_tiles(kk.tile_count(out), min_per_core=BLOCK)
    per_input, layouts, batched = [], [], []
    for t, layout in zip(inputs, s["inputs"], strict=True):
        if t.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"fusion: {t.dtype} input")
        lead = math.prod(t.shape[:-2]) if t.dim() > 2 else 1
        layouts.append(LAYOUTS[layout])
        batched.append(int(lead > 1))
        per_input.append([t, layouts[-1], batched[-1]])
    unpack = [i for i, t in enumerate(inputs) if t.dtype == torch.float32]
    cbs = [
        kk.CB(i, t, tiles=2 if lay in (2, 4) and not bat else 2 * BLOCK)
        for i, (t, lay, bat) in enumerate(zip(inputs, layouts, batched, strict=True))
    ]
    cbs.append(kk.CB(16, out, tiles=2 * BLOCK))
    defines = {"NUM_INPUTS": len(inputs)}
    compute_source = _compute_source(layouts, batched, s["out"], body(s["ops"]))

    def reader(core):
        start, count = split[core]
        args = [start, count, BLOCK, rt, ct]
        for item in per_input:
            args += item
        return args

    return kk.Program(
        cbs=cbs,
        reader=kk.Kernel(KERNELS / "reader.cpp", inputs, reader, defines=defines),
        compute=kk.Kernel(
            compute_source,
            runtime_args=lambda c: [*split[c], ct],
            unpack_fp32=unpack,
        ),
        writer=kk.Kernel(KERNELS / "writer.cpp", [out], lambda c: [out, *split[c], ct]),
        outputs=[out],
        cores=split,
    )


def run(inputs, layouts, ops, out_slot, sizes, dtype):
    """Runs a chain on tt tensors; returns the output tensor."""
    spec = json.dumps({"inputs": layouts, "ops": ops, "out": out_slot})
    code = {torch.bfloat16: 15, torch.float32: 6}[dtype]
    return torch.ops.ktt_fusion.chain(list(inputs), spec, list(sizes), code)
