"""The GDN decode step of the metal level on the CPU
(examples/kernels/dsl_metal/gdn_decode_km.py): it verifies with bf16 and fp32 gate
parameters, stays on the vector unit in fp32, and its launch values follow the shapes.
"""

import struct

import pytest
from km_programs import MACHINE, example

import kohakutt.dsl.metal as km


def _program(gates):
    dtypes = {"state": "float32", "qkv": "bfloat16", "a": "bfloat16", "b": "bfloat16"}
    dtypes.update({"a_log": gates, "dt_bias": gates, "o": "bfloat16"})
    return example("gdn_decode_km").gdn_decode_program.trace(MACHINE, dtypes, kt=4)


@pytest.mark.parametrize("gates", ["bfloat16", "float32"])
def test_verifies_within_l1(gates):
    program = _program(gates)
    km.verify(program)
    assert program.footprint() <= 96 * 1024


def test_state_stays_fp32_on_the_vector_unit():
    program = _program("bfloat16")
    compute = km.source(program, "compute")
    # The state and the fp32 intermediates unpack to fp32 registers; no matmul.
    assert "matmul" not in compute and "reduce_tile" not in compute
    assert "copy_tile(cb_state, 3, 0);" in compute
    assert "transpose_dest<DST_ACCUM_MODE>(0);" in compute
    writer = km.source(program, "dm1")
    assert (
        "noc_async_write_page((((((seq * hv) + head) * 4) + 3) * vt) + vb, state"
        in (writer)
    )


def test_launch_values():
    import torch

    gdn = example("gdn_decode_km").gdn_decode
    for hv, ct in ((32, 256), (48, 320)):
        state = torch.empty(2, hv, 128, 128, device="meta")
        qkv = torch.empty(2, 1, ct * 32, dtype=torch.bfloat16, device="meta")
        gate = torch.empty(2, 1, hv, dtype=torch.bfloat16, device="meta")
        param = torch.empty(hv, dtype=torch.bfloat16, device="meta")
        call = gdn(state, qkv, gate, gate, param, param)
        values = call.runtime
        assert values["units"] == 2 * hv * 4 and values["group"] == hv // 16
        assert values["ct"] == ct and values["koff"] == 64 and values["voff"] == 128
        assert values["gt"] == -(-hv // 32) and call.static == {"kt": 4}
        scale = struct.unpack("<f", struct.pack("<I", values["scale"]))[0]
        assert scale == pytest.approx(128**-0.5, rel=1e-7)
        assert call.outputs["o"] == ([2, 1, hv * 128], torch.bfloat16)
