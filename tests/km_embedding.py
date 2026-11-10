"""The backend's row gather (csrc/aten/index/ktt_embedding) written at the metal
level, for tests: out [B, n, D] = table [B, V, D] rows at int32 indices (torch op
``kohakutt_test::embedding_km``)."""

import math

import kohakutt.dsl.metal as km

# bf16 half rows are copied from their 64-byte staging slots into the output tile.
STAGED_ROWS = """
const auto* id_words = reinterpret_cast<volatile uint32_t*>(ids);
for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t r = id_words[(row >> 4) * 256 + (row & 15)] % 32;
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t src = (((r >> 4) * 2 + half) * 256 + (r & 15) * 16) * 2;
        const uint32_t dst = (((row >> 4) * 2 + half) * 256 + (row & 15) * 16) * 2;
        const auto* from = reinterpret_cast<volatile uint32_t*>(
            stage + (row * 2 + half) * 64 + (src & 63));
        auto* to = reinterpret_cast<volatile uint32_t*>(out_tile + dst);
        for (uint32_t i = 0; i < 8; ++i) {
            to[i] = from[i];
        }
    }
}
"""


def half_row(row, half):
    """Element offset of columns half * 16 .. + 15 of row `row` in a 32x32 tile
    stored as four 16x16 faces."""
    return ((row >> 4) * 2 + half) * 256 + (row & 15) * 16


@km.program("embedding")
def embedding_program(p, table: km.TensorRef, index: km.TensorRef, out: km.Out):
    """One unit is an output tile: 32 rows of one column tile of one table slice.
    Runtime values: `units`, the row count `n` and its `blocks` of 32, the column
    tiles `dt` and row tiles `vt` of a slice."""
    n, dt, vt = p.runtime("n"), p.runtime("dt"), p.runtime("vt")
    blocks = p.runtime("blocks")
    units = p.runtime("units")
    first, step = p.stride(units)
    elem = table.page_bytes // 1024
    cb_index = p.cb(0, like=index, pages=1)
    cb_out = p.cb(1, like=out, pages=1)
    # bf16 only: 64 slots of 64 bytes, one per half row.
    cb_stage = p.cb(2, dtype="float32", pages=1, name="stage") if elem == 2 else None

    @p.dm0
    def gather(k):
        ids, out_tile = cb_index.reserve(1), cb_out.reserve(1)
        stage = cb_stage.reserve(1) if cb_stage else None
        loaded = k.var("loaded", 0xFFFFFFFF)
        for u in k.range(units, start=first, step=step, name="u"):
            col = k.let("col", u % dt)
            block = k.let("block", (u // dt) % blocks)
            b = k.let("b", u // dt // blocks)
            with k.when(block.ne(loaded)):
                k.read(index.page(block), ids)
                k.reads_done()
                k.set(loaded, block)
            rows = k.let("rows", k.min(n - block * 32, 32))
            for i in k.range(rows, name="row"):
                word = (i >> 4) * 256 + (i & 15)
                # An index outside the table is not checked on the device.
                row_id = k.let("id", ids.load32(word).bounded(0, vt * 32))
                page = k.let("page", (b * vt + row_id // 32) * dt + col)
                r = k.let("r", row_id % 32)
                for h in k.range(2, name="half"):
                    src = k.let("src", half_row(r, h) * elem)
                    if elem == 4:
                        dst = out_tile.at(half_row(i, h) * elem)
                    else:
                        dst = stage.at((i * 2 + h) * 64, like=src)
                    k.read(table.span(page, src, 16 * elem), dst)
            k.reads_done()
            if elem == 2:
                k.inline(
                    STAGED_ROWS,
                    uses=[
                        km.CBUse(cb_out, written=cb_out.page_bytes),
                        km.CBUse(cb_stage),
                        km.CBUse(cb_index),
                    ],
                    args={
                        "rows": rows,
                        "ids": ids,
                        "stage": stage,
                        "out_tile": out_tile,
                    },
                )
            k.write(out_tile, out.page((b * blocks + block) * dt + col))
            k.writes_done()


def plan(table_shape, n):
    """The runtime values of embedding_program for a table of device shape
    `table_shape` ([..., V, D]) and `n` indices."""
    batch = math.prod(table_shape[:-2])
    dt, vt = math.ceil(table_shape[-1] / 32), math.ceil(table_shape[-2] / 32)
    blocks = math.ceil(n / 32)
    return {"n": n, "dt": dt, "vt": vt, "blocks": blocks, "units": batch * blocks * dt}


def register():
    """Registers the test op (once per process); returns its namespace."""
    import torch

    if "kohakutt_test::embedding_km" not in torch._C._dispatch_get_all_op_names():
        schema = "kohakutt_test::embedding_km(Tensor table, Tensor index) -> Tensor"

        @km.op(schema, embedding_program)
        def embedding(table, index):
            flat = index.dim() == 1 or (index.dim() == 2 and index.shape[0] == 1)
            if index.dtype != torch.int32 or not flat or table.dim() < 2:
                raise ValueError(
                    "embedding_km: the index is an int32 tensor of shape [n] or "
                    "[1, n] (stored as one row of int32 tiles), the table [..., V, D]"
                )
            n = index.numel()
            sizes = [*table.shape[:-2], n, table.shape[-1]]
            return km.Call(
                inputs={"table": table, "index": index},
                outputs={"out": (sizes, table.dtype)},
                runtime=plan(table.shape, n),
            )

    return torch.ops.kohakutt_test
