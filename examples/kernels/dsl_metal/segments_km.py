"""Packing ragged segments at the metal level: a job list per core (a km table, written
at every launch) copies tile ranges of x into out. Torch op
``kohakutt_example::segments_km``; prints its agreement with the backend's own cat.
Usage: python examples/kernels/dsl_metal/segments_km.py [--segments N]
"""

import argparse

import torch

import kohakutt
import kohakutt.dsl.metal as km

# Jobs per core, and the most tiles one job copies.
ROWS, MOST = 16, 4096
SRC, DST, TILES = 0, 1, 2


@km.program("segments")
def segments_program(p, x: km.TensorRef, out: km.Out):
    """Each core runs its jobs (source tile, destination tile, tile count): the reader
    reads the tiles, the compute copies them, the writer writes them in place."""
    jobs = p.table(
        "jobs",
        width=3,
        rows=ROWS,
        most={TILES: MOST},
        pages={SRC: (x, TILES), DST: (out, TILES)},
    )
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)

    k = p.kernel("dm0")
    with k.loop(jobs.used, name="j") as j, k.loop(jobs[j, TILES], name="t") as t:
        slot = k.reserve(cb_x, 1)
        k.read(x.page(jobs[j, SRC] + t), slot)
        k.reads_done()
        k.push(cb_x, 1)

    c = p.kernel("compute", unpack_fp32=[cb_x] if x.dtype == "float32" else [])
    c.startup(cb_x, None, cb_out)
    c.init("copy", cb_x)
    with c.loop(jobs.used, name="j") as j, c.loop(jobs[j, TILES], name="t"):
        c.wait(cb_x, 1)
        c.acquire()
        c.copy(cb_x, 0, 0)
        c.reserve(cb_out, 1)
        c.pack(0, cb_out)
        c.release()
        c.push(cb_out, 1)
        c.pop(cb_x, 1)

    w = p.kernel("dm1")
    with w.loop(jobs.used, name="j") as j, w.loop(jobs[j, TILES], name="t") as t:
        w.wait(cb_out, 1)
        w.write(cb_out, out.page(jobs[j, DST] + t))
        w.flushed()
        w.pop(cb_out, 1)
    w.writes_done()


def plan(segments, cores, rows_of_x):
    """Jobs per core for `segments` ((first tile row, tile rows) of x, packed in order)
    on `cores` cores of a tensor `rows_of_x` tile rows high whose rows are one tile
    wide: segments longer than MOST are cut; jobs are dealt out in turn."""
    if not segments:
        raise ValueError("segments_km: at least one segment")
    jobs, dst = [], 0
    for first, count in segments:
        if first < 0 or count < 1 or first + count > rows_of_x:
            raise ValueError(f"segments_km: segment {(first, count)} is outside x")
        while count:
            n = min(count, MOST)
            jobs.append([first, dst, n])
            first, dst, count = first + n, dst + n, count - n
    if len(jobs) > cores * ROWS:
        raise ValueError(f"segments_km: {len(jobs)} jobs, at most {cores * ROWS}")
    return [jobs[c::cores] for c in range(cores)], dst


DTYPES = {"x": "bfloat16", "out": "bfloat16"}
SCHEMA = "kohakutt_example::segments_km(Tensor x, int[] segments) -> Tensor"


@km.op(SCHEMA, segments_program)
def segments(x, flat):
    """out = cat(x[32 f : 32 (f + n)] for (f, n) in pairs of `flat`) for a tt tensor
    x [R, 32] (R a multiple of 32)."""
    if x.dim() != 2 or x.shape[1] != 32 or x.shape[0] % 32:
        raise ValueError("segments_km: x is [R, 32] with R a multiple of 32")
    cols, rows = torch.ops.ktt.compute_grid()
    pairs = list(zip(flat[::2], flat[1::2], strict=True))
    tables, tiles = plan(pairs, cols * rows, x.shape[0] // 32)
    return km.Call(
        inputs={"x": x},
        outputs={"out": ([tiles * 32, 32], x.dtype)},
        tables={"jobs": tables},
    )


def main():
    """Packs random segments with the program and with the backend's cat."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=300)
    args = parser.parse_args()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096 * 32, 32, generator=g).bfloat16()
    starts = torch.randint(0, 4000, (args.segments,), generator=g).tolist()
    sizes = torch.randint(1, 96, (args.segments,), generator=g).tolist()
    flat = [v for f, n in zip(starts, sizes, strict=True) for v in (f, n)]
    dev = kohakutt.device(0)
    try:
        xt = x.to(dev)
        got = torch.ops.kohakutt_example.segments_km(xt, flat).cpu()
        want = torch.cat(
            [xt[32 * f : 32 * (f + n)] for f, n in zip(starts, sizes)]
        ).cpu()
        same = torch.equal(got, want)
        print(f"segments_km {args.segments} segments: equal to cat: {same}")
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
