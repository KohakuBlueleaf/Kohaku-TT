"""A copy program x -> out whose three kernels each run their per-tile work in a named
device-profiler zone behind a watcher waypoint. Run as a script, it copies a tensor
and prints whether the copy is exact (for runs with TT_METAL_DEVICE_PROFILER=1)."""

import kohakutt.dsl.metal as km

ZONES = ("ktt-read", "ktt-copy", "ktt-write")


def zones_program(machine):
    """The program above; runtime value `tiles`."""
    p = km.Program("zones", machine)
    x, out = p.input("x", "bfloat16"), p.output("out", "bfloat16")
    start, count = p.split(p.runtime("tiles"))
    cb_x, cb_out = p.cb(0, like=x), p.cb(16, like=out)

    k = p.kernel("dm0")
    with k.loop(count) as i:
        k.waypoint("RD")
        slot = k.reserve(cb_x, 1)
        with k.zone(ZONES[0]):
            k.read(x.page(start + i), slot)
            k.reads_done()
        k.push(cb_x, 1)

    c = p.kernel("compute")
    c.startup(cb_x, None, cb_out)
    c.init("copy", cb_x)
    with c.loop(count):
        c.waypoint("CP")
        with c.zone(ZONES[1]):
            c.wait(cb_x, 1)
            c.acquire()
            c.copy(cb_x, 0, 0)
            c.reserve(cb_out, 1)
            c.pack(0, cb_out)
            c.release()
            c.push(cb_out, 1)
            c.pop(cb_x, 1)

    w = p.kernel("dm1")
    with w.loop(count) as i:
        w.waypoint("WR")
        with w.zone(ZONES[2]):
            w.wait(cb_out, 1)
            w.write(cb_out, out.page(start + i))
            w.flushed()
            w.pop(cb_out, 1)
    w.writes_done()
    return p


def main():
    """Copies a [64, 96] bf16 tensor through the program; prints `exact True|False`."""
    import torch

    import kohakutt

    dev = kohakutt.device(0)
    try:
        cols, rows = torch.ops.ktt.compute_grid()
        launcher = km.Launcher(zones_program(km.Machine(cols, rows)))
        x = torch.randn(64, 96).bfloat16()
        (out,) = launcher(
            {"x": x.to(dev)},
            outputs={"out": ([64, 96], torch.bfloat16)},
            runtime={"tiles": 6},
        )
        print("exact", torch.equal(out.cpu(), x), flush=True)
    finally:
        kohakutt.close()


def profiled(directory):
    """Runs main() under TT-Metal's device profiler writing to `directory` (set
    before TT-Metal loads); prints the program's zone names found in its log."""
    import csv
    import os
    from pathlib import Path

    os.environ.update(
        TT_METAL_DEVICE_PROFILER="1",
        TT_METAL_PROFILER_SYNC="1",
        TT_METAL_PROFILER_DIR=str(directory),
    )
    print("profiling into", directory, flush=True)
    main()
    log = Path(directory) / ".logs" / "profile_log_device.csv"
    with open(log, newline="") as f:
        next(f)
        names = {
            row["zone name"].strip() for row in csv.DictReader(f, skipinitialspace=True)
        }
    print("zones", sorted(set(ZONES) & names), "all", set(ZONES) <= names, flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--profiler":
        profiled(sys.argv[2])
    else:
        main()
