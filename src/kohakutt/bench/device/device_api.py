"""Experiment: torch.tt device semantics on a 1x1 or 1x2 mesh bound as tt:0.

Checks the torch.tt API, seeded random ops, memory statistics, and add/mm on the mesh.
    python -m kohakutt.bench.device.device_api [--mesh 1x2]
"""

import argparse
import json

import torch

import kohakutt


def checks():
    """(name, fn) pairs; fn returns True when tt behaves like a torch device should."""

    def api():
        return (
            torch.tt.is_available()
            and torch.tt.device_count() == 1
            and torch.tt.current_device() == 0
            and torch.tt.is_initialized()
            and torch.device("tt").type == "tt"
        )

    def seeded_random():
        torch.manual_seed(7)
        a = torch.randn(64, 32, device="tt").cpu()
        torch.manual_seed(7)
        b = torch.randn(64, 32, device="tt").cpu()
        c = torch.rand(64, 32, device="tt").cpu()
        return (
            torch.equal(a, b) and not torch.equal(b, c) and 0 <= c.min() < c.max() <= 1
        )

    def rng_state():
        state = torch.tt.get_rng_state()
        a = torch.randn(16, device="tt").cpu()
        torch.tt.set_rng_state(state)
        return torch.equal(a, torch.randn(16, device="tt").cpu())

    def memory():
        # The first zeros of a shape also fills the constant cache.
        torch.zeros(256, 256, device="tt")
        torch.tt.synchronize()
        before = torch.tt.memory_allocated()
        x = torch.zeros(256, 256, device="tt")
        during = torch.tt.memory_allocated()
        del x
        after = torch.tt.memory_allocated()
        return during >= before + 256 * 256 * 4 // 2 and after == before

    def compute():
        a = torch.randn(64, 64).bfloat16()
        b = torch.randn(64, 64).bfloat16()
        got = (a.to("tt") @ b.to("tt") + a.to("tt")).cpu().double()
        ref = a.double() @ b.double() + a.double()
        return (got - ref).abs().max().item() < 0.1 * ref.abs().max().item()

    return [
        ("torch.tt api", api),
        ("seeded randn/rand", seeded_random),
        ("rng state", rng_state),
        ("memory stats", memory),
        ("add + mm", compute),
    ]


def main():
    """Opens the mesh, runs every check and prints one JSON record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="1x1")
    args = parser.parse_args()
    rows, cols = (int(v) for v in args.mesh.split("x"))

    kohakutt.mesh(list(range(rows * cols)), shape=(rows, cols))
    results = {"mesh": args.mesh, "chips": torch.tt.mesh_size()}
    try:
        for name, fn in checks():
            try:
                results[name] = "ok" if fn() else "wrong result"
            except Exception as error:  # noqa: BLE001  (each check reports its failure)
                results[name] = (
                    f"{type(error).__name__}: {str(error).splitlines()[0][:140]}"
                )
    finally:
        kohakutt.close()
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
