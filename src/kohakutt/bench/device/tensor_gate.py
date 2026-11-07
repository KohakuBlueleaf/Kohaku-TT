"""Experiment: the TensorImpl gate checks on the current native build.

Runs the gate list (empty, to cpu/tt, detach, clone, logical view, transpose,
in-place through an alias, backward through mm/sum); prints ok or the failure per check.
    python -m kohakutt.bench.device.tensor_gate
"""

import json

import torch

import kohakutt


def checks():
    """(name, fn) pairs; fn returns True when the tt result matches CPU semantics."""
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(64, 64, generator=gen).bfloat16()
    b = torch.randn(64, 64, generator=gen).bfloat16()

    def empty():
        t = torch.empty(64, 64, device="tt", dtype=torch.bfloat16)
        return t.device.type == "tt" and tuple(t.shape) == (64, 64)

    def roundtrip():
        return torch.equal(a.to("tt").to("cpu"), a)

    def detach():
        return torch.equal(a.to("tt").detach().cpu(), a)

    def clone():
        return torch.equal(a.to("tt").clone().cpu(), a)

    def view():
        return torch.equal(a.to("tt").view(32, 128).cpu(), a.view(32, 128))

    def transpose():
        return torch.equal(a.to("tt").t().cpu(), a.t())

    def alias_inplace():
        x = a.to("tt")
        v = x.view(-1)
        x.add_(b.to("tt"))
        written = x.cpu()
        return torch.equal(v.cpu(), written.view(-1)) and not torch.equal(written, a)

    def backward():
        x = a.to("tt").requires_grad_()
        (x @ b.to("tt")).sum().backward()
        ref_x = a.double().requires_grad_()
        (ref_x @ b.double()).sum().backward()
        err = (x.grad.cpu().double() - ref_x.grad).abs().max().item()
        return err < 0.05 * ref_x.grad.abs().max().item()

    def dtypes():
        ints = torch.randint(-1000, 1000, (40, 5), generator=gen)
        mask = torch.rand(7, 33, generator=gen) > 0.5
        filled = torch.full((3, 4), 7, dtype=torch.long, device="tt").cpu()
        return (
            torch.equal(ints.to("tt").cpu(), ints)
            and torch.equal(ints.int().to("tt").cpu(), ints.int())
            and torch.equal(mask.to("tt").cpu(), mask)
            and filled.dtype == torch.long
            and torch.equal(filled, torch.full((3, 4), 7))
        )

    return [
        ("dtypes int64/int32/bool + full", dtypes),
        ("empty", empty),
        ("to/from cpu", roundtrip),
        ("detach", detach),
        ("clone", clone),
        ("view", view),
        ("transpose", transpose),
        ("in-place via alias", alias_inplace),
        ("backward mm/sum", backward),
    ]


def main():
    """Runs every check on device 0 and prints one JSON record."""
    kohakutt.open(0)
    results = {}
    try:
        for name, fn in checks():
            try:
                results[name] = "ok" if fn() else "wrong result"
            except Exception as error:  # noqa: BLE001  (each check reports its failure)
                results[name] = (
                    f"{type(error).__name__}: {str(error).splitlines()[0][:120]}"
                )
    finally:
        kohakutt.close()
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
