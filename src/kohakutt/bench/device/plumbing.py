"""Experiment: ecosystem plumbing on tt (print, save/load, pickle, module.to, errors,
opt-in CPU fallback).
    python -m kohakutt.bench.device.plumbing [--fallback]
"""

import argparse
import io
import json
import os
import pickle

import torch

import kohakutt


class Block(torch.nn.Module):
    """Linear + a buffer, to check that module.to and state_dict carry both."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32).bfloat16()
        self.register_buffer("scale", torch.full((32,), 2.0).bfloat16())

    def forward(self, x):
        return self.linear(x) * self.scale


def checks():
    """(name, fn) pairs; fn returns True when tt behaves like a torch device should."""
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(8, 16, generator=gen).bfloat16()

    def printing():
        t = a.to("tt")
        text = repr(t)
        return "device='tt:0'" in text and str(t) == text

    def save_load():
        buf = io.BytesIO()
        torch.save({"t": a.to("tt"), "v": a.to("tt")[2:5]}, buf)
        buf.seek(0)
        loaded = torch.load(buf)
        return (
            loaded["t"].device.type == "tt"
            and torch.equal(loaded["t"].cpu(), a)
            and torch.equal(loaded["v"].cpu(), a[2:5])
        )

    def state_dict():
        block = Block().to("tt")
        buf = io.BytesIO()
        torch.save(block.state_dict(), buf)
        buf.seek(0)
        fresh = Block()
        fresh.load_state_dict(torch.load(buf, map_location="cpu"))
        return block.scale.device.type == "tt" and torch.equal(
            fresh.linear.weight, block.linear.weight.cpu()
        )

    def pickling():
        return torch.equal(pickle.loads(pickle.dumps(a.to("tt"))).cpu(), a)

    def numpy_error():
        try:
            a.to("tt").numpy()
        except (TypeError, RuntimeError) as error:
            return "cpu" in str(error).lower()
        return False

    def unsupported_error():
        try:
            torch.igamma(a.abs().to("tt") + 0.1, a.abs().to("tt"))
        except NotImplementedError as error:
            return kohakutt.UNSUPPORTED in str(error) and "igamma" in str(error)
        return os.environ.get("KTT_CPU_FALLBACK") == "1"

    return [
        ("print", printing),
        ("save/load", save_load),
        ("state_dict", state_dict),
        ("pickle", pickling),
        ("numpy error", numpy_error),
        ("unsupported error", unsupported_error),
    ]


def fallback_checks():
    """Runs only with KTT_CPU_FALLBACK=1: a missing op runs on CPU, counted."""
    a = torch.randn(8, 16) * 4
    got = torch.special.bessel_j0(a.to("tt")).cpu()
    counts = kohakutt.fallback_counts()
    return {
        "bessel_j0 matches": bool(torch.equal(got, torch.special.bessel_j0(a))),
        "counted": counts,
    }


def main():
    """Runs the checks on device 0 and prints one JSON record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    kohakutt.open(0)
    results = {}
    try:
        if args.fallback:
            results = fallback_checks()
        else:
            for name, fn in checks():
                try:
                    results[name] = "ok" if fn() else "wrong result"
                except Exception as e:  # noqa: BLE001  (reported per check)
                    results[name] = (
                        f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                    )
    finally:
        kohakutt.close()
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
