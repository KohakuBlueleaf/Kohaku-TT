"""Models on 2 cards (1x2 mesh, automatic placement) against 1 card: first-call and
warm forward time, the 2-card / 1-card ratio, and the chips' AICLK (sysfs telemetry,
sampled every 50 ms during the warm runs). Models: Qwen2.5-0.5B (pretrained, HF_HOME)
and a Llama-shaped decoder from a config with random weights (no download).

    python experiments/mesh/mesh_bench.py
    python experiments/mesh/mesh_bench.py --seqs 512 2048 --layers 4 --repeat 5
"""

import argparse
import contextlib
import statistics
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

import kohakutt


class Aiclk:
    """AICLK samples (MHz) of the given chips while active."""

    def __init__(self, chips):
        self.paths = [
            Path(f"/sys/class/tenstorrent/tenstorrent!{c}/tt_aiclk") for c in chips
        ]
        self.samples = {c: [] for c in chips}
        self._stop = threading.Event()

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                for c, p in zip(self.samples, self.paths, strict=True):
                    with contextlib.suppress(OSError, ValueError):
                        self.samples[c].append(int(p.read_text().split()[0]))
                time.sleep(0.05)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        parts = []
        for c, s in self.samples.items():
            if s:
                parts.append(
                    f"chip {c} {min(s)}-{max(s)} MHz (median {statistics.median(s)})"
                )
        return ", ".join(parts) or "no samples"


def build(name, layers):
    if name == "qwen":
        return AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16
        )
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=layers,
        vocab_size=32000,
        max_position_embeddings=4096,
    )
    return LlamaForCausalLM(config).to(torch.bfloat16)


def inputs(seq):
    return torch.randint(0, 1000, (1, seq), generator=torch.Generator().manual_seed(1))


def run(name, layers, seq, repeat, dev, chips):
    model = build(name, layers).eval().to(dev)
    ids = inputs(seq).to(dev)
    with torch.no_grad():
        start = time.perf_counter()
        model(ids)
        torch.tt.synchronize()
        first = (time.perf_counter() - start) * 1e3
        for _ in range(2):
            model(ids)
        torch.tt.synchronize()
        with Aiclk(chips) as clk:
            start = time.perf_counter()
            for _ in range(repeat):
                logits = model(ids).logits
            torch.tt.synchronize()
            warm = (time.perf_counter() - start) * 1e3 / repeat
        last = logits[0, -1].cpu().double()
    del model
    return first, warm, clk.summary(), last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup", type=int, choices=(1, 2), help="cards; one per process"
    )
    parser.add_argument("--compare", action="store_true", help="ratios and statistics")
    parser.add_argument("--out", default="mesh_bench", help="results directory")
    parser.add_argument("--seqs", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [("qwen", 128)] + [("llama", s) for s in args.seqs]
    if args.compare:
        for name, seq in cases:
            one = torch.load(out / f"1_{name}_{seq}.pt")
            two = torch.load(out / f"2_{name}_{seq}.pt")
            cpu = build(name, args.layers).eval()
            with torch.no_grad():
                base = cpu(inputs(seq)).logits[0, -1].double()
                ruler = cpu.double()(inputs(seq)).logits[0, -1]
            del cpu
            print(
                f"{name} seq {seq}: warm 1 card {one['warm']:.1f} ms / 2 cards "
                f"{two['warm']:.1f} ms = {one['warm'] / two['warm']:.2f}x; first call "
                f"{one['first']:.0f} / {two['first']:.0f} ms"
            )
            for label, t in (
                ("1 card", one["last"]),
                ("2 cards", two["last"]),
                ("cpu", base),
            ):
                s = kohakutt.format_stats(kohakutt.diff_stats(t, ruler))
                print(f"    {label:7s} {s}")
        return
    chips = [0] if args.setup == 1 else [0, 1]
    dev = kohakutt.device(0) if args.setup == 1 else kohakutt.mesh(chips, shape=(1, 2))
    try:
        for name, seq in cases:
            first, warm, clk, last = run(
                name, args.layers, seq, args.repeat, dev, chips
            )
            torch.save(
                {"first": first, "warm": warm, "last": last},
                out / f"{args.setup}_{name}_{seq}.pt",
            )
            print(
                f"{args.setup} card(s) {name} seq {seq}: first call {first:.0f} ms, "
                f"warm "
                f"{warm:.1f} ms; AICLK {clk}",
                flush=True,
            )
    finally:
        kohakutt.close()


if __name__ == "__main__":
    main()
