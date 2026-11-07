"""MOSS-Image text-to-image on tt: Qwen3.5 text encoder -> DiT flow sampler (CFG) ->
FLUX.2 VAE decoder. Each model loads once: the text encoder encodes every prompt in one
batch and is freed, then the DiT and the VAE load; each prompt's (cfg, seed) images
sample as one batch, and each image is decoded and saved (PNG, latent, settings) as
soon as its batch is done.

    python examples/06_moss_image/generate.py --model MOSS-Image-Base --size 1024
    python examples/06_moss_image/generate.py --model MOSS-Image-Base \
        --prompts prompts/ --cfgs 2 3 4.5 --seeds 0 1 --out-dir images/
"""

import argparse
import gc
import itertools
import json
import re
import time
from pathlib import Path

import torch
from dit import load_dit
from PIL import Image
from text_encoder import load_text_encoder
from transformers import AutoTokenizer
from vae import load_vae

import kohakutt

SIZE_LINE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")
CANVAS = re.compile(r"canvas=(\d+)x(\d+)")


def sigmas(steps, shift=5.0, total=1000):
    """The MOSS-Image schedule: shifted sigmas snapped to the training grid, then 0."""

    def grid(count, s):
        values = torch.linspace(1, s / (total + s - 1), count)
        return s * values / (1 + (s - 1) * values)

    wanted = grid(steps, shift)
    training = grid(total, 1.0).flip(0)
    upper = torch.searchsorted(training, wanted).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    near_lower = (wanted - training[lower]).abs() <= (training[upper] - wanted).abs()
    snapped = torch.unique_consecutive(training[torch.where(near_lower, lower, upper)])
    return snapped.tolist() + [0.0]


def read_prompt(path, default):
    """(prompt, (width, height)) of a prompt file: a first line "WxH" is the size and
    is dropped; else a "canvas=WxH" in the prompt; else `default`."""
    text = Path(path).read_text().strip()
    first, _, rest = text.partition("\n")
    match = SIZE_LINE.match(first)
    if match:
        return rest.strip(), (int(match[1]), int(match[2]))
    match = CANVAS.search(text)
    if match:
        return text, (int(match[1]), int(match[2]))
    return text, default


def encode(tokenizer, model, prompts, device):
    """Final hidden states [1, T_i, 4096] of each chat-formatted prompt, from one
    right-padded batch."""
    ids = [
        tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )["input_ids"]
        for p in prompts
    ]
    batch = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long)
    for row, seq in enumerate(ids):
        batch[row, : len(seq)] = torch.tensor(seq)
    with torch.no_grad():
        states = model(batch.to(device))
    return [states[row : row + 1, : len(seq)] for row, seq in enumerate(ids)]


def stage(name, start, device):
    """Prints the stage time (device-synchronized); returns it."""
    kohakutt.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.1f} s", flush=True)
    return elapsed


def sample(dit, positive, negative, size, jobs, steps, trunk, device):
    """Final latents [n, 128, H/16, W/16] of the (cfg, seed) `jobs` of one prompt,
    sampled as one batch: Euler on the flow velocity with per-row CFG."""
    width, height = size
    rows, cols = height // 16, width // 16
    noise = [
        torch.randn(1, 128, rows, cols, generator=torch.Generator().manual_seed(seed))
        for _, seed in jobs
    ]
    x = torch.cat(noise).to(device)
    n = x.shape[0]
    scale = torch.tensor([cfg for cfg, _ in jobs]).view(n, 1, 1, 1).to(device)
    pos = positive.expand(n, -1, -1)
    neg = negative.expand(n, -1, -1)
    rope_pos = dit.rope_for(rows, cols, positive.shape[1], device)
    rope_neg = dit.rope_for(rows, cols, negative.shape[1], device)
    schedule = sigmas(steps)
    start = time.perf_counter()
    with torch.no_grad():
        # The text tokens do not depend on the step: refined once.
        pos = dit.embed_text(pos, rope_pos, rows * cols)
        neg = dit.embed_text(neg, rope_neg, rows * cols)
        for index, (sigma, nxt) in enumerate(zip(schedule[:-1], schedule[1:])):
            if index == 1:
                stage(f"  first step ({n} rows)", start, device)
                start = time.perf_counter()
            mod, head = dit.conditioning(sigma * 1000, device)
            cond = trunk(x, pos, rope_pos, mod, head)
            uncond = trunk(x, neg, rope_neg, mod, head)
            x = x + (nxt - sigma) * (uncond + scale * (cond - uncond))
    stage(f"  remaining {len(schedule) - 2} steps ({n} rows)", start, device)
    return x


def save_image(image, path):
    """Writes image [3, H, W] in [-1, 1] as an 8-bit PNG."""
    pixels = ((image.permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255)
    Image.fromarray(pixels.to(torch.uint8).numpy()).save(path)


def main():
    """Generates one image, or every (prompt file, cfg, seed) of a sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MOSS-Image dir (HF layout)")
    parser.add_argument("--prompt", default="a lighthouse on a rocky coast at sunset")
    parser.add_argument("--prompts", help="dir of prompt .txt files (a sweep)")
    parser.add_argument("--negative", default="", help="text, or a file path")
    parser.add_argument("--size", type=int, default=1024, help="square side")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfgs", type=float, nargs="+", default=[3.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--batch", type=int, default=0, help="max rows per DiT batch")
    parser.add_argument("--weights", choices=("bfp8", "bf16"), default="bfp8")
    parser.add_argument("--fidelity", choices=kohakutt.core.numerics.FIDELITIES)
    parser.add_argument("--approx", action="store_true", help="SFPU approximations")
    parser.add_argument("--attention", choices=kohakutt.core.numerics.ATTENTION)
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument(
        "--te-device",
        choices=("tt", "cpu"),
        default="tt",
        help="text encoder device (cpu: fp32 weights, embeddings moved to tt)",
    )
    parser.add_argument("--shard", default="0/1", help="i/n: this run's share of jobs")
    parser.add_argument(
        "--text-cache",
        help="text states file: read when it holds every prompt, else written after "
        "encoding",
    )
    parser.add_argument(
        "--compile", action="store_true", help="DiT trunk via torch.compile"
    )
    parser.add_argument("--out", default="moss.png", help="single-image output")
    parser.add_argument("--out-dir", default="moss", help="sweep output dir")
    args = parser.parse_args()
    root = Path(args.model)
    dev = kohakutt.device(args.chip)
    default = (args.width or args.size, args.height or args.size)
    negative_text = args.negative
    if negative_text and Path(negative_text).is_file():
        negative_text = Path(negative_text).read_text().strip()

    # Jobs per prompt: (cfg, seed) pairs, this shard's share.
    if args.prompts:
        files = sorted(Path(args.prompts).glob("*.txt"))
        prompts = {f.stem: read_prompt(f, default) for f in files}
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        prompts = {"prompt": (args.prompt, default)}
        out_dir = None
    every = list(itertools.product(prompts, args.cfgs, args.seeds))
    index, count = map(int, args.shard.split("/"))
    groups = {}
    for name, cfg, seed in every[index::count]:
        groups.setdefault(name, []).append((cfg, seed))

    scope = kohakutt.numerics(
        fidelity=args.fidelity, approx=args.approx or None, attention=args.attention
    )
    with scope:
        # Text: every prompt in one batch, or read from --text-cache; the text encoder
        # is freed before the DiT and the VAE load (all three do not fit the card at
        # once).
        fmt = None if args.weights == "bf16" else args.weights
        names = list(groups)
        texts = [prompts[n][0] for n in names] + [negative_text]
        cache = Path(args.text_cache) if args.text_cache else None
        cached = torch.load(cache) if cache and cache.exists() else None
        if cached and all(t in cached["texts"] for t in texts):
            states = dict(zip(cached["texts"], cached["states"], strict=True))
            embedded = [states[t] for t in texts]
            print(f"text states from {cache}", flush=True)
        else:
            start = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
            on_cpu = args.te_device == "cpu"
            te_dev = "cpu" if on_cpu else dev
            te = load_text_encoder(
                root / "text_encoder",
                te_dev,
                torch.float32 if on_cpu else torch.bfloat16,
            )
            if not on_cpu:
                kohakutt.block_float_linears(te, fmt)
            stage(f"text encoder load ({args.te_device})", start, dev)
            start = time.perf_counter()
            with torch.no_grad():
                embedded = [
                    e.cpu().bfloat16() for e in encode(tokenizer, te, texts, te_dev)
                ]
            stage(f"text encode ({len(texts)} texts)", start, dev)
            del te
            gc.collect()
            if cache:
                torch.save({"texts": texts, "states": embedded}, cache)
        embedded = [e.to(dev) for e in embedded]
        negative = embedded.pop()
        embedded = dict(zip(names, embedded, strict=True))

        start = time.perf_counter()
        dit = load_dit(root / "transformer", dev)
        kohakutt.block_float_linears(dit, fmt)
        vae = load_vae(root / "vae", dev)
        stage("DiT and VAE load", start, dev)

        trunk = dit.joint
        if args.compile:
            trunk = torch.compile(dit.joint, backend="ktt_tt", dynamic=False)
        for name, jobs in groups.items():
            size = prompts[name][1]
            width = args.batch or len(jobs)
            for part in range(0, len(jobs), width):
                rows = jobs[part : part + width]
                print(f"{name}: {size[0]}x{size[1]}, (cfg, seed) {rows}", flush=True)
                start = time.perf_counter()
                latents = sample(
                    dit, embedded[name], negative, size, rows, args.steps, trunk, dev
                )
                sampled = time.perf_counter() - start
                for row, (cfg, seed) in enumerate(rows):
                    label = f"{name}_cfg{cfg:g}_seed{seed}"
                    start = time.perf_counter()
                    with torch.no_grad():
                        image = vae(latents[row : row + 1]).cpu()
                    decoded = stage(f"  {label} VAE decode", start, dev)
                    path = out_dir / f"{label}.png" if out_dir else Path(args.out)
                    save_image(image[0], path)
                    torch.save(latents[row : row + 1].cpu(), path.with_suffix(".pt"))
                    meta = {
                        "prompt": name,
                        "width": size[0],
                        "height": size[1],
                        "cfg": cfg,
                        "seed": seed,
                        "steps": args.steps,
                        "weights": args.weights,
                        "numerics": str(kohakutt.current_numerics()),
                        "batch_rows": len(rows),
                        "batch_sample_s": sampled,
                        "vae_s": decoded,
                    }
                    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
                    print("  wrote", path, flush=True)


if __name__ == "__main__":
    main()
