"""One SDXL UNet forward and one VAE decode on a chip, timed; with --ref, compared with
the fp32 CPU outputs that `check_reference.py` saved. Weights come from the local
Hugging Face cache; the text conditioning and the latents are seeded random.
"""

import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from unet import load_unet
from vae import load_vae

import kohakutt

REPO = "KBlueLeaf/Kohaku-XL-Zeta"


def checkpoint_root(repo=REPO):
    """Local snapshot directory of the checkpoint (never downloads)."""
    index = hf_hub_download(repo, "model_index.json", local_files_only=True)
    return str(Path(index).parent)


def example_inputs(latent, dtype):
    """(sample, timestep, text [1, 77, 2048], pooled [1, 1280], time_ids [1, 6]);
    the random values are exact in bf16."""
    gen = torch.Generator().manual_seed(0)
    px = 8 * latent

    def randn(*shape):
        return torch.randn(*shape, generator=gen).bfloat16().to(dtype)

    return (
        randn(1, 4, latent, latent),
        torch.tensor([500.0]),
        randn(1, 77, 2048),
        randn(1, 1280),
        torch.tensor([[px, px, 0, 0, px, px]], dtype=dtype),
    )


def rel_l2(y, ref):
    y, ref = y.detach().cpu().double(), ref.double()
    return ((y - ref).norm() / ref.norm()).item()


def report(name, model, inputs, dev, ref=None, repeats=3):
    """Moves `model` and `inputs` to `dev`, times the forward, prints time and error."""
    model = model.to(dev)
    x = [t.to(dev) for t in inputs]
    times = []
    with torch.no_grad():
        for _ in range(repeats + 1):
            start = time.perf_counter()
            out = model(*x)
            kohakutt.synchronize(dev)
            times.append(time.perf_counter() - start)
    median = sorted(times[1:])[repeats // 2]
    line = f"{name}: first call {times[0]:.2f} s, median {median * 1e3:.1f} ms"
    if ref is not None:
        line += f", rel L2 vs fp32 CPU {rel_l2(out, ref):.3e}"
    print(f"{line}, output {tuple(out.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=REPO, help="an SDXL repo, diffusers layout")
    parser.add_argument("--latent", type=int, default=64, help="64: a 512 px image")
    parser.add_argument("--ref", help="the file check_reference.py wrote")
    parser.add_argument("--only", choices=("unet", "vae"))
    args = parser.parse_args()

    dev = kohakutt.device(0)
    root = checkpoint_root(args.model)
    ref = torch.load(args.ref) if args.ref else {}
    inputs = example_inputs(args.latent, torch.bfloat16)
    if args.only in (None, "unet"):
        report("unet", load_unet(root), inputs, dev, ref.get("unet"))
    if args.only in (None, "vae"):
        report("vae", load_vae(root), inputs[:1], dev, ref.get("vae"))
    kohakutt.close()


if __name__ == "__main__":
    main()
