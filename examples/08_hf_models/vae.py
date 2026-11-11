"""A diffusers AutoencoderKL on a chip: encode an image to latents and decode latents to
an image, timed, against fp32 on the CPU. Needs `diffusers` (pip install diffusers).
    python vae.py --model stabilityai/sdxl-vae --res 1024
"""

import argparse

import torch
from common import random_image, rel_l2, report, timed
from diffusers import AutoencoderKL

import kohakutt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--res", type=int, default=512, help="image side in pixels")
    args = parser.parse_args()

    dev = kohakutt.device(0)
    vae = AutoencoderKL.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    image = random_image(args.res)
    gen = torch.Generator().manual_seed(0)
    latents = torch.randn(1, 4, args.res // 8, args.res // 8, generator=gen)
    latents = latents.bfloat16().float()

    def encode(x):
        with torch.no_grad():
            return vae.encode(x).latent_dist.mode()

    def decode(z):
        with torch.no_grad():
            return vae.decode(z).sample

    ref_latents, ref_image = encode(image), decode(latents)

    vae = vae.to(torch.bfloat16).to(dev)
    x = image.to(torch.bfloat16).to(dev)
    z = latents.to(torch.bfloat16).to(dev)
    out, first, median = timed(lambda: encode(x), dev)
    report("encode", first, median, rel_l2(out, ref_latents))
    out, first, median = timed(lambda: decode(z), dev)
    report("decode", first, median, rel_l2(out, ref_image))
    print(f"latents {tuple(z.shape)} -> image {tuple(out.shape)}")
    kohakutt.close()


if __name__ == "__main__":
    main()
