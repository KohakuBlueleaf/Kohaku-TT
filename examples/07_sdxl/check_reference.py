"""CPU check of the SDXL modules against diffusers (fp32, same weights): one UNet
forward and one VAE decode. The fp32 outputs go to --out for `run.py --ref`.
"""

import argparse

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from run import REPO, checkpoint_root, example_inputs, rel_l2
from unet import load_unet
from vae import load_vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=REPO)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--out", default="ref_64.pt")
    args = parser.parse_args()

    root = checkpoint_root(args.model)
    sample, t, text, pooled, ids = example_inputs(args.latent, torch.float32)

    unet = load_unet(root, torch.float32)
    ref = UNet2DConditionModel.from_pretrained(root, subfolder="unet").eval()
    cond = {"text_embeds": pooled, "time_ids": ids}
    with torch.no_grad():
        mine = unet(sample, t, text, pooled, ids)
        theirs = ref.float()(sample, t, text, added_cond_kwargs=cond).sample
    print(f"unet: strict load ok; rel L2 vs diffusers {rel_l2(mine, theirs):.2e}")
    del ref

    vae = load_vae(root, torch.float32)
    ref = AutoencoderKL.from_pretrained(root, subfolder="vae").eval().float()
    with torch.no_grad():
        image = vae(sample)
        image_ref = ref.decode(sample / ref.config.scaling_factor).sample
    print(f"vae:  strict load ok; rel L2 vs diffusers {rel_l2(image, image_ref):.2e}")
    torch.save({"unet": mine, "vae": image}, args.out)


if __name__ == "__main__":
    main()
