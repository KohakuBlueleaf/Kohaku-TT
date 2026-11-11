"""Where the time of a forward goes: `kohakutt.tools.profile.OpProfile` times every aten
op on the chip (each one synchronized) and prints the top overloads.
    python profile_ops.py --model facebook/convnext-tiny-224 --res 512
"""

import argparse

import torch
from common import random_image
from transformers import AutoModel

import kohakutt
from kohakutt.tools.profile import OpProfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="facebook/dinov3-vits16plus-pretrain-lvd1689m"
    )
    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    dev = kohakutt.device(0)
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16).eval().to(dev)
    x = random_image(args.res).to(torch.bfloat16).to(dev)
    with torch.no_grad():
        model(pixel_values=x)  # first call: program builds, not profiled
        kohakutt.synchronize(dev)
        with OpProfile() as prof:
            model(pixel_values=x)
    print(prof.table(args.top))
    kohakutt.close()


if __name__ == "__main__":
    main()
