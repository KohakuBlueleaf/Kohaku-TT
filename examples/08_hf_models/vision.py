"""A vision encoder from transformers (DINOv3 ViT or ConvNeXt) on a chip: eager,
captured and (--compile) torch.compile forwards, timed, against an fp32 CPU forward.
    python vision.py --model facebook/dinov3-vits16plus-pretrain-lvd1689m --res 224
"""

import argparse

import torch
from common import random_image, rel_l2, report, timed, to_bf16
from transformers import AutoModel

import kohakutt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="facebook/dinov3-vits16plus-pretrain-lvd1689m"
    )
    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="also torch.compile")
    args = parser.parse_args()

    dev = kohakutt.device(0)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32).eval()
    pixels = random_image(args.res, args.batch)
    with torch.no_grad():
        ref = model(pixel_values=pixels).last_hidden_state

    # bf16 weights and input on the chip; the forward itself is unchanged
    model = to_bf16(model).to(dev)
    x = pixels.to(torch.bfloat16).to(dev)

    def forward(pixel_values):
        with torch.no_grad():
            return model(pixel_values=pixel_values).last_hidden_state

    out, first, median = timed(lambda: forward(x), dev)
    report("eager", first, median, rel_l2(out, ref))

    # capture records the forward once (after warm-up calls) and replays it as a trace
    replay = kohakutt.capture(forward, x)
    out, first, median = timed(lambda: replay(x), dev)
    report("captured", first, median, rel_l2(out, ref))
    replay.release()

    if args.compile:
        compiled = torch.compile(model, backend="ktt_tt")

        def compiled_forward():
            with torch.no_grad():
                return compiled(pixel_values=x).last_hidden_state

        out, first, median = timed(compiled_forward, dev)
        report("compiled", first, median, rel_l2(out, ref))
    print(f"output {tuple(out.shape)}")
    kohakutt.close()


if __name__ == "__main__":
    main()
