"""Training steps of a vision encoder from transformers on a chip: forward, backward and
AdamW, timed. The first step's loss and gradients are compared with the same step in
fp32 on the CPU.   python vision_train.py --model facebook/convnext-tiny-224
"""

import argparse
import time

import torch
from common import random_image, rel_l2, to_bf16
from transformers import AutoModel

import kohakutt


def loss_fn(model, pixels):
    """A stand-in objective: the mean square of the last hidden state."""
    return model(pixel_values=pixels).last_hidden_state.float().square().mean()


def grads(model):
    """All parameter gradients, flattened into one fp32 CPU vector."""
    found = [p.grad for p in model.parameters() if p.grad is not None]
    return torch.cat([g.float().cpu().flatten() for g in found])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/convnext-tiny-224")
    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    # open the device before any backward on the CPU
    dev = kohakutt.device(0)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32).train()
    pixels = random_image(args.res, args.batch)
    ref_loss = loss_fn(model, pixels)
    ref_loss.backward()
    ref_grads = grads(model)
    model.zero_grad(set_to_none=True)

    model = to_bf16(model).to(dev)
    x = pixels.to(torch.bfloat16).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(args.steps):
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, x)
        loss.backward()
        if step == 0:
            err = rel_l2(grads(model), ref_grads)
            print(
                f"first step: loss {loss.item():.5f} (CPU {ref_loss.item():.5f}), "
                f"gradients rel L2 vs CPU {err:.2e}"
            )
        opt.step()
        kohakutt.synchronize(dev)
        seconds = time.perf_counter() - start
        print(f"step {step} loss {loss.item():.5f}  {seconds * 1e3:8.1f} ms")
    kohakutt.close()


if __name__ == "__main__":
    main()
