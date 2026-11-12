"""The PixAI tagger v1.0 (a ViT-Det classifier from the repo's remote code) on a chip,
timed, against fp32 on the CPU. The model takes 1008 x 1008 inputs only.
    python tagger.py
"""

import argparse
import importlib
import sys
from pathlib import Path

import torch
from common import random_image, rel_l2, report, timed, to_bf16
from huggingface_hub import hf_hub_download
from stubs import install_stubs

import kohakutt


def load(repo):
    """The tagger's ViTDetCls from the local snapshot of `repo` (no download)."""
    install_stubs()
    config = hf_hub_download(repo, "config.json", local_files_only=True)
    root = str(Path(config).parent)
    sys.path.insert(0, root)
    module = importlib.import_module("tagger_pipeline")
    return module.ViTDetCls.from_pretrained(root, dtype=torch.float32).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pixai-labs/pixai-tagger-v1.0")
    args = parser.parse_args()

    dev = kohakutt.device(0)
    model = load(args.model)
    pixels = random_image(1008)
    with torch.no_grad():
        ref = model(pixels)

    model = to_bf16(model).to(dev)
    x = pixels.to(torch.bfloat16).to(dev)

    def forward():
        with torch.no_grad():
            return model(x)

    out, first, median = timed(forward, dev)
    report("eager", first, median, rel_l2(out, ref))
    top_tt = out.float().cpu()[0].topk(10).indices
    top_cpu = ref[0].topk(10).indices
    shared = len(set(top_tt.tolist()) & set(top_cpu.tolist()))
    print(f"logits {tuple(out.shape)}; top-10 tags shared with the CPU: {shared}/10")
    kohakutt.close()


if __name__ == "__main__":
    main()
