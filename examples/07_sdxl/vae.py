"""SDXL VAE decoder as plain PyTorch modules: latents [B, 4, h, w] -> RGB
[B, 3, 8 h, 8 w] in [-1, 1]. State-dict names follow the diffusers layout (decoder and
post_quant_conv only; the encoder keys are skipped when loading).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn


class Resnet(nn.Module):
    def __init__(self, cin, cout, eps=1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin, eps)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, cout, eps)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        h = self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))
        return (x if self.conv_shortcut is None else self.conv_shortcut(x)) + h


class Attention(nn.Module):
    """Single-head self-attention over the h w positions."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, dim, eps)
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.group_norm(x).flatten(2).transpose(1, 2)
        q, k, v = (proj(y)[:, None] for proj in (self.to_q, self.to_k, self.to_v))
        o = F.scaled_dot_product_attention(q, k, v)[:, 0]
        return x + self.to_out[0](o).transpose(1, 2).reshape(b, c, h, w)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UpBlock(nn.Module):
    def __init__(self, cin, cout, layers, upsample):
        super().__init__()
        self.resnets = nn.ModuleList(
            Resnet(cin if i == 0 else cout, cout) for i in range(layers)
        )
        self.upsamplers = nn.ModuleList([Upsample(cout)]) if upsample else None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        return x if self.upsamplers is None else self.upsamplers[0](x)


class MidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resnets = nn.ModuleList(Resnet(dim, dim) for _ in range(2))
        self.attentions = nn.ModuleList([Attention(dim)])

    def forward(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Decoder(nn.Module):
    def __init__(self, latent, chans, layers):
        super().__init__()
        rev = chans[::-1]
        self.conv_in = nn.Conv2d(latent, rev[0], 3, padding=1)
        self.mid_block = MidBlock(rev[0])
        self.up_blocks = nn.ModuleList(
            UpBlock(rev[max(i - 1, 0)], rev[i], layers + 1, i < len(rev) - 1)
            for i in range(len(rev))
        )
        self.conv_norm_out = nn.GroupNorm(32, chans[0], 1e-6)
        self.conv_out = nn.Conv2d(chans[0], 3, 3, padding=1)

    def forward(self, z):
        x = self.mid_block(self.conv_in(z))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class VaeDecoder(nn.Module):
    """forward(latents) takes the sampler's latents; the scaling factor is divided
    out first."""

    def __init__(self, cfg):
        super().__init__()
        latent = cfg["latent_channels"]
        self.scaling = cfg["scaling_factor"]
        self.post_quant_conv = nn.Conv2d(latent, latent, 1)
        self.decoder = Decoder(
            latent, cfg["block_out_channels"], cfg["layers_per_block"]
        )

    def forward(self, latents):
        return self.decoder(self.post_quant_conv(latents / self.scaling))


def load_vae(root, dtype=torch.bfloat16):
    """VaeDecoder on the CPU from `root`/vae; every decoder key must be matched."""
    root = Path(root) / "vae"
    with torch.device("meta"):
        model = VaeDecoder(json.loads((root / "config.json").read_text()))
    state = load_file(root / "diffusion_pytorch_model.safetensors")
    skip = ("encoder.", "quant_conv.")  # "post_quant_conv." does not match
    state = {k: v.to(dtype) for k, v in state.items() if not k.startswith(skip)}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()
