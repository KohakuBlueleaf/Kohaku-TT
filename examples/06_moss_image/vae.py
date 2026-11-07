"""FLUX.2 VAE decoder (the MOSS-Image image decoder): DiT latents [1, 128, h, w] ->
RGB [1, 3, 16 h, 16 w] in [-1, 1]. Convolutions in the weight dtype; GroupNorm, SiLU
and attention softmax in fp32. Weights from the checkpoint's vae/ dir (decoder only).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from dit import load_weights
from torch import nn


def group_norm_silu(x, norm, silu=True):
    """GroupNorm (+ SiLU) in fp32, back in x's dtype."""
    y = F.group_norm(
        x.float(), norm.num_groups, norm.weight.float(), norm.bias.float(), norm.eps
    )
    return (F.silu(y) if silu else y).to(x.dtype)


class ResnetBlock(nn.Module):
    def __init__(self, cin, cout, groups, eps):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin, eps)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, cout, eps)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        h = self.conv1(group_norm_silu(x, self.norm1))
        h = self.conv2(group_norm_silu(h, self.norm2))
        skip = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return (skip.float() + h.float()).to(x.dtype)


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention over H W tokens."""

    def __init__(self, channels, groups, eps):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x):
        b, c, h, w = x.shape
        y = group_norm_silu(x, self.group_norm, silu=False).flatten(2).transpose(1, 2)
        q, k, v = (proj(y)[:, None] for proj in (self.to_q, self.to_k, self.to_v))
        o = F.scaled_dot_product_attention(q, k, v)[:, 0]
        o = self.to_out[0](o).transpose(1, 2).reshape(b, c, h, w)
        return (x.float() + o.float()).to(x.dtype)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UpBlock(nn.Module):
    def __init__(self, cin, cout, layers, groups, eps, upsample):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(cin if i == 0 else cout, cout, groups, eps)
            for i in range(layers)
        )
        self.upsamplers = nn.ModuleList([Upsample(cout)]) if upsample else None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        return self.upsamplers[0](x) if self.upsamplers is not None else x


class MidBlock(nn.Module):
    def __init__(self, channels, groups, eps):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(channels, channels, groups, eps) for _ in range(2)
        )
        self.attentions = nn.ModuleList([AttentionBlock(channels, groups, eps)])

    def forward(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Decoder(nn.Module):
    def __init__(self, latent, channels, layers, groups, eps=1e-6):
        super().__init__()
        top = channels[-1]
        self.conv_in = nn.Conv2d(latent, top, 3, padding=1)
        self.mid_block = MidBlock(top, groups, eps)
        rev = list(reversed(channels))
        self.up_blocks = nn.ModuleList(
            UpBlock(
                rev[max(i - 1, 0)], rev[i], layers + 1, groups, eps, i < len(rev) - 1
            )
            for i in range(len(rev))
        )
        self.conv_norm_out = nn.GroupNorm(groups, channels[0], eps)
        self.conv_out = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(self, z):
        x = self.mid_block(self.conv_in(z))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(group_norm_silu(x, self.conv_norm_out))


class BatchNormStats(nn.Module):
    """Running mean / var of the DiT latent channels (used to de-normalize)."""

    def __init__(self, channels):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.int64))


class FluxVaeDecoder(nn.Module):
    """De-normalize, unpatchify 2 x 2, post-quant conv, decode."""

    def __init__(self, cfg):
        super().__init__()
        self.eps = cfg["batch_norm_eps"]
        latent = cfg["latent_channels"]
        self.decoder = Decoder(
            latent,
            cfg["decoder_block_out_channels"],
            cfg["layers_per_block"],
            cfg["norm_num_groups"],
        )
        self.post_quant_conv = nn.Conv2d(latent, latent, 1)
        self.bn = BatchNormStats(latent * 4)

    @torch.no_grad()
    def forward(self, latents):
        dtype = self.post_quant_conv.weight.dtype
        std = (self.bn.running_var.float() + self.eps).sqrt().view(1, -1, 1, 1)
        z = latents.float() * std + self.bn.running_mean.float().view(1, -1, 1, 1)
        image = self.decoder(self.post_quant_conv(F.pixel_shuffle(z, 2).to(dtype)))
        return image.float().clamp(-1, 1)


def load_vae(root, device, dtype=torch.bfloat16):
    """FluxVaeDecoder from the checkpoint's vae/ dir (decoder weights only), on
    `device`."""
    root = Path(root)
    with torch.device("meta"):
        model = FluxVaeDecoder(json.loads((root / "config.json").read_text()))
    load_weights(
        model, root, device, dtype, skip=("encoder.", "quant_conv."), keep_fp32=("bn.",)
    )
    return model.eval()
