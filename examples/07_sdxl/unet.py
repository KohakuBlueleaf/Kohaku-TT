"""SDXL UNet as plain PyTorch modules. State-dict names follow the diffusers layout,
so the checkpoint's unet/ weights load with `strict=True`. Every op is a torch op and
runs in the weight dtype.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn


def timestep_embedding(t, dim):
    """Sinusoidal embedding [B] -> [B, dim], cos half first (flip_sin_to_cos)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    return torch.cat([args.cos(), args.sin()], dim=-1)


class Resnet(nn.Module):
    def __init__(self, cin, cout, temb_dim, eps=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin, eps)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_dim, cout)
        self.norm2 = nn.GroupNorm(32, cout, eps)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return (x if self.conv_shortcut is None else self.conv_shortcut(x)) + h


class Attention(nn.Module):
    """Multi-head attention; keys / values from `context` (self-attention if None)."""

    def __init__(self, dim, context_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x, context=None):
        context = x if context is None else context
        b, n, c = x.shape

        def split(t):
            return t.view(b, -1, self.heads, c // self.heads).transpose(1, 2)

        q, k, v = (
            split(self.to_q(x)),
            split(self.to_k(context)),
            split(self.to_v(context)),
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out[0](out.transpose(1, 2).reshape(b, n, c))


class GEGLU(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Linear(cin, 2 * cout)

    def forward(self, x):
        h, gate = self.proj(x).chunk(2, dim=-1)
        return h * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.ModuleList(
            [GEGLU(dim, 4 * dim), nn.Identity(), nn.Linear(4 * dim, dim)]
        )

    def forward(self, x):
        return self.net[2](self.net[0](x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, heads):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))
        self.attn1 = Attention(dim, dim, heads)
        self.attn2 = Attention(dim, context_dim, heads)
        self.ff = FeedForward(dim)

    def forward(self, x, context):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))


class SpatialTransformer(nn.Module):
    def __init__(self, dim, context_dim, heads, depth):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim, 1e-6)
        self.proj_in = nn.Linear(dim, dim)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(dim, context_dim, heads) for _ in range(depth)
        )
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        b, c, h, w = x.shape
        y = self.proj_in(self.norm(x).flatten(2).transpose(1, 2))
        for block in self.transformer_blocks:
            y = block(y, context)
        y = self.proj_out(y).transpose(1, 2).reshape(b, c, h, w)
        return x + y


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Block(nn.Module):
    """A stage of one resnet (+ transformer when `depth` > 0) per entry of `cins`.
    With `skips`, each resnet first concatenates one skip activation (up stages)."""

    def __init__(self, cins, cout, temb_dim, depth, heads, context_dim):
        super().__init__()
        self.resnets = nn.ModuleList(Resnet(c, cout, temb_dim) for c in cins)
        self.attentions = (
            nn.ModuleList(
                SpatialTransformer(cout, context_dim, heads, depth) for _ in cins
            )
            if depth
            else None
        )

    def forward(self, x, temb, context, skips=None):
        outputs = []
        for i, resnet in enumerate(self.resnets):
            if skips is not None:
                x = torch.cat([x, skips.pop()], dim=1)
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x, context)
            outputs.append(x)
        return x, outputs


class DownBlock(Block):
    def __init__(self, *args, downsample):
        super().__init__(*args)
        self.downsamplers = nn.ModuleList([Downsample(args[1])]) if downsample else None

    def forward(self, x, temb, context):
        x, outputs = super().forward(x, temb, context)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            outputs.append(x)
        return x, outputs


class UpBlock(Block):
    def __init__(self, *args, upsample):
        super().__init__(*args)
        self.upsamplers = nn.ModuleList([Upsample(args[1])]) if upsample else None

    def forward(self, x, temb, context, skips):
        x, _ = super().forward(x, temb, context, skips)
        return self.upsamplers[0](x) if self.upsamplers is not None else x


class MidBlock(nn.Module):
    def __init__(self, dim, temb_dim, depth, heads, context_dim):
        super().__init__()
        self.resnets = nn.ModuleList(Resnet(dim, dim, temb_dim) for _ in range(2))
        self.attentions = nn.ModuleList(
            [SpatialTransformer(dim, context_dim, heads, depth)]
        )

    def forward(self, x, temb, context):
        x = self.resnets[0](x, temb)
        return self.resnets[1](self.attentions[0](x, context), temb)


class AddEmbedding(nn.Module):
    def __init__(self, cin, dim):
        super().__init__()
        self.linear_1, self.linear_2 = nn.Linear(cin, dim), nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class UNet(nn.Module):
    """forward(sample [B, 4, h, w], timestep [B], text [B, N, 2048], pooled
    [B, 1280], time_ids [B, 6]) -> noise prediction [B, 4, h, w]."""

    def __init__(self, cfg):
        super().__init__()
        chans = cfg["block_out_channels"]
        heads = cfg["attention_head_dim"]  # SDXL: the number of heads, 64 wide
        depths = cfg["transformer_layers_per_block"]
        layers, ctx = cfg["layers_per_block"], cfg["cross_attention_dim"]
        self.time_dim, self.add_dim = chans[0], cfg["addition_time_embed_dim"]
        temb = chans[0] * 4
        self.conv_in = nn.Conv2d(cfg["in_channels"], chans[0], 3, padding=1)
        self.time_embedding = AddEmbedding(chans[0], temb)
        self.add_embedding = AddEmbedding(
            cfg["projection_class_embeddings_input_dim"], temb
        )

        n = len(chans)
        has_attn = [t.startswith("CrossAttn") for t in cfg["down_block_types"]]
        self.down_blocks = nn.ModuleList()
        skip_ch, cin = [chans[0]], chans[0]
        for i, cout in enumerate(chans):
            depth = depths[i] if has_attn[i] else 0
            self.down_blocks.append(
                DownBlock(
                    [cin if j == 0 else cout for j in range(layers)],
                    cout,
                    temb,
                    depth,
                    heads[i],
                    ctx,
                    downsample=i < n - 1,
                )
            )
            skip_ch += [cout] * layers + ([cout] if i < n - 1 else [])
            cin = cout
        self.mid_block = MidBlock(chans[-1], temb, depths[-1], heads[-1], ctx)

        self.up_blocks = nn.ModuleList()
        prev = chans[-1]
        for i, cout in enumerate(reversed(chans)):
            k = n - 1 - i  # index of the matching down stage
            depth = depths[k] if has_attn[k] else 0
            cins = [
                (prev if j == 0 else cout) + skip_ch.pop() for j in range(layers + 1)
            ]
            self.up_blocks.append(
                UpBlock(cins, cout, temb, depth, heads[k], ctx, upsample=i < n - 1)
            )
            prev = cout
        self.conv_norm_out = nn.GroupNorm(32, chans[0], 1e-5)
        self.conv_out = nn.Conv2d(chans[0], cfg["out_channels"], 3, padding=1)

    def forward(self, sample, timestep, text, pooled, time_ids):
        b = sample.shape[0]
        dtype = sample.dtype
        t = timestep.expand(b)
        temb = self.time_embedding(timestep_embedding(t, self.time_dim).to(dtype))
        ids = timestep_embedding(time_ids.flatten(), self.add_dim).view(b, -1)
        temb = temb + self.add_embedding(torch.cat([pooled, ids.to(dtype)], dim=-1))

        x = self.conv_in(sample)
        skips = [x]
        for block in self.down_blocks:
            x, outputs = block(x, temb, text)
            skips += outputs
        x = self.mid_block(x, temb, text)
        for block in self.up_blocks:
            x = block(x, temb, text, skips)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


def load_unet(root, dtype=torch.bfloat16):
    """UNet on the CPU from `root`/unet (config.json + safetensors); every checkpoint
    key must match a parameter and the other way round."""
    root = Path(root) / "unet"
    with torch.device("meta"):
        model = UNet(json.loads((root / "config.json").read_text()))
    state = load_file(root / "diffusion_pytorch_model.safetensors")
    state = {k: v.to(dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()
