"""MOSS-Image DiT: 42 joint [image ; text] blocks with shared modulation, qk-RMSNorm,
4-axis RoPE and a SiTU FFN. State-dict names follow the official checkpoint. Linears,
the residual stream and every activation are in the weight dtype; norms, modulation,
RoPE and SiTU compute in fp32 inside one program each (torch.ops.ktt.rms_norm /
rotary, a fused SiTU chain on tt; their fp32 composites elsewhere).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

import kohakutt  # noqa: F401 (registers torch.ops.ktt)
from kohakutt import fusion


class RMSNorm(nn.Module):
    """x / rms(x) * weight over the last dim (fp32 math), in x's dtype."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, scale=None, shift=None):
        """With `scale` / `shift` ([D], shared by the batch): norm(x) * scale +
        shift."""
        weight = self.weight if scale is None else self.weight.float() * scale
        if shift is not None:
            shift = shift.to(x.dtype)
        return torch.ops.ktt.rms_norm(x, weight.to(x.dtype), shift, self.eps)


# SiTU as a fused chain: slots 0 = gate, 1 = hidden; ops (op, dst, a, b / scalar).
SITU_OPS = [
    ["muls", 2, 0, 0.25],
    ["tanh", 2, 2, None],
    ["sigmoid", 0, 0, None],
    ["mul", 2, 2, 0],
    ["muls", 2, 2, 4.0],
    ["muls", 3, 1, 0.04],
    ["tanh", 3, 3, None],
    ["muls", 3, 3, 25.0],
    ["mul", 0, 2, 3],
]


def situ(hidden, gate):
    """(4 tanh(g / 4) sigmoid(g)) * (25 tanh(h / 25)) (fp32 math), in hidden's
    dtype."""
    if hidden.device.type == "tt" and hidden.shape == gate.shape:
        return fusion.run(
            [gate, hidden],
            ["full", "full"],
            SITU_OPS,
            0,
            list(hidden.shape),
            hidden.dtype,
        )
    g, h = gate.float(), hidden.float()
    out = (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(h / 25.0))
    return out.to(hidden.dtype)


def split_heads(qkv, heads):
    """[B, S, 3 H Dh] fused projection -> q, k, v views [B, H, S, Dh]."""
    b, s, c = qkv.shape
    return qkv.view(b, s, 3, heads, c // (3 * heads)).permute(2, 0, 3, 1, 4).unbind(0)


def attention(q, k, v, rope):
    """Rotated attention of q / k / v [B, H, S, Dh] with rope = (cos, sin), each
    [1, 1, S, Dh] in q's dtype -> [B, S, H Dh]."""
    cos, sin = rope
    q, k = torch.ops.ktt.rotary(q, cos, sin), torch.ops.ktt.rotary(k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).flatten(2)


def gated(x, o, gate):
    """x + o * gate (fp32 math) in x's dtype; x / o [B, S, D], gate [D]."""
    gate = gate.to(x.dtype).view(1, 1, -1)
    if x.device.type == "tt" and x.shape == o.shape and x.dtype == o.dtype:
        return fusion.run(
            [x, o, gate],
            ["full", "full", "row"],
            [["mul", 1, 1, 2], ["add", 0, 0, 1]],
            0,
            list(x.shape),
            x.dtype,
        )
    return (x.float() + o.float() * gate.float()).to(x.dtype)


class TextBlock(nn.Module):
    """Text refinement block: qk-norm attention + SiTU FFN, no modulation."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = split_heads(self.qkv(self.norm1(x)), self.heads)
        x = x + self.proj(attention(self.q_norm(q), self.k_norm(k), v, rope))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    """Text features [B, S, 4096] -> refined text tokens [B, S, dim]."""

    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(nn.Module):
    """Joint block: modulated RMSNorm, qk-norm attention, gated residual; modulated
    RMSNorm, SiTU FFN, gated residual."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def fuse_qkv(self):
        """Replaces the q / k / v projections by one fused [3 D, D] projection."""
        p = self.proj
        weight = torch.cat([p.pop(name).weight.data for name in ("q", "k", "v")])
        p["qkv"] = nn.Linear(
            weight.shape[1], weight.shape[0], bias=False, device="meta"
        )
        p["qkv"].weight = nn.Parameter(weight, requires_grad=False)

    def split_ffn(self):
        """Replaces the fused [2 F, D] up / gate projection by its two [F, D] halves
        (hidden, gate)."""
        p = self.proj
        hidden, gate = p.pop("ffn_up_gate").weight.data.chunk(2, 0)
        for name, weight in (("ffn_hidden", hidden), ("ffn_gate", gate)):
            p[name] = nn.Linear(
                weight.shape[1], weight.shape[0], bias=False, device="meta"
            )
            p[name].weight = nn.Parameter(weight.clone(), requires_grad=False)

    def forward(self, x, rope, mod):
        """`mod`: (shift, 1 + scale, gate) for attention then FFN, each [D] (shared by
        the batch)."""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        p = self.proj
        dtype = p["o"].weight.dtype

        # Attention: modulated norm, qk-norm + RoPE, gated residual.
        y = self.norm1(x, scale_a, shift_a).to(dtype)
        q, k, v = split_heads(p["qkv"](y), self.heads)
        x = gated(x, p["o"](attention(self.q_norm(q), self.k_norm(k), v, rope)), gate_a)

        # FFN: modulated norm, SiTU, gated residual.
        y = self.norm2(x, scale_f, shift_f).to(dtype)
        hidden, gate = p["ffn_hidden"](y), p["ffn_gate"](y)
        return gated(x, p["ffn_down"](situ(hidden, gate)), gate_f)


class RoPE(nn.Module):
    """4-axis rotary phases (r, f, h, w) for image tokens followed by text tokens."""

    def __init__(self, head_dim, base):
        super().__init__()
        bands = head_dim // 8
        exponent = torch.arange(bands, dtype=torch.float32) / bands
        self.register_buffer("bands", 1 / (base**exponent), persistent=False)

    def forward(self, height, width, text_length):
        """(cos, sin), each [1, 1, H W + T, head_dim] on the CPU in fp32."""
        h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        image = (
            torch.stack(
                (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
            )
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4)
        text[:, 1] = torch.arange(text_length)
        phase = (torch.cat((image, text), 0).unsqueeze(-1) * self.bands.cpu()).flatten(
            1
        )
        return tuple(
            torch.cat((t, t), -1)[None, None] for t in (phase.cos(), phase.sin())
        )


class MossDiT(nn.Module):
    """MOSS-Image transformer: (latents [B, 128, h, w], sigma, text [B, T, 4096]) ->
    velocity [B, 128, h, w]."""

    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
    ):
        super().__init__()
        self.freq_dim = freq_dim
        # dtype of the joint blocks' residual stream (None: the weight dtype).
        self.residual_dtype = None
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList(
            [Block(dim, ffn_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def conditioning(self, timestep, device):
        """Per-step modulation [1, 6, D] and head modulation [1, 2, D], fp32."""
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
        phase = torch.tensor([float(timestep)], dtype=torch.float64)[:, None] * freq
        emb = torch.cat((phase.cos(), phase.sin()), -1).float().to(device)
        te = self.time_embedding
        time = te[2](
            F.silu(te[0](emb.to(te[0].weight.dtype)).float()).to(te[2].weight.dtype)
        ).float()
        mod = self.shared_block_modulation(
            time.to(self.shared_block_modulation.weight.dtype)
        )
        head = self.video_head_modulation(
            time.to(self.video_head_modulation.weight.dtype)
        )
        return mod.float().view(1, 6, -1), head.float().view(1, 2, -1)

    def forward(self, latents, timestep, text, rope, skip_layers=()):
        """Velocity for one sample; `rope` from rope_for(); skipped blocks are
        identity."""
        mod, head = self.conditioning(timestep, latents.device)
        return self.trunk(latents, text, rope, mod, head, skip_layers)

    def trunk(self, latents, text, rope, mod, head, skip_layers=()):
        """Velocity [B, 128, h, w] of latents [B, 128, h, w] with text [B, T, 4096],
        given the step's modulation (from conditioning()); a fixed-shape graph for
        torch.compile, reused across steps."""
        n = latents.shape[2] * latents.shape[3]
        return self.joint(
            latents, self.embed_text(text, rope, n), rope, mod, head, skip_layers
        )

    def embed_text(self, text, rope, n):
        """Refined text tokens [B, T, D] of text [B, T, 4096] (independent of the
        latents and the step); `rope` covers n image tokens then the text."""
        return self.text_embedder(text, tuple(t[:, :, n:] for t in rope))

    def joint(self, latents, text, rope, mod, head, skip_layers=()):
        """trunk() with the text already through embed_text()."""
        batch, _, height, width = latents.shape
        dtype = self.video_embedder.weight.dtype
        video = latents.flatten(2).transpose(1, 2).to(dtype)
        video = self.video_embedder(video)
        n = height * width
        x = torch.cat((video, text), 1).to(self.residual_dtype or dtype)

        # The shared modulation, once: (shift, 1 + scale, gate) per sub-block, [D].
        m = mod[0]
        mod = (m[0], 1 + m[1], m[2], m[3], 1 + m[4], m[5])
        for index, layer in enumerate(self.layers):
            if index not in skip_layers:
                x = layer(x, rope, mod)
        x = self.final_norm_video(x[:, :n], 1 + head[0, 1], head[0, 0]).to(dtype)
        x = self.final_linear_video(x)
        return x.float().transpose(1, 2).reshape(batch, -1, height, width)

    def rope_for(self, height, width, text_length, device):
        """RoPE (cos, sin) for a latent grid and a text length, on `device` in the
        weight dtype."""
        dtype = self.video_embedder.weight.dtype
        return tuple(
            t.to(dtype).to(device) for t in self.rope(height, width, text_length)
        )


def load_dit(root, device, dtype=torch.bfloat16):
    """MossDiT from the checkpoint's transformer/ dir, its weights on `device`."""
    root = Path(root)
    with torch.device("meta"):
        model = MossDiT(
            **{
                k: v
                for k, v in json.loads((root / "config.json").read_text()).items()
                if not k.startswith("_")
            }
        )
    model.rope = RoPE(
        model.layers[0].proj["q"].weight.shape[0] // model.layers[0].heads, 10000.0
    )
    load_weights(
        model,
        root,
        device,
        dtype,
        keep_fp32=(
            "time_embedding.",
            "shared_block_modulation.",
            "video_head_modulation.",
        ),
    )
    # One fused q / k / v projection.
    for layer in model.layers:
        layer.fuse_qkv()
        layer.split_ffn()
    return model.eval()


def load_weights(model, root, device, dtype, prefix="", skip=(), keep_fp32=()):
    """Streams every safetensors shard under `root` into `model` (meta-initialized):
    keys under `prefix` (stripped), not starting with a `skip` prefix, go straight to
    `device`; floating tensors in `dtype`, or fp32 under a `keep_fp32` prefix."""
    for path in sorted(Path(root).glob("*.safetensors")):
        state = {}
        with safe_open(path, "pt") as f:
            for key in f.keys():  # noqa: SIM118 (safetensors handle, not a dict)
                name = key[len(prefix) :]
                if not key.startswith(prefix) or name.startswith(tuple(skip)):
                    continue
                t = f.get_tensor(key)
                if t.is_floating_point():
                    t = t.to(
                        torch.float32 if name.startswith(tuple(keep_fp32)) else dtype
                    )
                state[name] = t.to(device)
        model.load_state_dict(state, strict=False, assign=True)
    missing = [
        n for n, p in [*model.named_parameters(), *model.named_buffers()] if p.is_meta
    ]
    if missing:
        raise ValueError(
            f"weights missing for {len(missing)} tensors, e.g. {missing[:3]}"
        )
