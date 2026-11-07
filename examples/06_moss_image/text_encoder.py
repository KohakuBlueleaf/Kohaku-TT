"""Qwen3.5 text model (the MOSS-Image text encoder): Gated DeltaNet and gated
full-attention layers, token ids [B, T] -> final hidden states [B, T, 4096]. Weights
from the checkpoint's text_encoder/ dir; linears in the weight dtype, vector math in
fp32.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from dit import load_weights
from torch import nn

CHUNK = 64


def zero_centred_norm(x, weight, eps):
    """Qwen3.5 RMSNorm: x / rms(x) * (1 + weight), fp32."""
    xf = x.float()
    return (
        xf
        * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        * (1.0 + weight.float())
    )


def rotary(x, cos, sin):
    """Partial RoPE on the first cos.shape[-1] dims of [T, H, Dh] (fp32)."""
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    half = rot // 2
    turned = torch.cat((-xr[..., half:], xr[..., :half]), -1)
    return torch.cat((xr * cos + turned * sin, xp), -1)


class Attention(nn.Module):
    """Gated causal GQA attention with qk-norm and partial RoPE."""

    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads = (
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
        )
        self.head_dim, self.eps = cfg["head_dim"], cfg["rms_norm_eps"]
        dim = cfg["hidden_size"]
        self.q_proj = nn.Linear(dim, 2 * self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.Module()
        self.q_norm.weight = nn.Parameter(torch.zeros(self.head_dim))
        self.k_norm = nn.Module()
        self.k_norm.weight = nn.Parameter(torch.zeros(self.head_dim))

    def forward(self, x, cos, sin):
        """[B, T, D] -> [B, T, D]."""
        b, t = x.shape[:2]
        hd, dtype = self.head_dim, x.dtype
        qg = self.q_proj(x).view(b, t, self.heads, 2 * hd)
        gate = qg[..., hd:].reshape(b, t, -1)
        q = zero_centred_norm(qg[..., :hd], self.q_norm.weight, self.eps)
        k = zero_centred_norm(
            self.k_proj(x).view(b, t, self.kv_heads, hd), self.k_norm.weight, self.eps
        )
        v = self.v_proj(x).view(b, t, self.kv_heads, hd)
        q, k = rotary(q, cos, sin).to(dtype), rotary(k, cos, sin).to(dtype)
        groups = self.heads // self.kv_heads
        q, k, v = (a.transpose(1, 2).contiguous() for a in (q, k, v))  # [B, H, T, Dh]
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        o = (o.reshape(b, t, -1).float() * torch.sigmoid(gate.float())).to(dtype)
        return self.o_proj(o)


def unit_lower_inverse(a):
    """(I + a)^-1 for strictly lower [..., C, C] a (fp32)."""
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    return torch.linalg.solve_triangular(
        eye + a, eye.expand_as(a), upper=False, unitriangular=True
    )


def gated_delta_rule_reference(q, k, v, g, beta, dtype):
    """Chunked gated delta rule on [..., L, D] q / k / v and [..., L] g / beta (L a
    multiple of CHUNK; leading dims e.g. [B, H]), fp32 vector math, matmul operands in
    `dtype`, one chunk at a time; returns [..., L, Dv] fp32 (reference for
    gated_delta_rule)."""
    *lead, length, dk = k.shape
    dv, nc = v.shape[-1], length // CHUNK
    scale = dk**-0.5
    q, k, v = (t.reshape(*lead, nc, CHUNK, t.shape[-1]) for t in (q, k, v))
    g, beta = g.reshape(*lead, nc, CHUNK), beta.reshape(*lead, nc, CHUNK)

    def mm(a, b):
        return (a.to(dtype) @ b.to(dtype)).float()

    rk = torch.rsqrt(k.square().sum(-1) + 1e-6)
    rq = torch.rsqrt(q.square().sum(-1) + 1e-6) * scale
    later = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device).tril(-1)
    low = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device).tril()
    decay = (g.unsqueeze(-1) * later).cumsum(-2)  # D[i, j] = sum_{j < t <= i} g_t
    kk = mm(k, k.transpose(-1, -2)) * rk.unsqueeze(-1) * rk.unsqueeze(-2)
    t_inv = unit_lower_inverse(
        torch.where(later, kk * beta.unsqueeze(-1) * decay.exp(), 0.0)
    )
    qk = mm(q, k.transpose(-1, -2)) * rq.unsqueeze(-1) * rk.unsqueeze(-2)
    qk = torch.where(low, qk * decay.exp(), 0.0)
    gc = g.cumsum(-1)
    suffix = (g.unsqueeze(-1) * later).sum(-2)
    fy = (rk * beta * gc.exp()).unsqueeze(-1)
    fq = (rq * gc.exp()).unsqueeze(-1)
    fk = (rk * suffix.exp()).unsqueeze(-1)
    tail = g.sum(-1).exp()

    # One chunk at a time: the recurrent state [..., Dk, Dv] carries across chunks.
    state = q.new_zeros(*lead, dk, dv)
    outs = []
    for c in range(nc):
        kc, qc = k[..., c, :, :], q[..., c, :, :]
        y = v[..., c, :, :] * beta[..., c, :].unsqueeze(-1) - fy[..., c, :, :] * mm(
            kc, state
        )
        vn = mm(t_inv[..., c, :, :], y)
        outs.append(fq[..., c, :, :] * mm(qc, state) + mm(qk[..., c, :, :], vn))
        state = state * tail[..., c, None, None] + mm(
            kc.transpose(-1, -2), fk[..., c, :, :] * vn
        )
    return torch.stack(outs, -3).reshape(*lead, length, dv)


def gated_delta_rule(q, k, v, g, beta, dtype):
    """Chunked gated delta rule on [..., L, D] q / k / v and [..., L] g / beta (L a
    multiple of CHUNK; leading dims e.g. [B, H]), fp32 vector math, matmul operands in
    `dtype`; returns [..., L, Dv] fp32. u = T (beta v) and w = T (f_y k) for every chunk
    at once; the recurrence over chunks is v_new = u - w S and S = tail S + (f_k k)^T
    v_new; the output o = (f_q q) S_c + (gated q k^T) v_new for every chunk after it.
    Chunks lead the layout (a chunk is a contiguous slice)."""
    *lead, length, dk = k.shape
    dv, nc = v.shape[-1], length // CHUNK
    scale = dk**-0.5
    rank = len(lead)

    def chunks_first(t):
        t = t.reshape(*lead, nc, CHUNK, *t.shape[rank + 1 :])
        return t.movedim(rank, 0).contiguous()

    q, k, v = chunks_first(q), chunks_first(k), chunks_first(v)
    g, beta = chunks_first(g), chunks_first(beta)

    def mm(a, b):
        return (a.to(dtype) @ b.to(dtype)).float()

    rk = torch.rsqrt(k.square().sum(-1) + 1e-6)
    rq = torch.rsqrt(q.square().sum(-1) + 1e-6) * scale
    later = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device).tril(-1)
    low = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device).tril()
    decay = (g.unsqueeze(-1) * later).cumsum(-2)
    kk = mm(k, k.transpose(-1, -2)) * rk.unsqueeze(-1) * rk.unsqueeze(-2)
    t_inv = unit_lower_inverse(
        torch.where(later, kk * beta.unsqueeze(-1) * decay.exp(), 0.0)
    )
    qk = mm(q, k.transpose(-1, -2)) * rq.unsqueeze(-1) * rk.unsqueeze(-2)
    qk = torch.where(low, qk * decay.exp(), 0.0)
    gc = g.cumsum(-1)
    suffix = (g.unsqueeze(-1) * later).sum(-2)
    u = mm(t_inv, v.float() * beta.unsqueeze(-1))
    w = mm(t_inv, (rk * beta * gc.exp()).unsqueeze(-1) * k.float())
    kd_t = ((rk * suffix.exp()).unsqueeze(-1) * k.float()).transpose(-1, -2)
    tail = g.sum(-1).exp()[..., None, None]

    state = q.new_zeros(*lead, dk, dv, dtype=torch.float32)
    states, news = [], []
    for c in range(nc):
        states.append(state)
        vn = u[c] - mm(w[c], state)
        news.append(vn)
        state = state * tail[c] + mm(kd_t[c], vn)
    fq = (rq * gc.exp()).unsqueeze(-1) * q.float()
    o = mm(fq, torch.stack(states)) + mm(qk, torch.stack(news))
    return o.movedim(0, rank).reshape(*lead, length, dv)


class DeltaNet(nn.Module):
    """Gated DeltaNet layer: causal depthwise conv + SiLU on qkv, chunked gated delta
    rule, gated RMSNorm, output projection."""

    def __init__(self, cfg):
        super().__init__()
        dim = cfg["hidden_size"]
        self.k_heads, self.v_heads = (
            cfg["linear_num_key_heads"],
            cfg["linear_num_value_heads"],
        )
        self.dk, self.dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        self.eps = cfg["rms_norm_eps"]
        key_dim, value_dim = self.k_heads * self.dk, self.v_heads * self.dv
        self.split = (key_dim, key_dim, value_dim)
        width = cfg["linear_conv_kernel_dim"]
        self.in_proj_qkv = nn.Linear(dim, 2 * key_dim + value_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, value_dim, bias=False)
        self.in_proj_a = nn.Linear(dim, self.v_heads, bias=False)
        self.in_proj_b = nn.Linear(dim, self.v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            2 * key_dim + value_dim,
            2 * key_dim + value_dim,
            width,
            groups=2 * key_dim + value_dim,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.zeros(self.v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.v_heads))
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(self.dv))
        self.out_proj = nn.Linear(value_dim, dim, bias=False)

    def forward(self, x):
        """[B, T, D] -> [B, T, D]."""
        b, t = x.shape[:2]
        dtype = x.dtype

        # Causal depthwise conv over time (taps t - W + 1 .. t), then SiLU, fp32.
        qkv = self.in_proj_qkv(x).float()
        w = self.conv1d.weight[:, 0].float()
        width = w.shape[1]
        padded = F.pad(qkv, (0, 0, width - 1, 0))
        conv = sum(padded[:, j : j + t] * w[:, j] for j in range(width))
        q, k, v = F.silu(conv).split(self.split, -1)

        # Gates per value head; q / k shared by groups of value heads.
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias.float()
        )
        beta = torch.sigmoid(self.in_proj_b(x).float())
        groups = self.v_heads // self.k_heads
        q = q.contiguous().view(b, t, self.k_heads, self.dk)
        k = k.contiguous().view(b, t, self.k_heads, self.dk)
        v = v.contiguous().view(b, t, self.v_heads, self.dv)

        # Chunked rule on [B, H, L, D], the sequence padded to a multiple of CHUNK;
        # heads move first before the padding and the q / k group repeat (sequence
        # and head dim stay the trailing dims).
        length = math.ceil(t / CHUNK) * CHUNK

        def heads_first(a):
            return F.pad(a.transpose(1, 2), (0, 0, 0, length - t))

        g_pad = F.pad(g, (0, 0, 0, length - t)).transpose(1, 2)
        b_pad = F.pad(beta, (0, 0, 0, length - t)).transpose(1, 2)
        o = gated_delta_rule(
            heads_first(q).repeat_interleave(groups, 1),
            heads_first(k).repeat_interleave(groups, 1),
            heads_first(v),
            g_pad,
            b_pad,
            dtype,
        )
        o = o.transpose(1, 2)[:, :t]

        # Gated RMSNorm per value head, then the output projection.
        z = self.in_proj_z(x).float().view(b, t, self.v_heads, self.dv)
        o = (
            o
            * torch.rsqrt(o.square().mean(-1, keepdim=True) + self.eps)
            * self.norm.weight.float()
        )
        o = (o * F.silu(z)).reshape(b, t, -1).to(dtype)
        return self.out_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, inner = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.up_proj = nn.Linear(dim, inner, bias=False)
        self.down_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        h = F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()
        return self.down_proj(h.to(x.dtype))


class Layer(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.eps = cfg["rms_norm_eps"]
        dim = cfg["hidden_size"]
        self.input_layernorm = nn.Module()
        self.input_layernorm.weight = nn.Parameter(torch.zeros(dim))
        self.post_attention_layernorm = nn.Module()
        self.post_attention_layernorm.weight = nn.Parameter(torch.zeros(dim))
        if kind == "full_attention":
            self.self_attn = Attention(cfg)
        else:
            self.linear_attn = DeltaNet(cfg)
        self.mlp = MLP(cfg)

    def forward(self, h, cos, sin):
        dtype = h.dtype
        x = zero_centred_norm(h, self.input_layernorm.weight, self.eps).to(dtype)
        mixed = (
            self.self_attn(x, cos, sin)
            if hasattr(self, "self_attn")
            else self.linear_attn(x)
        )
        h = (h.float() + mixed.float()).to(dtype)
        x = zero_centred_norm(h, self.post_attention_layernorm.weight, self.eps).to(
            dtype
        )
        return (h.float() + self.mlp(x).float()).to(dtype)


class QwenTextModel(nn.Module):
    """Token ids [B, T] -> final hidden states [B, T, hidden]. Causal: a prompt
    right-padded to T gives its own states at its own positions."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg["layer_types"])
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.zeros(cfg["hidden_size"]))

    def rope(self, length, device):
        """cos / sin [T, 1, rot] for positions 0..T-1 (text-only mRoPE = plain RoPE)."""
        rope = self.cfg["rope_parameters"]
        rot = int(self.cfg["head_dim"] * rope["partial_rotary_factor"])
        inv = 1.0 / rope["rope_theta"] ** (
            torch.arange(0, rot, 2, dtype=torch.float64) / rot
        )
        phase = torch.arange(length, dtype=torch.float64)[:, None] * inv
        phase = torch.cat((phase, phase), -1).float()
        return phase.cos()[:, None].to(device), phase.sin()[:, None].to(device)

    def forward(self, ids):
        h = self.embed_tokens(ids)
        cos, sin = self.rope(ids.shape[-1], h.device)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return zero_centred_norm(h, self.norm.weight, self.cfg["rms_norm_eps"]).to(
            h.dtype
        )


def load_text_encoder(root, device, dtype=torch.bfloat16):
    """QwenTextModel from the checkpoint's text_encoder/ dir, weights on `device`."""
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text())["text_config"]
    with torch.device("meta"):
        model = QwenTextModel(cfg)
    load_weights(model, root, device, dtype, prefix="language_model.")
    return model.eval()
