"""Small models: MLP and convnet (with training), ViT-style encoder, GPT-style decoder.
Models are functions of a dict of parameter Tensors, so the same numpy weights run on
any device. A `Model` builds inputs from a seed; `train` uses tinygrad's AdamW.
"""

from __future__ import annotations

import numpy as np

from kohakutt.tinygrad_backend import compat as C

Tensor = C.Tensor


def _w(rng, *shape, scale=None):
    scale = (
        scale
        if scale is not None
        else 1.0 / np.sqrt(shape[-1] if len(shape) > 1 else 1)
    )
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _linear(P, name, x):
    return x @ P[name + ".w"].T + P[name + ".b"]


def _layernorm(P, name, x):
    return x.layernorm() * P[name + ".g"] + P[name + ".be"]


def _attention(P, name, x, heads, causal, mask=None):
    B, T, D = x.shape
    qkv = (
        _linear(P, name + ".qkv", x)
        .reshape(B, T, 3, heads, D // heads)
        .permute(2, 0, 3, 1, 4)
    )
    a = qkv[0].scaled_dot_product_attention(
        qkv[1], qkv[2], attn_mask=mask, is_causal=causal
    )
    return _linear(P, name + ".proj", a.transpose(1, 2).reshape(B, T, D))


def _block(P, name, x, heads, causal, mask=None):
    x = x + _attention(
        P, name + ".att", _layernorm(P, name + ".ln1", x), heads, causal, mask
    )
    return x + _linear(
        P,
        name + ".fc2",
        _linear(P, name + ".fc1", _layernorm(P, name + ".ln2", x)).gelu(),
    )


def _block_params(rng, name, D):
    p = {}
    for ln in ("ln1", "ln2"):
        p[f"{name}.{ln}.g"], p[f"{name}.{ln}.be"] = np.ones(D, np.float32), np.zeros(
            D, np.float32
        )
    p[f"{name}.att.qkv.w"], p[f"{name}.att.qkv.b"] = _w(rng, 3 * D, D), np.zeros(
        3 * D, np.float32
    )
    p[f"{name}.att.proj.w"], p[f"{name}.att.proj.b"] = _w(rng, D, D), np.zeros(
        D, np.float32
    )
    p[f"{name}.fc1.w"], p[f"{name}.fc1.b"] = _w(rng, 4 * D, D), np.zeros(
        4 * D, np.float32
    )
    p[f"{name}.fc2.w"], p[f"{name}.fc2.b"] = _w(rng, D, 4 * D), np.zeros(D, np.float32)
    return p


def _batchnorm(P, name, x, eps=1e-5):
    axes = (0, 2, 3)
    mean = x.mean(axis=axes, keepdim=True)
    var = ((x - mean) ** 2).mean(axis=axes, keepdim=True)
    shape = (1, -1, 1, 1)
    return (x - mean) * (var + eps).rsqrt() * P[name + ".g"].reshape(shape) + P[
        name + ".be"
    ].reshape(shape)


class Model:
    """A model kind: numpy parameters and inputs, a forward function, a loss."""

    def __init__(self, name, params, inputs, forward, targets=None):
        self.name, self.params, self.inputs, self.forward, self.targets = (
            name,
            params,
            inputs,
            forward,
            targets,
        )

    def tensors(self, device):
        return {k: Tensor(v, device=device).realize() for k, v in self.params.items()}

    def run(self, device):
        """Forward logits on `device` as a numpy array."""
        P = self.tensors(device)
        x = Tensor(self.inputs, device=device).realize()
        return self.forward(P, x).numpy()

    def train(self, device, steps, lr=1e-2):
        """(losses, final parameters) of `steps` AdamW steps on `device`."""
        P = self.tensors(device)
        x = Tensor(self.inputs, device=device).realize()
        y = Tensor(self.targets, device=device).realize()
        opt = C.nn.optim.AdamW(list(P.values()), lr=lr)
        losses = []
        with C.Context(TRAINING=1):
            for _ in range(steps):
                opt.zero_grad()
                logits = self.forward(P, x)
                loss = logits.reshape(
                    -1, logits.shape[-1]
                ).sparse_categorical_crossentropy(y.reshape(-1))
                loss.backward()
                Tensor.realize(loss, *[p.grad for p in P.values()])
                losses.append(
                    float(loss.item())
                )  # read before the step: the optimizer assigns the parameters in place
                opt.step()
        return losses, {k: v.numpy() for k, v in P.items()}


def mlp(seed=0):
    rng = np.random.default_rng(seed)
    p = {
        "l1.w": _w(rng, 128, 64),
        "l1.b": np.zeros(128, np.float32),
        "l2.w": _w(rng, 10, 128),
        "l2.b": np.zeros(10, np.float32),
    }
    x = rng.standard_normal((32, 64)).astype(np.float32)
    y = rng.integers(0, 10, 32).astype(np.int32)
    return Model(
        "mlp", p, x, lambda P, x: _linear(P, "l2", _linear(P, "l1", x).silu()), y
    )


def cnn(seed=0):
    rng = np.random.default_rng(seed)
    p = {
        "c1.w": _w(rng, 16, 3, 3, 3, scale=0.2),
        "bn1.g": np.ones(16, np.float32),
        "bn1.be": np.zeros(16, np.float32),
        "c2.w": _w(rng, 32, 16, 3, 3, scale=0.1),
        "bn2.g": np.ones(32, np.float32),
        "bn2.be": np.zeros(32, np.float32),
        "fc.w": _w(rng, 10, 32),
        "fc.b": np.zeros(10, np.float32),
    }
    x = rng.standard_normal((16, 3, 16, 16)).astype(np.float32)
    y = rng.integers(0, 10, 16).astype(np.int32)

    def forward(P, x):
        # no conv bias: before batch norm its gradient is zero and Adam amplifies noise
        x = _batchnorm(P, "bn1", x.conv2d(P["c1.w"], padding=1)).relu().max_pool2d(2)
        x = _batchnorm(P, "bn2", x.conv2d(P["c2.w"], padding=1)).relu()
        return _linear(P, "fc", x.mean(axis=(2, 3)))

    return Model("cnn", p, x, forward, y)


def vit(seed=0, D=128, depth=2, heads=4, image=32, patch=4):
    """Patch-embedding conv, positions, `depth` pre-norm encoder blocks, pool, head."""
    rng = np.random.default_rng(seed)
    n = (image // patch) ** 2
    p = {
        "patch.w": _w(rng, D, 3, patch, patch, scale=0.1),
        "patch.b": np.zeros(D, np.float32),
        "pos": _w(rng, n, D, scale=0.02),
    }
    for i in range(depth):
        p.update(_block_params(rng, f"b{i}", D))
    p.update(
        {
            "lnf.g": np.ones(D, np.float32),
            "lnf.be": np.zeros(D, np.float32),
            "head.w": _w(rng, 10, D),
            "head.b": np.zeros(10, np.float32),
        }
    )
    x = rng.standard_normal((2, 3, image, image)).astype(np.float32)

    def forward(P, x):
        h = x.conv2d(P["patch.w"], P["patch.b"], stride=patch)  # [B, D, g, g]
        h = h.flatten(2).transpose(1, 2) + P["pos"]
        for i in range(depth):
            h = _block(P, f"b{i}", h, heads, causal=False)
        return _linear(P, "head", _layernorm(P, "lnf", h).mean(axis=1))

    return Model("vit", p, x, forward, rng.integers(0, 10, 2).astype(np.int32))


def gpt(seed=0, D=128, depth=2, heads=4, T=32, vocab=128, explicit_mask=False):
    """Token + position embeddings, `depth` causal pre-norm decoder blocks, LM head."""
    rng = np.random.default_rng(seed)
    p = {"wte": _w(rng, vocab, D, scale=0.1), "wpe": _w(rng, T, D, scale=0.02)}
    for i in range(depth):
        p.update(_block_params(rng, f"b{i}", D))
    p.update(
        {
            "lnf.g": np.ones(D, np.float32),
            "lnf.be": np.zeros(D, np.float32),
            "head.w": _w(rng, vocab, D),
            "head.b": np.zeros(vocab, np.float32),
        }
    )
    ids = rng.integers(0, vocab, (2, T)).astype(np.int32)

    def forward(P, ids):
        h = C.nn.Embedding(vocab, D)
        h.weight = P["wte"]
        x = h(ids) + P["wpe"]
        mask = (
            Tensor.ones(T, T, device=ids.device).tril().cast(C.dtypes.bool)
            if explicit_mask
            else None
        )
        for i in range(depth):
            x = _block(P, f"b{i}", x, heads, causal=not explicit_mask, mask=mask)
        return _linear(P, "head", _layernorm(P, "lnf", x))

    name = "gpt (explicit mask)" if explicit_mask else "gpt"
    return Model(name, p, ids, forward, rng.integers(0, vocab, (2, T)).astype(np.int32))


MODELS = {
    "mlp": mlp,
    "cnn": cnn,
    "vit": vit,
    "gpt": gpt,
    "gpt_mask": lambda: gpt(explicit_mask=True),
}
