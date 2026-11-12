"""Device-fed decode loops: each step's inputs are made on the device from the previous
step's output (no upload per step), and the host reads the emitted values in batches
(one download per `read_every` steps), so the host stays ahead of the device.
"""

import inspect

import torch


def _tuple(values):
    """values as a tuple of inputs (a bare tensor is one input)."""
    return (values,) if isinstance(values, torch.Tensor) else tuple(values)


def _count(value, name, least):
    """value when it is an int (not a bool) of at least `least`, else ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ValueError(f"{name} must be an int >= {least}, got {value!r}")
    return value


class DecodeLoop:
    """Runs out = step(*inputs), then inputs = feedback(out, inputs), on the device.

    emit(out) is the device tensor kept per step (default: a clone of out); the host
    reads the kept tensors every `read_every` steps, so a stop acts up to read_every - 1
    steps late (those steps run and are dropped; then reset() before the next run)."""

    def __init__(self, step, inputs, feedback, emit=None, read_every=8):
        self.step = step
        self.feedback = feedback
        self.emit = (lambda out: out.clone()) if emit is None else emit
        self.read_every = _count(read_every, "read_every", 1)
        self.reset(inputs)

    def reset(self, inputs):
        """Starts again from `inputs` (after a stop with dropped steps or an error)."""
        self.inputs = _tuple(inputs)
        self.steps = 0  # steps run, the dropped ones included
        self.stale = False  # inputs do not follow the last returned value

    def run(self, steps, stop=None, ahead=()):
        """CPU copies of the device values `ahead`, then of up to `steps` steps' emitted
        values, in order; stop(value) True ends the loop after that value."""
        _count(steps, "steps", 0)
        if self.stale:
            raise RuntimeError(
                "DecodeLoop: steps were dropped or failed; reset() first"
            )
        values, pending = [], list(ahead)
        self.stale = True  # until this run ends with every step's value returned
        for _ in range(steps):
            out = self.step(*self.inputs)
            pending.append(self.emit(out))
            self.inputs = _tuple(self.feedback(out, self.inputs))
            self.steps += 1
            if len(pending) >= self.read_every:
                dropped = self._read(pending, values, stop)
                if dropped is not None:
                    self.stale = dropped > 0
                    return values
        dropped = self._read(pending, values, stop)
        self.stale = bool(dropped)
        return values

    @staticmethod
    def _read(pending, values, stop):
        """Moves the pending values to `values` with one download. When stop ended the
        loop: the number of values dropped after the stopping one, else None."""
        if not pending:
            return None
        host = torch.stack(pending).cpu()
        pending.clear()
        for i, value in enumerate(host):
            values.append(value)
            if stop is not None and stop(value):
                return len(host) - i - 1
        return None


def _model_of(forward):
    """The module behind forward (a module, its compiled form or a bound method)."""
    for name in ("_torchdynamo_orig_callable", "_orig_mod", "__self__"):
        inner = getattr(forward, name, None)
        if inner is not None and inner is not forward:
            return _model_of(inner)
    return forward


def _takes(forward, name):
    """True when the model's forward has a parameter `name`."""
    model = _model_of(forward)
    target = getattr(model, "forward", model)
    try:
        return name in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


def _ids(value, name):
    """value (None, int, list or tensor of token ids) as a 1-D long tensor, or None."""
    if value is None:
        return None
    ids = torch.as_tensor(value).flatten().long()
    if ids.numel() == 0 or bool((ids < 0).any()):
        raise ValueError(f"{name} must hold token ids >= 0, got {value!r}")
    return ids


def greedy_decode(
    forward,
    input_ids,
    cache,
    max_new_tokens,
    eos_token_id=None,
    pad_token_id=None,
    attention_mask=None,
    read_every=8,
):
    """(batch, new) CPU greedy tokens of forward (a causal LM or its compiled form) on
    tt input_ids with a static `cache` (reset here), as HF's greedy generate: a 0/1
    attention_mask (left padding), rows after their end token filled with pad_token_id
    (default: the model's generation_config.pad_token_id, else the first end token)."""
    batch, length = input_ids.shape
    _count(max_new_tokens, "max_new_tokens", 0)
    capacity = cache.get_max_length() if getattr(cache, "is_compileable", False) else -1
    if capacity < 0:
        raise ValueError(
            "greedy_decode needs a static cache (transformers StaticCache)"
        )
    if length + max_new_tokens - 1 > capacity:
        raise ValueError(
            f"prompt {length} + max_new_tokens {max_new_tokens} - 1 exceeds the "
            f"cache's {capacity} positions"
        )
    if max_new_tokens == 0:
        return torch.empty(batch, 0, dtype=torch.long)
    eos = _ids(eos_token_id, "eos_token_id")
    config = getattr(_model_of(forward), "generation_config", None)
    if pad_token_id is None and config is not None:
        pad_token_id = config.pad_token_id
    pad = _ids(pad_token_id, "pad_token_id")
    pad = pad[0] if pad is not None else (eos[0] if eos is not None else None)
    if attention_mask is None and pad is not None and (eos is None or pad not in eos):
        # HF infers the mask from pad tokens in the prompt (one read of the prompt)
        host = input_ids.cpu()
        if bool((host == pad).any()):
            attention_mask = (host != pad).long()
    device = input_ids.device
    extra = {}
    if attention_mask is not None:
        mask = attention_mask.cpu().long()
        if tuple(mask.shape) != (batch, length):
            raise ValueError("attention_mask must have the shape of input_ids")
        # one fixed-shape mask over the whole cache; positions count real tokens
        full = torch.cat([mask, mask.new_ones(batch, capacity - length)], dim=1)
        start = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
        extra = {"attention_mask": full.to(device)}
    prefill = dict(extra)
    if attention_mask is not None:
        prefill["position_ids"] = start.to(device)
    if _takes(forward, "logits_to_keep"):
        prefill["logits_to_keep"] = 1
    with torch.no_grad():
        cache.reset()  # zeroed in place (a trace keeps its tensors), lengths at 0
        logits = forward(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=torch.arange(length).to(device),
            use_cache=True,
            **prefill,
        ).logits
        first = logits[:, -1:].argmax(-1)

        def step(token, position, rows=None):
            kwargs = dict(extra)
            if rows is not None:
                kwargs["position_ids"] = rows
            out = forward(
                input_ids=token,
                past_key_values=cache,
                cache_position=position,
                use_cache=True,
                **kwargs,
            ).logits
            return out[:, -1:].argmax(-1)

        inputs = [first, torch.tensor([length]).to(device)]
        if attention_mask is not None:
            inputs.append(mask.sum(-1, keepdim=True).to(device))
        done = torch.zeros(batch, dtype=torch.bool)

        def stop(token):
            # token: (batch, 1); every row has emitted an end token
            if eos is None:
                return False
            done.logical_or_(torch.isin(token[:, 0].long(), eos))
            return bool(done.all())

        loop = DecodeLoop(
            step,
            inputs,
            lambda out, ins: (out, *(x + 1 for x in ins[1:])),
            emit=lambda out: out,  # a fresh argmax per step
            read_every=read_every,
        )
        rows = loop.run(max_new_tokens - 1, stop, ahead=(first,))
    tokens = torch.cat(rows, dim=1).long()
    if eos is not None:
        # after a row's first end token: the pad token
        seen = torch.isin(tokens, eos).long().cumsum(1)
        after = torch.cat([seen[:, :1] * 0, seen[:, :-1]], dim=1) > 0
        tokens = torch.where(after, pad, tokens)
    return tokens
