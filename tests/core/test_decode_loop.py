"""Device-fed decode loops on CPU tensors: values in order with one read per
`read_every` steps, stop and error semantics, argument checks, and greedy_decode equal
to HF's static-cache greedy generate (end tokens, pad ids, masks, batches)."""

import pytest
import torch

from kohakutt.torch_backend.decode_loop import DecodeLoop, greedy_decode

try:
    from transformers import DynamicCache, Qwen2ForCausalLM, StaticCache
except ImportError:  # the HF comparisons below skip without transformers
    DynamicCache = Qwen2ForCausalLM = StaticCache = None


def _inc(read_every=4, emit=None):
    return DecodeLoop(
        lambda x: x + 1,
        torch.tensor([0]),
        lambda out, _: out,
        emit=emit,
        read_every=read_every,
    )


@pytest.mark.parametrize("read_every", [1, 3, 8])
def test_values_in_order_one_read_per_batch(monkeypatch, read_every):
    loop, reads = _inc(read_every), []
    stack = torch.stack

    def counted(values):
        reads.append(len(values))
        return stack(values)

    monkeypatch.setattr(torch, "stack", counted)
    values = loop.run(10, ahead=(torch.tensor([0]),))
    assert [int(v) for v in values] == list(range(11))
    assert sum(reads) == 11 and max(reads) <= max(read_every, 2)  # ahead + 1 step
    assert loop.steps == 10
    assert [int(v) for v in loop.run(2)] == [11, 12]  # continues


def test_stop_drops_later_values_and_asks_for_reset():
    loop = _inc(4)
    values = loop.run(10, stop=lambda v: int(v) == 5)
    assert [int(v) for v in values] == [1, 2, 3, 4, 5]
    assert loop.steps == 8  # the read after step 8 saw the stop
    with pytest.raises(RuntimeError, match="reset"):
        loop.run(1)
    loop.reset(torch.tensor([5]))
    assert [int(v) for v in loop.run(2)] == [6, 7]
    # a stop on the last value read drops nothing: the loop may go on
    assert [int(v) for v in loop.run(2, stop=lambda v: int(v) == 9)] == [8, 9]
    assert [int(v) for v in loop.run(1)] == [10]


def test_error_mid_run_asks_for_reset():
    def step(x):
        if int(x) == 3:
            raise RuntimeError("boom")
        return x + 1

    loop = DecodeLoop(step, torch.tensor([0]), lambda o, _: o, read_every=8)
    with pytest.raises(RuntimeError, match="boom"):
        loop.run(10)
    with pytest.raises(RuntimeError, match="reset"):
        loop.run(1)


def test_default_emit_copies_a_reused_output():
    buffer = torch.zeros(1)

    def step(x):
        buffer.copy_(x + 1)  # the same tensor every step (as a trace's own output)
        return buffer

    loop = DecodeLoop(step, torch.zeros(1), lambda o, _: o.clone(), read_every=4)
    assert [int(v) for v in loop.run(8)] == list(range(1, 9))


def test_bare_tensor_inputs_are_one_input():
    seen = []

    def step(x):
        seen.append(tuple(x.shape))
        return x + 1

    loop = DecodeLoop(step, torch.zeros(3, 2), lambda o, _: o, read_every=2)
    loop.run(3)
    assert seen == [(3, 2)] * 3


@pytest.mark.parametrize("read_every", [0, 2.5, True, -1])
def test_read_every_must_be_a_positive_int(read_every):
    with pytest.raises(ValueError):
        DecodeLoop(lambda x: x, (), lambda o, i: i, read_every=read_every)


def _model():
    pytest.importorskip("transformers")
    from kohakutt.bench.models.census import hf_config

    torch.manual_seed(0)
    return Qwen2ForCausalLM(hf_config("qwen")).eval()


def _hf(model, ids, n, eos=None, pad=None, mask=None):
    kwargs = {"max_new_tokens": n, "do_sample": False}
    kwargs["cache_implementation"] = "static"
    if eos is not None:
        kwargs["eos_token_id"] = eos
    if pad is not None:
        kwargs["pad_token_id"] = pad
    if mask is not None:
        kwargs["attention_mask"] = mask
    with torch.no_grad():
        return model.generate(ids, **kwargs)[:, ids.shape[1] :]


def _ids(batch, length=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 512, (batch, length), generator=g)


def _ends(model, ids, n):
    """Tokens of an unstopped run, to pick end tokens that occur."""
    return _hf(model, ids, n)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("n,read_every", [(1, 4), (3, 8), (13, 4), (12, 1)])
def test_greedy_decode_equals_hf_generate(batch, n, read_every):
    model, ids = _model(), _ids(batch)
    cache = StaticCache(config=model.config, max_cache_len=8 + n)
    want = _hf(model, ids, n)
    for _ in range(2):  # the cache is reset and reused
        got = greedy_decode(model, ids, cache, n, read_every=read_every)
        assert torch.equal(got, want) and got.dtype == torch.long


@pytest.mark.parametrize("pick", ["first", "middle", "per_row", "never"])
@pytest.mark.parametrize("read_every", [1, 3, 8])
def test_end_tokens_and_padding_equal_hf(pick, read_every):
    model, ids = _model(), _ids(3)
    free = _ends(model, ids, 12)
    if pick == "first":  # every row's first token ends it
        eos = sorted({int(t) for t in free[:, 0]})
    elif pick == "middle":
        eos = int(free[0, 3])
    elif pick == "per_row":  # rows end at different steps on a list of ids
        eos = [int(free[0, 2]), int(free[1, 6]), int(free[2, 9])]
    else:
        eos = 511 if 511 not in free else 510
    pad = 7  # differs from every end id
    cache = StaticCache(config=model.config, max_cache_len=20)
    got = greedy_decode(
        model, ids, cache, 12, eos_token_id=eos, pad_token_id=pad, read_every=read_every
    )
    assert torch.equal(got, _hf(model, ids, 12, eos=eos, pad=pad))


def test_default_pad_is_the_first_end_token():
    model, ids = _model(), _ids(2)
    free = _ends(model, ids, 10)
    eos = [int(free[1, 5]), int(free[0, 2])]
    cache = StaticCache(config=model.config, max_cache_len=18)
    model.generation_config.pad_token_id = None
    got = greedy_decode(model, ids, cache, 10, eos_token_id=eos)
    assert torch.equal(got, _hf(model, ids, 10, eos=eos, pad=eos[0]))


def test_cache_reuse_after_an_early_stop():
    model, ids = _model(), _ids(1)
    free = _ends(model, ids, 12)
    cache = StaticCache(config=model.config, max_cache_len=20)
    eos = int(free[0, 2])
    greedy_decode(model, ids, cache, 12, eos_token_id=eos, pad_token_id=0)
    shorter = ids[:, 2:]
    assert torch.equal(
        greedy_decode(model, shorter, cache, 12), _hf(model, shorter, 12)
    )


@pytest.mark.parametrize("inferred", [False, True])
def test_left_padded_batch_with_mask(inferred):
    model, ids = _model(), _ids(3)
    pad = 3
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    mask[2, :1] = 0
    ids = ids.masked_fill(mask == 0, pad)
    cache = StaticCache(config=model.config, max_cache_len=20)
    want = _hf(model, ids, 10, pad=pad, mask=mask)
    got = greedy_decode(
        model,
        ids,
        cache,
        10,
        pad_token_id=pad,
        attention_mask=None if inferred else mask,
    )
    assert torch.equal(got, want)


def test_capacity_and_argument_checks():
    model, ids = _model(), _ids(1)
    small = StaticCache(config=model.config, max_cache_len=12)
    with pytest.raises(ValueError, match="exceeds"):
        greedy_decode(model, ids, small, 6)  # needs 8 + 6 - 1 = 13 positions
    assert greedy_decode(model, ids, small, 5).shape == (1, 5)  # exactly 12
    with pytest.raises(ValueError, match="static cache"):
        greedy_decode(model, ids, DynamicCache(config=model.config), 4)
    with pytest.raises(ValueError, match="eos_token_id"):
        greedy_decode(model, ids, small, 4, eos_token_id=[])
    with pytest.raises(ValueError, match="max_new_tokens"):
        greedy_decode(model, ids, small, 2.0)
    with pytest.raises(ValueError, match="attention_mask"):
        greedy_decode(model, ids, small, 2, attention_mask=torch.ones(1, 3))
    assert greedy_decode(model, ids, small, 0).shape == (1, 0)
