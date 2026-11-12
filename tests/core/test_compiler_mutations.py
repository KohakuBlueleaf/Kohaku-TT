"""ktt_tt's in-place inputs (used in place by its traces, never copied): a graph's
placeholders written by an in-place method, a setitem, or an op whose schema writes an
argument (`Tensor(a!)`, out= included), directly or through a view."""

import operator

import torch
import torch.fx as fx

from kohakutt.torch_backend.compiler import _mutated_inputs


def _graph(build):
    g = fx.Graph()
    a, b, c = (g.placeholder(n) for n in "abc")
    g.output(build(g, a, b, c))
    return fx.GraphModule(torch.nn.Module(), g)


def test_schema_writes_are_in_place_inputs():
    def build(g, a, b, c):
        view = g.call_method("view", (c, -1))
        return g.call_function(torch.ops.aten.add.out, (a, b), {"out": view})

    assert _mutated_inputs(_graph(build)) == {2}


def test_packet_and_overload_targets():
    for target in (torch.ops.aten.index_put_, torch.ops.aten.index_put_.default):

        def build(g, a, b, c, target=target):
            return g.call_function(target, (b, [a], c))

        assert _mutated_inputs(_graph(build)) == {1}


def test_methods_setitem_and_pure_ops():
    def build(g, a, b, c):
        g.call_method("add_", (a, 1))
        g.call_function(operator.setitem, (b, 0, 1.0))
        return g.call_function(torch.ops.aten.add.Tensor, (a, c))

    assert _mutated_inputs(_graph(build)) == {0, 1}
