"""Python-side torch behavior for tt tensors that torch provides only for its built-in
devices: printing, pickling / torch.save, tied weights across Module.to("tt"), and the
optimizers' capturable check in a capture."""

import torch

_format_data = torch._tensor_str._tensor_str
_reduce_ex_internal = torch.Tensor._reduce_ex_internal
_module_apply = torch.nn.Module._apply
_health_check = torch.optim.Optimizer._accelerator_graph_capture_health_check


def _tensor_str(self, indent):
    """torch's data formatter, on a CPU copy for tt tensors."""
    return _format_data(self.cpu() if self.device.type == "tt" else self, indent)


def _reduce_tt(self, proto):
    """torch's tensor reduce; tt tensors become CPU copies rebuilt on their device."""
    if self.device.type != "tt":
        return _reduce_ex_internal(self, proto)
    return (
        torch._utils._rebuild_device_tensor_from_cpu_tensor,
        (self.cpu(), self.dtype, str(self.device), self.requires_grad),
    )


_apply_depth = 0


def _apply_keeping_ties(self, fn, recurse=True):
    """torch's Module._apply, then every slot that shared a Parameter shares again; the
    outermost call then places marked tensors at their marks (kohakutt.device.marks)."""
    global _apply_depth
    slots = {}
    for owner in self.modules():
        for name, param in owner._parameters.items():
            if param is not None:
                slots.setdefault(id(param), []).append((owner, name))
    _apply_depth += 1
    try:
        result = _module_apply(self, fn, recurse)
    finally:
        _apply_depth -= 1
    for shared in slots.values():
        if len(shared) > 1:
            first_owner, first_name = shared[0]
            kept = first_owner._parameters[first_name]
            for owner, name in shared[1:]:
                owner._parameters[name] = kept
    if _apply_depth == 0:
        from kohakutt.device.marks import apply_marks

        apply_marks(self)
    return result


def _capture_health_check(self):
    """torch's check, and in a kohakutt capture: every param group is capturable (a
    CPU step count would be baked into the trace)."""
    from kohakutt.torch_backend.trace import capturing

    if capturing() and not all(g.get("capturable") for g in self.param_groups):
        raise RuntimeError(
            f"kohakutt.capture of {type(self).__name__}.step() with capturable=False: "
            "its step count stays on the CPU and the trace would replay a stale one; "
            f"construct it with {type(self).__name__}(..., capturable=True)"
        )
    _health_check(self)


def install():
    """Installs the four patches (printing, pickling, tied weights, the optimizers'
    capture check)."""
    torch._tensor_str._tensor_str = _tensor_str
    torch.Tensor._reduce_ex_internal = _reduce_tt
    torch.nn.Module._apply = _apply_keeping_ties
    torch.optim.Optimizer._accelerator_graph_capture_health_check = (
        _capture_health_check
    )
