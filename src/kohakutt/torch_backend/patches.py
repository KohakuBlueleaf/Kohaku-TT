"""Python-side torch behavior for tt tensors that torch provides only for its built-in
devices: printing, pickling / torch.save, and tied weights across Module.to("tt")."""

import torch

_format_data = torch._tensor_str._tensor_str
_reduce_ex_internal = torch.Tensor._reduce_ex_internal
_module_apply = torch.nn.Module._apply


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
    if _apply_depth == 0 and any(
        "_ktt_marks" in m.__dict__ or "_ktt_default" in m.__dict__
        for m in self.modules()
    ):
        from kohakutt.device.marks import apply_marks

        apply_marks(self)
    return result


def install():
    """Installs the three patches (printing, pickling, tied weights)."""
    torch._tensor_str._tensor_str = _tensor_str
    torch.Tensor._reduce_ex_internal = _reduce_tt
    torch.nn.Module._apply = _apply_keeping_ties
