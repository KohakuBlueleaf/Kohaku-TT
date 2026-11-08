"""torch.tt: the device module of the "tt" backend (the counterpart of torch.cuda).

tt:0 is the one logical mesh this process opened (1x1, or 1xN with replicated tensors);
there is no tt:1. Seeds and RNG state belong to the default tt generator.
"""

import torch


def is_available():
    """True: the native backend is loaded."""
    return True


def is_initialized():
    """True when a mesh is open."""
    return torch.ops.ktt.mesh_size() > 0


def _is_in_bad_fork():
    """False: the backend keeps no fork-unsafe state torch has to check."""
    return False


def device_count():
    """1: the open mesh is the single tt device."""
    return 1


def current_device():
    """0: the only tt device."""
    return 0


def set_device(device):
    """Accepts tt:0 (or 0) only."""
    index = torch.device(device).index if not isinstance(device, int) else device
    if index not in (0, None):
        raise ValueError(f"tt has one device (tt:0, the open mesh); got {device}")


def mesh_size():
    """Number of chips in the open mesh (0 when closed)."""
    return torch.ops.ktt.mesh_size()


def synchronize(device=None):
    """Waits until all queued device work is done."""
    torch.ops.ktt.synchronize()


def memory_allocated(device=None):
    """Device bytes held by live tt tensors (padded), summed over the mesh's chips (a
    replicated tensor counts once per chip)."""
    return torch.ops.ktt.memory_stats()[0] * max(1, mesh_size())


def max_memory_allocated(device=None):
    """Peak of memory_allocated() since the last reset."""
    return torch.ops.ktt.memory_stats()[1] * max(1, mesh_size())


def mem_get_info(device=None):
    """(free, total) DRAM bytes summed over the mesh's chips."""
    free, total = torch.ops.ktt.mem_get_info()
    return free, total


def mesh_stats(device=None):
    """Mesh fallbacks since load or the last reset: {gathers, gather_bytes} (ops without
    a placement rule that gathered an input to replicated), {device_gathers} (of those,
    through TT-NN collectives) and {collective_failures} (collectives that raised; the
    host path ran instead)."""
    gathers, nbytes, device_gathers, failures = torch.ops.ktt.mesh_counters()
    return {
        "gathers": gathers,
        "gather_bytes": nbytes,
        "device_gathers": device_gathers,
        "collective_failures": failures,
    }


def reset_mesh_stats(device=None):
    """Zeroes mesh_stats()."""
    torch.ops.ktt.reset_mesh_counters()


def reset_peak_memory_stats(device=None):
    """Resets max_memory_allocated() to the current allocation."""
    torch.ops.ktt.reset_peak_memory()


def manual_seed(seed):
    """Seeds the default tt generator."""
    torch.ops.ktt.manual_seed(int(seed))


def manual_seed_all(seed):
    """Seeds the default tt generator (one device)."""
    manual_seed(seed)


def initial_seed():
    """The seed of the default tt generator."""
    return torch.ops.ktt.initial_seed()


def get_rng_state(device="tt"):
    """The default tt generator's state as a uint8 CPU tensor."""
    return torch.ops.ktt.get_rng_state()


def set_rng_state(new_state, device="tt"):
    """Restores the default tt generator's state."""
    torch.ops.ktt.set_rng_state(new_state)
