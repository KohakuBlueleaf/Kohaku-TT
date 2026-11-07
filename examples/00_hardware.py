"""The cards on this host (read-only), then chip 0 as a torch device and its compute
grid. Usage: python examples/00_hardware.py
"""

import torch

import kohakutt
from kohakutt.core.doctor import format_report, run_doctor

print(format_report(run_doctor()))

dev = kohakutt.device(0)  # chip 0 as a torch.device
cols, rows = kohakutt.compute_grid(dev)
print(f"{dev} = {kohakutt.mesh_size(dev)} chip, {cols}x{rows} Tensix compute grid")
print("torch.tt.is_available():", torch.tt.is_available())
