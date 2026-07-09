import torch
import torch.nn as nn

try:
    import torch_npu
except Exception:
    torch_npu = None


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, x, cos, sin):
        if torch_npu is not None and getattr(x.device, "type", None) == "npu":
            self._last_backend = "torch_npu_npu_rotary_mul"
            self._last_error = None
            return torch_npu.npu_rotary_mul(x, cos, sin)
        self._last_backend = "torch_formula_reference"
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x_rot = torch.cat((-x2, x1), dim=-1)
        return cos * x + sin * x_rot
