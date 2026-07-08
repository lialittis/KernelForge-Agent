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

    def forward(self, combined):
        if torch_npu is not None and hasattr(torch_npu, "npu_swiglu"):
            try:
                output = torch_npu.npu_swiglu(combined, dim=-1)
                self._last_backend = "torch_npu_npu_swiglu"
                self._last_error = None
                return output
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"

        x, y = torch.chunk(combined, 2, dim=-1)
        self._last_backend = "torch_reference_silu_mul"
        return torch.nn.functional.silu(x) * y
