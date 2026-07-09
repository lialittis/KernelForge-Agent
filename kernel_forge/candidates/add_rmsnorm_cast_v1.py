import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, epsilon=1e-6, target_dtype=torch.float16):
        super().__init__()
        self.epsilon = epsilon
        self.target_dtype = target_dtype
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, x, residual, gamma):
        self._last_backend = "torch_reference"
        self._last_error = None
        x_added = x + residual
        variance = x_added.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        output = (x_added * rstd) * gamma
        return output.to(self.target_dtype)
