import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, input_tensor):
        self._last_backend = "torch_reference"
        self._last_error = None
        return torch.softmax(input_tensor, dim=self.dim)
