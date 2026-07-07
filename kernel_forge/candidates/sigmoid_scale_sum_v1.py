import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, x, bias):
        self._last_backend = "torch_reference"
        self._last_error = None
        return torch.sum(torch.sigmoid(x * 2.0 + bias), dim=-1, keepdim=True)
