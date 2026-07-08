import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, bias):
        return torch.matmul(a, b) + bias
