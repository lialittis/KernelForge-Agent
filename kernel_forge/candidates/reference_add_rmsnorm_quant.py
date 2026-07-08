import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x, residual, gamma, scale, zero_point):
        x_added = x + residual
        rstd = torch.rsqrt(x_added.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        output = (x_added * rstd) * gamma
        return torch.round(output / scale + zero_point).clamp(-128, 127).to(torch.int8)
