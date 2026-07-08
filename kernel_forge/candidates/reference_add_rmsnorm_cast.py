import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, epsilon=1e-6, target_dtype=torch.float16):
        super().__init__()
        self.epsilon = epsilon
        self.target_dtype = target_dtype

    def forward(self, x, residual, gamma):
        x_added = x + residual
        variance = x_added.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        x_normalized = x_added * rstd
        output = x_normalized * gamma
        return output.to(self.target_dtype)
