import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, eps=1e-6, norm_before_gate=True, is_rms_norm=True):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.is_rms_norm = is_rms_norm

    def forward(self, x, weight, z=None):
        if self.is_rms_norm:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps) * weight
        else:
            mean = x.mean(dim=-1, keepdim=True)
            variance = x.var(dim=-1, keepdim=True, unbiased=False)
            x_normed = (x - mean) * torch.rsqrt(variance + self.eps) * weight

        if z is not None:
            gate = torch.sigmoid(z)
            if self.norm_before_gate:
                return x_normed * gate
            x_gated = x * gate
            variance = x_gated.pow(2).mean(dim=-1, keepdim=True)
            return x_gated * torch.rsqrt(variance + self.eps) * weight
        return x_normed
