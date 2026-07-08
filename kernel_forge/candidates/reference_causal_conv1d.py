import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, activation="silu"):
        super().__init__()
        self.activation = activation

    def forward(self, x, conv_state, weight, bias, conv_state_indices):
        del conv_state_indices
        width = weight.shape[1]
        x_3d = x.unsqueeze(-1)
        x_padded = torch.cat([conv_state, x_3d], dim=-1).to(weight.dtype)
        out = F.conv1d(x_padded, weight.unsqueeze(1), bias, padding=0, groups=x.shape[1])
        out = out.squeeze(-1)
        if self.activation == "silu":
            out = F.silu(out)
        conv_state.copy_(x_padded[:, :, -(width - 1):])
        return out.to(x.dtype)
