import torch.nn as nn
import torch_npu


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, cos, sin):
        return torch_npu.npu_rotary_mul(x, cos, sin)
