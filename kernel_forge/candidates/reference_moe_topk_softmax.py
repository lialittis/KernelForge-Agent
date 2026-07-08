import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, top_k=2):
        super().__init__()
        self.top_k = top_k

    def forward(self, gating_logits):
        gating_probs = torch.nn.functional.softmax(gating_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(gating_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)
        return top_k_probs, top_k_indices
