import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, sm_scale, page_size):
        super().__init__()
        self.sm_scale = sm_scale
        self.page_size = page_size

    def forward(self, q, k_nope_buffer, k_rope_buffer, v_buffer, seq_lens, block_table):
        batch = q.shape[0]
        num_q_heads = q.shape[1]
        qk_nope_dim = k_nope_buffer.shape[-1]
        qk_rope_dim = k_rope_buffer.shape[-1]
        num_kv_heads = k_nope_buffer.shape[2]
        v_dim = v_buffer.shape[-1]

        outputs = []
        for i in range(batch):
            kv_len = seq_lens[i].item()
            q_i = q[i:i + 1]
            q_nope = q_i[:, :, :qk_nope_dim]
            q_rope = q_i[:, :, qk_nope_dim:]

            num_blocks = (kv_len + self.page_size - 1) // self.page_size
            block_ids = block_table[i, :num_blocks]

            k_nope = k_nope_buffer[block_ids].view(-1, num_kv_heads, qk_nope_dim)[:kv_len]
            k_rope = k_rope_buffer[block_ids].view(-1, num_kv_heads, qk_rope_dim)[:kv_len]
            v = v_buffer[block_ids].view(-1, num_kv_heads, v_dim)[:kv_len]

            if num_q_heads != num_kv_heads:
                rep_factor = num_q_heads // num_kv_heads
                k_nope = torch.repeat_interleave(k_nope, rep_factor, dim=1)
                k_rope = torch.repeat_interleave(k_rope, rep_factor, dim=1)
                v = torch.repeat_interleave(v, rep_factor, dim=1)

            qk_nope = torch.einsum("qhd,khd->hqk", q_nope, k_nope).float()
            qk_rope = torch.einsum("qhd,khd->hqk", q_rope, k_rope).float()
            qk = (qk_nope + qk_rope) * self.sm_scale
            score = torch.softmax(qk, dim=-1).to(v.dtype)
            out_i = torch.einsum("hqk,khd->qhd", score, v)
            outputs.append(out_i)

        return torch.cat(outputs, dim=0)
