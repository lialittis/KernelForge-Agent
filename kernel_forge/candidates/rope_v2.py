import torch
import torch.nn as nn

try:
    import torch_npu
except Exception:
    torch_npu = None

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_BLOCK_SIZE = 128
_HAS_TRITON = (
    triton is not None
    and tl is not None
    and hasattr(triton, "jit")
    and hasattr(tl, "arange")
    and hasattr(tl, "where")
    and hasattr(tl, "constexpr")
)


if _HAS_TRITON:

    @triton.jit
    def _rope_row_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (offsets < head_dim)
        col_is_first_half = offsets < half_dim
        partner_offsets = tl.where(col_is_first_half, offsets + half_dim, offsets - half_dim)
        rotate_sign = tl.where(col_is_first_half, -1.0, 1.0)

        row_base = row * head_dim
        seq = row - (row // seq_len) * seq_len
        trig_base = seq * head_dim
        x = tl.load(x_ptr + row_base + offsets, mask=mask, other=0.0)
        x_partner = tl.load(x_ptr + row_base + partner_offsets, mask=mask, other=0.0)
        c = tl.load(cos_ptr + trig_base + offsets, mask=mask, other=0.0)
        s = tl.load(sin_ptr + trig_base + offsets, mask=mask, other=0.0)
        y = c * x + s * (rotate_sign * x_partner)
        tl.store(out_ptr + row_base + offsets, y, mask=mask)

else:
    _rope_row_kernel = None


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._disable_triton = False
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, x, cos, sin):
        if not self._can_try_triton(x, cos, sin):
            return self._torch_reference(x, cos, sin, "torch_fallback_unavailable")

        output = torch.empty_like(x)
        n_rows = x.numel() // x.shape[-1]
        try:
            _rope_row_kernel[(n_rows,)](
                x,
                cos,
                sin,
                output,
                n_rows,
                x.shape[-2],
                x.shape[-1],
                x.shape[-1] // 2,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
            self._last_backend = "triton_rope_row_bs128"
            self._last_error = None
            return output
        except Exception as exc:
            self._disable_triton = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._torch_reference(x, cos, sin, "torch_fallback_after_error")

    def _can_try_triton(self, x, cos, sin):
        if self._disable_triton or _rope_row_kernel is None:
            return False
        if getattr(x.device, "type", None) != "npu":
            return False
        return (
            x.is_contiguous()
            and cos.is_contiguous()
            and sin.is_contiguous()
            and x.ndim == 4
            and cos.ndim == 4
            and sin.ndim == 4
            and cos.shape[0] == 1
            and cos.shape[1] == 1
            and sin.shape[0] == 1
            and sin.shape[1] == 1
            and cos.shape[-2:] == x.shape[-2:]
            and sin.shape[-2:] == x.shape[-2:]
            and x.shape[-1] <= _BLOCK_SIZE
            and x.shape[-1] % 2 == 0
            and x.dtype == torch.float16
            and cos.dtype == torch.float16
            and sin.dtype == torch.float16
        )

    def _torch_reference(self, x, cos, sin, reason):
        self._last_backend = reason
        if torch_npu is not None and getattr(x.device, "type", None) == "npu":
            return torch_npu.npu_rotary_mul(x, cos, sin)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x_rot = torch.cat((-x2, x1), dim=-1)
        return cos * x + sin * x_rot
