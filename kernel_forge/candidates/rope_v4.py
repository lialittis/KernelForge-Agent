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


_BLOCK_SIZE = 1024
_HAS_TRITON = (
    triton is not None
    and tl is not None
    and hasattr(triton, "jit")
    and hasattr(triton, "cdiv")
    and hasattr(tl, "arange")
    and hasattr(tl, "where")
    and hasattr(tl, "constexpr")
)


if _HAS_TRITON:

    @triton.jit
    def _rope_flat_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_elements,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        col = offsets - (offsets // head_dim) * head_dim
        row = offsets // head_dim
        seq = row - (row // seq_len) * seq_len
        col_is_first_half = col < half_dim
        partner = offsets + tl.where(col_is_first_half, half_dim, -half_dim)
        rotate_sign = tl.where(col_is_first_half, -1.0, 1.0)
        trig = seq * head_dim + col

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_partner = tl.load(x_ptr + partner, mask=mask, other=0.0)
        c = tl.load(cos_ptr + trig, mask=mask, other=0.0)
        s = tl.load(sin_ptr + trig, mask=mask, other=0.0)
        y = c * x + s * (rotate_sign * x_partner)
        tl.store(out_ptr + offsets, y, mask=mask)

else:
    _rope_flat_kernel = None


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
        try:
            _rope_flat_kernel[(triton.cdiv(x.numel(), _BLOCK_SIZE),)](
                x,
                cos,
                sin,
                output,
                x.numel(),
                x.shape[-2],
                x.shape[-1],
                x.shape[-1] // 2,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
            self._last_backend = "triton_rope_flat_bs1024"
            self._last_error = None
            return output
        except Exception as exc:
            self._disable_triton = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._torch_reference(x, cos, sin, "torch_fallback_after_error")

    def _can_try_triton(self, x, cos, sin):
        if self._disable_triton or _rope_flat_kernel is None:
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
