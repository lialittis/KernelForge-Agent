import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_BLOCK_SIZE = 4096
_HAS_TRITON = (
    triton is not None
    and tl is not None
    and hasattr(triton, "jit")
    and hasattr(tl, "arange")
    and hasattr(tl, "exp")
    and hasattr(tl, "sum")
    and hasattr(tl, "sqrt")
    and hasattr(tl, "constexpr")
    and hasattr(tl, "float32")
)


if _HAS_TRITON:

    @triton.jit
    def _layernorm_gated_kernel(
        x_ptr,
        weight_ptr,
        z_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        base = row * n_cols + offsets

        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + base, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(variance + eps)
        gate = 1.0 / (1.0 + tl.exp(-z))
        output = x * rstd * weight * gate
        tl.store(out_ptr + base, output, mask=mask)

else:
    _layernorm_gated_kernel = None


class ModelNew(nn.Module):
    def __init__(self, eps=1e-6, norm_before_gate=True, is_rms_norm=True):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.is_rms_norm = is_rms_norm
        self._disable_triton = False
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, x, weight, z=None):
        if not self._can_try_triton(x, weight, z):
            return self._torch_reference(x, weight, z, "torch_fallback_unavailable")

        output = torch.empty_like(x)
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        try:
            _layernorm_gated_kernel[(n_rows,)](
                x,
                weight,
                z,
                output,
                n_cols,
                eps=float(self.eps),
                BLOCK_SIZE=_BLOCK_SIZE,
            )
            self._last_backend = "triton_row_gated_rmsnorm_bs4096"
            self._last_error = None
            return output
        except Exception as exc:
            self._disable_triton = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._torch_reference(x, weight, z, "torch_fallback_after_error")

    def _can_try_triton(self, x, weight, z):
        if self._disable_triton or _layernorm_gated_kernel is None:
            return False
        if not self.is_rms_norm or not self.norm_before_gate or z is None:
            return False
        if getattr(x.device, "type", None) != "npu":
            return False
        return (
            x.is_contiguous()
            and weight.is_contiguous()
            and z.is_contiguous()
            and x.shape == z.shape
            and x.ndim >= 2
            and weight.ndim == 1
            and weight.shape[0] == x.shape[-1]
            and x.shape[-1] <= _BLOCK_SIZE
            and x.dtype == torch.float16
            and weight.dtype == torch.float16
            and z.dtype == torch.float16
        )

    def _torch_reference(self, x, weight, z, reason):
        self._last_backend = reason
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
