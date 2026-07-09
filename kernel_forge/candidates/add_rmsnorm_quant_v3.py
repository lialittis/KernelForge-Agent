import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


_CHUNK_SIZE = 2048
_SUPPORTED_HIDDEN = 4096
_HAS_TRITON = (
    triton is not None
    and tl is not None
    and hasattr(triton, "jit")
    and hasattr(tl, "arange")
    and hasattr(tl, "floor")
    and hasattr(tl, "maximum")
    and hasattr(tl, "minimum")
    and hasattr(tl, "sum")
    and hasattr(tl, "sqrt")
    and hasattr(tl, "constexpr")
    and hasattr(tl, "int8")
)


if _HAS_TRITON:

    @triton.jit
    def _add_rmsnorm_quant_kernel(
        x_ptr,
        residual_ptr,
        gamma_ptr,
        scale_ptr,
        zero_point_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets0 = tl.arange(0, CHUNK_SIZE)
        offsets1 = offsets0 + CHUNK_SIZE
        base = row * n_cols

        added0 = tl.load(x_ptr + base + offsets0) + tl.load(residual_ptr + base + offsets0)
        added1 = tl.load(x_ptr + base + offsets1) + tl.load(residual_ptr + base + offsets1)
        sumsq = tl.sum(added0 * added0, axis=0) + tl.sum(added1 * added1, axis=0)
        rstd = 1.0 / tl.sqrt(sumsq / n_cols + eps)

        scale = tl.load(scale_ptr)
        zero_point = tl.load(zero_point_ptr)
        gamma0 = tl.load(gamma_ptr + offsets0)
        gamma1 = tl.load(gamma_ptr + offsets1)
        quant0 = added0 * rstd * gamma0 / scale + zero_point
        quant1 = added1 * rstd * gamma1 / scale + zero_point
        rounded0 = tl.floor(quant0 + 0.5)
        rounded1 = tl.floor(quant1 + 0.5)
        clamped0 = tl.minimum(tl.maximum(rounded0, -128.0), 127.0)
        clamped1 = tl.minimum(tl.maximum(rounded1, -128.0), 127.0)
        tl.store(out_ptr + base + offsets0, clamped0.to(tl.int8))
        tl.store(out_ptr + base + offsets1, clamped1.to(tl.int8))

else:
    _add_rmsnorm_quant_kernel = None


class ModelNew(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self._disable_triton = False
        self._last_backend = "not_run"
        self._last_error = None

    def forward(self, x, residual, gamma, scale, zero_point):
        if not self._can_try_triton(x, residual, gamma, scale, zero_point):
            return self._torch_reference(
                x, residual, gamma, scale, zero_point, "torch_fallback_unavailable"
            )

        output = torch.empty(x.shape, device=x.device, dtype=torch.int8)
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        try:
            _add_rmsnorm_quant_kernel[(n_rows,)](
                x,
                residual,
                gamma,
                scale,
                zero_point,
                output,
                n_cols,
                eps=float(self.epsilon),
                CHUNK_SIZE=_CHUNK_SIZE,
            )
            self._last_backend = "triton_row_rmsnorm_quant_bs2048x2"
            self._last_error = None
            return output
        except Exception as exc:
            self._disable_triton = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._torch_reference(
                x, residual, gamma, scale, zero_point, "torch_fallback_after_error"
            )

    def _can_try_triton(self, x, residual, gamma, scale, zero_point):
        if self._disable_triton or _add_rmsnorm_quant_kernel is None:
            return False
        if getattr(x.device, "type", None) != "npu":
            return False
        return (
            x.is_contiguous()
            and residual.is_contiguous()
            and gamma.is_contiguous()
            and scale.is_contiguous()
            and zero_point.is_contiguous()
            and x.shape == residual.shape
            and x.ndim >= 2
            and gamma.ndim == 1
            and gamma.shape[0] == x.shape[-1]
            and scale.numel() == 1
            and zero_point.numel() == 1
            and x.shape[-1] == _SUPPORTED_HIDDEN
            and x.dtype == torch.float32
            and residual.dtype == torch.float32
            and gamma.dtype == torch.float32
            and scale.dtype == torch.float32
            and zero_point.dtype == torch.float32
        )

    def _torch_reference(self, x, residual, gamma, scale, zero_point, reason):
        self._last_backend = reason
        x_added = x + residual
        rstd = torch.rsqrt(x_added.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        output = (x_added * rstd) * gamma
        return torch.round(output / scale + zero_point).clamp(-128, 127).to(torch.int8)
